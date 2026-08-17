# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPU tests for Inkling's paged-KV prefill attention.

``inkling_prefill_attention`` reads K/V from the paged cache, so one code path
serves a fresh context and a later chunk of a chunked prefill. Three properties
are worth separating, because they fail for different reasons:

1. **Correctness against an independent oracle.** The oracle here is a dense
   fp32 attention written straight from the definition, not the previous packed
   kernel. The packed kernel is gone, and using a Triton kernel to check a
   Triton kernel would share any misreading of the causal / window / bias
   contract between them.
2. **Chunk invariance.** Splitting a prompt must not change the answer for the
   tokens after the split. This is the property the feature exists to provide.
3. **Page fidelity.** The kernel must honour the page table it is handed. With
   ``block_ids = range(n)`` a kernel that ignored the table entirely and
   treated pages as contiguous would pass everything else in this file, so the
   realistic-layout tests carry explicit negative controls.

Everything runs over the local (sliding-window) and global (full-causal) layer
shapes and with the relative-position bias on and off: the bias indexing is the
part most likely to be wrong across a chunk boundary, because it is the one
quantity indexed by the LOCAL query row while its distance is GLOBAL.
"""

import pytest
import torch

pytest.importorskip("triton")

from tensorrt_llm._torch.attention_backend.inkling.kernels import (  # noqa: E402
    build_page_table,
    inkling_prefill_attention,
    write_kv_cache_hnd,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Inkling's text tower: head_dim 128, 64 q heads / 8 kv heads global, 16 local.
HEAD_DIM = 128
NUM_HEADS = 8
# 32 is ``tokens_per_block``'s default and therefore the page size every recorded
# Inkling accuracy run used. It is also smaller than the kernel's untuned
# BLOCK_N of 64, which is exactly the case that forces BLOCK_N to follow the
# page size -- see the wrapper.
PAGE_SIZE = 32
DTYPE = torch.bfloat16
# The real model uses 512; a smaller window keeps the tests fast while still
# forcing the kernel's whole-tile skip path (lo > 0) to run.
WINDOW = 96
REL_EXTENT = 8
SM_SCALE = HEAD_DIM**-1.0


def _rand(*shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=DTYPE)


def _make_case(total_len, num_kv_heads, *, has_rel, seed=0):
    q = _rand(total_len, NUM_HEADS, HEAD_DIM, seed=seed)
    k = _rand(total_len, num_kv_heads, HEAD_DIM, seed=seed + 1)
    v = _rand(total_len, num_kv_heads, HEAD_DIM, seed=seed + 2)
    rel = None
    if has_rel:
        rel = _rand(total_len, NUM_HEADS, REL_EXTENT, seed=seed + 3).float().contiguous()
    return q, k, v, rel


def _empty_cache(num_pages, num_kv_heads, page_size=PAGE_SIZE):
    shape = (num_pages, num_kv_heads, page_size, HEAD_DIM)
    return (
        torch.zeros(shape, device="cuda", dtype=DTYPE),
        torch.zeros(shape, device="cuda", dtype=DTYPE),
    )


def _reference(q, k_all, v_all, rel, window, q_offset):
    """Dense fp32 attention, written from the definition.

    Args:
        q: ``[n_q, H, D]`` queries, sitting at global positions
            ``q_offset + [0, n_q)``.
        k_all, v_all: ``[total, Hkv, D]`` -- every key/value of the prompt,
            including the part the kernel would read from the cached prefix.
        rel: ``[n_q, H, REL_EXTENT]`` fp32 indexed by the LOCAL query row, or
            None.
        window: sliding-window radius (inclusive), -1 to disable.

    fp32 throughout and no tiling, so it shares no arithmetic with the kernel:
    a shared bug would have to be a shared misreading of the contract, which is
    what the explicit position algebra below is here to prevent.
    """
    n_q = q.shape[0]
    total = k_all.shape[0]
    num_kv_heads = k_all.shape[1]
    group = NUM_HEADS // num_kv_heads

    qf = q.float()
    kf = k_all.float().repeat_interleave(group, dim=1)  # [total, H, D]
    vf = v_all.float().repeat_interleave(group, dim=1)

    # [H, n_q, total]
    scores = torch.einsum("ihd,jhd->hij", qf, kf) * SM_SCALE

    qpos = torch.arange(n_q, device=q.device) + q_offset  # global
    kpos = torch.arange(total, device=q.device)  # global
    dist = qpos[:, None] - kpos[None, :]  # [n_q, total]

    if rel is not None:
        idx = dist.clamp(0, REL_EXTENT - 1)
        # rel is indexed by the LOCAL row, its distance by the GLOBAL position.
        bias = torch.gather(rel.permute(1, 0, 2), 2, idx.unsqueeze(0).expand(NUM_HEADS, -1, -1))
        scores = scores + torch.where(
            ((dist >= 0) & (dist < REL_EXTENT)).unsqueeze(0), bias, torch.zeros_like(bias)
        )

    allowed = dist >= 0
    if window >= 0:
        allowed = allowed & (dist <= window)
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("hij,jhd->ihd", p, vf)  # [n_q, H, D]
    return out


def _run_chunk(q, k, v, rel, num_cached, k_cache, v_cache, block_ids, window, page_size=PAGE_SIZE):
    """Write this chunk's K/V into the pages, then run the prefill kernel.

    Mirrors ``InklingTritonAttention._run_context``: the write happens first, so
    by the time the kernel launches the prefix and the new tokens are both in
    the pages and the kernel reads all of them the same way.
    """
    new_len = q.shape[0]
    write_kv_cache_hnd(k_cache, v_cache, k, v, block_ids, num_cached, page_size)
    cu = torch.tensor([0, new_len], dtype=torch.int32, device="cuda")
    nc = torch.tensor([num_cached], dtype=torch.int32, device="cuda")
    total = num_cached + new_len
    max_pages = (total + page_size - 1) // page_size
    # build_page_table writes len(blocks) entries into a row of width
    # max_pages, so hand it exactly the pages this request spans.
    page_table = build_page_table([block_ids[:max_pages]], max_pages, "cuda")
    return inkling_prefill_attention(
        q,
        k_cache,
        v_cache,
        cu,
        nc,
        page_table,
        page_size,
        new_len,
        SM_SCALE,
        rel,
        REL_EXTENT if rel is not None else 0,
        window,
    )


def _assert_close(got, want, what, rtol=2e-2, atol=2e-2, min_cos=0.999):
    got_f, want_f = got.float(), want.float()
    max_abs = (got_f - want_f).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(got_f.flatten(), want_f.flatten(), dim=0).item()
    assert torch.allclose(got_f, want_f, rtol=rtol, atol=atol), (
        f"{what}: max_abs={max_abs:.4g} cos={cos:.6f}"
    )
    assert cos > min_cos, f"{what}: cos={cos:.6f}"


# ---------------------------------------------------------------------------
# 1. Correctness against the dense oracle.
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("window", [-1, WINDOW], ids=["global", "local"])
@pytest.mark.parametrize("has_rel", [False, True], ids=["norel", "rel"])
@pytest.mark.parametrize("total_len", [1, 31, 32, 63, 64, 65, 200])
def test_a_fresh_context_matches_the_dense_oracle(window, has_rel, total_len):
    """``num_cached == 0`` is the path every recorded accuracy number used.

    Lengths straddle both the page size (32) and the query tile (64), because
    the masked tail of a partial tile is where an off-by-one in the causal or
    window bound hides.
    """
    num_kv_heads = 8
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=has_rel)
    k_cache, v_cache = _empty_cache(16, num_kv_heads)

    got = _run_chunk(q, k, v, rel, 0, k_cache, v_cache, list(range(16)), window)
    want = _reference(q, k, v, rel, window, 0)
    _assert_close(got, want, f"len={total_len} window={window} rel={has_rel}")


@requires_gpu
@pytest.mark.parametrize("window", [-1, WINDOW], ids=["global", "local"])
@pytest.mark.parametrize("has_rel", [False, True], ids=["norel", "rel"])
@pytest.mark.parametrize("num_cached", [1, 31, 32, 33, 64, 100])
def test_a_cached_chunk_matches_the_dense_oracle(window, has_rel, num_cached):
    """The chunked case against the oracle directly, rather than only against
    an unsplit run of the same kernel.

    A same-kernel comparison cannot see an error that is symmetric in the two
    runs -- a wrong-but-consistent global position, say. The oracle can.
    ``num_cached`` values that are not multiples of the page size are the point:
    they put the chunk boundary inside a page.
    """
    total_len = 160
    num_kv_heads = 8
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=has_rel, seed=5)
    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    blocks = list(range(16))

    _run_chunk(
        q[:num_cached],
        k[:num_cached],
        v[:num_cached],
        None if rel is None else rel[:num_cached].contiguous(),
        0,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    got = _run_chunk(
        q[num_cached:],
        k[num_cached:],
        v[num_cached:],
        None if rel is None else rel[num_cached:].contiguous(),
        num_cached,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    want = _reference(
        q[num_cached:], k, v, None if rel is None else rel[num_cached:], window, num_cached
    )
    _assert_close(got, want, f"cached={num_cached} window={window} rel={has_rel}")


# ---------------------------------------------------------------------------
# 2. Chunk invariance.
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("window", [-1, WINDOW], ids=["global", "local"])
@pytest.mark.parametrize("has_rel", [False, True], ids=["norel", "rel"])
@pytest.mark.parametrize("split", [1, 2, 7, 32, 64, 65, 128])
def test_a_split_prompt_matches_one_shot_prefill(window, has_rel, split):
    """Splitting at ``split`` must not change the answer after the split.

    ``total_len`` 200 with WINDOW 96 and REL_EXTENT 8 puts splits on both sides
    of the window edge and of the bias extent; 32 is a page boundary and 64/65
    straddle the query tile.
    """
    total_len = 200
    num_kv_heads = 8
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=has_rel, seed=11)

    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    blocks = list(range(16))
    want = _run_chunk(q, k, v, rel, 0, k_cache, v_cache, blocks, window)[split:]

    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    # Chunk 1 seeds the cache; its own output is not under test here.
    _run_chunk(
        q[:split],
        k[:split],
        v[:split],
        None if rel is None else rel[:split].contiguous(),
        0,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    got = _run_chunk(
        q[split:],
        k[split:],
        v[split:],
        None if rel is None else rel[split:].contiguous(),
        split,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    _assert_close(got, want, f"split={split} window={window} rel={has_rel}")


@requires_gpu
def test_three_chunks_are_the_same_as_one():
    """More than one boundary, and boundaries that are not page-aligned: the
    page table has to be walked, not just offset once."""
    total_len, num_kv_heads, window = 200, 8, WINDOW
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=23)
    want = _reference(q, k, v, rel, window, 0)

    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    blocks = list(range(16))
    outs, lo = [], 0
    for hi in (37, 100, total_len):  # 37 and 100 are not multiples of PAGE_SIZE
        outs.append(
            _run_chunk(
                q[lo:hi],
                k[lo:hi],
                v[lo:hi],
                rel[lo:hi].contiguous(),
                lo,
                k_cache,
                v_cache,
                blocks,
                window,
            )
        )
        lo = hi
    _assert_close(torch.cat(outs, dim=0), want, "three chunks")


@requires_gpu
def test_grouped_query_attention_maps_heads_correctly():
    """Inkling's local layers carry 16 KV heads against 64 query heads; the
    kernel must fold q->kv as ``cur_head // kv_group_num`` or the split output
    is wrong in a way parity at ``kv_group_num == 1`` cannot see."""
    total_len, num_kv_heads, split, window = 128, 2, 33, -1
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=31)
    want = _reference(q[split:], k, v, rel[split:], window, split)

    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    blocks = list(range(16))
    _run_chunk(
        q[:split],
        k[:split],
        v[:split],
        rel[:split].contiguous(),
        0,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    got = _run_chunk(
        q[split:],
        k[split:],
        v[split:],
        rel[split:].contiguous(),
        split,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    _assert_close(got, want, "gqa split")


# ---------------------------------------------------------------------------
# 3. Page fidelity, and the page-size constraint the tiling imposes.
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("num_cached", [17, 32, 70])
def test_page_sizes_and_unaligned_boundaries(page_size, num_cached):
    """BLOCK_N follows the page size so a key tile stays inside one page. That
    makes ``page_size`` a correctness parameter, not a tuning one: 32 is below
    the kernel's natural BLOCK_N of 64 and forces the narrow tile, 128 runs
    several tiles per page. ``num_cached`` values that are not multiples of any
    of them put the chunk boundary mid-page."""
    total_len, num_kv_heads, window = 200, 8, WINDOW
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=53)
    num_pages = (total_len + page_size - 1) // page_size + 2
    k_cache, v_cache = _empty_cache(num_pages, num_kv_heads, page_size)
    blocks = list(range(num_pages))

    _run_chunk(
        q[:num_cached],
        k[:num_cached],
        v[:num_cached],
        rel[:num_cached].contiguous(),
        0,
        k_cache,
        v_cache,
        blocks,
        window,
        page_size,
    )
    got = _run_chunk(
        q[num_cached:],
        k[num_cached:],
        v[num_cached:],
        rel[num_cached:].contiguous(),
        num_cached,
        k_cache,
        v_cache,
        blocks,
        window,
        page_size,
    )
    want = _reference(q[num_cached:], k, v, rel[num_cached:], window, num_cached)
    _assert_close(got, want, f"page_size={page_size} cached={num_cached}")


@requires_gpu
@pytest.mark.parametrize("page_size", [48, 96])
def test_a_page_size_the_tiling_cannot_serve_is_refused(page_size):
    """A page the key tile cannot divide would make the scalar page id wrong for
    part of the tile -- wrong ADDRESSES, not wrong numbers, so nothing
    downstream would flag it. It must fail loudly at the wrapper instead.

    Two shapes, because they fail for different reasons and an assert that
    catches only one is worse than none: 96 breaks divisibility against a
    BLOCK_N of 64, while 48 divides itself cleanly and instead breaks
    ``tl.arange``'s power-of-two requirement. The first version of this assert
    checked only divisibility, so 48 sailed past it and died inside Triton
    naming neither the page size nor the setting behind it.
    """
    num_kv_heads = 8
    q, k, v, _ = _make_case(64, num_kv_heads, has_rel=False, seed=57)
    k_cache, v_cache = _empty_cache(8, num_kv_heads, page_size)
    with pytest.raises(AssertionError, match="tokens_per_block"):
        _run_chunk(q, k, v, None, 0, k_cache, v_cache, list(range(8)), -1, page_size)


@requires_gpu
@pytest.mark.parametrize(
    "blocks",
    [
        [9, 2, 14, 5, 11, 0, 7, 3],  # unordered
        [31, 30, 29, 28, 27, 26, 25, 24],  # high, descending, never touches 0
        [4, 12, 20, 28, 36, 44, 52, 60],  # strided, sparse
    ],
    ids=["unordered", "high_descending", "strided"],
)
def test_chunk_invariance_holds_on_a_realistic_page_layout(blocks):
    """Every other test here hands the kernel ``range(n)``: contiguous,
    zero-based, ascending. A real KVCacheManagerV2 hands out whatever is free --
    sparse, unordered, and never starting at 0 in a warm pool.

    ``write_kv_cache_hnd`` and the kernel must agree on the SAME mapping from
    absolute position to (page, offset). A disagreement shows up here and
    nowhere else in this file.
    """
    total_len, num_kv_heads, split, window = 200, 8, 37, WINDOW
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=71)
    want = _reference(q[split:], k, v, rel[split:], window, split)

    k_cache, v_cache = _empty_cache(64, num_kv_heads)
    _run_chunk(
        q[:split],
        k[:split],
        v[:split],
        rel[:split].contiguous(),
        0,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    got = _run_chunk(
        q[split:],
        k[split:],
        v[split:],
        rel[split:].contiguous(),
        split,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    _assert_close(got, want, f"pages={blocks[:4]}...")


@requires_gpu
def test_a_contiguous_page_assumption_would_be_caught():
    """Negative control for the test above: writing with one layout and reading
    with another must NOT agree. If it does, the kernel is ignoring the page
    table and the test above proves nothing."""
    total_len, num_kv_heads, split, window = 200, 8, 37, -1
    q, k, v, _ = _make_case(total_len, num_kv_heads, has_rel=False, seed=73)
    k_cache, v_cache = _empty_cache(64, num_kv_heads)
    write_blocks = [9, 2, 14, 5, 11, 0, 7, 3]

    _run_chunk(q[:split], k[:split], v[:split], None, 0, k_cache, v_cache, write_blocks, window)
    honest = _run_chunk(
        q[split:], k[split:], v[split:], None, split, k_cache, v_cache, write_blocks, window
    )
    # Same cache, but the reader is told a different page order.
    cu = torch.tensor([0, total_len - split], dtype=torch.int32, device="cuda")
    nc = torch.tensor([split], dtype=torch.int32, device="cuda")
    max_pages = (total_len + PAGE_SIZE - 1) // PAGE_SIZE
    wrong_table = build_page_table([list(range(max_pages))], max_pages, "cuda")
    wrong = inkling_prefill_attention(
        q[split:],
        k_cache,
        v_cache,
        cu,
        nc,
        wrong_table,
        PAGE_SIZE,
        total_len - split,
        SM_SCALE,
        None,
        0,
        window,
    )
    assert not torch.allclose(honest.float(), wrong.float(), rtol=2e-2, atol=2e-2), (
        "reading with the wrong page order changed nothing -- the kernel is not "
        "using the page table"
    )


@requires_gpu
def test_ignoring_the_cached_prefix_would_be_caught():
    """Negative control for chunk invariance: the old behaviour -- attending
    only to the chunk's own tokens -- must be visibly different. Without this,
    a passing invariance test could mean the prefix simply does not matter for
    this input."""
    total_len, num_kv_heads, split, window = 200, 8, 37, -1
    q, k, v, _ = _make_case(total_len, num_kv_heads, has_rel=False, seed=79)
    with_prefix = _reference(q[split:], k, v, None, window, split)
    without_prefix = _reference(q[split:], k[split:], v[split:], None, window, 0)
    assert not torch.allclose(with_prefix.float(), without_prefix.float(), rtol=2e-2, atol=2e-2), (
        "the cached prefix makes no difference on this input -- pick a harder one"
    )


# ---------------------------------------------------------------------------
# 4. The conv half of the same property. Attention is only one of the two
# places a split prompt can lose its history; the four depthwise short convs
# are the other. causal_conv1d_fn writes the trailing kernel-1 window into the
# state pool on every context call, so a later chunk only has to DECLARE that
# it has one (has_initial_state) for it to be consumed.
# ---------------------------------------------------------------------------
SCONV_KERNEL = 4
SCONV_CHANNELS = 256


def _conv_state_pool(rows=1):
    return torch.zeros(rows, SCONV_CHANNELS, SCONV_KERNEL - 1, device="cuda", dtype=DTYPE)


def _run_conv_chunk(x, conv_w, state, has_initial):
    """One varlen context call through causal_conv1d_fn, state updated in place."""
    from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn

    n = x.shape[0]
    xt = x.transpose(0, 1).contiguous()  # [channels, tokens]
    y = causal_conv1d_fn(
        xt,
        conv_w,
        None,
        query_start_loc=torch.tensor([0, n], dtype=torch.int32, device="cuda"),
        cache_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        has_initial_state=torch.tensor([has_initial], dtype=torch.bool, device="cuda"),
        conv_states=state,
        activation=None,
    )
    return y.transpose(0, 1).contiguous()


@requires_gpu
@pytest.mark.parametrize("split", [1, 2, 3, 4, 17, 64])
def test_the_short_conv_carries_its_window_across_a_chunk_boundary(split):
    """Splits at 1..3 are the interesting ones: shorter than the kernel-1
    window, so the second chunk's first outputs depend on tokens the first chunk
    owned. With has_initial_state=False these were convolved against zeros."""
    total_len = 128
    x = _rand(total_len, SCONV_CHANNELS, seed=41)
    conv_w = _rand(SCONV_CHANNELS, SCONV_KERNEL, seed=42)

    state = _conv_state_pool()
    want = _run_conv_chunk(x, conv_w, state, False)

    state = _conv_state_pool()
    _run_conv_chunk(x[:split], conv_w, state, False)
    got = _run_conv_chunk(x[split:], conv_w, state, True)
    _assert_close(got, want[split:], f"conv split={split}")


@requires_gpu
def test_declaring_no_initial_state_on_a_later_chunk_is_visibly_wrong():
    """The negative control: without has_initial_state the second chunk really
    does differ. If this ever passes, the test above proves nothing."""
    total_len, split = 128, 2
    x = _rand(total_len, SCONV_CHANNELS, seed=43)
    conv_w = _rand(SCONV_CHANNELS, SCONV_KERNEL, seed=44)

    state = _conv_state_pool()
    want = _run_conv_chunk(x, conv_w, state, False)[split:]

    state = _conv_state_pool()
    _run_conv_chunk(x[:split], conv_w, state, False)
    wrong = _run_conv_chunk(x[split:], conv_w, state, False)  # the old behaviour

    assert not torch.allclose(wrong.float(), want.float(), rtol=2e-2, atol=2e-2), (
        "dropping the carried window changed nothing -- the parity test above "
        "is not exercising what it claims"
    )


# ---------------------------------------------------------------------------
# 5. Measure, do not only bound.
# ---------------------------------------------------------------------------
@requires_gpu
def test_report_the_split_divergence_magnitude():
    """The tests above bound the split-vs-unsplit difference at 2e-2 and stop.

    A correct kernel still re-aligns query tiles at the chunk boundary, so the
    online-softmax accumulation order differs and bf16 rounding compounds. That
    explanation is only credible if the per-call difference is actually
    rounding-sized, so this test prints it next to the ``num_cached == 0``
    floor, where the two runs read the same keys in the same tile order.

    Recorded here because an earlier attempt at this feature saw the head of the
    end-to-end logprob distribution move ~0.35 (jobs 6046462 / 6047277) and had
    no per-layer number to compare it against.
    """
    total_len, num_kv_heads, window = 200, 8, WINDOW
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=91)
    oracle = _reference(q, k, v, rel, window, 0)

    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    fresh = _run_chunk(q, k, v, rel, 0, k_cache, v_cache, list(range(16)), window)
    f_abs = (fresh.float() - oracle.float()).abs().max().item()
    print(f"\n  num_cached==0 vs oracle : max_abs={f_abs:.3e}")

    for split in (37, 64, 100):
        k_cache, v_cache = _empty_cache(16, num_kv_heads)
        blocks = list(range(16))
        _run_chunk(
            q[:split],
            k[:split],
            v[:split],
            rel[:split].contiguous(),
            0,
            k_cache,
            v_cache,
            blocks,
            window,
        )
        got = _run_chunk(
            q[split:],
            k[split:],
            v[split:],
            rel[split:].contiguous(),
            split,
            k_cache,
            v_cache,
            blocks,
            window,
        )
        tail = oracle[split:].float()
        max_abs = (got.float() - tail).abs().max().item()
        rel_rms = ((got.float() - tail).pow(2).mean().sqrt() / tail.pow(2).mean().sqrt()).item()
        cos = torch.nn.functional.cosine_similarity(
            got.float().flatten(), tail.flatten(), dim=0
        ).item()
        print(
            f"  split={split:4d} vs oracle  : max_abs={max_abs:.3e} "
            f"rel_rms={rel_rms:.3e} cos={cos:.8f}"
        )
        assert cos > 0.999, f"split={split} cos={cos}"

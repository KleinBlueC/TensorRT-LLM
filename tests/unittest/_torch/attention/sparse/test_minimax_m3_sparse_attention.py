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
"""MiniMax-M3 Triton sparse-attention (MSA) kernel unit tests.

The golden references below reimplement the MiniMax-M3 SGLang sparse ops
semantics (``minimax_sparse_ops/decode/{flash_with_topk_idx,topk_sparse}.py`` and
their PyTorch references) so the tests do not depend on an installed SGLang or on
a newer ``transformers`` than the repo pins. The kernels under test live in
``tensorrt_llm/_torch/attention_backend/sparse/minimax_m3_kernels.py``.

Coverage:
* ``test_index_topk_decode``  -- top-k block selection matches the reference set,
  across GQA ratios, seq patterns, init/local retention and topk sizes.
* ``test_sparse_gqa_decode``  -- sparse GQA output matches attending only to the
  selected blocks.
* ``test_sparse_attention_decode`` -- the full index->select->attend pipeline.
* ``test_long_pruned_block_drop`` -- >=4096-token context proves the MSA
  drop regime (eligible non-local/non-init blocks are actually dropped) and the
  pruned output still matches the reference.
* ``test_m3_config_shapes`` -- checkpoint-scale head counts/dims (64 q / 4 kv /
  4 index heads, head_dim 128, index_dim 128, block 128, topk 16).
* ``test_cuda_graph_capture_replay`` -- the sparse decode path is CUDA-graph
  hard-path safe (capture + replay reproduces eager output on two inputs).
"""

import pytest
import torch

triton = pytest.importorskip("triton")

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels import (  # noqa: E402
    minimax_m3_index_topk,
    minimax_m3_sparse_attention_decode,
    minimax_m3_sparse_attention_prefill,
    minimax_m3_sparse_gqa,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MiniMax-M3 sparse kernels require CUDA"
)

DEVICE = "cuda"
RTOL = 5e-3
ATOL = 5e-3


# ---------------------------------------------------------------------------
# Golden references (MiniMax-M3 SGLang semantics, reimplemented locally)
# ---------------------------------------------------------------------------
def ref_index_topk_decode(
    index_q,
    index_k_cache,
    req_to_token,
    seq_lens,
    slot_ids,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    idx_sm_scale=None,
):
    """Reference top-k block selection for the index branch (decode)."""
    batch, num_index_heads, index_dim = index_q.shape
    if idx_sm_scale is None:
        idx_sm_scale = index_dim**-0.5
    max_sl = int(seq_lens.max().item())
    rows = req_to_token[slot_ids.long()][:, :max_sl].long()  # [batch, max_sl]
    k = index_k_cache[rows, 0, :].float()  # [batch, max_sl, index_dim]
    qk = torch.einsum("bhd,bnd->bhn", index_q.float(), k) * idx_sm_scale
    seq_mask = torch.arange(max_sl, device=index_q.device)[None, :] < seq_lens[:, None]
    qk = qk.masked_fill(~seq_mask[:, None, :], float("-inf"))

    max_num_blocks = (max_sl + block_size - 1) // block_size
    padded = max_num_blocks * block_size
    qk_padded = torch.full((batch, num_index_heads, padded), float("-inf"), device=index_q.device)
    qk_padded[:, :, :max_sl] = qk
    block_scores = (
        qk_padded.reshape(batch, num_index_heads, max_num_blocks, block_size).max(dim=-1).values
    )

    topk_idx = torch.full(
        (num_index_heads, batch, topk), -1, dtype=torch.int32, device=index_q.device
    )
    for b in range(batch):
        num_blocks = (int(seq_lens[b].item()) + block_size - 1) // block_size
        bs_b = block_scores[b, :, :num_blocks].clone()
        if init_blocks > 0:
            bs_b[:, :init_blocks] = 1e30
        if local_blocks > 0:
            local_start = max(0, num_blocks - local_blocks)
            bs_b[:, local_start:num_blocks] = 1e29
        actual_topk = min(topk, num_blocks)
        _, tidx = bs_b.topk(actual_topk, dim=-1)
        topk_idx[:, b, :actual_topk] = tidx.to(torch.int32)
    return topk_idx


def ref_sparse_gqa_decode(
    q, k_cache, v_cache, req_to_token, seq_lens, slot_ids, block_size, topk_idx, sm_scale=None
):
    """Reference sparse GQA (decode): attend only to the selected blocks."""
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    gqa_group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    o = torch.zeros(batch, num_q_heads, head_dim, dtype=q.dtype, device=q.device)
    for b in range(batch):
        seq_len = int(seq_lens[b].item())
        row = req_to_token[int(slot_ids[b].item())]
        for kh in range(num_kv_heads):
            slots = []
            for blk in topk_idx[kh, b, :].tolist():
                if blk < 0:
                    continue
                start = blk * block_size
                end = min(start + block_size, seq_len)
                if start >= seq_len:
                    continue
                slots.append(row[start:end].long())
            if not slots:
                continue
            sel = torch.cat(slots, dim=0)
            k_sel = k_cache[sel, kh, :].float()  # [n, hd]
            v_sel = v_cache[sel, kh, :].float()
            for g in range(gqa_group_size):
                qh = kh * gqa_group_size + g
                q_vec = q[b, qh, :].float()
                scores = torch.matmul(q_vec, k_sel.T) * sm_scale
                attn = torch.softmax(scores, dim=-1)
                o[b, qh, :] = torch.matmul(attn.to(q.dtype), v_sel.to(q.dtype))
    return o


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------
def build_decode_inputs(
    batch,
    num_q_heads,
    num_kv_heads,
    num_index_heads,
    head_dim,
    index_dim,
    seq_lens_list,
    paged=True,
    seed=42,
    dtype=torch.bfloat16,
):
    torch.manual_seed(seed)
    max_kv_len = max(seq_lens_list)
    max_slots = batch * max_kv_len
    q = torch.randn(batch, num_q_heads, head_dim, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    index_q = torch.randn(batch, num_index_heads, index_dim, dtype=dtype, device=DEVICE)
    index_k_cache = torch.randn(max_slots, 1, index_dim, dtype=dtype, device=DEVICE)
    req_to_token = torch.zeros(batch, max_kv_len, dtype=torch.int32, device=DEVICE)
    slot_ids = torch.arange(batch, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    for i in range(batch):
        base = i * max_kv_len
        if paged:
            req_to_token[i, :max_kv_len] = (torch.randperm(max_kv_len, device=DEVICE) + base).to(
                torch.int32
            )
        else:
            req_to_token[i, :max_kv_len] = torch.arange(base, base + max_kv_len, device=DEVICE).to(
                torch.int32
            )
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        index_q=index_q,
        index_k_cache=index_k_cache,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
    )


def _assert_topk_sets_match(topk_kernel, topk_ref, seq_lens, block_size, topk):
    num_index_heads, batch, _ = topk_kernel.shape
    for h in range(num_index_heads):
        for b in range(batch):
            num_blocks = (int(seq_lens[b].item()) + block_size - 1) // block_size
            actual_k = min(topk, num_blocks)
            set_kernel = set(topk_kernel[h, b, :actual_k].tolist())
            set_ref = set(topk_ref[h, b, :actual_k].tolist())
            assert set_kernel == set_ref, (
                f"topk mismatch h={h} b={b}: kernel={sorted(set_kernel)} ref={sorted(set_ref)}"
            )
            # invalid tail must be sentinel -1
            if actual_k < topk:
                tail = topk_kernel[h, b, actual_k:]
                assert (tail == -1).all(), f"sentinel fail h={h} b={b}: {tail}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _case(bs, nqh, nkh, nih, hd, idxd, blk, tk, ib, lb, seq_pat):
    return pytest.param(
        bs,
        nqh,
        nkh,
        nih,
        hd,
        idxd,
        blk,
        tk,
        ib,
        lb,
        seq_pat,
        id=f"bs{bs}_q{nqh}kv{nkh}ix{nih}_hd{hd}ixd{idxd}_blk{blk}_tk{tk}"
        f"_init{ib}_local{lb}_{seq_pat}",
    )


def _seq_lens(pattern, batch, block_size):
    if pattern == "aligned":
        return [block_size * 8] * batch
    if pattern == "unaligned":
        base = [513, 1023, 257, 769]
        return (base * ((batch + 3) // 4))[:batch]
    if pattern == "short":  # topk >= num_blocks -> dense-equivalent
        return [block_size * 2] * batch
    if pattern == "mixed":
        base = [block_size, block_size * 6, block_size * 20, block_size * 3]
        return (base * ((batch + 3) // 4))[:batch]
    raise ValueError(pattern)


# num_index_heads == num_kv_heads (MiniMax-M3 contract for the full pipeline);
# the standalone GQA kernel is additionally exercised with wider GQA ratios.
INDEX_CASES = [
    _case(2, 8, 4, 4, 128, 128, 128, 16, 0, 1, "aligned"),
    _case(4, 8, 4, 4, 128, 128, 128, 16, 0, 1, "unaligned"),
    _case(2, 8, 4, 4, 128, 128, 128, 16, 0, 1, "short"),
    _case(4, 16, 4, 4, 128, 128, 128, 16, 2, 1, "unaligned"),
    _case(3, 8, 4, 4, 128, 128, 128, 8, 0, 2, "mixed"),
    _case(2, 8, 2, 2, 128, 128, 64, 16, 0, 1, "aligned"),
]


@pytest.mark.parametrize("bs,nqh,nkh,nih,hd,idxd,blk,tk,ib,lb,seq_pat", INDEX_CASES)
def test_index_topk_decode(bs, nqh, nkh, nih, hd, idxd, blk, tk, ib, lb, seq_pat):
    """Top-k selected block ids match the reference set."""
    inp = build_decode_inputs(bs, nqh, nkh, nih, hd, idxd, _seq_lens(seq_pat, bs, blk))
    topk_kernel = minimax_m3_index_topk(
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        tk,
        init_blocks=ib,
        local_blocks=lb,
    )
    topk_ref = ref_index_topk_decode(
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        tk,
        ib,
        lb,
    )
    _assert_topk_sets_match(topk_kernel, topk_ref, inp["seq_lens"], blk, tk)


# The sparse GQA kernel is backend-shared, so cover wider GQA ratios / head dims.
GQA_CASES = [
    _case(2, 8, 1, 1, 128, 128, 128, 16, 0, 0, "aligned"),
    _case(4, 8, 1, 1, 128, 128, 128, 16, 0, 0, "unaligned"),
    _case(2, 32, 8, 8, 128, 128, 128, 16, 0, 0, "aligned"),
    _case(2, 64, 4, 4, 128, 128, 128, 16, 0, 0, "aligned"),
    _case(2, 8, 4, 4, 64, 64, 64, 16, 0, 0, "aligned"),
    _case(2, 8, 1, 1, 128, 128, 128, 1, 0, 0, "aligned"),
]


@pytest.mark.parametrize("bs,nqh,nkh,nih,hd,idxd,blk,tk,ib,lb,seq_pat", GQA_CASES)
def test_sparse_gqa_decode(bs, nqh, nkh, nih, hd, idxd, blk, tk, ib, lb, seq_pat):
    """Sparse GQA output matches attending to exactly the selected blocks."""
    inp = build_decode_inputs(bs, nqh, nkh, nih, hd, idxd, _seq_lens(seq_pat, bs, blk))
    seq_lens = inp["seq_lens"]
    # random valid topk selection per kv head/batch (left-packed, -1 padded)
    num_blocks = [(int(s.item()) + blk - 1) // blk for s in seq_lens]
    topk_idx = torch.full((nkh, bs, tk), -1, dtype=torch.int32, device=DEVICE)
    for kh in range(nkh):
        for b in range(bs):
            ak = min(tk, num_blocks[b])
            perm = torch.randperm(num_blocks[b], device=DEVICE)[:ak]
            topk_idx[kh, b, :ak] = perm.to(torch.int32)

    o_kernel = minimax_m3_sparse_gqa(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["req_to_token"],
        seq_lens,
        inp["slot_ids"],
        blk,
        topk_idx,
    )
    o_ref = ref_sparse_gqa_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["req_to_token"],
        seq_lens,
        inp["slot_ids"],
        blk,
        topk_idx,
    ).to(o_kernel.dtype)
    max_diff = (o_kernel.float() - o_ref.float()).abs().max().item()
    assert torch.allclose(o_kernel.float(), o_ref.float(), rtol=RTOL, atol=ATOL), (
        f"max abs diff {max_diff:.4e}"
    )


@pytest.mark.parametrize("bs,nqh,nkh,nih,hd,idxd,blk,tk,ib,lb,seq_pat", INDEX_CASES)
def test_sparse_attention_decode(bs, nqh, nkh, nih, hd, idxd, blk, tk, ib, lb, seq_pat):
    """Full index->select->attend pipeline vs the reference on the same set."""
    inp = build_decode_inputs(bs, nqh, nkh, nih, hd, idxd, _seq_lens(seq_pat, bs, blk))
    o_kernel, topk_kernel = minimax_m3_sparse_attention_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        tk,
        init_blocks=ib,
        local_blocks=lb,
    )
    # Validate the selection matches the reference, then validate the GQA output
    # given the kernel's own selection (robust to score near-ties).
    topk_ref = ref_index_topk_decode(
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        tk,
        ib,
        lb,
    )
    _assert_topk_sets_match(topk_kernel, topk_ref, inp["seq_lens"], blk, tk)
    o_ref = ref_sparse_gqa_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        topk_kernel,
    ).to(o_kernel.dtype)
    max_diff = (o_kernel.float() - o_ref.float()).abs().max().item()
    assert torch.allclose(o_kernel.float(), o_ref.float(), rtol=RTOL, atol=ATOL), (
        f"max abs diff {max_diff:.4e}"
    )


def test_long_pruned_block_drop():
    """>=4096-token context: MSA must actually drop eligible blocks.

    With topk=16, block_size=128 and a 4096-token context there are 32 KV
    blocks, so 16 blocks are dropped per query. Assert at least one dropped block
    is neither an init nor a local block (the defining MSA regime, not the
    dense-equivalent all-block case), and that the pruned output matches the
    reference restricted to the same selection.
    """
    bs, nqh, nkh, nih = 2, 64, 4, 4
    hd = idxd = 128
    blk, tk, ib, lb = 128, 16, 0, 1
    seq_len = 4096
    num_blocks = seq_len // blk  # 32
    assert num_blocks > tk, "test must exercise the block-drop regime"

    inp = build_decode_inputs(bs, nqh, nkh, nih, hd, idxd, [seq_len] * bs)
    o_kernel, topk_kernel = minimax_m3_sparse_attention_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        tk,
        init_blocks=ib,
        local_blocks=lb,
    )

    # Exactly topk blocks are kept, so 32-16 are dropped; at least one dropped
    # block must be a normal (non-init, non-local) block.
    for h in range(nkh):
        for b in range(bs):
            selected = set(topk_kernel[h, b].tolist()) - {-1}
            assert len(selected) == tk, f"expected {tk} selected blocks, got {len(selected)}"
            all_blocks = set(range(num_blocks))
            dropped = all_blocks - selected
            eligible = all_blocks - set(range(ib)) - {num_blocks - 1}
            dropped_eligible = dropped & eligible
            assert dropped_eligible, (
                f"no eligible block dropped h={h} b={b}: selected={sorted(selected)}"
            )
            # local block is always retained
            assert (num_blocks - 1) in selected, "local block was dropped"

    topk_ref = ref_index_topk_decode(
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        tk,
        ib,
        lb,
    )
    _assert_topk_sets_match(topk_kernel, topk_ref, inp["seq_lens"], blk, tk)
    o_ref = ref_sparse_gqa_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        topk_kernel,
    ).to(o_kernel.dtype)
    max_diff = (o_kernel.float() - o_ref.float()).abs().max().item()
    assert torch.allclose(o_kernel.float(), o_ref.float(), rtol=RTOL, atol=ATOL), (
        f"max abs diff {max_diff:.4e}"
    )


@pytest.mark.parametrize("seq_len", [1024, 4096])
def test_m3_config_shapes(seq_len):
    """Checkpoint-scale MiniMax-M3 attention dims (64 q / 4 kv / 4 index)."""
    bs, nqh, nkh, nih = 2, 64, 4, 4
    hd = idxd = 128
    blk, tk, ib, lb = 128, 16, 0, 1
    inp = build_decode_inputs(bs, nqh, nkh, nih, hd, idxd, [seq_len] * bs)
    o_kernel, topk_kernel = minimax_m3_sparse_attention_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        tk,
        init_blocks=ib,
        local_blocks=lb,
    )
    assert o_kernel.shape == (bs, nqh, hd)
    assert not torch.isnan(o_kernel).any()
    o_ref = ref_sparse_gqa_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["req_to_token"],
        inp["seq_lens"],
        inp["slot_ids"],
        blk,
        topk_kernel,
    ).to(o_kernel.dtype)
    max_diff = (o_kernel.float() - o_ref.float()).abs().max().item()
    assert torch.allclose(o_kernel.float(), o_ref.float(), rtol=RTOL, atol=ATOL), (
        f"max abs diff {max_diff:.4e}"
    )


def test_cuda_graph_capture_replay():
    """The sparse decode path is CUDA-graph hard-path safe.

    Capture the full index->select->attend pipeline with a fixed grid
    (``max_num_blocks`` supplied so no host sync), then replay on two different
    inputs and require the replayed output to match an eager recompute. This is
    the kernel-level CUDA-graph hard-path evidence the runtime backend depends
    on.
    """
    bs, nqh, nkh, nih = 2, 64, 4, 4
    hd = idxd = 128
    blk, tk, ib, lb = 128, 16, 0, 1
    seq_len = 2048
    max_num_blocks = seq_len // blk

    inp = build_decode_inputs(bs, nqh, nkh, nih, hd, idxd, [seq_len] * bs, seed=7)
    # Static buffers reused across capture + replay.
    q = inp["q"].clone()
    index_q = inp["index_q"].clone()

    def run():
        o, _ = minimax_m3_sparse_attention_decode(
            q,
            inp["k_cache"],
            inp["v_cache"],
            index_q,
            inp["index_k_cache"],
            inp["req_to_token"],
            inp["seq_lens"],
            inp["slot_ids"],
            blk,
            tk,
            init_blocks=ib,
            local_blocks=lb,
            max_num_blocks=max_num_blocks,
        )
        return o

    # Warmup on a side stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = run()

    for seed in (7, 99):
        new = build_decode_inputs(bs, nqh, nkh, nih, hd, idxd, [seq_len] * bs, seed=seed)
        q.copy_(new["q"])
        index_q.copy_(new["index_q"])
        # NB: k/v/index caches + req_to_token are shared tensors (not copied),
        # so an eager recompute uses the same buffers the graph replays over.
        g.replay()
        torch.cuda.synchronize()
        o_replay = static_out.clone()
        o_eager = run()
        max_diff = (o_replay.float() - o_eager.float()).abs().max().item()
        assert torch.allclose(o_replay.float(), o_eager.float(), rtol=RTOL, atol=ATOL), (
            f"cuda-graph replay mismatch (seed={seed}) max abs diff {max_diff:.4e}"
        )


# ---------------------------------------------------------------------------
# Prefill (per-token causal MSA). MiniMax-M3 uses block_size_q == 1, so prefill
# is per-query decode with a per-query causal seq_len = abs_pos + 1.
# ---------------------------------------------------------------------------
def _prefill_query_metadata(cu_seqlens, prefix_lens, slot_ids, total_q):
    """Mirror the wrapper's varlen -> per-query causal metadata lowering."""
    batch = cu_seqlens.shape[0] - 1
    extend_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    req_id = torch.repeat_interleave(
        torch.arange(batch, device=DEVICE, dtype=torch.int64), extend_lens
    )
    starts = cu_seqlens[:-1].to(torch.int64)
    local_idx = torch.arange(total_q, device=DEVICE, dtype=torch.int64) - starts[req_id]
    q_abs_pos = prefix_lens.to(torch.int64)[req_id] + local_idx
    q_seq_lens = (q_abs_pos + 1).to(torch.int32)
    q_slot_ids = slot_ids.to(torch.int64)[req_id]
    return q_seq_lens, q_slot_ids


def build_prefill_inputs(
    extend_list,
    prefix_list,
    num_q_heads,
    num_kv_heads,
    num_index_heads,
    head_dim,
    index_dim,
    seed=42,
    dtype=torch.bfloat16,
):
    torch.manual_seed(seed)
    batch = len(extend_list)
    total_q = sum(extend_list)
    max_kv_len = max(p + e for p, e in zip(prefix_list, extend_list))
    max_slots = batch * max_kv_len
    q = torch.randn(total_q, num_q_heads, head_dim, dtype=dtype, device=DEVICE)
    index_q = torch.randn(total_q, num_index_heads, index_dim, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    index_k_cache = torch.randn(max_slots, 1, index_dim, dtype=dtype, device=DEVICE)
    req_to_token = torch.zeros(batch, max_kv_len, dtype=torch.int32, device=DEVICE)
    for i in range(batch):
        base = i * max_kv_len
        req_to_token[i, :max_kv_len] = (torch.randperm(max_kv_len, device=DEVICE) + base).to(
            torch.int32
        )
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    cu_seqlens[1:] = torch.tensor(extend_list, dtype=torch.int32, device=DEVICE).cumsum(0)
    prefix_lens = torch.tensor(prefix_list, dtype=torch.int32, device=DEVICE)
    slot_ids = torch.arange(batch, dtype=torch.int64, device=DEVICE)
    return dict(
        q=q,
        index_q=index_q,
        k_cache=k_cache,
        v_cache=v_cache,
        index_k_cache=index_k_cache,
        req_to_token=req_to_token,
        cu_seqlens=cu_seqlens,
        prefix_lens=prefix_lens,
        slot_ids=slot_ids,
        total_q=total_q,
    )


def ref_dense_causal_attention(
    q, k_cache, v_cache, req_to_token, q_seq_lens, q_slot_ids, sm_scale=None
):
    """Standard causal attention (independent of the MSA path).

    For the short/dense-equivalent regime (all blocks selected) the MSA output
    must equal this, which cross-checks the causal masking without reusing the
    sparse reference.
    """
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    gqa = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    o = torch.zeros_like(q)
    for i in range(total_q):
        sl = int(q_seq_lens[i].item())
        slots = req_to_token[int(q_slot_ids[i].item())][:sl].long()
        for kh in range(num_kv_heads):
            k_sel = k_cache[slots, kh, :].float()
            v_sel = v_cache[slots, kh, :].float()
            for g in range(gqa):
                qh = kh * gqa + g
                scores = (q[i, qh, :].float() @ k_sel.T) * sm_scale
                attn = scores.softmax(dim=-1)
                o[i, qh, :] = attn.to(q.dtype) @ v_sel.to(q.dtype)
    return o


def _run_and_check_prefill(inp, blk, tk, ib, lb):
    o_kernel, topk_kernel = minimax_m3_sparse_attention_prefill(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        inp["cu_seqlens"],
        inp["prefix_lens"],
        inp["slot_ids"],
        blk,
        tk,
        init_blocks=ib,
        local_blocks=lb,
    )
    q_seq_lens, q_slot_ids = _prefill_query_metadata(
        inp["cu_seqlens"], inp["prefix_lens"], inp["slot_ids"], inp["total_q"]
    )
    topk_ref = ref_index_topk_decode(
        inp["index_q"],
        inp["index_k_cache"],
        inp["req_to_token"],
        q_seq_lens,
        q_slot_ids,
        blk,
        tk,
        ib,
        lb,
    )
    _assert_topk_sets_match(topk_kernel, topk_ref, q_seq_lens, blk, tk)
    o_ref = ref_sparse_gqa_decode(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["req_to_token"],
        q_seq_lens,
        q_slot_ids,
        blk,
        topk_kernel,
    ).to(o_kernel.dtype)
    # Short-context prefill queries (small causal p) attend to only a few tokens,
    # so the bf16 output carries ~1-2 ULP (~1.5e-2 at magnitude ~1) of rounding
    # between the kernel's online softmax and the reference's batch softmax. Use a
    # bf16-appropriate tolerance; block selection is validated exactly by the
    # topk-set match above and correctness is cross-checked against dense causal
    # attention in test_prefill_dense_causal_equiv.
    max_diff = (o_kernel.float() - o_ref.float()).abs().max().item()
    assert torch.allclose(o_kernel.float(), o_ref.float(), rtol=1e-2, atol=2e-2), (
        f"prefill max abs diff {max_diff:.4e}"
    )
    return o_kernel, topk_kernel, q_seq_lens, q_slot_ids


@pytest.mark.parametrize(
    "extend_list,prefix_list",
    [
        ([200], [0]),  # single-request pure prefill
        ([120, 200, 80], [0, 0, 0]),  # varlen pure prefill
        ([64, 96], [500, 1000]),  # chunked prefill / cache reuse (prefix > 0)
    ],
)
def test_sparse_attention_prefill(extend_list, prefix_list):
    """Per-token causal MSA prefill matches the per-query reference."""
    inp = build_prefill_inputs(extend_list, prefix_list, 8, 4, 4, 128, 128)
    _run_and_check_prefill(inp, 128, 16, 0, 1)


def test_prefill_dense_causal_equiv():
    """Short prefill (all blocks selected) must equal standard causal attention."""
    inp = build_prefill_inputs([200], [0], 8, 4, 4, 128, 128)
    o_kernel, _, q_seq_lens, q_slot_ids = _run_and_check_prefill(inp, 128, 16, 0, 1)
    o_dense = ref_dense_causal_attention(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        inp["req_to_token"],
        q_seq_lens,
        q_slot_ids,
    ).to(o_kernel.dtype)
    # bf16 short-context tolerance (see _run_and_check_prefill).
    max_diff = (o_kernel.float() - o_dense.float()).abs().max().item()
    assert torch.allclose(o_kernel.float(), o_dense.float(), rtol=1e-2, atol=2e-2), (
        f"prefill vs dense-causal max abs diff {max_diff:.4e}"
    )


def test_prefill_long_pruned_block_drop():
    """Prefill queries with a long causal prefix must drop eligible blocks."""
    blk, tk, ib, lb = 128, 16, 0, 1
    prefix = 4000
    extend = 8
    inp = build_prefill_inputs([extend], [prefix], 8, 4, 4, 128, 128)
    o_kernel, topk_kernel, q_seq_lens, _ = _run_and_check_prefill(inp, blk, tk, ib, lb)
    # Each query at abs pos p>=4000 has >16 causal blocks, so blocks must drop.
    nkh = inp["k_cache"].shape[1]
    for i in range(inp["total_q"]):
        num_blocks = (int(q_seq_lens[i].item()) + blk - 1) // blk
        assert num_blocks > tk, "test must exercise the block-drop regime"
        for h in range(nkh):
            selected = set(topk_kernel[h, i].tolist()) - {-1}
            assert len(selected) == tk
            dropped = set(range(num_blocks)) - selected
            eligible = dropped - {num_blocks - 1} - set(range(ib))
            assert eligible, f"no eligible block dropped q={i} h={h}"
            assert (num_blocks - 1) in selected, "local block dropped"


@pytest.mark.parametrize("prefix", [1024, 3000])
def test_m3_config_prefill(prefix):
    """Checkpoint-scale prefill dims (64 q / 4 kv / 4 index), small token count."""
    inp = build_prefill_inputs([4], [prefix], 64, 4, 4, 128, 128)
    o_kernel, _, _, _ = _run_and_check_prefill(inp, 128, 16, 0, 1)
    assert o_kernel.shape == (inp["total_q"], 64, 128)
    assert not torch.isnan(o_kernel).any()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))

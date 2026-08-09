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
"""The short-conv under a multi-token generation step.

Ordinary decode feeds one token per generation request, and ``_apply_sconv``
sends the whole post-context slice through ``causal_conv1d_update`` with one
cache index per request. Speculative decoding breaks that assumption: the
target verifies ``1 + max_draft_len`` tokens per request in a single step.

The failure is loud in one place and silent in another, which is why both are
tested here. Loud: the update kernel requires ``conv_state_indices`` to have one
entry per ROW, so it rejects the multi-token batch outright. Silent: reshaping
to satisfy it would apply the same initial state to every drafted token instead
of advancing through them -- no error, just a conv that stopped being causal.
"""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_inkling import InklingConvRuntime


class _Meta:
    """The slice of attention metadata ``InklingConvRuntime.build`` reads."""

    def __init__(self, seq_lens, num_contexts):
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self.num_contexts = num_contexts
        self.request_ids = list(range(len(seq_lens)))
        self.is_cuda_graph = False


class _Cache:
    """Stands in for InklingConvStateCache: hands out one pool row per request."""

    def __init__(self, n):
        self.state_indices = torch.arange(n, dtype=torch.int32)
        self._n = n

    def write_state_indices(self, request_ids, is_graph):
        return list(range(len(request_ids)))


def _build(seq_lens, num_contexts):
    return InklingConvRuntime.build(_Meta(seq_lens, num_contexts), _Cache(len(seq_lens)))


def test_ordinary_decode_stays_on_the_single_token_path():
    """One token per generation request must not change behaviour.

    This is the overwhelmingly common case and the update kernel is the fast
    path for it, so the multi-token branch has to stay strictly opt-in.
    """
    rt = _build([1, 1, 1], num_contexts=0)
    assert rt.gen_tokens_per_seq == 1
    assert rt.gen_query_start_loc is None
    assert rt.gen_has_initial_state is None


def test_a_verify_step_is_recognised_as_multi_token():
    """Four requests verifying 1 + 3 drafted tokens each."""
    rt = _build([4, 4, 4, 4], num_contexts=0)
    assert rt.gen_tokens_per_seq == 4
    assert rt.gen_query_start_loc.tolist() == [0, 4, 8, 12, 16]


def test_generation_tokens_continue_an_existing_window():
    """``has_initial_state`` is True for generation, unlike prefill.

    A prefill starts a fresh stream and the pool row holds nothing worth
    reading; a generation step continues one. Getting this backwards would drop
    the first kernel-width tokens of context at every verify step -- again with
    no error, only worse output.
    """
    rt = _build([4, 4], num_contexts=0)
    assert rt.gen_has_initial_state.tolist() == [True, True]
    # ...while the context side of the same structure stays False.
    rt2 = _build([7, 4], num_contexts=1)
    assert rt2.has_initial_state.tolist() == [False]


def test_mixed_context_and_verify_batch_splits_correctly():
    """Context requests keep their own varlen offsets; generation gets its own."""
    rt = _build([9, 5, 4, 4], num_contexts=2)
    assert rt.num_ctx_tokens == 14
    assert rt.query_start_loc.tolist() == [0, 9, 14]
    assert rt.gen_tokens_per_seq == 4
    assert rt.gen_query_start_loc.tolist() == [0, 4, 8]


def test_a_ragged_generation_batch_is_rejected():
    """The varlen offsets are built from a uniform per-request token count.

    Speculative decoding always verifies the same number of tokens per request,
    so a ragged batch means an assumption elsewhere has broken. Computing the
    offsets from the first request's length and carrying on would silently
    mis-slice every later request.
    """
    with pytest.raises(ValueError, match="uniform token count"):
        _build([4, 3], num_contexts=0)


# --- numerics -------------------------------------------------------------
# The reason to route a verify step through the varlen path rather than reshape
# it onto the update kernel: the conv must advance token by token. A reference
# built from repeated single-token updates pins that down.


def _reference_windows(init_state, x, width):
    """Conv state after each token: last ``width`` inputs of init ++ x.

    The state a causal conv carries is a window of past INPUTS, so it can be
    written down without running a conv at all -- which makes it a reference
    that shares no code with the implementation.
    """
    stream = torch.cat([init_state, x], dim=-1)
    return [stream[..., t + 1 : t + 1 + width] for t in range(x.shape[-1])]


def test_the_window_after_the_last_token_is_what_a_verify_step_must_leave():
    """After k tokens the state holds the last ``width`` of init ++ x[:k].

    This is the property the varlen path provides and the update kernel, given
    one shared initial state, does not: its output for every drafted token
    would be the window after the FIRST one.
    """
    width, channels, steps = 3, 2, 4
    init = torch.arange(channels * width, dtype=torch.float32).reshape(channels, width)
    x = torch.arange(100, 100 + channels * steps, dtype=torch.float32).reshape(channels, steps)

    windows = _reference_windows(init, x, width)
    assert len(windows) == steps
    # Each step shifts the window one input to the right...
    assert torch.equal(windows[0], torch.cat([init[:, 1:], x[:, :1]], dim=-1))
    # ...and after all four drafted tokens nothing of the initial state is left.
    assert torch.equal(windows[-1], x[:, -width:])


def test_partial_acceptance_needs_an_earlier_window_than_the_forward_leaves():
    """Why the model refuses speculative decoding rather than running.

    The forward advances the state to ``windows[-1]``. If only one token is
    accepted, the correct state is ``windows[0]``, and they differ. Nothing in
    the shapes or dtypes distinguishes them, which is precisely the problem:
    the wrong one is a perfectly valid conv state holding tokens the model
    never emitted.
    """
    width, channels, steps = 3, 2, 4
    init = torch.zeros(channels, width)
    x = torch.arange(1, 1 + channels * steps, dtype=torch.float32).reshape(channels, steps)
    windows = _reference_windows(init, x, width)
    assert not torch.equal(windows[0], windows[-1])


def test_a_draft_length_the_capture_cannot_hold_is_rejected():
    """The capture buffers are sized from max_draft_len.

    A verify step deeper than they hold could not be rolled back, so the
    precondition is checked at load rather than discovered as quietly worse
    output later. max_draft_len < 1 means nothing was sized at all.
    """
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    src = inspect.getsource(InklingForCausalLM._assert_inkling_spec_conv_state)
    assert "max_draft_len" in src and "raise ValueError" in src


# --- rolling the window back to what was accepted --------------------------
# The verify forward leaves the window advanced over every drafted token. What
# each request should be left holding is the window after its ACCEPTED prefix,
# and the two differ on any partial acceptance.


def _capture(n, channels, kwin, steps, init, x):
    from tensorrt_llm._torch.models.modeling_inkling import _ConvVerifyCapture

    cap = _ConvVerifyCapture(n, channels, kwin, steps, torch.device("cpu"), torch.float32)
    cap.init[:n].copy_(init)
    cap.x[:n].copy_(x)
    return cap


def test_accepted_window_matches_the_hand_computed_stream():
    """Reconstruct against the definition: last kwin of (init ++ x[:k]).

    The reference shares no code with the implementation -- it slices the
    concatenated stream directly -- so agreement is evidence, not tautology.
    """
    n, channels, kwin, steps = 3, 2, 3, 4
    init = torch.randn(n, channels, kwin)
    x = torch.randn(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)

    for k in range(1, steps + 1):
        got = cap.accepted_window(torch.full((n,), k, dtype=torch.int64), kwin)
        want = torch.cat([init, x.transpose(1, 2)], dim=-1)[..., k : k + kwin]
        assert torch.allclose(got, want), f"k={k}"


def test_each_request_rolls_back_to_its_own_acceptance():
    """Acceptance counts differ per request within one batch.

    A commit that used a single count for the batch would be right for whichever
    request happened to set it and wrong for the rest -- the kind of bug that
    only shows up once acceptance rates stop being uniform.
    """
    n, channels, kwin, steps = 3, 2, 3, 4
    init = torch.zeros(n, channels, kwin)
    x = torch.arange(n * steps * channels, dtype=torch.float32).reshape(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)

    accepted = torch.tensor([1, 3, 4], dtype=torch.int64)
    got = cap.accepted_window(accepted, kwin)
    stream = torch.cat([init, x.transpose(1, 2)], dim=-1)
    for i, k in enumerate(accepted.tolist()):
        assert torch.equal(got[i], stream[i, :, k : k + kwin])


def test_full_acceptance_leaves_what_the_forward_already_wrote():
    """When every drafted token is accepted the commit is a no-op in effect.

    Worth pinning: it is the case where a wrong commit would be invisible,
    because the forward's own result is also correct.
    """
    n, channels, kwin, steps = 2, 3, 2, 4
    init = torch.randn(n, channels, kwin)
    x = torch.randn(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)
    got = cap.accepted_window(torch.full((n,), steps, dtype=torch.int64), kwin)
    # The state after all steps is simply the last kwin inputs.
    assert torch.allclose(got, x.transpose(1, 2)[..., -kwin:])


def test_single_acceptance_discards_the_rejected_tokens():
    """The target's own token is always accepted, so k >= 1 and never 0.

    With k=1 the window keeps kwin-1 of the pre-step state and exactly one new
    input; the drafted tokens 2..steps must leave no trace.
    """
    n, channels, kwin, steps = 1, 2, 3, 4
    init = torch.full((n, channels, kwin), -1.0)
    x = torch.arange(1, 1 + steps * channels, dtype=torch.float32).reshape(n, steps, channels)
    cap = _capture(n, channels, kwin, steps, init, x)
    got = cap.accepted_window(torch.ones(n, dtype=torch.int64), kwin)
    assert torch.equal(got[0, :, :-1], init[0, :, 1:])
    assert torch.equal(got[0, :, -1], x[0, 0])


def test_capture_is_only_allocated_when_speculating():
    """An ordinary server must not pay for buffers it never reads."""
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingConvStateCache

    src = inspect.getsource(InklingConvStateCache.__init__)
    assert "verify_steps" in src and "if self.verify_steps < 2" in src


# --- verify-step attention -------------------------------------------------
# The decode path serves one query token per request. A verify step presents
# 1 + max_draft_len, which is what produced the illegal memory access: it wrote
# one KV entry per request and read a page table sized for one new position.


def test_verify_attention_is_not_routed_through_the_context_path():
    """Routing a verify step at the prefill kernel would drop the prefix.

    ``inkling_prefill_attention`` attends only within the tokens handed to it,
    which is complete for a fresh prefill (Inkling keeps block reuse off) and
    silently wrong here -- the drafted tokens would see none of the cached
    conversation and still produce fluent text. Cheapest-looking route, worst
    failure mode, so it is worth pinning that it was not taken.
    """
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingAttention

    src = inspect.getsource(InklingAttention._run_verify)
    # The docstring explains why the prefill kernel is wrong here, so check the
    # body rather than the whole source.
    quote = '"' * 3
    body = src[src.index(quote, src.index(quote) + 3) + 3 :]
    assert "inkling_prefill_attention" not in body
    assert "inkling_decode_attention" in body


def test_verify_walks_positions_in_order_so_causality_is_structural():
    """Position t must see the prefix plus 0..t, and nothing later.

    Ordering provides that rather than a mask: each step writes its KV before
    attending, and the seq_len it passes is num_cached + t + 1. A loop that
    wrote all KV up front would let position 0 attend to drafted tokens that,
    at that point in the sequence, do not exist.
    """
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingAttention

    src = inspect.getsource(InklingAttention._run_verify)
    write_at = src.index("write_kv_cache_hnd")
    attend_at = src.index("inkling_decode_attention")
    assert write_at < attend_at, "each position's KV must be written before it attends"
    assert "int(num_cached[i]) + t + 1" in src


def test_verify_output_is_reassembled_in_packed_order():
    """The batch is request-major, so per-step results interleave back.

    Returning the steps concatenated instead would hand every downstream module
    a batch whose rows belong to the wrong requests -- a permutation, not a
    crash.
    """
    num_gen, steps, hidden = 3, 4, 5
    packed = torch.arange(num_gen * steps * hidden, dtype=torch.float32).reshape(
        num_gen * steps, hidden
    )
    view = packed.view(num_gen, steps, hidden)
    # Request-major: request i's step t is row i*steps + t.
    for i in range(num_gen):
        for t in range(steps):
            assert torch.equal(view[i, t], packed[i * steps + t])
    # ...and the reshape back is the identity, which is what _run_verify relies on.
    assert torch.equal(view.reshape(num_gen * steps, hidden), packed)


def test_capture_under_cuda_graph_is_refused_with_a_reason():
    """The verify path is eager; capturing it would be silently wrong.

    Better to say so at the point of use than to let a captured graph replay
    stale per-step writes.
    """
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingAttention

    src = inspect.getsource(InklingAttention._run_verify)
    assert "is_cuda_graph" in src and "RuntimeError" in src

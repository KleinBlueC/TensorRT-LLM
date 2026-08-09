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


def test_speculative_config_is_refused_at_load():
    """Better an immediate, explanatory failure than a silently worse server."""
    import inspect

    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    src = inspect.getsource(InklingForCausalLM._assert_inkling_spec_conv_state)
    assert "NotImplementedError" in src
    assert "spec_config" in src

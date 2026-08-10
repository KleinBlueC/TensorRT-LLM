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
"""How much KV a request must have reserved before its FIRST verify step.

The context phase reserves ``prompt + num_extra_kv_tokens``. From the second
step on, the scheduler grows a generation request by ``1 + draft`` per step and
the margin is comfortable, so this is the one place the reservation has to be
right.

The generic one-engine reserve is ``max_draft_len - 1``
(``get_num_extra_kv_tokens``), which is two short of the ``1 + max_draft_len``
positions an Inkling verify step writes. Being short is invisible unless the
last drafted position lands on a page boundary -- which is why the end-to-end
runs passed for weeks and one 5-shot GSM8K prompt did not (job 6026096:
prompt 669, capacity 671, writing 669..672, page_size 32).
"""

import pytest

from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens


def _pages(last_pos: int, page_size: int) -> int:
    """Pages needed to hold positions 0..last_pos, as write_kv_cache_hnd indexes."""
    return last_pos // page_size + 1


def _reserved_pages(prompt: int, extra: int, page_size: int) -> int:
    """Pages the context phase's reservation of ``prompt + extra`` materialises.

    Blocks follow capacity in KVCacheManagerV2 (``div_up(capacity, page)``), so
    the page count is derivable from the token count alone.
    """
    return -(-(prompt + extra) // page_size)


@pytest.mark.parametrize("max_draft_len", [1, 2, 3, 4, 7])
def test_a_verify_step_writes_one_more_than_the_generic_reserve(max_draft_len):
    """The gap this fix closes, stated as the arithmetic that produced it.

    A verify step presents ``1 + max_draft_len`` tokens and writes every one of
    them, so the first one needs that many positions past the prompt. The
    generic reserve stops ``2`` short of it.
    """

    class _SpecConfig:
        pass

    spec_config = _SpecConfig()
    spec_config.max_draft_len = max_draft_len
    spec_config.spec_dec_mode = type("_Mode", (), {"use_one_engine": staticmethod(lambda: True)})()

    generic = get_num_extra_kv_tokens(spec_config)
    needed = 1 + max_draft_len
    assert generic == max_draft_len - 1
    assert needed - generic == 2


@pytest.mark.parametrize("page_size", [16, 32, 128])
def test_the_generic_reserve_is_short_exactly_at_a_page_boundary(page_size):
    """Why this was not caught earlier: it needs the prompt to line up.

    Over a sweep of prompt lengths the generic reserve is enough for most and
    short for those where the last drafted position opens a new page. The fixed
    reserve is enough for every one of them -- that difference, not the average,
    is the regression.
    """
    max_draft_len = 3
    generic = max_draft_len - 1
    fixed = 1 + max_draft_len
    short_for = []
    for prompt in range(page_size * 4, page_size * 8):
        last_pos = prompt + max_draft_len  # positions prompt .. prompt + draft
        need = _pages(last_pos, page_size)
        if need > _reserved_pages(prompt, generic, page_size):
            # Never more than one page short: the reservation misses by two
            # tokens, so it can only ever miss the page those two open.
            assert need == _reserved_pages(prompt, generic, page_size) + 1
            short_for.append(prompt)
        assert need <= _reserved_pages(prompt, fixed, page_size)
    # It is an alignment, not a size: the same two residues in every page cycle
    # (job 6026096's prompt of 669 is 669 % 32 == 29, the first of them).
    assert short_for, "the sweep must contain the failing alignment"
    assert {p % page_size for p in short_for} == {page_size - 3, page_size - 2}


def test_the_manager_raises_the_reservation_over_the_generic_one(monkeypatch):
    """``InklingHybridCacheManager`` takes the larger of the two.

    Asserted through the constructor rather than by re-deriving the number,
    because the bug was that the generic value reached the context phase
    unchanged.
    """
    from tensorrt_llm._torch.attention_backend.inkling import cache_manager as cm

    captured = {}

    def _fake_super_init(self, *args, **kwargs):
        self.num_extra_kv_tokens = get_num_extra_kv_tokens(kwargs.get("spec_config"))
        captured["generic"] = self.num_extra_kv_tokens

    class _FakeConvCache:
        def __init__(self, *args, **kwargs):
            captured["verify_steps"] = kwargs.get("verify_steps")

    monkeypatch.setattr(cm.KVCacheManagerV2, "__init__", _fake_super_init)
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_inkling.InklingConvStateCache",
        _FakeConvCache,
    )

    class _Mapping:
        enable_attention_dp = False
        tp_size = 1

    class _Text:
        num_hidden_layers = 66
        torch_dtype = None

    class _Pretrained:
        text_config = _Text()

    class _SpecConfig:
        max_draft_len = 3
        spec_dec_mode = type("_Mode", (), {"use_one_engine": staticmethod(lambda: True)})()

    mgr = cm.InklingHybridCacheManager(
        pretrained_config=_Pretrained(),
        mapping=_Mapping(),
        max_batch_size=4,
        spec_config=_SpecConfig(),
    )

    assert captured["generic"] == 2  # max_draft_len - 1, the generic reserve
    assert captured["verify_steps"] == 4  # 1 + max_draft_len, what a step writes
    assert mgr.num_extra_kv_tokens == 4

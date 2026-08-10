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
"""LoRA on Inkling.

The failure mode this guards against is not a crash. TRT-LLM wires LoRA into
the shared ``Attention`` and ``GatedMLP`` modules, and Inkling uses neither: its
attention subclasses ``Attention`` but overrides the forward, and its MLP is its
own. So the base builds ``splitted_qkv_lora`` / ``fused_qkv_lora`` / ``o_lora``
and then never calls them -- an adapter that loads, allocates, and contributes
nothing, which from the outside is indistinguishable from an adapter that simply
does not help.

Every assertion here is about a path that would otherwise be silent.
"""

import inspect

import pytest


def _src(obj):
    return inspect.getsource(obj)


# --- the adapter actually reaches the projections --------------------------


def test_qkv_lora_is_applied_in_the_overridden_projection():
    """The base applies it in a forward Inkling does not use."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingAttention

    src = _src(InklingAttention._project)
    assert "splitted_qkv_lora" in src and "fused_qkv_lora" in src
    assert "lora_params" in src


def test_qkv_lora_is_added_before_the_split_and_the_short_convs():
    """Order is not cosmetic here.

    The adapter was trained against the fused qkv output, and Inkling's k/v
    short-convs run on the split k and v. Adding the delta after the split would
    put it on the wrong side of a convolution -- numerically different, and
    nothing about it raises.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingAttention

    src = _src(InklingAttention._project)
    assert src.index("fused_qkv_lora") < src.index("split_qkv")
    assert src.index("fused_qkv_lora") < src.index("_apply_sconv")


def test_o_proj_receives_lora_params():
    """``o_lora`` is attached to the Linear, which applies it -- if it is told.

    The Linear no-ops when ``lora_params`` is absent, so forgetting to pass it
    is exactly the silent case.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingAttention

    assert "self.o_proj(attn_out, lora_params=lora_params)" in _src(InklingAttention.forward)


def test_dense_mlp_builds_and_uses_both_adapters():
    from tensorrt_llm._torch.models.modeling_inkling import InklingDenseMLP

    init, fwd = _src(InklingDenseMLP.__init__), _src(InklingDenseMLP.forward)
    assert "MLP_GATE_UP" in init and "MLP_4H_TO_H" in init
    assert init.count("lora=self.") == 2
    assert fwd.count("lora_params=lora_params") == 2


def test_the_fused_gate_up_adapter_covers_the_full_width():
    """One LoraModuleType over 2*inter, matching the fusion.

    Sizing it to ``inter`` would apply the adapter to half the fused output and
    leave the rest untouched: a plausible-looking tensor, silently wrong.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingDenseMLP

    assert "[2 * inter]" in _src(InklingDenseMLP.__init__)


def test_lora_modules_are_only_built_when_an_adapter_is_configured():
    """An unused module in every layer's state_dict misstates what is supported."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingDenseMLP

    src = _src(InklingDenseMLP.__init__)
    assert "if model_config.lora_config is not None:" in src


# --- the parameters reach the layers ---------------------------------------


def test_lora_params_are_threaded_from_the_model_down():
    """Model -> layer -> attention/MLP. A break anywhere is silent."""
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingAttention,
        InklingDecoderLayer,
        InklingModel,
    )

    assert 'kwargs.get("lora_params")' in _src(InklingModel.forward)
    layer = _src(InklingDecoderLayer.forward)
    assert "lora_params=lora_params" in layer
    # both the stateless and the state-pool branch
    assert layer.count("lora_params") >= 4
    assert "lora_params" in _src(InklingAttention.forward)


def test_both_mlp_branches_are_covered():
    """The stateless branch and the conv-pool branch each call _run_mlp."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingDecoderLayer

    src = _src(InklingDecoderLayer.forward)
    assert src.count("_run_mlp(") == 2
    assert src.count("all_rank_num_tokens, lora_params") == 2


# --- what Inkling cannot serve ---------------------------------------------


class _LoraConfig:
    def __init__(self, targets):
        self.lora_target_modules = targets


class _ModelConfig:
    def __init__(self, targets):
        self.lora_config = _LoraConfig(targets)


@pytest.mark.parametrize(
    "targets",
    [["moe_h_to_4h"], ["experts"], ["attn_qkv", "moe_4h_to_h"], ["r_proj"]],
)
def test_unsupported_targets_are_rejected_at_load(targets):
    """Routed experts are NVFP4; r_proj has no LoRA module type.

    ``check_moe_lora_supported`` permits expert LoRA only on bf16/fp16 or
    per-tensor-FP8 base weights, and the routed experts are the only quantized
    part of an Inkling checkpoint. Dropping such a target instead of refusing it
    gives an adapter that loads, occupies memory and does nothing.
    """
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    with pytest.raises(ValueError, match="cannot serve LoRA"):
        InklingForCausalLM._assert_inkling_lora_supported(_ModelConfig(targets))


@pytest.mark.parametrize(
    "targets", [["attn_q", "attn_k", "attn_v"], ["attn_dense"], ["mlp_h_to_4h"], []]
)
def test_supported_targets_pass(targets):
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    InklingForCausalLM._assert_inkling_lora_supported(_ModelConfig(targets))


def test_no_lora_config_is_not_an_error():
    """The overwhelmingly common case: no adapter at all."""
    from tensorrt_llm._torch.models.modeling_inkling import InklingForCausalLM

    class _NoLora:
        lora_config = None

    InklingForCausalLM._assert_inkling_lora_supported(_NoLora())

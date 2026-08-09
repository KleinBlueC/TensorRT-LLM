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
"""The interface MTPWorker calls the draft chain with.

``MTPWorker.__call__`` does, per depth:

    hidden = mtp_layer(embed_tokens=draft_model.embed_tokens, **draft_inputs)
    logits = mtp_layer.shared_head(hidden, draft_model.lm_head, attn_metadata)

where ``draft_inputs`` carries ``input_ids``, ``position_ids``,
``hidden_states`` and ``attn_metadata``. A signature that does not match is a
TypeError deep inside the speculative loop, on a multi-GPU run, after several
minutes of model load -- the most expensive place to discover a keyword name.

These assertions are signature-level on purpose: they run without a GPU, a
checkpoint or a built extension, so the contract is checked in seconds rather
than at the end of an end-to-end job.
"""

import inspect

from tensorrt_llm._torch.configs.inkling import InklingConfig
from tensorrt_llm._torch.models.modeling_inkling import InklingMTPBlock, InklingMTPHead

# The keys MTPWorker builds in prepare_drafter_inputs and forwards as **kwargs.
_DRAFT_INPUT_KEYS = {"input_ids", "position_ids", "hidden_states", "attn_metadata"}


def test_block_accepts_every_draft_input_key():
    """Each key MTPWorker passes must be a named parameter or reach **kwargs."""
    params = inspect.signature(InklingMTPBlock.forward).parameters
    named = set(params) - {"self"}
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = _DRAFT_INPUT_KEYS - named
    assert not missing or has_var_kw, (
        f"MTPWorker passes {sorted(missing)} which the block neither names nor "
        "absorbs into **kwargs"
    )


def test_block_takes_embed_tokens_as_a_keyword():
    """The worker passes ``embed_tokens=`` explicitly, not positionally.

    It hands over the TARGET model's embedding table -- the draft chain shares
    it rather than owning a second copy -- so the name has to match exactly.
    """
    params = inspect.signature(InklingMTPBlock.forward).parameters
    assert "embed_tokens" in params
    assert params["embed_tokens"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def test_shared_head_signature_matches_the_worker_call():
    """``shared_head(hidden_states, lm_head, attn_metadata)``, positionally."""
    params = list(inspect.signature(InklingMTPHead.forward).parameters.values())[1:]
    positional = [p.name for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[:3] == ["hidden_states", "lm_head", "attn_metadata"], (
        f"MTPWorker calls shared_head positionally; got {positional[:3]}"
    )


def test_block_exposes_shared_head_attribute():
    """The worker reaches ``mtp_layer.shared_head`` by attribute name."""
    assert "shared_head" in inspect.getsource(InklingMTPBlock.__init__)


def test_head_norm_is_built_only_when_the_checkpoint_declares_it():
    """Both shipped checkpoints set chain_hidden_post_norm False.

    Building the norm unconditionally would create a parameter with no
    checkpoint tensor behind it, which the loader then has to explain away.
    """
    src = inspect.getsource(InklingMTPHead.__init__)
    assert "use_norm" in src
    src_block = inspect.getsource(InklingMTPBlock.__init__)
    assert "chain_hidden_post_norm" in src_block


# --- how MTPForCausalLM constructs the chain -------------------------------
# It does, for each depth:
#   mtp_layer(model_config, layer_idx + start_layer_idx, model.aux_stream_dict)
# with start_layer_idx = the TARGET's num_hidden_layers, and reads the chain
# depth from pretrained_config.num_nextn_predict_layers. Inkling declares the
# depth on mtp_config, so the mirroring is what makes the framework able to
# build the chain at all.

_CKPT_MTP = {"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2, 4, 5, 6, 7]}


def test_chain_depth_is_visible_under_the_framework_name():
    """MTPForCausalLM reads ``pretrained_config.num_nextn_predict_layers``.

    Inkling declares it on ``mtp_config``. Without the mirror the framework
    reads None and builds a zero-depth chain -- speculative decoding silently
    does nothing rather than failing.
    """
    text = InklingConfig(text_config={}, mtp_config=dict(_CKPT_MTP)).text_config
    assert text.num_nextn_predict_layers == 8


def test_block_constructor_takes_the_frameworks_three_positionals():
    """``mtp_layer(model_config, layer_idx, aux_stream_dict)``."""
    params = list(inspect.signature(InklingMTPBlock.__init__).parameters.values())[1:]
    positional = [p.name for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[:3] == ["model_config", "depth", "aux_stream_dict"], (
        f"MTPForCausalLM constructs layers positionally; got {positional[:3]}"
    )
    assert params[2].default is None, "aux_stream_dict must stay optional for direct construction"


def test_offset_layer_index_maps_back_onto_the_chain():
    """The framework passes ``depth + target_num_hidden_layers``.

    A 66-layer trunk means depth 0 arrives as 66. Indexing the chain's geometry
    with 66 would read past the chain and treat every depth as global -- wrong
    windows on every banded depth, and no crash to show it.
    """
    src = inspect.getsource(InklingMTPBlock.__init__)
    assert "%" in src and "_mtp_num_depths" in src, (
        "the offset start_layer_idx must be folded back onto the chain's own indexing"
    )


def test_inkling_is_registered_in_the_mtp_dispatch_table():
    """`get_draft_model` picks the MTP class by model_type."""
    import tensorrt_llm._torch.models.modeling_speculative as spec

    src = inspect.getsource(spec.MTPForCausalLM.__init__)
    assert "inkling_mm_model" in src and "InklingMTPBlock" in src

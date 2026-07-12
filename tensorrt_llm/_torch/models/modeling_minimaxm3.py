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
"""MiniMax-M3 text tower for the TensorRT-LLM PyTorch backend.

MiniMax-M3 is a hybrid sparse-attention MoE model. Its decoder is split into
two regimes by the checkpoint's ``sparse_attention_config``/``moe_layer_freq``
schedules:

* Leading dense layers (0-2): plain GQA attention + a dense gated MLP.
* Trailing layers (3-59): GQA attention **plus** a low-dimensional "index"
  branch that scores 128-token KV blocks and selects the top-k blocks (MiniMax
  Sparse Attention, MSA), and a sparse MoE FFN with a shared expert.

The released checkpoint is multimodal
(``MiniMaxM3SparseForConditionalGeneration`` / ``model_type=minimax_m3_vl``);
for text-only bring-up only the text tower (``MiniMaxM3SparseForCausalLM``) is
built. Config normalization that routes the unmodified multimodal checkpoint to
this text architecture lives in ``pyexecutor/config_utils.py``.

Scope note (sparse-attention runtime): this module builds and weight-loads the
full model, including the per-layer index-branch projections/norms, and runs the
main GQA attention through the selected attention backend. The MSA block
selection over the index side cache (Triton kernels + ``KVCacheManagerV2`` K-only
side pool + CUDA-graph hard path) is layered on top of these modules by the
MiniMax-M3 sparse attention backend; the index-branch modules constructed here
are the projection/norm contract that backend consumes.
"""

from typing import Callable, Dict, List, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..distributed import AllReduce
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import MiniMaxM2MoeRoutingMethod, create_moe
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..utils import ActivationType, AuxStreamType
from .modeling_minimaxm2 import _EScoreCorrectionBiasHolder
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model

# HF weight prefix of the language tower inside the multimodal checkpoint. All
# text weights are stored as ``language_model.<...>``; the vision tower and the
# multimodal projector live under sibling prefixes and are not part of the
# text-only bring-up.
_LANGUAGE_MODEL_PREFIX = "language_model."

# Checkpoint prefixes that belong to the (deferred) vision path. They are
# explicitly excluded from text weight loading so the exclusion is auditable
# rather than an accidental silent drop.
_NON_TEXT_PREFIXES = (
    "vision_tower.",
    "multi_modal_projector.",
    "patch_merge_mlp.",
)


def _sparse_attention_config(config: PretrainedConfig) -> Optional[dict]:
    """The nested sparse-attention schedule dict, or None for a dense model."""
    return getattr(config, "sparse_attention_config", None)


def _sparse_schedule(config: PretrainedConfig, key: str) -> Optional[List[int]]:
    sac = _sparse_attention_config(config)
    if not sac:
        return None
    # sparse_attention_config survives config normalization as a plain dict.
    if isinstance(sac, dict):
        return sac.get(key)
    return getattr(sac, key, None)


def is_sparse_attention_layer(config: PretrainedConfig, layer_idx: int) -> bool:
    """Whether layer ``layer_idx`` runs MiniMax Sparse Attention (MSA)."""
    freq = _sparse_schedule(config, "sparse_attention_freq")
    if not freq:
        return False
    return bool(freq[layer_idx])


def disable_index_value(config: PretrainedConfig, layer_idx: int) -> bool:
    """Whether a sparse layer omits the index value/output branch.

    For the released MiniMax-M3 checkpoint every sparse layer sets this, so the
    index branch is K-only: it selects blocks but contributes no ``index_o_proj``
    value output (there is no ``index_v_proj``/``index_o_proj`` weight).
    """
    div = _sparse_schedule(config, "sparse_disable_index_value")
    if not div:
        return False
    return bool(div[layer_idx])


def is_moe_layer(config: PretrainedConfig, layer_idx: int) -> bool:
    """Whether layer ``layer_idx`` uses a sparse MoE FFN (else a dense MLP)."""
    freq = getattr(config, "moe_layer_freq", None)
    if freq is None:
        return True
    return bool(freq[layer_idx])


def _make_swigluoai_activation(
    alpha: float, limit: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """MiniMax ``swigluoai`` gated activation (clamped SwiGLU, OAI form).

    Applied to the fused ``[gate | up]`` output of a :class:`GatedMLP`
    (``gate_up_proj`` produces the two halves chunked, not interleaved):

        gate = clamp(gate, max=limit)
        up   = clamp(up, min=-limit, max=limit)
        out  = gate * sigmoid(alpha * gate) * (up + 1)

    This matches the clamped-SwiGLU semantics the fused MoE applies to the
    routed experts via ``swiglu_alpha``/``swiglu_beta``/``swiglu_limit`` so the
    dense MLP, the shared expert and the routed experts share one activation.
    """

    def activation(fused_gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = fused_gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return gate * torch.sigmoid(alpha * gate) * (up + 1.0)

    return activation


class MiniMaxM3MoE(nn.Module):
    """Sparse MoE FFN for MiniMax-M3 trailing layers.

    Routing: fp32 sigmoid scores + ``e_score_correction_bias`` for top-k
    selection, renormalized un-biased scores for the routing weights (reusing
    :class:`MiniMaxM2MoeRoutingMethod`, routing type ``MiniMax2``), then a
    ``routed_scaling_factor`` applied on the routed output. The MiniMax2 fused
    routing kernel fixes its in-kernel route-scale to 1.0, so the scale is
    applied here on the output (backend-agnostic, mirrors Step3p7). Experts are
    clamped SwiGLU (``swigluoai``); a single shared expert is added to the routed
    output before the tensor-parallel all-reduce.
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        # fp32 router gate (checkpoint stores block_sparse_moe.gate.weight in fp32).
        self.gate = Linear(
            self.hidden_dim, self.num_experts, bias=False, dtype=torch.float32, quant_config=None
        )

        # Per-(local-)expert clamped-SwiGLU parameters for the fused MoE. The
        # MoE kernel expects one value per local expert slot (experts sharded by
        # expert-parallel size); mirror GPT-OSS's construction.
        ep_size = max(1, getattr(self.mapping, "moe_ep_size", 1))
        num_local_experts = self.num_experts // ep_size
        alpha = float(getattr(config, "swiglu_alpha", 1.702))
        limit = float(getattr(config, "swiglu_limit", 7.0))
        swiglu_alpha = torch.full((num_local_experts,), alpha, dtype=torch.float32)
        swiglu_beta = torch.ones((num_local_experts,), dtype=torch.float32)
        swiglu_limit = torch.full((num_local_experts,), limit, dtype=torch.float32)
        if torch.cuda.is_available():
            swiglu_alpha = swiglu_alpha.cuda()
            swiglu_beta = swiglu_beta.cuda()
            swiglu_limit = swiglu_limit.cuda()

        # Routed experts. reduce_results=False: the shared expert is summed in
        # first and a single all-reduce closes the layer (see forward).
        self.experts = create_moe(
            routing_method=MiniMaxM2MoeRoutingMethod(
                top_k=self.top_k,
                num_experts=self.num_experts,
                callable_e_score_correction_bias=lambda: self.e_score_correction_bias.e_score_correction_bias,
            ),
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=config.intermediate_size,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer_idx,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            activation_type=ActivationType.Swiglu,
        )

        # Holder gives the generic loader a narrow prefix so mark_consumed does
        # not clobber gate/experts before they load (see MiniMax-M2, #11119).
        self.e_score_correction_bias = _EScoreCorrectionBiasHolder(self.num_experts)

        # One shared expert (clamped SwiGLU), added to the routed output.
        shared_intermediate = getattr(config, "shared_intermediate_size", config.intermediate_size)
        self.shared_experts = GatedMLP(
            hidden_size=self.hidden_dim,
            intermediate_size=shared_intermediate,
            bias=False,
            activation=_make_swigluoai_activation(alpha, limit),
            dtype=config.torch_dtype,
            config=model_config,
            reduce_output=False,
            is_shared_expert=True,
        )

        self.all_reduce = AllReduce(mapping=self.mapping)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        router_logits = self.gate(hidden_states.to(torch.float32))
        routed = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False,
        )
        if self.routed_scaling_factor != 1.0:
            routed = routed * self.routed_scaling_factor
        shared = self.shared_experts(hidden_states)
        final_hidden_states = routed + shared

        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            final_hidden_states = self.all_reduce(final_hidden_states)
        return final_hidden_states


class MiniMaxM3Attention(Attention):
    """MiniMax-M3 attention with per-head Gemma QK norm and partial RoPE.

    Both dense and sparse layers apply a per-head Gemma RMS norm to Q and K
    (over ``head_dim``) before RoPE, and use partial NeoX RoPE (``rotary_dim``
    dims of each head). Sparse layers additionally build the low-dimensional
    "index" branch: separate ``index_q_proj``/``index_k_proj`` projections and
    per-head Gemma norms whose output the MiniMax-M3 sparse attention backend
    scores to pick top-k KV blocks. For the released checkpoint the index branch
    is K-only (``sparse_disable_index_value`` set), so there is no index value
    or index-output projection.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        self.pretrained_config = config
        self.is_sparse_attention_layer = is_sparse_attention_layer(config, layer_idx)
        self.disable_index_value = self.is_sparse_attention_layer and disable_index_value(
            config, layer_idx
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            rope_fusion=True,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

        # MiniMax-M3 uses per-head Gemma RMS norm on Q/K (qk_norm_type
        # 'per_head', use_gemma_norm). head_dim norm does not span the TP shard,
        # so no cross-rank reduction is needed (unlike MiniMax-M2's per-layer
        # norm).
        assert getattr(config, "qk_norm_type", "per_head") == "per_head", (
            f"MiniMax-M3 attention only supports qk_norm_type='per_head', "
            f"got {getattr(config, 'qk_norm_type', None)!r}"
        )
        use_gemma = bool(getattr(config, "use_gemma_norm", True))
        self.q_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=use_gemma,
        )
        self.k_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=use_gemma,
        )

        if self.is_sparse_attention_layer:
            sac = _sparse_attention_config(config)
            self.num_index_heads = int(sac["sparse_num_index_heads"])
            self.index_head_dim = int(sac["sparse_index_dim"])
            self.index_topk_blocks = int(sac["sparse_topk_blocks"])
            self.index_block_size = int(sac["sparse_block_size"])
            # Index Q: one projection over all index heads, sharded column-wise
            # so each rank holds a whole number of index heads (parity with the
            # main Q sharding). Index K: a single replicated head (GQA-style), so
            # it is not tensor-parallel sharded.
            self.index_q_proj = Linear(
                config.hidden_size,
                self.num_index_heads * self.index_head_dim,
                bias=False,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=model_config.get_quant_config(),
            )
            self.index_k_proj = Linear(
                config.hidden_size,
                self.index_head_dim,
                bias=False,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
            )
            self.index_q_norm = RMSNorm(
                hidden_size=self.index_head_dim,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
                use_gemma=use_gemma,
            )
            self.index_k_norm = RMSNorm(
                hidden_size=self.index_head_dim,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
                use_gemma=use_gemma,
            )
            # K-only index branch: value/output projections are absent for the
            # released checkpoint. Guarded rather than silently supported so a
            # future index-value checkpoint fails loudly instead of mis-loading.
            assert self.disable_index_value, (
                "MiniMax-M3 sparse layers are expected to disable the index "
                "value branch (sparse_disable_index_value); an index-value "
                "checkpoint is not yet supported."
            )

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head Gemma RMS norm over ``head_dim`` for Q and K."""
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        return q, k

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        # QK norm runs before RoPE; RoPE itself is fused into the attention op
        # (rope_fusion=True), so super().apply_rope is a no-op that returns the
        # fused (q, k, v) for the kernel to rotate with partial rotary_dim.
        q, k, v = self.split_qkv(q, k, v)
        q, k = self.apply_qk_norm(q, k)
        return super().apply_rope(q, k, v, position_ids)


class MiniMaxM3DecoderLayer(DecoderLayer):
    """A MiniMax-M3 decoder layer: (sparse|dense) attention + (MoE|dense) MLP."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping

        self.self_attn = MiniMaxM3Attention(model_config=model_config, layer_idx=layer_idx)

        if is_moe_layer(config, layer_idx):
            self.mlp = MiniMaxM3MoE(
                model_config=model_config, aux_stream=aux_stream, layer_idx=layer_idx
            )
        else:
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                bias=False,
                activation=_make_swigluoai_activation(
                    float(getattr(config, "swiglu_alpha", 1.702)),
                    float(getattr(config, "swiglu_limit", 7.0)),
                ),
                dtype=config.torch_dtype,
                config=model_config,
            )

        use_gemma = bool(getattr(config, "use_gemma_norm", True))
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=use_gemma,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=use_gemma,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if isinstance(self.mlp, MiniMaxM3MoE):
            hidden_states = self.mlp(hidden_states, attn_metadata)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiniMaxM3Model(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        # bf16 weights/KV cache unless an fp8/fp4 KV cache is configured.
        quant_config = model_config.quant_config
        if quant_config is None or (
            (not quant_config.quant_mode.has_fp8_kv_cache())
            and (not quant_config.quant_mode.has_fp4_kv_cache())
        ):
            model_config.pretrained_config.torch_dtype = torch.bfloat16
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            enable_torch_compile_for_embedding=model_config.enable_torch_compile_for_embedding,
        )

        self.layers = nn.ModuleList(
            [
                MiniMaxM3DecoderLayer(model_config, layer_idx, self.aux_stream)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        use_gemma = bool(getattr(config, "use_gemma_norm", True))
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=use_gemma,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the "
                "same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("MiniMaxM3SparseForCausalLM")
class MiniMaxM3SparseForCausalLM(DecoderModelForCausalLM[MiniMaxM3Model, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(
            MiniMaxM3Model(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: Dict, weight_mapper=None):
        """Load the text tower from the multimodal MiniMax-M3 checkpoint.

        All text weights are stored under a ``language_model.`` prefix; the
        vision tower and multimodal projector are explicitly excluded (text-only
        bring-up). Once the prefix is stripped and non-text keys dropped, the
        remaining keys match this module tree exactly, so the generic loader
        handles the QKV / gate-up fusion, the MoE experts, the shared expert and
        the ``e_score_correction_bias`` holder without any per-key remapping.
        """
        text_weights = {}
        for name, weight in weights.items():
            if name.startswith(_LANGUAGE_MODEL_PREFIX):
                text_weights[name[len(_LANGUAGE_MODEL_PREFIX) :]] = weight
            elif name.startswith(_NON_TEXT_PREFIXES):
                # Vision / projector weights: deferred (text-only bring-up).
                continue
            else:
                # Anything else is unexpected for this checkpoint; keep it so the
                # generic loader surfaces it rather than silently dropping.
                text_weights[name] = weight
        super().load_weights(text_weights, weight_mapper=weight_mapper)

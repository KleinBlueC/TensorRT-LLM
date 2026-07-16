"""ChatGLM3-6B (THUDM/chatglm3-6b) for the TensorRT-LLM PyTorch (``_torch``) flow.

Semantic source of truth: the HuggingFace remote-code model
``modeling_chatglm.py`` shipped with the checkpoint. The ChatGLM3 decoder is a
standard pre-norm transformer with a few family-specific contracts that this
file wires onto existing TensorRT-LLM ``_torch`` modules:

* ``tensor_geometry`` -- fused QKV projection ``[Q: 32*128, K: 2*128, V: 2*128]``
  (multi-query attention, 2 KV groups) with a QKV bias but no output-projection
  bias. Maps directly onto ``Attention``'s fused ``qkv_proj``.
* ``positional_encoding`` -- interleaved (GPT-J style) RoPE applied to only the
  first 64 of each 128-dim head (``partial_rotary_factor = 0.5``), base 10000.
  Expressed as ``PositionEmbeddingType.rope_gptj`` with ``is_neox=False``.
* ``schedule_or_mask`` -- plain causal attention; scale is ``1/sqrt(head_dim)``.
  ChatGLM's ``apply_query_key_layer_scaling`` coefficient cancels out in both
  the SDPA and the non-SDPA reference paths, so no extra per-layer QK scale is
  applied (``q_scaling`` is left at its default of 1.0).
* ``projection_topology`` -- SwiGLU MLP: ``dense_h_to_4h`` emits ``[gate, up]``
  and the block computes ``silu(gate) * up``. This matches ``GatedMLP``'s fused
  ``gate_up_proj`` layout exactly (gate = first half, up = second half).
* ``output_semantics`` -- final RMSNorm before an untied output projection.

The checkpoint uses non-standard HF config field names
(``num_layers``/``multi_query_group_num``/``ffn_hidden_size``/
``padded_vocab_size``/``seq_length``/``layernorm_epsilon``/``kv_channels``).
:func:`normalize_chatglm_config` mirrors these onto the canonical names that the
attention module, MLP, RMSNorm, ``RopeParams.from_config`` and the
``ModelConfig`` -> C++ ``ModelConfig`` conversion (which sizes the KV cache)
expect. Dispatch keys off ``config.architectures[0] == "ChatGLMModel"``.
"""

from typing import Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model


def normalize_chatglm_config(config: PretrainedConfig) -> PretrainedConfig:
    """Mirror ChatGLM's HF config field names onto the canonical names used by
    the TensorRT-LLM ``_torch`` stack. Mutates ``config`` in place (idempotent)
    and returns it.

    This does not change any checkpoint value; it only exposes the same values
    under the names ``Attention`` / ``GatedMLP`` / ``RMSNorm`` /
    ``RopeParams.from_config`` / ``ModelConfig.get_bindings_model_config`` read.
    """
    # dtype: chatglm3-6b ships fp16 weights; guarantee a real torch.dtype so the
    # ``ModelConfig.torch_dtype`` property does not silently fall back to bf16.
    dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if not isinstance(dtype, torch.dtype):
        dtype = torch.float16
    config.torch_dtype = dtype

    # Layer / head / hidden-size names.
    config.num_hidden_layers = config.num_layers
    if getattr(config, "multi_query_attention", False):
        config.num_key_value_heads = config.multi_query_group_num
    else:
        config.num_key_value_heads = config.num_attention_heads
    config.head_dim = config.kv_channels
    config.intermediate_size = config.ffn_hidden_size
    config.vocab_size = config.padded_vocab_size
    config.max_position_embeddings = config.seq_length
    config.rms_norm_eps = config.layernorm_epsilon

    # Positional encoding: interleaved GPT-J RoPE over the first
    # kv_channels/2 == 64 dims of each 128-dim head, base 10000. ``rope_ratio``
    # is 1 for the base chatglm3-6b checkpoint (present only on the long-context
    # variants), but is honored here for robustness.
    config.partial_rotary_factor = 0.5
    config.rope_theta = 10000.0 * float(getattr(config, "rope_ratio", 1.0))

    # Untied output projection (checkpoint has a separate transformer.output_layer).
    config.tie_word_embeddings = False
    return config


class ChatGLMAttention(Attention):
    """ChatGLM3 self-attention: fused biased QKV, MQA (2 KV heads), and
    interleaved partial GPT-J RoPE. Everything else is the stock ``Attention``.
    """

    def __init__(
        self, model_config: ModelConfig[PretrainedConfig], layer_idx: Optional[int] = None
    ):
        config = model_config.pretrained_config
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=RopeParams.from_config(config),
            # ChatGLM3 rotates interleaved (2i, 2i+1) pairs, i.e. GPT-J style.
            is_neox=False,
        )
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            # HF: query_key_value bias = add_bias_linear or add_qkv_bias.
            bias=bool(config.add_bias_linear or config.add_qkv_bias),
            pos_embd_params=pos_embd_params,
            # Apply RoPE in the module (verified equivalent to the source for
            # partial interleaved RoPE) rather than fusing it into the backend
            # kernel; keeps the frequency/pairing contract explicit and is
            # CUDA-graph safe.
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            # HF: self_attention.dense bias = add_bias_linear (False).
            dense_bias=bool(config.add_bias_linear),
            config=model_config,
            head_dim=config.head_dim,
        )


class ChatGLMDecoderLayer(DecoderLayer):
    """Pre-norm decoder block. With ``apply_residual_connection_post_layernorm``
    False (the chatglm3-6b setting), the source residual order is exactly the
    standard fused add+RMSNorm pattern:

        h1 = x  + attn(input_layernorm(x))
        h2 = h1 + mlp(post_attention_layernorm(h1))
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config

        self.self_attn = ChatGLMAttention(model_config, layer_idx=layer_idx)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=bool(config.add_bias_linear),
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ):
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
        hidden_states = self.mlp(hidden_states, **kwargs)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
        return hidden_states, residual


class ChatGLMModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                ChatGLMDecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
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
                spec_metadata=spec_metadata,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("ChatGLMModel")
class ChatGLMForCausalLM(DecoderModelForCausalLM[ChatGLMModel, PretrainedConfig]):
    """ChatGLM3-6B causal LM.

    Registered under the checkpoint's ``architectures[0] == "ChatGLMModel"``
    even though the class name here is ``...ForCausalLM``; ``_torch`` dispatch
    keys off the ``architectures`` string, and the base ChatGLM checkpoint only
    ever runs as a decoder-only causal LM in this flow.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        normalize_chatglm_config(model_config.pretrained_config)
        super().__init__(
            ChatGLMModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: Dict, **kwargs):
        """Translate HF ChatGLM weight keys onto the canonical ``_torch``
        module paths, then defer to the standard loader.

        The fused ``query_key_value`` and ``dense_h_to_4h`` tensors are split
        into ``q/k/v`` and ``gate/up`` respectively -- the byte layout already
        matches (Q|K|V and gate|up ordering), so the standard fused-linear
        loader re-concatenates them without any reordering.
        """
        super().load_weights(remap_chatglm_weights(weights, self.config), **kwargs)


def remap_chatglm_weights(weights: Dict, config: PretrainedConfig) -> Dict:
    """Rename HF ChatGLM state-dict keys to TensorRT-LLM ``_torch`` module paths.

    Raises ``KeyError`` on any unmapped source key so that a checkpoint layout
    change is a hard failure rather than a silent partial load. The only source
    tensor intentionally skipped is the derived ``rotary_pos_emb.inv_freq``
    buffer, which TensorRT-LLM regenerates from ``RopeParams``.
    """
    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    ffn = config.intermediate_size

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in weights.items():
        if key == "transformer.embedding.word_embeddings.weight":
            remapped["model.embed_tokens.weight"] = value
        elif key == "transformer.encoder.final_layernorm.weight":
            remapped["model.norm.weight"] = value
        elif key == "transformer.output_layer.weight":
            remapped["lm_head.weight"] = value
        elif key.startswith("transformer.rotary_pos_emb"):
            # Derived buffer (inv_freq); regenerated from RopeParams.
            continue
        elif key.startswith("transformer.encoder.layers."):
            idx, sub = key[len("transformer.encoder.layers.") :].split(".", 1)
            prefix = f"model.layers.{idx}"
            if sub == "self_attention.query_key_value.weight":
                q, k, v = torch.split(value[:], [q_size, kv_size, kv_size], dim=0)
                remapped[f"{prefix}.self_attn.q_proj.weight"] = q
                remapped[f"{prefix}.self_attn.k_proj.weight"] = k
                remapped[f"{prefix}.self_attn.v_proj.weight"] = v
            elif sub == "self_attention.query_key_value.bias":
                q, k, v = torch.split(value[:], [q_size, kv_size, kv_size], dim=0)
                remapped[f"{prefix}.self_attn.q_proj.bias"] = q
                remapped[f"{prefix}.self_attn.k_proj.bias"] = k
                remapped[f"{prefix}.self_attn.v_proj.bias"] = v
            elif sub == "self_attention.dense.weight":
                remapped[f"{prefix}.self_attn.o_proj.weight"] = value
            elif sub == "mlp.dense_h_to_4h.weight":
                gate, up = torch.split(value[:], [ffn, ffn], dim=0)
                remapped[f"{prefix}.mlp.gate_proj.weight"] = gate
                remapped[f"{prefix}.mlp.up_proj.weight"] = up
            elif sub == "mlp.dense_4h_to_h.weight":
                remapped[f"{prefix}.mlp.down_proj.weight"] = value
            elif sub == "input_layernorm.weight":
                remapped[f"{prefix}.input_layernorm.weight"] = value
            elif sub == "post_attention_layernorm.weight":
                remapped[f"{prefix}.post_attention_layernorm.weight"] = value
            else:
                raise KeyError(f"Unmapped ChatGLM source weight: {key}")
        else:
            raise KeyError(f"Unmapped ChatGLM source weight: {key}")
    return remapped

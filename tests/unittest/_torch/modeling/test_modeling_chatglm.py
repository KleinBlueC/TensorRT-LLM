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
"""Unit tests for the ChatGLM3-6B ``_torch`` bring-up.

Single unit-test bucket for the model, covering three distinct concerns:

* registration + config normalization (the checkpoint routes to
  ``ChatGLMForCausalLM`` via the auto-model path and its ChatGLM-specific field
  names are mirrored onto the canonical ones);
* strict weight accounting (every trainable source tensor maps to exactly one
  canonical ``_torch`` key);
* checkpoint-scale forward parity vs the HF source model on the TRTLLM backend +
  ``KVCacheManagerV2``, for both the eager and CUDA-graph configs.

The checkpoint is resolved under ``llm_models_root()``; a checkpoint-backed test
fails loudly rather than skipping when it is absent.
"""

import glob
import os

import pytest
import torch
from utils.llm_data import llm_models_root


def _chatglm3_path():
    return os.path.join(llm_models_root(check=True), "chatglm3-6b")


# chatglm3-6b config.json reference dimensions.
NUM_LAYERS = 28
NUM_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
FFN = 13696
VOCAB = 65024
Q_SIZE = NUM_HEADS * HEAD_DIM  # 4096
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 256


def _load_bin_state_dict(ckpt):
    """Full state dict from the ``.bin`` shard set (chatglm3-6b's shipped
    safetensors set is missing a shard, so ``.bin`` is the complete format)."""
    shards = sorted(glob.glob(os.path.join(ckpt, "pytorch_model-*.bin")))
    assert shards, f"No pytorch_model-*.bin shards under {ckpt}"
    state_dict = {}
    for shard in shards:
        state_dict.update(torch.load(shard, map_location="cpu", weights_only=True))
    return state_dict


def _hf_source_config():
    # Backfill the legacy generation/output defaults the checkpoint's 2023 remote
    # modeling code reads at construction; transformers>=5 moved them to
    # GenerationConfig. Correctness-neutral under greedy decoding.
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(_chatglm3_path(), trust_remote_code=True)
    for name, value in {
        "max_length": getattr(cfg, "seq_length", 8192),
        "max_new_tokens": None,
        "min_length": 0,
        "use_cache": True,
        "do_sample": False,
        "num_beams": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
        "is_encoder_decoder": False,
        "problem_type": None,
    }.items():
        if not hasattr(cfg, name):
            setattr(cfg, name, value)
    return cfg


def _build_hf_model():
    # Build the remote architecture from config + load the complete .bin weights
    # directly: transformers 5.5.x from_pretrained finalization references
    # all_tied_weights_keys, absent on this 2023 remote model class.
    from transformers import AutoModelForCausalLM

    ckpt = _chatglm3_path()
    model = AutoModelForCausalLM.from_config(_hf_source_config(), trust_remote_code=True)
    missing, unexpected = model.load_state_dict(_load_bin_state_dict(ckpt), strict=False)
    assert not missing, f"HF source model missing weights: {sorted(missing)[:8]}"
    assert set(unexpected) <= {"transformer.rotary_pos_emb.inv_freq"}, (
        f"unexpected source keys: {sorted(unexpected)[:8]}"
    )
    return model.to(torch.float16).cuda().eval()


def _build_trtllm_model(backend="TRTLLM"):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_chatglm import ChatGLMForCausalLM

    ckpt = _chatglm3_path()
    model_config = ModelConfig.from_pretrained(ckpt, trust_remote_code=True, attn_backend=backend)
    with torch.device("cuda"):
        model = ChatGLMForCausalLM(model_config).to("cuda").eval()
    model.load_weights(_load_bin_state_dict(ckpt))
    model.post_load_weights()
    return model, model_config


def _metrics(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (
        (a - b).abs().max().item(),
        (a - b).abs().mean().item(),
        torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    )


def test_chatglm3_registration_and_config_normalization():
    """The checkpoint routes to ``ChatGLMForCausalLM`` via the auto-model path
    and ``normalize_chatglm_config`` mirrors ChatGLM's field names onto the
    canonical ones the ``_torch`` attention/MLP/KV-cache stack reads."""
    import tensorrt_llm  # noqa: F401  (populate the model registry)
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_chatglm import (
        ChatGLMForCausalLM,
        _prepare_chatglm_model_config,
        normalize_chatglm_config,
    )
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

    assert MODEL_CLASS_MAPPING.get("ChatGLMModel") is ChatGLMForCausalLM

    model_config = ModelConfig.from_pretrained(_chatglm3_path(), trust_remote_code=True)
    model_config.extra_attrs["allreduce_dtype"] = None
    cfg = _prepare_chatglm_model_config(model_config)
    assert cfg.architectures[0] == "ChatGLMModel"
    normalize_chatglm_config(cfg)
    assert cfg.num_hidden_layers == NUM_LAYERS
    assert cfg.num_attention_heads == NUM_HEADS
    assert cfg.num_key_value_heads == NUM_KV_HEADS
    assert cfg.head_dim == HEAD_DIM
    assert cfg.intermediate_size == FFN
    assert cfg.vocab_size == VOCAB
    assert cfg.max_position_embeddings == 8192
    assert abs(cfg.rope_theta - 10000.0) < 1e-6
    assert abs(cfg.partial_rotary_factor - 0.5) < 1e-9
    assert cfg.tie_word_embeddings is False
    assert cfg.torch_dtype == torch.float16
    assert model_config.extra_attrs["allreduce_dtype"] == torch.float16


def test_chatglm3_weight_accounting():
    """Every trainable source tensor maps to exactly one canonical ``_torch``
    key (only the derived ``rotary_pos_emb.inv_freq`` buffer is dropped), and the
    fused QKV / gate-up splits partition the source tensors exactly."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_chatglm import (
        normalize_chatglm_config,
        remap_chatglm_weights,
    )

    cfg = ModelConfig.from_pretrained(_chatglm3_path(), trust_remote_code=True).pretrained_config
    normalize_chatglm_config(cfg)

    state_dict = _load_bin_state_dict(_chatglm3_path())
    source_keys = set(state_dict)
    assert "transformer.rotary_pos_emb.inv_freq" in source_keys
    trainable_source = source_keys - {"transformer.rotary_pos_emb.inv_freq"}

    remapped = remap_chatglm_weights(state_dict, cfg)
    assert "transformer.rotary_pos_emb.inv_freq" not in remapped
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            assert f"{p}.{suffix}" in remapped, f"missing remapped {p}.{suffix}"
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        assert name in remapped

    qkv_w = state_dict["transformer.encoder.layers.0.self_attention.query_key_value.weight"]
    assert qkv_w.shape[0] == Q_SIZE + 2 * KV_SIZE
    assert remapped["model.layers.0.self_attn.q_proj.weight"].shape[0] == Q_SIZE
    assert remapped["model.layers.0.self_attn.k_proj.weight"].shape[0] == KV_SIZE
    assert remapped["model.layers.0.self_attn.v_proj.weight"].shape[0] == KV_SIZE
    h_to_4h = state_dict["transformer.encoder.layers.0.mlp.dense_h_to_4h.weight"]
    assert h_to_4h.shape[0] == 2 * FFN
    assert remapped["model.layers.0.mlp.gate_proj.weight"].shape[0] == FFN
    assert remapped["model.layers.0.mlp.up_proj.weight"].shape[0] == FFN
    # 7 trainable tensors per layer (fused QKV weight+bias, dense, fused MLP
    # h_to_4h, 4h_to_h, 2 layernorms) + 3 global (embedding, final norm, lm_head).
    assert len(trainable_source) == 7 * NUM_LAYERS + 3


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ChatGLM3 forward-parity test requires CUDA."
)
@pytest.mark.parametrize(
    "use_cuda_graph",
    [False, True],
    ids=["baseline_nograph", "enabled_cudagraph"],
)
def test_chatglm3_forward_parity_vs_hf(use_cuda_graph, request):
    """Checkpoint-scale final-logit parity vs the HF source model on the TRTLLM
    backend + ``KVCacheManagerV2``, over a prefill and a decode/cache-reuse step.

    The enabled entry drives the decode step through a real CUDA-graph
    capture/replay (hard-path evidence: a silent eager fallback leaves the graph
    store empty and fails the assertion). Correctness is greedy-argmax token
    equality plus a cosine floor on the final logits.
    """
    from _torch.helpers import create_mock_cuda_graph_runner

    import tensorrt_llm
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    backend = "TRTLLM"
    device = torch.device("cuda")
    model, model_config = _build_trtllm_model(backend)
    hf_model = _build_hf_model()

    cfg = model.config
    assert cfg.num_attention_heads == NUM_HEADS
    assert cfg.num_key_value_heads == NUM_KV_HEADS
    assert cfg.head_dim == HEAD_DIM

    metadata_cls = get_attention_backend(backend).Metadata
    tokens_per_block = 128
    num_blocks = 8
    max_seq_len = num_blocks * tokens_per_block
    kv_cache_manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=max_seq_len, use_kv_cache_manager_v2=True),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=model_config.mapping,
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    request.addfinalizer(kv_cache_manager.shutdown)

    # ---- Context / prefill ----
    input_ids = torch.tensor(
        [64790, 64792, 790, 30951, 517, 30910, 30939], dtype=torch.int, device=device
    )
    request_ids = [1]
    prompt_lens = [input_ids.size(-1)]
    kv_cache_manager.add_dummy_requests(request_ids, [input_ids.size(-1)])

    ctx_metadata = metadata_cls(
        seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
        num_contexts=1,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
        max_num_requests=1,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
    )
    position_ids = torch.arange(0, input_ids.size(-1), device=device).unsqueeze(0)
    with torch.inference_mode():
        ctx_metadata.prepare()
        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attn_metadata=ctx_metadata
        )
        hf_out = hf_model.forward(
            input_ids=input_ids.unsqueeze(0), position_ids=position_ids, use_cache=True
        )
    hf_last = hf_out.logits[:, -1].float()
    max_abs, mean_abs, cos = _metrics(logits, hf_last)
    print(f"[prefill] max_abs={max_abs:.4f} mean_abs={mean_abs:.5f} cos={cos:.6f}")
    assert cos > 0.99, f"prefill cosine too low: {cos}"
    assert torch.argmax(logits, dim=-1).item() == torch.argmax(hf_last, dim=-1).item(), (
        "prefill greedy-argmax token mismatch vs HF"
    )

    # ---- Decode / cache reuse ----
    next_tok = torch.argmax(hf_last, dim=-1).to(torch.int)
    gen_input_ids = next_tok.view(1)
    gen_metadata = metadata_cls(
        seq_lens=torch.tensor([1], dtype=torch.int),
        num_contexts=0,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[input_ids.size(-1)]
        ),
        max_num_requests=1,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
    )
    gen_position_ids = torch.tensor([input_ids.size(-1)], device=device).unsqueeze(0)

    graph_runner = create_mock_cuda_graph_runner(1) if use_cuda_graph else None
    if graph_runner is not None:
        request.addfinalizer(graph_runner.clear)
    if use_cuda_graph:
        gen_metadata = gen_metadata.create_cuda_graph_metadata(1)

    def run_decode():
        gen_metadata.prepare()
        if not use_cuda_graph:
            return model.forward(
                input_ids=gen_input_ids, position_ids=gen_position_ids, attn_metadata=gen_metadata
            )
        inputs = {
            "input_ids": gen_input_ids,
            "position_ids": gen_position_ids,
            "attn_metadata": gen_metadata,
        }
        key = (1, 0, False)
        graph_runner.capture(key, lambda inp: model.forward(**inp), inputs)
        assert key in graph_runner.graphs, (
            "decode CUDA graph was not captured (silent eager fallback)"
        )
        out = None
        for _ in range(2):  # capture once, replay twice
            gen_metadata.prepare()
            out = graph_runner.replay(key, inputs)
        return out

    with torch.inference_mode():
        gen_logits = run_decode()
        hf_gen = hf_model.forward(
            input_ids=gen_input_ids.unsqueeze(0),
            position_ids=gen_position_ids,
            past_key_values=hf_out.past_key_values,
            use_cache=True,
        )
    hf_gen_last = hf_gen.logits[:, -1].float()
    g_max_abs, g_mean_abs, g_cos = _metrics(gen_logits, hf_gen_last)
    print(
        f"[decode] cuda_graph={use_cuda_graph} max_abs={g_max_abs:.4f} "
        f"mean_abs={g_mean_abs:.5f} cos={g_cos:.6f}"
    )
    assert g_cos > 0.99, f"decode cosine too low: {g_cos}"
    assert torch.argmax(gen_logits, dim=-1).item() == torch.argmax(hf_gen_last, dim=-1).item(), (
        "decode greedy-argmax token mismatch vs HF"
    )

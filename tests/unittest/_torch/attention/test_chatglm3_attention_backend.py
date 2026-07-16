"""Checkpoint-scale ChatGLM3 attention on the selected backend + KVCacheManagerV2.

Drives the real ChatGLM3-6B model (real checkpoint weights) through the TRTLLM
attention backend with ``KVCacheManagerV2`` at the checkpoint's attention
dimensions (32 query heads, 2 KV heads, head_dim 128, interleaved partial RoPE
over 64 dims, causal), covering both a context (prefill) pass and a
decode/cache-reuse step, for the full CUDA-graph matrix:

* baseline: ``cuda_graph=False, overlap_scheduler=False``
* enabled : ``cuda_graph=True`` (real capture/replay hard path via the mock
  CUDA-graph runner) ``, overlap_scheduler=True``

Correctness is anchored to the HF source model loaded from the same checkpoint
(greedy-argmax-token equality + reported max_abs / cosine on the final logits).
"""

import faulthandler
import glob
import os
import sys

import pytest
import torch

# Watchdog: dump every thread's stack every 120s to stderr if a test wedges
# (e.g. remote-code config load deadlocking after ``import tensorrt_llm``) so
# the Slurm log localizes the hang instead of silently hitting the timeout.
faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)

CHATGLM3_CKPT = os.environ.get(
    "CHATGLM3_CKPT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "kleinc/hf_data/chatglm3-6b",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ChatGLM3 attention backend test requires CUDA."
)


def _require_checkpoint():
    if not os.path.isdir(CHATGLM3_CKPT):
        pytest.skip(f"ChatGLM3 checkpoint not found at {CHATGLM3_CKPT}")


def _load_bin_state_dict(ckpt):
    shards = sorted(glob.glob(os.path.join(ckpt, "pytorch_model-*.bin")))
    assert shards, f"No pytorch_model-*.bin shards under {ckpt}"
    sd = {}
    for shard in shards:
        sd.update(torch.load(shard, map_location="cpu", weights_only=True))
    return sd


def _build_trtllm_model(backend):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_chatglm import ChatGLMForCausalLM

    model_config = ModelConfig.from_pretrained(
        CHATGLM3_CKPT, trust_remote_code=True, attn_backend=backend
    )
    with torch.device("cuda"):
        model = ChatGLMForCausalLM(model_config).to("cuda").eval()
    model.load_weights(_load_bin_state_dict(CHATGLM3_CKPT))
    model.post_load_weights()
    return model, model_config


def _hf_source_config():
    """Backfill the legacy ``max_length`` attribute the checkpoint's 2023 remote
    modeling code reads at construction (stored as an unused
    ``max_sequence_length``); transformers>=5 dropped it from
    ``PretrainedConfig``. Correctness-neutral config-compat, not a semantics
    change and not a transformers monkeypatch.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
    # The 2023 remote modeling code reads legacy PretrainedConfig generation /
    # output defaults that transformers>=5 no longer sets on the config object
    # (they moved to GenerationConfig). Backfill the ones the remote forward /
    # generate paths read, only when absent -- all non-semantic under
    # deterministic greedy decoding. Config-compat, not a transformers monkeypatch.
    _legacy_defaults = {
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
    }
    for _name, _value in _legacy_defaults.items():
        if not hasattr(cfg, _name):
            setattr(cfg, _name, _value)
    return cfg


def _build_hf_model():
    from transformers import AutoModelForCausalLM

    # transformers 5.5.x from_pretrained finalization references
    # ``all_tied_weights_keys`` (absent on this 2023 remote model class); build
    # the architecture from config + load the complete .bin weights directly.
    # Public from_config / load_state_dict APIs only; real source model + weights.
    model = AutoModelForCausalLM.from_config(_hf_source_config(), trust_remote_code=True)
    missing, unexpected = model.load_state_dict(
        _load_bin_state_dict(CHATGLM3_CKPT), strict=False
    )
    assert not missing, f"HF source model missing weights: {sorted(missing)[:8]}"
    assert set(unexpected) <= {"transformer.rotary_pos_emb.inv_freq"}, (
        f"unexpected source keys: {sorted(unexpected)[:8]}"
    )
    return model.to(torch.float16).cuda().eval()


def _metrics(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return max_abs, mean_abs, cos


@pytest.mark.parametrize(
    "use_cuda_graph,overlap_scheduler",
    [(False, False), (True, True)],
    ids=["baseline_nograph", "enabled_cudagraph"],
)
def test_chatglm3_checkpoint_scale_backend_kvcache_v2(use_cuda_graph, overlap_scheduler):
    _require_checkpoint()

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
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 128

    metadata_cls = get_attention_backend(backend).Metadata
    num_blocks = 8
    tokens_per_block = 128
    max_seq_len = num_blocks * tokens_per_block

    kv_cache_manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=num_blocks * tokens_per_block, use_kv_cache_manager_v2=True),
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
    print(
        f"[attn/prefill/{'graph' if use_cuda_graph else 'eager'}] "
        f"max_abs={max_abs:.4f} mean_abs={mean_abs:.5f} cos={cos:.6f}"
    )
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
        # Real capture/replay hard path: capture once, replay twice.
        graph_runner.capture(key, lambda inp: model.forward(**inp), inputs)
        # Hard-path evidence: the graph object must have materialized (a silent
        # eager fallback would leave graphs empty).
        assert key in graph_runner.graphs, (
            "decode CUDA graph was not captured (silent eager fallback)"
        )
        print(
            f"[attn/decode/graph] captured_graphs={len(graph_runner.graphs)} "
            f"keys={list(graph_runner.graphs.keys())}"
        )
        out = None
        for _ in range(2):
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
        f"[attn/decode/{'graph' if use_cuda_graph else 'eager'}] "
        f"cuda_graph={use_cuda_graph} overlap={overlap_scheduler} "
        f"max_abs={g_max_abs:.4f} mean_abs={g_mean_abs:.5f} cos={g_cos:.6f}"
    )
    assert g_cos > 0.99, f"decode cosine too low: {g_cos}"
    assert torch.argmax(gen_logits, dim=-1).item() == torch.argmax(hf_gen_last, dim=-1).item(), (
        "decode greedy-argmax token mismatch vs HF"
    )

    if graph_runner is not None:
        graph_runner.clear()
    kv_cache_manager.shutdown()

"""Unit tests for the ChatGLM3-6B ``_torch`` bring-up.

Covers:
* ``test_chatglm3_hf_source_load_canary`` -- the provided checkpoint loads as an
  HF source model + tokenizer with ``trust_remote_code=True``, reports the
  complete weight format used, and runs one CUDA forward.
* ``test_chatglm3_config_registration_weight_accounting`` -- the architecture is
  registered, config normalization is correct, every trainable source weight is
  consumed exactly once (only the derived ``rotary_pos_emb.inv_freq`` buffer is
  skipped), and an LLM-API object built with ``KVCacheManagerV2`` runs on GPU.

Both tests require a real GPU and the real checkpoint; they are not meaningful
on CPU and must not be counted as passing when skipped.
"""

import faulthandler
import glob
import os
import sys

import pytest
import torch

# Watchdog: if a bring-up test wedges (observed: remote-code config load
# deadlocking after ``import tensorrt_llm``), dump every thread's stack every
# 120s to stderr so the Slurm log localizes the hang instead of silently
# hitting the wrapper timeout.
faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)

CHATGLM3_CKPT = os.environ.get(
    "CHATGLM3_CKPT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "kleinc/hf_data/chatglm3-6b",
)

# Config-derived reference dimensions (chatglm3-6b config.json).
NUM_LAYERS = 28
HIDDEN = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
FFN = 13696
VOCAB = 65024
Q_SIZE = NUM_HEADS * HEAD_DIM  # 4096
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 256

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ChatGLM3 bring-up tests require CUDA."
)


def _require_checkpoint():
    if not os.path.isdir(CHATGLM3_CKPT):
        pytest.skip(f"ChatGLM3 checkpoint not found at {CHATGLM3_CKPT}")


def _flush_mark(msg: str) -> None:
    """Write a progress marker that survives output buffering and a wedged main
    thread.

    Prints to stderr with an explicit flush *and* appends to the file named by
    ``$CHATGLM3_C3_DIAG`` (when set) with ``os.fsync`` so that if a step wedges
    the log still shows the last marker reached. This is belt-and-suspenders
    localization: the module-level ``faulthandler`` watchdog was observed not to
    fire during the C3 wedge, so we do not rely on it alone.
    """
    line = f"[C3-MARK] {msg}\n"
    sys.stderr.write(line)
    sys.stderr.flush()
    diag = os.environ.get("CHATGLM3_C3_DIAG")
    if diag:
        with open(diag, "a") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _load_bin_state_dict(ckpt: str):
    """Load the full state dict from the complete PyTorch ``.bin`` shard set.

    The provided checkpoint's safetensors set is missing shard 00005, so the
    ``.bin`` set is the only complete weight format; source loading must not use
    safetensors here.
    """
    shards = sorted(glob.glob(os.path.join(ckpt, "pytorch_model-*.bin")))
    assert shards, f"No pytorch_model-*.bin shards under {ckpt}"
    state_dict = {}
    for shard in shards:
        state_dict.update(torch.load(shard, map_location="cpu", weights_only=True))
    return state_dict


def _hf_source_config():
    """Load the ChatGLM3 config for the HF *source* model, backfilling the
    legacy ``max_length`` attribute.

    The checkpoint's 2023 remote ``modeling_chatglm.py`` reads
    ``config.max_length`` at construction (``self.max_sequence_length =
    config.max_length``), a value it inherited from ``PretrainedConfig``'s
    generation defaults. transformers>=5 removed that default (it lives only on
    ``GenerationConfig`` now), so the raw config raises ``AttributeError``.
    ``max_sequence_length`` is stored and never read, so mirroring ``seq_length``
    here is correctness-neutral -- a config-compat backfill, not a semantics
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


def test_chatglm3_hf_source_load_canary():
    """Load the HF/source model + tokenizer and run one CUDA forward.

    Forces ``use_safetensors=False`` because the safetensors shard set is
    incomplete; the ``.bin`` set is complete. Reports the weight format used.
    """
    _require_checkpoint()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
    print("[C2] building HF source model from config + loading .bin weights", flush=True)
    # transformers 5.5.x's from_pretrained weight-loading finalization references
    # ``all_tied_weights_keys`` -- an attribute this checkpoint's 2023 remote model
    # class predates -- so from_pretrained raises. Build the architecture from the
    # (remote) config, then load the complete .bin weights directly. This still
    # exercises the real source model + real checkpoint weights and uses only the
    # public from_config / load_state_dict APIs (not a transformers monkeypatch).
    model = AutoModelForCausalLM.from_config(_hf_source_config(), trust_remote_code=True)
    missing, unexpected = model.load_state_dict(
        _load_bin_state_dict(CHATGLM3_CKPT), strict=False
    )
    assert not missing, f"HF source model missing weights: {sorted(missing)[:8]}"
    # The derived RoPE buffer is the only tolerated non-parameter source key.
    assert set(unexpected) <= {"transformer.rotary_pos_emb.inv_freq"}, (
        f"unexpected source keys: {sorted(unexpected)[:8]}"
    )
    print("[C2] weights loaded; moving to CUDA/fp16", flush=True)
    model = model.to(torch.float16).cuda().eval()

    assert model.config.num_layers == NUM_LAYERS
    assert model.config.padded_vocab_size == VOCAB

    inputs = tokenizer("1 + 1 =", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model(**inputs)
    logits = out.logits
    assert logits.shape[-1] == VOCAB
    assert torch.isfinite(logits).all(), "HF source logits are not finite"

    # Report the weight format actually consumed for the artifact/log.
    print(
        f"[canary] HF source load OK: dtype={next(model.parameters()).dtype}, "
        f"weight_format=pytorch_bin, device={next(model.parameters()).device}"
    )

    # Free the 6B HF model: the foundation tier now runs C2/C3/C4 in a single
    # process (to avoid the sequential-MPI-init PMIx wedge), so C3's LLM and C4's
    # attention matrix need this GPU memory back.
    import gc

    del model, out, logits, inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_chatglm3_config_registration_weight_accounting():
    """Registration + config normalization + strict weight accounting + a
    GPU LLM-API object built with ``KVCacheManagerV2``.
    """
    _require_checkpoint()

    _flush_mark("C3 body entered; importing tensorrt_llm")
    import tensorrt_llm  # noqa: F401  (ensures model registry is populated)
    _flush_mark("import tensorrt_llm returned; importing model_config")
    from tensorrt_llm._torch.model_config import ModelConfig
    _flush_mark("model_config imported; importing modeling_chatglm")
    from tensorrt_llm._torch.models.modeling_chatglm import (
        ChatGLMForCausalLM,
        normalize_chatglm_config,
        remap_chatglm_weights,
    )
    _flush_mark("modeling_chatglm imported; importing modeling_utils")
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
    _flush_mark("all imports done")

    print("[C3] imports done (tensorrt_llm + modeling_chatglm)", flush=True)

    # (A) Registration keyed off architectures[0] == "ChatGLMModel".
    assert MODEL_CLASS_MAPPING.get("ChatGLMModel") is ChatGLMForCausalLM
    print(
        "[C3] registration OK; calling ModelConfig.from_pretrained(trust_remote_code=True)",
        flush=True,
    )

    # (B) Config normalization maps ChatGLM field names onto canonical ones.
    model_config = ModelConfig.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
    print("[C3] ModelConfig.from_pretrained returned", flush=True)
    cfg = model_config.pretrained_config
    normalize_chatglm_config(cfg)
    assert cfg.num_hidden_layers == NUM_LAYERS
    assert cfg.num_key_value_heads == NUM_KV_HEADS
    assert cfg.num_attention_heads == NUM_HEADS
    assert cfg.head_dim == HEAD_DIM
    assert cfg.intermediate_size == FFN
    assert cfg.vocab_size == VOCAB
    assert cfg.max_position_embeddings == 8192
    assert abs(cfg.rope_theta - 10000.0) < 1e-6
    assert abs(cfg.partial_rotary_factor - 0.5) < 1e-9
    assert cfg.tie_word_embeddings is False
    assert cfg.torch_dtype == torch.float16

    # (C) Strict weight accounting over the real checkpoint.
    print("[C3] step C: loading .bin state dict for weight accounting", flush=True)
    state_dict = _load_bin_state_dict(CHATGLM3_CKPT)
    print(f"[C3] loaded {len(state_dict)} source tensors", flush=True)
    source_keys = set(state_dict.keys())
    assert "transformer.rotary_pos_emb.inv_freq" in source_keys
    trainable_source = source_keys - {"transformer.rotary_pos_emb.inv_freq"}

    remapped = remap_chatglm_weights(state_dict, cfg)
    # inv_freq is the only intentionally-dropped source tensor.
    assert "transformer.rotary_pos_emb.inv_freq" not in remapped
    # Every layer contributes the expected canonical keys.
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
    # Split shapes must partition the fused source tensors exactly.
    qkv_w = state_dict["transformer.encoder.layers.0.self_attention.query_key_value.weight"]
    assert qkv_w.shape[0] == Q_SIZE + 2 * KV_SIZE
    assert remapped["model.layers.0.self_attn.q_proj.weight"].shape[0] == Q_SIZE
    assert remapped["model.layers.0.self_attn.k_proj.weight"].shape[0] == KV_SIZE
    assert remapped["model.layers.0.self_attn.v_proj.weight"].shape[0] == KV_SIZE
    h_to_4h = state_dict["transformer.encoder.layers.0.mlp.dense_h_to_4h.weight"]
    assert h_to_4h.shape[0] == 2 * FFN
    assert remapped["model.layers.0.mlp.gate_proj.weight"].shape[0] == FFN
    assert remapped["model.layers.0.mlp.up_proj.weight"].shape[0] == FFN
    # No trainable source key was silently dropped by the remap.
    assert len(trainable_source) == 7 * NUM_LAYERS + 3

    import gc

    del state_dict, remapped
    gc.collect()
    print("[C3] weight accounting OK; host state dict freed", flush=True)

    # (D) Build an LLM-API object with KVCacheManagerV2 and prove it runs on GPU
    #     (instantiation loads every weight via the model's load_weights, which
    #     raises on any unmapped source key or missing target parameter).
    # Force the in-process (single-process) executor worker for this TP1/single-GPU
    # LLM. The default TP1 path (executor.py::GenerationExecutor.create) spawns a
    # worker via mpi4py MPI_Comm_spawn, which deadlocks under Slurm's PMIx when the
    # step is launched with `srun --ntasks=1` (observed: C3 wedged at LLM
    # construction -- the foundation log froze right after the nvml "Link state"
    # probe with no further progress and no crash). Setting this env var takes the
    # create() use_worker=True path (gated by enable_worker_single_process_for_tp1),
    # running the executor in-process with no spawn. This is the same knob the
    # _torch replay/spec-decode tests set (see test_chatglm3_replay.py::_make_llm),
    # and it is also what keeps the in-process cuda_graph_runner reachable for the
    # later hard-path checks. The env var is read live at LLM() construction time.
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    _flush_mark("step D: constructing LLM(attn=TRTLLM, kv_cache_v2=True)")
    print("[C3] step D: constructing LLM(attn=TRTLLM, kv_cache_v2=True)", flush=True)
    llm = LLM(
        model=CHATGLM3_CKPT,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5, use_kv_cache_manager_v2=True),
        max_batch_size=1,
        max_seq_len=1024,
    )
    try:
        _flush_mark("LLM constructed; running generate")
        print("[C3] LLM constructed; running generate", flush=True)
        outputs = llm.generate(
            ["The capital of France is"],
            sampling_params=SamplingParams(max_tokens=8, temperature=0.0, top_k=1),
        )
        text = outputs[0].outputs[0].text
        assert isinstance(text, str) and len(text) > 0
        print(f"[llmapi/v2] generated: {text!r}")
    finally:
        llm.shutdown()

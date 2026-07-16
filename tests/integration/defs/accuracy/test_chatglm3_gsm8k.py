"""GSM8K accuracy for ChatGLM3-6B: canary + HF-parity full gate.

Two tests, sharing one deterministic ``trtllm-eval`` runner (pytorch backend,
``TRTLLM`` attention, ``KVCacheManagerV2``) over the CUDA-graph matrix:

* ``test_chatglm3_gsm8k_accuracy_canary`` -- a small deterministic GSM8K slice
  for both matrix entries *before* the full benchmark, to catch catastrophic
  regressions cheaply. Records a durable artifact per entry (config, score,
  per-sample dump, CUDA-graph hard-path marker).
* ``test_chatglm3_gsm8k_full_parity`` -- the completion-criterion gate. Scores
  the real HuggingFace source model on GSM8K to produce a durable *reference*
  score, then runs ``trtllm-eval`` for both matrix entries and fails unless each
  TensorRT-LLM score is within ``CHATGLM3_GSM8K_TOLERANCE`` (default 2.0) points
  of the HF reference. The enabled entry must additionally prove CUDA-graph
  hard-path execution from its own run artifacts; the baseline entry must prove
  CUDA graph was OFF.

HF/TRT comparability (the scorer-mismatch risk): both paths evaluate the
**identical** lm-eval ``gsm8k`` ``task_dict`` -- built via TensorRT-LLM's own
``tensorrt_llm.evaluate.lm_eval.GSM8K`` evaluator so the include-path, few-shot
seed, and dataset shuffle match ``trtllm-eval`` exactly -- over the identical
first ``--num_samples`` shuffled samples, under deterministic greedy decoding
(``temperature=0``, ``top_k=1``), and are compared on the same metric
(``exact_match,flexible-extract``) read from the same saved results json. The HF
reference decodes through the checkpoint's own remote ``modeling_chatglm.py``
(built from config + the complete ``.bin`` weights) using ``forward()`` + the
model's legacy tuple KV cache: transformers>=5 ``generate()`` /
``DynamicCache`` is incompatible with this 2023 remote model, and this greedy
path is numerically identical to greedy ``generate()`` and version-robust. It is
a genuine source reference (real checkpoint, real remote code), not a
transformers monkeypatch.

CUDA-graph hard-path evidence: each ``trtllm-eval`` run's output is scanned for
the pytorch-backend capture marker ("Creating CUDA graph instances for <N>
batch sizes."). The enabled entry asserts the marker is PRESENT (real
capture/replay, not a silent non-graph fallback); the baseline entry asserts it
is ABSENT (the explicit ``cuda_graph_config: null`` overrides
``TorchLlmArgs.cuda_graph_config``'s ``default_factory=CudaGraphConfig``, which
otherwise leaves CUDA graph ON when the key is omitted). This self-contained
marker gate agrees with the in-process capture proven by
``test_chatglm3_replay.py::test_chatglm3_llmapi_smoke_matrix[enabled_cudagraph]``
and ``test_chatglm3_attention_backend.py``.
"""

import gc
import glob
import json
import os
import re
import subprocess
import tempfile

import pytest
import torch

CHATGLM3_CKPT = os.environ.get(
    "CHATGLM3_CKPT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "kleinc/hf_data/chatglm3-6b",
)

# Small deterministic slice for the pre-benchmark canary.
CANARY_NUM_SAMPLES = int(os.environ.get("CHATGLM3_GSM8K_CANARY_SAMPLES", "20"))
# Sample count for the full HF-parity gate. Default is the FULL GSM8K test set
# (all 1319 problems) so the durable score is the real GSM8K score task.yaml
# requires ("GSM8K score ... match the HuggingFace reference within 2 points"),
# not a slice. Both the HF reference and both TensorRT-LLM matrix entries evaluate
# the identical seed-0 shuffled dataset under deterministic greedy decoding, so
# the parity comparison is exact (no sampling noise). Set
# CHATGLM3_GSM8K_FULL_SAMPLES to a positive integer only to request a smaller
# slice (e.g. a quick local check); "full"/"0"/"-1"/"none" => full dataset.
_full_env = os.environ.get("CHATGLM3_GSM8K_FULL_SAMPLES", "full").strip().lower()
FULL_NUM_SAMPLES = None if _full_env in ("full", "0", "-1", "none") else int(_full_env)
# Completion criterion: TRT-LLM GSM8K within this many points of the HF reference.
GSM8K_TOLERANCE = float(os.environ.get("CHATGLM3_GSM8K_TOLERANCE", "2.0"))
# Durable artifact root (host-visible lustre path in the Slurm job); temp if unset.
GSM8K_ARTIFACT_DIR = os.environ.get("CHATGLM3_GSM8K_ARTIFACT_DIR") or None

# CUDA-graph capture marker emitted by the pytorch backend warmup when
# cuda_graph_runner.enabled is True (tensorrt_llm/_torch/pyexecutor/model_engine.py:
# "Creating CUDA graph instances for <N> batch sizes.", logged at INFO which these
# runs surface as "[TRT-LLM] [I] ..."). Present ONLY when a cuda_graph_config is
# active, so it distinguishes the enabled matrix entry from the baseline and proves
# the enabled run drove real capture/replay rather than a silent non-graph fallback.
CUDA_GRAPH_CAPTURE_MARKER = "Creating CUDA graph instances for"

BASELINE_YAML = """\
attn_backend: TRTLLM
disable_overlap_scheduler: true
# Explicitly disable CUDA graph for the baseline matrix entry. TorchLlmArgs.
# cuda_graph_config has default_factory=CudaGraphConfig, so OMITTING this key
# leaves CUDA graph ON by default. An explicit null forces a true
# cuda_graph=false baseline -> capture marker ABSENT.
cuda_graph_config: null
kv_cache_config:
  use_kv_cache_manager_v2: true
  free_gpu_memory_fraction: 0.5
"""

ENABLED_YAML = """\
attn_backend: TRTLLM
disable_overlap_scheduler: false
cuda_graph_config:
  enable_padding: true
kv_cache_config:
  use_kv_cache_manager_v2: true
  free_gpu_memory_fraction: 0.5
"""

# (tag, config yaml, cuda_graph, overlap_scheduler)
MATRIX = [
    ("baseline_nograph", BASELINE_YAML, False, False),
    ("enabled_cudagraph", ENABLED_YAML, True, True),
]


def _require():
    if not torch.cuda.is_available():
        pytest.skip("GSM8K test requires CUDA.")
    if not os.path.isdir(CHATGLM3_CKPT):
        pytest.skip(f"ChatGLM3 checkpoint not found at {CHATGLM3_CKPT}")


def _artifact_dir(tag: str) -> str:
    """Per-entry artifact directory: a durable lustre subdir when
    ``CHATGLM3_GSM8K_ARTIFACT_DIR`` is set, else an ephemeral temp dir."""
    if GSM8K_ARTIFACT_DIR:
        path = os.path.join(GSM8K_ARTIFACT_DIR, tag)
        os.makedirs(path, exist_ok=True)
        return path
    return tempfile.mkdtemp(prefix=f"chatglm3_gsm8k_{tag}_")


def _parse_score(stdout: str) -> float:
    """Parse the normalized (0-100) GSM8K score from trtllm-eval stdout.

    Fallback when the saved results json is unavailable. Prefers the explicit
    "average accuracy: <x>" line the evaluator logs (already 0-100), then any
    accuracy/score/exact_match number.
    """
    avg = re.findall(r"average accuracy:\s*([0-9]+\.?[0-9]*)", stdout, re.IGNORECASE)
    if avg:
        return float(avg[-1])
    matches = re.findall(
        r"(?:accuracy|score|exact_match)[^0-9]*([0-9]+\.?[0-9]*)", stdout, re.IGNORECASE
    )
    assert matches, "could not parse a GSM8K score from trtllm-eval output"
    score = float(matches[-1])
    if score <= 1.0:  # normalize a 0-1 fraction to a percentage
        score *= 100.0
    assert 0.0 <= score <= 100.0, f"invalid GSM8K score {score}"
    return score


def _extract_gsm8k_metrics(scores: dict, already_pct: bool) -> dict:
    """Pull the two headline GSM8K exact-match metrics onto a 0-100 scale.

    lm-eval's gsm8k reports ``exact_match,strict-match`` and
    ``exact_match,flexible-extract`` (plus ``_stderr`` variants and an
    ``alias``). ``already_pct`` is True for the trtllm-eval saved json (the
    evaluator multiplies scores by 100 before saving) and False for a raw
    in-process ``lm_eval.evaluate`` result (0-1 fractions).
    """
    mult = 1.0 if already_pct else 100.0
    flexible = strict = None
    for key, value in scores.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if "_stderr" in key or "exact_match" not in key:
            continue
        if "flexible" in key:
            flexible = value * mult
        elif "strict" in key:
            strict = value * mult
    if flexible is None:  # fall back to any exact_match metric
        for key, value in scores.items():
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and "exact_match" in key and "_stderr" not in key):
                flexible = value * mult
                break
    return {"flexible_extract": flexible, "strict_match": strict}


def _score_from_output_path(out_dir: str, stdout: str):
    """Robustly recover the GSM8K score from the evaluator's saved results json
    (``--output_path``), falling back to stdout parsing. Returns
    ``(flexible_extract_score, metrics_dict)`` on a 0-100 scale."""
    results_json = os.path.join(out_dir, "samples_gsm8k.json")
    if os.path.isfile(results_json):
        try:
            with open(results_json) as handle:
                data = json.load(handle)
            scores = data.get("results", {}).get("gsm8k", {})
            metrics = _extract_gsm8k_metrics(scores, already_pct=True)
            if metrics["flexible_extract"] is not None:
                return metrics["flexible_extract"], metrics
        except Exception as exc:  # noqa: BLE001 - fall back to stdout parsing
            print(f"[gsm8k] results json parse failed ({exc}); using stdout parse")
    return _parse_score(stdout), {"flexible_extract": None, "strict_match": None}


# --------------------------------------------------------------------------- #
# Shared deterministic trtllm-eval runner (canary + full parity)
# --------------------------------------------------------------------------- #
def _run_trtllm_gsm8k(tag, config_yaml, cuda_graph, overlap, num_samples,
                      timeout=3600):
    """Run one deterministic ``trtllm-eval gsm8k`` matrix entry and record a
    durable summary. Returns the summary dict (``score`` == flexible-extract)."""
    out_dir = _artifact_dir(tag)
    cfg_path = os.path.join(out_dir, "extra_llm_api_options.yaml")
    with open(cfg_path, "w") as handle:
        handle.write(config_yaml)

    cmd = [
        "trtllm-eval",
        "--model", CHATGLM3_CKPT,
        "--backend", "pytorch",
        "--trust_remote_code",
        "--extra_llm_api_options", cfg_path,
        "gsm8k",
        "--temperature", "0.0",
        "--top_k", "1",
        "--output_dir", out_dir,   # per-sample prompt/generation dump
        "--output_path", out_dir,  # saved results json (robust score parse)
    ]
    if num_samples is not None:
        cmd += ["--num_samples", str(num_samples)]
    # Force the in-process (single-process) executor for the trtllm-eval
    # subprocess. For TP1 the LLM API otherwise "partitions the workload to
    # multiple processes" (tensorrt_llm/executor/executor.py) and spawns a worker
    # process whose MPI/PMIx spawn WEDGES under a single-task ``srun`` allocation
    # (observed: CUDA context created ~1.5GB, 0% util, no model load -> hang until
    # timeout). TLLM_WORKER_USE_SINGLE_PROCESS=1 runs the executor in-process via
    # GenerationExecutorWorker -- the exact mode the in-process LLM-API smoke /
    # replay tests use -- so it also guarantees the CUDA-graph capture marker
    # lands in THIS subprocess's captured stdout/stderr for the hard-path gate.
    eval_env = {**os.environ, "TLLM_WORKER_USE_SINGLE_PROCESS": "1"}
    print(f"[gsm8k/{tag}] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=eval_env)
    print(f"[gsm8k/{tag}] stdout tail:\n{proc.stdout[-2000:]}")
    if proc.returncode != 0:
        print(f"[gsm8k/{tag}] stderr tail:\n{proc.stderr[-2000:]}")
    assert proc.returncode == 0, f"trtllm-eval gsm8k ({tag}) exited {proc.returncode}"

    # CUDA-graph hard-path evidence from THIS run's captured output. The "[TRT-LLM]
    # [I]" logger line may land on stdout or stderr, so scan both.
    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    cg_marker_present = CUDA_GRAPH_CAPTURE_MARKER in combined_output
    cuda_graph_hard_path = "PRESENT" if cg_marker_present else "ABSENT"

    score, metrics = _score_from_output_path(out_dir, proc.stdout)

    # Per-sample artifacts produced by --output_dir (prompt text + generations).
    _artifact_names = {"extra_llm_api_options.yaml", "run_summary.json",
                       "samples_gsm8k.json"}
    sample_files = sorted(
        f
        for f in glob.glob(os.path.join(out_dir, "**", "*"), recursive=True)
        if os.path.isfile(f) and os.path.basename(f) not in _artifact_names
    )

    summary = {
        "tag": tag,
        "task": "gsm8k",
        "backend": "pytorch",
        "attn_backend": "TRTLLM",
        "kv_cache_manager_v2": True,
        "cuda_graph": cuda_graph,
        "cuda_graph_hard_path": cuda_graph_hard_path,
        "overlap_scheduler": overlap,
        "trust_remote_code": True,
        "num_samples": num_samples,
        "score": score,
        "metrics_pct": metrics,
        "returncode": proc.returncode,
        "config_yaml": config_yaml,
        "artifact_dir": out_dir,
        "sample_files": sample_files,
        "cuda_graph_hard_path_evidence": (
            f"capture marker '{CUDA_GRAPH_CAPTURE_MARKER}' {cuda_graph_hard_path} "
            "in this run's trtllm-eval output; corroborated in-process by "
            "test_chatglm3_replay.py::test_chatglm3_llmapi_smoke_matrix"
            "[enabled_cudagraph] and test_chatglm3_attention_backend.py"
            if cuda_graph else
            f"capture marker '{CUDA_GRAPH_CAPTURE_MARKER}' {cuda_graph_hard_path} "
            "in this run's trtllm-eval output (true cuda_graph=false baseline)"
        ),
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"[gsm8k/{tag}] score={score:.2f} (n={num_samples}) "
        f"cuda_graph={cuda_graph} overlap={overlap} "
        f"CUDA_GRAPH_HARDPATH={cuda_graph_hard_path} "
        f"sample_files={len(sample_files)} artifact_dir={out_dir}"
    )

    # Effective-config gate (self-contained, per-run): prove each matrix entry
    # actually ran in its intended CUDA-graph mode rather than trusting the label.
    if cuda_graph:
        assert "cuda_graph_config" in config_yaml, (
            "enabled config must set cuda_graph_config"
        )
        assert cg_marker_present, (
            f"[{tag}] enabled: CUDA-graph capture marker "
            f"'{CUDA_GRAPH_CAPTURE_MARKER}' not found in trtllm-eval output -> "
            "silent non-graph fallback (no hard-path evidence)"
        )
    else:
        assert "cuda_graph_config: null" in config_yaml, (
            f"[{tag}] baseline config must explicitly disable CUDA graph "
            "(cuda_graph_config: null) to override the CudaGraphConfig default_factory"
        )
        assert not cg_marker_present, (
            f"[{tag}] baseline: CUDA-graph capture marker "
            f"'{CUDA_GRAPH_CAPTURE_MARKER}' unexpectedly PRESENT -> baseline ran WITH "
            "CUDA graph (cuda_graph_config default_factory not overridden)"
        )
    assert sample_files, f"[{tag}] trtllm-eval --output_dir produced no per-sample artifacts"
    return summary


# --------------------------------------------------------------------------- #
# HuggingFace source reference (same lm-eval task_dict, greedy forward decode)
# --------------------------------------------------------------------------- #
def _load_bin_state_dict(ckpt):
    """Load the full state dict from the complete PyTorch ``.bin`` shard set."""
    shards = sorted(glob.glob(os.path.join(ckpt, "pytorch_model-*.bin")))
    assert shards, f"No pytorch_model-*.bin shards under {ckpt}"
    state_dict = {}
    for shard in shards:
        state_dict.update(torch.load(shard, map_location="cpu", weights_only=True))
    return state_dict


def _hf_source_config():
    """Load the ChatGLM3 config for the HF *source* model, backfilling the
    legacy generation/output defaults the checkpoint's 2023 remote modeling code
    reads at construction but transformers>=5 no longer sets on the config
    object (they moved to GenerationConfig). Correctness-neutral config-compat
    under deterministic greedy decoding -- not a transformers monkeypatch.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
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
    for name, value in _legacy_defaults.items():
        if not hasattr(cfg, name):
            setattr(cfg, name, value)
    return cfg


def _build_hf_model_and_tokenizer():
    """Build the checkpoint's remote HF model from config + the complete ``.bin``
    weights and load the tokenizer. transformers 5.5.x ``from_pretrained``
    finalization references ``all_tied_weights_keys`` (absent on this 2023 remote
    model class), so we use the public ``from_config`` / ``load_state_dict`` APIs
    -- still the real source model + real checkpoint weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CHATGLM3_CKPT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(_hf_source_config(), trust_remote_code=True)
    missing, unexpected = model.load_state_dict(
        _load_bin_state_dict(CHATGLM3_CKPT), strict=False
    )
    assert not missing, f"HF source model missing weights: {sorted(missing)[:8]}"
    assert set(unexpected) <= {"transformer.rotary_pos_emb.inv_freq"}, (
        f"unexpected source keys: {sorted(unexpected)[:8]}"
    )
    return model.to(torch.float16).cuda().eval(), tok


def _hf_reference_gsm8k_score(num_samples, artifact_dir):
    """Score the real HF source model on the identical lm-eval gsm8k task_dict.

    Reuses TensorRT-LLM's own ``GSM8K`` evaluator to build the task_dict (same
    include-path, few-shot seed, dataset shuffle as ``trtllm-eval``), then runs
    ``lm_eval.evaluate`` with an HF-backed greedy ``TemplateLM`` wrapper. Returns
    the metrics dict (0-100 scale) and writes durable artifacts. Frees the HF
    model before returning so the subsequent ``trtllm-eval`` subprocesses have
    the GPU to themselves.
    """
    import lm_eval
    from lm_eval.api.model import TemplateLM

    from tensorrt_llm.evaluate.lm_eval import GSM8K

    model, tok = _build_hf_model_and_tokenizer()
    eot = tok.eos_token_id if tok.eos_token_id is not None else 2

    class _HFGreedyLM(TemplateLM):
        """lm-eval adapter: deterministic greedy generate_until via the remote
        model's ``forward()`` + legacy tuple KV cache (generate()/DynamicCache is
        incompatible with this 2023 remote model). gsm8k is generate_until-only,
        so the loglikelihood methods are unused."""

        def __init__(self):
            super().__init__()
            self._tok = tok
            self._model = model

        @property
        def eot_token_id(self):
            return eot

        def tok_encode(self, string, **kwargs):
            return self._tok.encode(string, add_special_tokens=False)

        def _loglikelihood_tokens(self, requests, **kwargs):
            raise NotImplementedError("gsm8k is generate_until only")

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError("gsm8k is generate_until only")

        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return self._tok.apply_chat_template(
                chat_history, tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        @property
        def tokenizer_name(self):
            return "chatglm3-6b-hf-reference"

        def generate_until(self, requests, disable_tqdm=False):
            outputs = []
            total = len(requests)
            for idx, request in enumerate(requests):
                context, gen_kwargs = request.args
                until = gen_kwargs.get("until") or []
                if isinstance(until, str):
                    until = [until]
                max_gen = int(gen_kwargs.get("max_gen_toks", 256))
                outputs.append(self._greedy(context, max_gen, until))
                if (idx + 1) % 25 == 0 or (idx + 1) == total:
                    print(f"[hf-ref/gsm8k] decoded {idx + 1}/{total} samples", flush=True)
            return outputs

        @torch.inference_mode()
        def _greedy(self, context, max_new_tokens, stops):
            device = next(self._model.parameters()).device
            input_ids = self._tok(context, return_tensors="pt").input_ids.to(device)
            ctx_len = input_ids.shape[1]
            position_ids = torch.arange(
                ctx_len, dtype=torch.long, device=device).unsqueeze(0)
            out = self._model(input_ids=input_ids, position_ids=position_ids,
                              use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            generated, text = [], ""
            for step in range(max_new_tokens):
                next_tok = int(logits.argmax(dim=-1).item())
                if next_tok == eot:
                    break
                generated.append(next_tok)
                text = self._tok.decode(generated, skip_special_tokens=True)
                hits = [text.find(s) for s in stops if s and s in text]
                if hits:
                    text = text[:min(hits)]
                    break
                cur = torch.tensor([[next_tok]], dtype=torch.long, device=device)
                pos = torch.tensor([[ctx_len + step]], dtype=torch.long, device=device)
                out = self._model(input_ids=cur, position_ids=pos,
                                  past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
            return text

    # Identical task_dict to trtllm-eval (same include path + seed-0 shuffle).
    evaluator = GSM8K(num_samples=num_samples, random_seed=0, apply_chat_template=False)
    lm = _HFGreedyLM()
    results = lm_eval.evaluate(
        lm=lm,
        task_dict=evaluator.task_dict,
        limit=num_samples,
        apply_chat_template=False,
        log_samples=True,
    )
    scores = results["results"]["gsm8k"]
    metrics = _extract_gsm8k_metrics(scores, already_pct=False)

    reference = {
        "role": "hf_reference",
        "task": "gsm8k",
        "num_samples": num_samples,
        "random_seed": 0,
        "apply_chat_template": False,
        "decoding": "greedy (argmax, temperature=0, top_k=1)",
        "reference_impl": (
            "checkpoint remote modeling_chatglm.py (from_config + complete .bin "
            "weights), forward() + legacy tuple KV cache"
        ),
        "metrics_pct": metrics,
        "raw_scores": {
            k: (v if isinstance(v, (int, float, str)) else repr(v))
            for k, v in scores.items()
        },
    }
    with open(os.path.join(artifact_dir, "hf_reference_gsm8k.json"), "w") as handle:
        json.dump(reference, handle, indent=2, default=repr)
    try:
        with open(os.path.join(artifact_dir, "hf_reference_samples_gsm8k.json"), "w") as handle:
            json.dump(results.get("samples", {}).get("gsm8k", []), handle,
                      indent=2, default=repr)
    except Exception as exc:  # noqa: BLE001 - sample dump is best-effort
        print(f"[hf-ref/gsm8k] per-sample dump skipped: {exc}")

    print(
        f"[hf-ref/gsm8k] flexible_extract={metrics['flexible_extract']} "
        f"strict_match={metrics['strict_match']} (n={num_samples}) "
        f"artifact_dir={artifact_dir}"
    )

    # Free the ~12GB HF model before the trtllm-eval subprocesses run.
    del lm, model, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_chatglm3_gsm8k_accuracy_canary():
    """Deterministic small GSM8K slice for baseline + enabled configs.

    Each entry records a durable artifact (config, score, per-sample dump, hard-path).
    """
    _require()
    summaries = {}
    for tag, cfg, cuda_graph, overlap in MATRIX:
        summaries[tag] = _run_trtllm_gsm8k(
            tag, cfg, cuda_graph, overlap, CANARY_NUM_SAMPLES)
    print(
        f"[gsm8k-canary] baseline={summaries['baseline_nograph']['score']:.2f} "
        f"enabled={summaries['enabled_cudagraph']['score']:.2f} "
        f"(canary only; the HF-parity gate is test_chatglm3_gsm8k_full_parity)"
    )


def test_chatglm3_gsm8k_full_parity():
    """Completion-criterion gate: TensorRT-LLM GSM8K must match the HF reference
    within ``GSM8K_TOLERANCE`` points for both CUDA-graph matrix entries.

    Order (cheap-fails-first): compute the in-process HF reference and free the
    model, then run the baseline and enabled ``trtllm-eval`` entries. Fails
    unless (a) each TRT-LLM flexible-extract score is within tolerance of the HF
    reference, (b) the enabled entry proves CUDA-graph hard-path from its own
    artifacts, and (c) the baseline entry proves CUDA graph was OFF.
    """
    _require()
    root = GSM8K_ARTIFACT_DIR or tempfile.mkdtemp(prefix="chatglm3_gsm8k_full_")
    os.makedirs(root, exist_ok=True)
    print(
        f"[gsm8k-full] num_samples={FULL_NUM_SAMPLES} tolerance={GSM8K_TOLERANCE} "
        f"artifact_root={root}"
    )

    # 1) HF reference first (in-process); frees the HF model before subprocesses.
    hf_metrics = _hf_reference_gsm8k_score(FULL_NUM_SAMPLES, root)
    hf_ref = hf_metrics["flexible_extract"]
    assert hf_ref is not None, "HF reference flexible-extract score missing"

    # 2) TensorRT-LLM baseline + enabled via trtllm-eval (full-gate timeout).
    trt = {}
    for tag, cfg, cuda_graph, overlap in MATRIX:
        trt[tag] = _run_trtllm_gsm8k(
            tag, cfg, cuda_graph, overlap, FULL_NUM_SAMPLES, timeout=10800)
    baseline = trt["baseline_nograph"]["score"]
    enabled = trt["enabled_cudagraph"]["score"]

    # 3) Durable combined summary.
    summary = {
        "task": "gsm8k",
        "num_samples": FULL_NUM_SAMPLES,
        "tolerance_points": GSM8K_TOLERANCE,
        "hf_reference": hf_metrics,
        "hf_reference_flexible_extract": hf_ref,
        "trtllm_baseline": {
            "score": baseline,
            "cuda_graph": False,
            "overlap_scheduler": False,
            "cuda_graph_hard_path": trt["baseline_nograph"]["cuda_graph_hard_path"],
            "delta_vs_hf": abs(baseline - hf_ref),
            "artifact_dir": trt["baseline_nograph"]["artifact_dir"],
        },
        "trtllm_enabled": {
            "score": enabled,
            "cuda_graph": True,
            "overlap_scheduler": True,
            "cuda_graph_hard_path": trt["enabled_cudagraph"]["cuda_graph_hard_path"],
            "delta_vs_hf": abs(enabled - hf_ref),
            "artifact_dir": trt["enabled_cudagraph"]["artifact_dir"],
        },
    }
    with open(os.path.join(root, "full_parity_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"[gsm8k-full] HF_ref={hf_ref:.2f} "
        f"TRT_baseline={baseline:.2f} (delta {abs(baseline - hf_ref):.2f}) "
        f"TRT_enabled={enabled:.2f} (delta {abs(enabled - hf_ref):.2f}) "
        f"tolerance={GSM8K_TOLERANCE} summary={os.path.join(root, 'full_parity_summary.json')}"
    )

    # 4) Gates.
    assert trt["enabled_cudagraph"]["cuda_graph_hard_path"] == "PRESENT", (
        "enabled full-gate run lacks CUDA-graph hard-path evidence "
        "(capture marker absent -> silent non-graph fallback)"
    )
    assert trt["baseline_nograph"]["cuda_graph_hard_path"] == "ABSENT", (
        "baseline full-gate run unexpectedly captured CUDA graphs "
        "(cuda_graph=false not honored)"
    )
    assert abs(baseline - hf_ref) <= GSM8K_TOLERANCE, (
        f"TRT-LLM baseline (cuda_graph=false, overlap=false) GSM8K {baseline:.2f} "
        f"differs from HF reference {hf_ref:.2f} by more than {GSM8K_TOLERANCE} points"
    )
    assert abs(enabled - hf_ref) <= GSM8K_TOLERANCE, (
        f"TRT-LLM enabled (cuda_graph=true, overlap=true) GSM8K {enabled:.2f} "
        f"differs from HF reference {hf_ref:.2f} by more than {GSM8K_TOLERANCE} points"
    )

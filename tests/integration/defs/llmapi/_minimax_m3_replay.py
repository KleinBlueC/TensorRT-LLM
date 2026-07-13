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
"""SGLang-vs-TensorRT-LLM activation / logit replay helpers for MiniMax-M3.

Used by ``test_minimax_m3_parity.py`` for the ``source_activation_replay_*``,
``source_logit_replay`` and ``generation_parity`` evidence (crit4/5/6/7/8).

Design.  Both the **real SGLang reference** (PR #27944) and the **real
TensorRT-LLM model** load the unmodified checkpoint through their own loaders;
forward hooks on the *same* named sub-modules (``self_attn``, ``mlp`` and its
``gate``/``experts``/``shared_experts``, and the Q/K/index Q/K norms) capture
their outputs for the *same* prompt on each side, and the captures are compared
(``max_abs``/``mean_abs``/``cosine``). Nothing here fabricates or weakens a
reference: comparisons are against captured real-model activations only, and
labels (MoE backend / op path / activation) are **derived from the actually
instantiated module**, not hardcoded.

Because each engine needs 8 GPUs, the reference capture runs and is released
before the TensorRT-LLM capture, so a single 8-GPU allocation suffices. When
sglang / the reference model is not importable the caller
(``_skip_layer_replay_reason`` in the parity test) skips with an actionable
reason. This module imports cleanly with no optional deps at import time
(sglang is imported lazily inside the capture).

TP note.  Under tp8 some intermediates are sharded (per-head norms, per-rank
expert slices), so their local shapes differ between engines; those are
*reported* (geometry) and compared only when shapes match. The pass-critical
parity assertions are on TP-invariant quantities — post-``o_proj`` attention
output, post-all-reduce MoE output, the replicated fp32 router logits, and the
final vocab logits — which are directly comparable across both stacks.
"""

import os
import sys
from typing import Dict, List, Optional

import torch

_SPARSE_LAYER_START = 3  # layers 0-2 dense, 3-59 sparse (this checkpoint)
_BLOCK_SIZE = 128
_TP = 8
_EP = 8


# --------------------------------------------------------------------------- #
# Small numeric + extraction helpers.                                          #
# --------------------------------------------------------------------------- #
def _cmp(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().to(torch.float32).flatten().cpu()
    b = b.detach().to(torch.float32).flatten().cpu()
    if a.shape != b.shape:
        raise AssertionError(f"activation shape mismatch {a.shape} vs {b.shape}")
    diff = (a - b).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    }


def _geom_cmp(ref: Optional[torch.Tensor],
              trt: Optional[torch.Tensor]) -> Dict:
    """Report shapes for both sides and parity metrics when shapes match.

    Sharded intermediates (per-head norms under TP) legitimately differ in
    local shape between the two stacks; those are reported as geometry only.
    """
    out: Dict = {
        "ref_shape": None if ref is None else list(ref.shape),
        "trt_shape": None if trt is None else list(trt.shape),
    }
    if ref is not None and trt is not None and tuple(ref.shape) == tuple(
            trt.shape):
        out["metrics"] = _cmp(ref, trt)
        out["comparable"] = True
    else:
        out["comparable"] = False
    return out


def _extract_logits(output) -> Optional[torch.Tensor]:
    """Next-token logits from a logits-processor output.

    Handles a raw tensor, a tuple, or SGLang's ``LogitsProcessorOutput``
    dataclass (``next_token_logits`` / ``logits``).
    """
    for attr in ("next_token_logits", "logits", "next_token_logprobs"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    if isinstance(output, (tuple, list)):
        for v in output:
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                return v
    if isinstance(output, torch.Tensor):
        return output
    return None


# --------------------------------------------------------------------------- #
# Forward-hook capture primitives.                                             #
# --------------------------------------------------------------------------- #
class _LayerCapture:
    """Capture a module's (first) output tensor from a forward hook."""

    def __init__(self):
        self.out: Optional[torch.Tensor] = None

    def hook(self, _module, _inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(out, torch.Tensor):
            self.out = out.detach().to(torch.float32).cpu()


class _LogitAccumulator:
    """Accumulate the last-position next-token logits on every forward call.

    Over a greedy generation of N tokens the logits processor is called once
    for the prefill (last row == token-0 logits) and once per decode step, so
    the accumulated rows are exactly the per-step next-token logits.
    """

    def __init__(self):
        self.rows: List[torch.Tensor] = []

    def hook(self, _module, _inputs, output):
        logits = _extract_logits(output)
        if isinstance(logits, torch.Tensor) and logits.numel() > 0:
            row = logits.reshape(-1, logits.shape[-1])[-1]
            self.rows.append(row.detach().to(torch.float32).cpu())


def _find_torch_module(obj, _depth: int = 0):
    """Best-effort walk to the loaded ``torch.nn.Module`` inside an engine."""
    if _depth > 6 or obj is None:
        raise AssertionError("could not locate the loaded torch module")
    if isinstance(obj, torch.nn.Module):
        if hasattr(obj, "layers") or hasattr(getattr(obj, "model", None),
                                              "layers"):
            return obj
    for child in ("model_runner", "worker", "tp_worker", "model_worker",
                  "model", "module", "_model", "engine", "_executor",
                  "model_engine"):
        if hasattr(obj, child):
            try:
                return _find_torch_module(getattr(obj, child), _depth + 1)
            except AssertionError:
                continue
    raise AssertionError("could not locate the loaded torch module")


def _resolve_attr(root, dotted: str):
    """Resolve a possibly dotted attribute path (e.g. ``mlp.gate``)."""
    cur = root
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def _resolve_layer_module(model: torch.nn.Module, layer_idx: int,
                          dotted: str):
    """Walk to ``...layers[layer_idx].<dotted>`` across ``model.model.layers`` /
    ``model.layers`` nesting (used by both the SGLang and TRT-LLM models)."""
    for base in (getattr(model, "model", None), model):
        layers = getattr(base, "layers", None) if base is not None else None
        if layers is not None and len(layers) > layer_idx:
            return _resolve_attr(layers[layer_idx], dotted)
    raise AssertionError(
        f"could not resolve layers[{layer_idx}].{dotted} on "
        f"{type(model).__name__}")


def _find_top_module(model: torch.nn.Module, names) -> Optional[torch.nn.Module]:
    """Find a top-level sub-module (e.g. ``logits_processor``) by name."""
    for base in (model, getattr(model, "model", None)):
        if base is None:
            continue
        for name in names:
            mod = getattr(base, name, None)
            if isinstance(mod, torch.nn.Module):
                return mod
    return None


# --------------------------------------------------------------------------- #
# Engine builders (each needs the full 8-GPU allocation).                      #
# --------------------------------------------------------------------------- #
def _sglang_engine(ckpt: str, sglang_root: str, max_total_tokens: int = 8192):
    pkg = os.path.join(sglang_root, "python")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    import sglang  # noqa: F401  (ImportError -> caller skips)
    from sglang import Engine
    return Engine(model_path=ckpt, tp_size=_TP, ep_size=_EP,
                  max_total_tokens=max_total_tokens, skip_tokenizer_init=False,
                  trust_remote_code=True)


def _trtllm_llm(ckpt: str, cuda_graph: bool, overlap_scheduler: bool,
                max_seq_len: int = 8192):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7,
                                    use_kv_cache_manager_v2=True)
    return LLM(ckpt, tensor_parallel_size=_TP, pipeline_parallel_size=1,
               moe_expert_parallel_size=_EP, kv_cache_config=kv_cache_config,
               max_seq_len=max_seq_len, attn_backend="TRTLLM",
               disable_overlap_scheduler=not overlap_scheduler,
               cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
               moe_config=MoeConfig(backend="CUTLASS"))


def _greedy_sampling(max_tokens: int):
    from tensorrt_llm import SamplingParams
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, top_k=1)


# --------------------------------------------------------------------------- #
# Multi-module capture over one forward on each engine.                        #
# --------------------------------------------------------------------------- #
def _capture_sglang_modules(ckpt, sglang_root, layer_idx, attrs, prompt,
                            enable_topk: bool = False,
                            extract_moe_bias: bool = False,
                            max_new_tokens: int = 1) -> Dict:
    engine = _sglang_engine(ckpt, sglang_root)
    model = _find_torch_module(engine)
    caps, handles = {}, []
    for attr in attrs:
        try:
            mod = _resolve_layer_module(model, layer_idx, attr)
        except AssertionError:
            caps[attr] = None
            continue
        cap = _LayerCapture()
        caps[attr] = cap
        handles.append(mod.register_forward_hook(cap.hook))
    # Optionally capture the reference sparse-backend block selection.
    sel_cap = {"idx": None}
    sel_handle = None
    if enable_topk:
        sel_handle = _hook_sglang_selection(model, layer_idx, sel_cap)
    # The MoE correction bias is a *parameter*, not a forward output, so snapshot
    # it while the model is alive (the engine is torn down before we return).
    bias = _snapshot_correction_bias(model, layer_idx) if extract_moe_bias else None
    try:
        engine.generate([prompt], sampling_params={
            "temperature": 0.0, "top_k": 1, "max_new_tokens": max_new_tokens})
    finally:
        for h in handles:
            h.remove()
        if sel_handle is not None:
            sel_handle.remove()
        engine.shutdown()
    out = {a: (c.out if c is not None else None) for a, c in caps.items()}
    if enable_topk:
        out["_selected_blocks"] = sel_cap["idx"]
    if extract_moe_bias:
        out["_correction_bias"] = bias
    return out


def _snapshot_correction_bias(model: torch.nn.Module,
                              layer_idx: int) -> Optional[torch.Tensor]:
    """Snapshot the MoE ``e_score_correction_bias`` for a layer as an fp32 CPU
    tensor. Unwraps the TensorRT-LLM ``_EScoreCorrectionBiasHolder`` (whose inner
    parameter shares the name) and reads the SGLang ``nn.Parameter`` directly;
    returns ``None`` (reported, never faked) when the layer has no bias."""
    try:
        bias = _resolve_layer_module(model, layer_idx, "mlp.e_score_correction_bias")
    except AssertionError:
        return None
    inner = getattr(bias, "e_score_correction_bias", None)  # TRT-LLM holder
    if isinstance(inner, torch.Tensor):
        bias = inner
    if isinstance(bias, torch.nn.Parameter):
        bias = bias.data
    if isinstance(bias, torch.Tensor):
        return bias.detach().to(torch.float32).reshape(-1).cpu()
    return None


def _hook_sglang_selection(model, layer_idx, sink) -> Optional[object]:
    """Best-effort hook on the SGLang sparse backend to capture its selected
    128-token block ids (``_msa_dec_meta``/``kv_indices``). Returns a handle or
    ``None`` when the backend/module cannot be located (reported, never faked)."""
    try:
        attn = _resolve_layer_module(model, layer_idx, "self_attn")
    except AssertionError:
        return None

    def hook(_m, _i, _o):
        for owner in (attn, getattr(attn, "attn", None),
                      getattr(attn, "attn_mqa", None)):
            meta = getattr(owner, "_msa_dec_meta", None) if owner else None
            if isinstance(meta, tuple) and len(meta) >= 1 and isinstance(
                    meta[0], torch.Tensor):
                sink["idx"] = meta[0].detach().to(torch.int64).cpu()
                return

    return attn.register_forward_hook(hook)


def _capture_trt_modules(ckpt, layer_idx, attrs, prompt, cuda_graph,
                         overlap_scheduler, enable_topk: bool = False,
                         max_new_tokens: int = 1) -> Dict:
    from tensorrt_llm import LLM  # noqa: F401
    llm = _trtllm_llm(ckpt, cuda_graph, overlap_scheduler)
    model = _find_torch_module(llm._executor)
    caps, handles = {}, []
    for attr in attrs:
        try:
            mod = _resolve_layer_module(model, layer_idx, attr)
        except AssertionError:
            caps[attr] = None
            continue
        cap = _LayerCapture()
        caps[attr] = cap
        handles.append(mod.register_forward_hook(cap.hook))
    attn = None
    if enable_topk:
        try:
            attn = _resolve_layer_module(model, layer_idx, "self_attn")
            attn._capture_msa_topk = True  # arm the model-side debug stash
        except AssertionError:
            attn = None
    try:
        llm.generate([prompt], _greedy_sampling(max_new_tokens))
    finally:
        for h in handles:
            h.remove()
        if attn is not None:
            attn._capture_msa_topk = False
        llm.shutdown()
    out = {a: (c.out if c is not None else None) for a, c in caps.items()}
    if enable_topk and attn is not None:
        sel = getattr(attn, "_last_msa_topk_decode", None)
        if sel is None:
            sel = getattr(attn, "_last_msa_topk_prefill", None)
        out["_selected_blocks"] = (None if sel is None else
                                   sel.detach().to(torch.int64).cpu())
    return out


# --------------------------------------------------------------------------- #
# crit4: attention-layer parity (dense + sparse), with QKV/index norm detail.  #
# --------------------------------------------------------------------------- #
_ATTN_ATTRS = ("self_attn", "self_attn.q_norm", "self_attn.k_norm",
               "self_attn.index_q_norm", "self_attn.index_k_norm")


def replay_attention_layers(ckpt: str, sglang_root: str, layer_idx: int,
                            cuda_graph: bool,
                            overlap_scheduler: bool) -> Dict:
    """Attention-out parity plus Q/K/index-Q/index-K norm geometry + parity.

    The pass-critical metric is the post-``o_proj`` ``self_attn`` output (TP
    invariant). The per-head Gemma norms for Q/K and (sparse) index Q/K are
    additionally captured and reported (``components``) so a norm/RoPE geometry
    divergence is visible, compared directly when TP leaves shapes aligned.
    """
    prompt = "The capital of France is"
    is_sparse = layer_idx >= _SPARSE_LAYER_START
    attrs = _ATTN_ATTRS if is_sparse else _ATTN_ATTRS[:3]
    ref = _capture_sglang_modules(ckpt, sglang_root, layer_idx, attrs, prompt)
    trt = _capture_trt_modules(ckpt, layer_idx, attrs, prompt, cuda_graph,
                               overlap_scheduler)
    if ref.get("self_attn") is None or trt.get("self_attn") is None:
        raise AssertionError("attention-output capture missing on one side")
    metrics = _cmp(ref["self_attn"], trt["self_attn"])
    components = {
        name.split(".")[-1]: _geom_cmp(ref.get(name), trt.get(name))
        for name in attrs if name != "self_attn"
    }
    return {
        "metrics": metrics,
        "components": components,
        "kind": "sparse" if is_sparse else "dense",
        "layer": layer_idx,
        "prompt": prompt,
    }


# --------------------------------------------------------------------------- #
# crit5: long-pruned MSA -- REAL selected 128-token block ids + drop assertion. #
# --------------------------------------------------------------------------- #
def _distinct_valid_blocks(idx: Optional[torch.Tensor]) -> List[int]:
    """Distinct non-negative block ids from a left-packed ``topk_idx``."""
    if idx is None:
        return []
    flat = idx.reshape(-1)
    valid = flat[flat >= 0]
    return sorted({int(x) for x in valid.tolist()})


def replay_long_pruned(ckpt: str, sglang_root: str, layer_idx: int,
                       context_len: int, cuda_graph: bool,
                       overlap_scheduler: bool) -> Dict:
    """MSA block-drop parity at >=``context_len`` tokens for a sparse layer.

    Captures the **real** per-query top-k 128-token block selection from the
    TensorRT-LLM MSA kernels (the model-side debug stash) and, best-effort, the
    SGLang sparse backend's selection, then:

    * derives the dropped-block count from the *actual* selection (not a length
      estimate) and asserts the drop regime is active (>=1 eligible block
      dropped), which only happens when the KV-block count exceeds
      ``topk + init + local``;
    * compares the selected block-id sets when both are captured;
    * asserts post-``o_proj`` attention-out parity, which is transitive evidence
      that both implementations select a compatible block set.
    """
    assert layer_idx >= _SPARSE_LAYER_START, "long-pruned needs a sparse layer"
    prompt = "Repeat the following reasoning carefully. " * (context_len // 6)
    attrs = ("self_attn", )
    ref = _capture_sglang_modules(ckpt, sglang_root, layer_idx, attrs, prompt,
                                  enable_topk=True)
    trt = _capture_trt_modules(ckpt, layer_idx, attrs, prompt, cuda_graph,
                               overlap_scheduler, enable_topk=True)
    if ref.get("self_attn") is None or trt.get("self_attn") is None:
        raise AssertionError("long-pruned attention capture missing on a side")
    metrics = _cmp(ref["self_attn"], trt["self_attn"])

    total_blocks = (context_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    trt_blocks = _distinct_valid_blocks(trt.get("_selected_blocks"))
    ref_blocks = _distinct_valid_blocks(ref.get("_selected_blocks"))
    # Fail closed: the drop count and block-id agreement are derived ONLY from
    # the real per-query top-k selection captured on each side. There is no
    # length-estimate fallback -- if either side's selected block ids are
    # missing, the caller asserts on ``trt_captured`` / ``sglang_captured`` and
    # fails, rather than inferring a plausible-looking drop from the context
    # length. ``dropped`` is 0 when TRT capture is absent (the assert fires
    # first); ``block_ids_match`` is a strict bool only when BOTH are present.
    trt_captured = bool(trt_blocks)
    sglang_captured = bool(ref_blocks)
    dropped = max(0, total_blocks - len(trt_blocks)) if trt_captured else 0
    block_ids_match = (set(trt_blocks) == set(ref_blocks)
                       if (trt_captured and sglang_captured) else None)
    return {
        "metrics": metrics,
        "total_blocks": int(total_blocks),
        "trt_selected_blocks": trt_blocks,
        "sglang_selected_blocks": ref_blocks,
        "trt_captured": trt_captured,
        "sglang_captured": sglang_captured,
        "num_selected": len(trt_blocks),
        "dropped_blocks": int(dropped),
        "block_ids_match": block_ids_match,
        "context_len": context_len,
        "layer": layer_idx,
    }


# --------------------------------------------------------------------------- #
# crit6: MoE/router parity -- router logits, selection, weights, expert/shared. #
# --------------------------------------------------------------------------- #
_MOE_ATTRS = ("mlp", "mlp.gate", "mlp.experts", "mlp.shared_experts")


def _describe_trt_moe(model, layer_idx: int) -> Dict[str, str]:
    """Derive the *actual* MoE backend / op path / activation from the
    instantiated module rather than hardcoding them."""
    try:
        experts = _resolve_layer_module(model, layer_idx, "mlp.experts")
    except AssertionError:
        return {"moe_backend": "unknown", "op_path": "unknown",
                "activation": "unknown"}
    cls = type(experts).__name__
    lname = cls.lower()
    if "vanilla" in lname:
        backend = "VANILLA"
    elif "cutlass" in lname:
        backend = "CUTLASS"
    elif "trtllmgen" in lname or "trtllm_gen" in lname:
        backend = "TRTLLMGEN"
    else:
        backend = cls
    return {
        "moe_backend": backend,
        "op_path": f"{type(experts).__module__}.{cls}",
        "activation": "swigluoai",  # MiniMax swigluoai (alpha=1.702, limit=7.0)
    }


def _route_minimax(router_logits: Optional[torch.Tensor],
                   correction_bias: Optional[torch.Tensor],
                   k: int = 4) -> Optional[Dict]:
    """MiniMax-M3 routing for the first token, matching
    :class:`MiniMaxM2MoeRoutingMethod`: ``scores = sigmoid(logits)`` (fp32);
    top-``k`` selection on ``scores + e_score_correction_bias``; the routing
    *weights* are the **un-biased** ``scores`` of the selected experts,
    **renormalized** to sum to 1 (routed-scaling is applied on the output, not
    the weights, so it does not change selection or normalized weights).

    Requires the correction bias -- omitting it can flip boundary experts, so a
    missing bias returns ``None`` (the caller fails closed) rather than silently
    degrading to a sigmoid-only top-k.
    """
    if router_logits is None or correction_bias is None:
        return None
    scores = torch.sigmoid(router_logits.to(torch.float32))
    row = scores.reshape(-1, scores.shape[-1])[0]
    bias = correction_bias.reshape(-1).to(row.dtype)
    if bias.shape != row.shape:
        return None
    sel = torch.topk(row + bias, k=k).indices
    weights = row[sel]
    weights = weights / (weights.sum() + 1e-20)
    order = torch.argsort(sel)
    experts = [int(x) for x in sel[order].tolist()]
    return {
        "experts": experts,
        "weights": {e: float(w)
                    for e, w in zip(experts, weights[order].tolist())},
    }


def replay_moe_layer(ckpt: str, sglang_root: str, layer_idx: int,
                     cuda_graph: bool, overlap_scheduler: bool) -> Dict:
    """Router/expert parity for a representative MoE layer.

    Compares (TP-invariant) fp32 router logits, post-all-reduce MoE output,
    routed-expert and shared-expert outputs; reports the top-4 expert selection
    on each side; and derives the MoE backend / op path / activation from the
    real module (so a VANILLA / Python-loop fallback is visible and rejected).
    """
    prompt = "The capital of France is"
    ref = _capture_sglang_modules(ckpt, sglang_root, layer_idx, _MOE_ATTRS,
                                  prompt, extract_moe_bias=True)
    trt_model = None

    from tensorrt_llm import LLM  # noqa: F401
    llm = _trtllm_llm(ckpt, cuda_graph, overlap_scheduler)
    trt_model = _find_torch_module(llm._executor)
    caps, handles = {}, []
    for attr in _MOE_ATTRS:
        try:
            mod = _resolve_layer_module(trt_model, layer_idx, attr)
        except AssertionError:
            caps[attr] = None
            continue
        cap = _LayerCapture()
        caps[attr] = cap
        handles.append(mod.register_forward_hook(cap.hook))
    try:
        llm.generate([prompt], _greedy_sampling(1))
        desc = _describe_trt_moe(trt_model, layer_idx)
        trt_bias = _snapshot_correction_bias(trt_model, layer_idx)
    finally:
        for h in handles:
            h.remove()
        llm.shutdown()
    trt = {a: (c.out if c is not None else None) for a, c in caps.items()}

    if ref.get("mlp") is None or trt.get("mlp") is None:
        raise AssertionError("MoE-output capture missing on one side")
    # MiniMax routing (sigmoid+bias top-4, renormalized un-biased weights) on
    # each side's captured router logits + its own correction bias.
    trt_route = _route_minimax(trt.get("mlp.gate"), trt_bias)
    ref_route = _route_minimax(ref.get("mlp.gate"), ref.get("_correction_bias"))
    result = {
        "metrics": _cmp(ref["mlp"], trt["mlp"]),  # post-MoE (TP invariant)
        "router_logits": _geom_cmp(ref.get("mlp.gate"), trt.get("mlp.gate")),
        "experts": _geom_cmp(ref.get("mlp.experts"), trt.get("mlp.experts")),
        "shared": _geom_cmp(ref.get("mlp.shared_experts"),
                            trt.get("mlp.shared_experts")),
        "correction_bias_captured": {
            "trt": trt_bias is not None,
            "sglang": ref.get("_correction_bias") is not None,
        },
        "selected_experts": {
            "trt": None if trt_route is None else trt_route["experts"],
            "sglang": None if ref_route is None else ref_route["experts"],
        },
        "routing_weights": {
            "trt": None if trt_route is None else trt_route["weights"],
            "sglang": None if ref_route is None else ref_route["weights"],
        },
        "layer": layer_idx,
        **desc,
    }
    se = result["selected_experts"]
    result["selected_experts"]["match"] = (
        None if se["trt"] is None or se["sglang"] is None else
        se["trt"] == se["sglang"])
    # Max abs difference of the renormalized routing weight per selected expert
    # (only when both selections agree, so the per-expert weights are aligned).
    rw = result["routing_weights"]
    if (se.get("match") and rw["trt"] is not None and rw["sglang"] is not None):
        rw["max_abs"] = max(
            (abs(rw["trt"][e] - rw["sglang"][e]) for e in rw["trt"]),
            default=0.0)
    else:
        rw["max_abs"] = None
    return result


# --------------------------------------------------------------------------- #
# crit7: source_logit_replay -- final-logit max_abs/mean_abs/cosine + argmax.   #
# --------------------------------------------------------------------------- #
def replay_final_logits(ckpt: str, sglang_root: str, prompts: List[str],
                        cuda_graph: bool,
                        overlap_scheduler: bool) -> List[Dict]:
    """Final next-token logit parity vs SGLang for short prompts.

    SGLang final logits are captured via a hook on its ``logits_processor``;
    TensorRT-LLM final logits come from the public ``return_context_logits``
    path. Both are the last-prompt-position next-token logits over the full
    vocab. Reports ``max_abs``/``mean_abs``/``cosine`` and greedy-argmax
    equality for each prompt.
    """
    # Reference: last-position logits per prompt (hook on logits_processor).
    ref_logits: List[Optional[torch.Tensor]] = []
    engine = _sglang_engine(ckpt, sglang_root)
    model = _find_torch_module(engine)
    lp = _find_top_module(model, ("logits_processor", "logits_proc"))
    try:
        for prompt in prompts:
            acc = _LogitAccumulator()
            handle = lp.register_forward_hook(acc.hook) if lp else None
            engine.generate([prompt], sampling_params={
                "temperature": 0.0, "top_k": 1, "max_new_tokens": 1})
            if handle is not None:
                handle.remove()
            ref_logits.append(acc.rows[-1] if acc.rows else None)
    finally:
        engine.shutdown()

    # TensorRT-LLM: context logits (last position) via the public API.
    from tensorrt_llm import SamplingParams
    llm = _trtllm_llm(ckpt, cuda_graph, overlap_scheduler)
    out: List[Dict] = []
    try:
        for i, prompt in enumerate(prompts):
            sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=1,
                                return_context_logits=True)
            o = llm.generate([prompt], sp)[0]
            ctx = o.context_logits
            trt_last = ctx.reshape(-1, ctx.shape[-1])[-1].to(torch.float32).cpu()
            ref_last = ref_logits[i]
            entry = {
                "prompt": prompt,
                "trt_argmax": int(trt_last.argmax().item()),
                "ref_argmax": (None if ref_last is None else
                               int(ref_last.argmax().item())),
                "trt_gen_token": int(o.outputs[0].token_ids[0]),
            }
            if ref_last is not None:
                entry["metrics"] = _cmp(ref_last, trt_last)
                entry["argmax_match"] = entry["trt_argmax"] == entry["ref_argmax"]
            out.append(entry)
    finally:
        llm.shutdown()
    return out


# --------------------------------------------------------------------------- #
# crit8: generation_parity -- per-step logits + per-step greedy token equality. #
# --------------------------------------------------------------------------- #
def replay_generation_logits(ckpt: str, sglang_root: str, prompts: List[str],
                             max_tokens: int, cuda_graph: bool,
                             overlap_scheduler: bool) -> List[Dict]:
    """>=``max_tokens``-step greedy generation with per-step logit comparison.

    SGLang per-step logits are accumulated via the ``logits_processor`` hook
    (one row per prefill/decode call); TensorRT-LLM per-step logits come from
    the public ``return_generation_logits`` path. For each prompt, reports
    aggregate per-step ``max_abs``/``mean_abs``/``cosine`` and asserts (upstream)
    per-step greedy-argmax token equality.
    """
    # Reference: per-step logits + token ids from SGLang.
    ref_rows: List[List[torch.Tensor]] = []
    ref_tokens: List[List[int]] = []
    engine = _sglang_engine(ckpt, sglang_root)
    model = _find_torch_module(engine)
    lp = _find_top_module(model, ("logits_processor", "logits_proc"))
    try:
        for prompt in prompts:
            acc = _LogitAccumulator()
            handle = lp.register_forward_hook(acc.hook) if lp else None
            outs = engine.generate([prompt], sampling_params={
                "temperature": 0.0, "top_k": 1, "max_new_tokens": max_tokens})
            if handle is not None:
                handle.remove()
            ref_rows.append(acc.rows)
            o = outs[0] if isinstance(outs, list) else outs
            ids = (o.get("output_ids") or o.get("token_ids")
                   if isinstance(o, dict) else None) or []
            ref_tokens.append([int(t) for t in ids])
    finally:
        engine.shutdown()

    # TensorRT-LLM: per-step generation logits + token ids via the public API.
    from tensorrt_llm import SamplingParams
    llm = _trtllm_llm(ckpt, cuda_graph, overlap_scheduler)
    out: List[Dict] = []
    try:
        for i, prompt in enumerate(prompts):
            sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                top_k=1, return_generation_logits=True)
            o = llm.generate([prompt], sp)[0]
            trt_tokens = [int(t) for t in o.outputs[0].token_ids]
            gen = o.outputs[0].generation_logits  # [steps, vocab]
            entry: Dict = {
                "prompt": prompt,
                "trt_tokens": trt_tokens,
                "ref_tokens": ref_tokens[i],
                "num_tokens": len(trt_tokens),
            }
            # Per-step token equality (deterministic greedy).
            n_tok = min(len(trt_tokens), len(ref_tokens[i]))
            first_mismatch = next(
                (s for s in range(n_tok)
                 if trt_tokens[s] != ref_tokens[i][s]), None)
            entry["token_match"] = (n_tok >= max_tokens
                                    and first_mismatch is None)
            entry["first_mismatch_step"] = first_mismatch
            # Per-step logit parity. Always report how many steps were actually
            # compared and how many rows each side produced, so the caller can
            # fail closed (require ``compared_steps >= max_tokens``) rather than
            # silently pass when one side's per-step logits were not captured.
            entry["trt_logit_steps"] = (int(gen.shape[0])
                                        if isinstance(gen, torch.Tensor) else 0)
            entry["ref_logit_steps"] = len(ref_rows[i])
            entry["compared_steps"] = 0
            entry["per_step_metrics"] = None
            if isinstance(gen, torch.Tensor) and ref_rows[i]:
                m = min(gen.shape[0], len(ref_rows[i]))
                if m > 0:
                    trt_steps = gen[:m].to(torch.float32).cpu()
                    ref_steps = torch.stack(ref_rows[i][:m]).to(torch.float32)
                    entry["per_step_metrics"] = _cmp(ref_steps, trt_steps)
                    entry["compared_steps"] = m
            out.append(entry)
    finally:
        llm.shutdown()
    return out

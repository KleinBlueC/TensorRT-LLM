"""Config-recognition tests for the MiniMax-M3 text bring-up.

These tests are CPU-only. They exercise the multimodal -> text-tower config
normalization in ``config_utils`` (the ``_MiniMaxM3ConfigCompat`` shim and the
``load_pretrained_config`` routing branch). They do **not** construct the model:
building MiniMax-M3 needs a GPU plus the sparse-attention/MoE stack, and the
registry / weight-accounting tests are shipped alongside the model
implementation.

The released MiniMax-M3 checkpoint is multimodal
(``MiniMaxM3SparseForConditionalGeneration`` / ``model_type=minimax_m3_vl``);
for text-only GSM8K bring-up only the text tower
(``MiniMaxM3SparseForCausalLM``) must be resolved, from the *unmodified*
checkpoint, without ``trust_remote_code``.
"""

import json
import os
import tempfile
import unittest

import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.pyexecutor.config_utils import (
    _MiniMaxM3ConfigCompat,
    load_pretrained_config,
)


def _sparse_schedule(num_layers: int, num_dense: int) -> list[int]:
    """0 for the leading dense layers, 1 for the trailing sparse/MoE layers."""
    return [0] * num_dense + [1] * (num_layers - num_dense)


def _make_text_config(num_layers: int = 60, num_dense: int = 3) -> dict:
    """A structurally-real MiniMax-M3 text_config (dims match the checkpoint)."""
    freq = _sparse_schedule(num_layers, num_dense)
    return {
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "hidden_size": 6144,
        "intermediate_size": 3072,
        "dense_intermediate_size": 12288,
        "shared_intermediate_size": 3072,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 200064,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-06,
        "use_gemma_norm": True,
        "rope_theta": 5000000,
        "rotary_dim": 64,
        "partial_rotary_factor": 0.5,
        "hidden_act": "swigluoai",
        "use_qk_norm": True,
        "qk_norm_type": "per_head",
        "tie_word_embeddings": False,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "n_shared_experts": 1,
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "swiglu_alpha": 1.702,
        "swiglu_limit": 7.0,
        "routed_scaling_factor": 2.0,
        "moe_layer_freq": list(freq),
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_index_dim": 128,
            "sparse_num_index_heads": 4,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "sparse_score_type": "max",
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_disable_index_value": list(freq),
            "sparse_attention_freq": list(freq),
        },
    }


def _make_multimodal_config(num_layers: int = 60, num_dense: int = 3) -> dict:
    """The top-level multimodal config.json shape (text_config nested)."""
    return {
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "model_type": "minimax_m3_vl",
        "auto_map": {"AutoConfig": "configuration_minimax_m3_vl.MiniMaxM3VLConfig"},
        "torch_dtype": "bfloat16",
        "transformers_version": "4.52.4",
        "image_token_index": 200025,
        "video_token_index": 200026,
        "vision_config": {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "model_type": "clip_vision_model",
        },
        "text_config": _make_text_config(num_layers, num_dense),
    }


class TestMiniMaxM3Config(unittest.TestCase):
    """CPU-only config-recognition tests for the MiniMax-M3 text bring-up."""

    def _assert_text_dims(self, get):
        """`get(key)` reads either a dict (`d.get`) or a config (`getattr`)."""
        self.assertEqual(get("hidden_size"), 6144)
        self.assertEqual(get("num_attention_heads"), 64)
        self.assertEqual(get("num_key_value_heads"), 4)
        self.assertEqual(get("head_dim"), 128)
        self.assertEqual(get("vocab_size"), 200064)
        self.assertEqual(get("max_position_embeddings"), 1048576)
        self.assertEqual(get("rope_theta"), 5000000)
        self.assertEqual(get("rotary_dim"), 64)
        self.assertEqual(get("partial_rotary_factor"), 0.5)
        self.assertTrue(get("use_qk_norm"))
        self.assertEqual(get("qk_norm_type"), "per_head")
        self.assertTrue(get("use_gemma_norm"))
        self.assertEqual(get("hidden_act"), "swigluoai")
        self.assertEqual(get("num_local_experts"), 128)
        self.assertEqual(get("num_experts_per_tok"), 4)
        self.assertEqual(get("n_shared_experts"), 1)
        self.assertEqual(get("scoring_func"), "sigmoid")
        self.assertTrue(get("use_routing_bias"))
        self.assertAlmostEqual(get("swiglu_alpha"), 1.702)
        self.assertAlmostEqual(get("swiglu_limit"), 7.0)
        self.assertAlmostEqual(get("routed_scaling_factor"), 2.0)
        self.assertEqual(get("intermediate_size"), 3072)
        self.assertEqual(get("dense_intermediate_size"), 12288)
        self.assertEqual(get("shared_intermediate_size"), 3072)

    def _assert_schedules(self, sparse_attention_config, moe_layer_freq, num_layers, num_dense):
        # sparse_attention_config may be a dict (raw) or an attribute holding a
        # dict (resolved config); both index the same way.
        freq = sparse_attention_config["sparse_attention_freq"]
        self.assertEqual(len(freq), num_layers)
        self.assertEqual(list(moe_layer_freq), list(freq))
        # Dense leading layers, sparse/MoE trailing layers.
        for i in range(num_dense):
            self.assertEqual(freq[i], 0, f"layer {i} should be dense attention")
            self.assertEqual(moe_layer_freq[i], 0, f"layer {i} should be dense MLP")
        for i in range(num_dense, num_layers):
            self.assertEqual(freq[i], 1, f"layer {i} should be sparse (MSA)")
            self.assertEqual(moe_layer_freq[i], 1, f"layer {i} should be MoE")
        self.assertEqual(sparse_attention_config["sparse_topk_blocks"], 16)
        self.assertEqual(sparse_attention_config["sparse_block_size"], 128)
        self.assertEqual(sparse_attention_config["sparse_index_dim"], 128)
        self.assertEqual(sparse_attention_config["sparse_num_index_heads"], 4)
        self.assertEqual(sparse_attention_config["sparse_score_type"], "max")
        self.assertEqual(sparse_attention_config["sparse_init_block"], 0)
        self.assertEqual(sparse_attention_config["sparse_local_block"], 1)

    def test_minimax_m3_config_normalization_extracts_text_tower(self):
        """normalize() pulls the text tower out of the multimodal config."""
        num_layers, num_dense = 60, 3
        raw = _make_multimodal_config(num_layers, num_dense)
        text = _MiniMaxM3ConfigCompat.normalize(raw)

        self.assertEqual(text["architectures"], ["MiniMaxM3SparseForCausalLM"])
        # Vision-only fields are left behind.
        self.assertNotIn("vision_config", text)
        self.assertNotIn("image_token_index", text)
        # Top-level dtype is inherited into the text config.
        self.assertEqual(text["torch_dtype"], "bfloat16")

        self._assert_text_dims(text.get)
        self._assert_schedules(
            text["sparse_attention_config"], text["moe_layer_freq"], num_layers, num_dense
        )
        # normalize() must not mutate the caller's nested text_config.
        self.assertNotIn("torch_dtype", raw["text_config"])

    def test_minimax_m3_config_normalization_passthrough_text_only(self):
        """A pre-extracted text-only config is used as-is (arch still forced)."""
        text_only = _make_text_config(num_layers=6, num_dense=3)
        text_only["architectures"] = ["SomeOtherArch"]
        out = _MiniMaxM3ConfigCompat.normalize(text_only)
        self.assertEqual(out["architectures"], ["MiniMaxM3SparseForCausalLM"])
        self.assertEqual(out["hidden_size"], 6144)

    def test_minimax_m3_config_load_pretrained_from_written_config(self):
        """load_pretrained_config routes a written multimodal config.json to text."""
        num_layers, num_dense = 6, 3
        raw = _make_multimodal_config(num_layers, num_dense)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(raw, f)
            cfg = load_pretrained_config(d)

        self.assertEqual(cfg.architectures, ["MiniMaxM3SparseForCausalLM"])
        # Vision tower is dropped from the resolved text config.
        self.assertIsNone(getattr(cfg, "vision_config", None))
        # dtype inherited from the top level (kept, exact form is
        # transformers-version dependent).
        self.assertIsNotNone(getattr(cfg, "torch_dtype", None))

        self._assert_text_dims(lambda k: getattr(cfg, k, None))
        self._assert_schedules(
            cfg.sparse_attention_config, cfg.moe_layer_freq, num_layers, num_dense
        )

    def test_minimax_m3_model_config_derives_sparse_attention_config(self):
        """ModelConfig.from_pretrained auto-derives the MiniMax-M3 sparse config
        so the runtime selects MiniMaxM3CacheManager (KVCacheManagerV2 + index
        side pool) and the MiniMax-M3 attention backend for the unmodified
        checkpoint -- mirrors the DSA auto-derivation in model_config.py."""
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

        num_layers, num_dense = 6, 3
        raw = _make_multimodal_config(num_layers, num_dense)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(raw, f)
            model_config = ModelConfig.from_pretrained(d, trust_remote_code=False)

        sac = model_config.sparse_attention_config
        self.assertIsInstance(sac, MiniMaxM3SparseAttentionConfig)
        self.assertEqual(sac.algorithm, "minimax_m3")
        self.assertEqual(sac.index_head_dim, 128)
        self.assertEqual(sac.num_index_heads, 4)
        self.assertEqual(sac.topk_blocks, 16)
        self.assertEqual(sac.block_size, 128)
        self.assertTrue(sac.disable_index_value)

    def test_minimax_m3_config_from_real_checkpoint(self):
        """The unmodified MiniMax-M3 checkpoint resolves to the text causal LM."""
        root = llm_models_root()
        if root is None:
            self.skipTest("llm_models_root() is unavailable in this environment")
        ckpt = os.path.join(str(root), "MiniMax-M3")
        # Fail loudly (do not skip) when the models root exists but the M3
        # checkpoint is missing — that is a real misconfiguration, not a
        # "no models" environment.
        self.assertTrue(os.path.isdir(ckpt), f"MiniMax-M3 checkpoint not found at {ckpt}")

        cfg = load_pretrained_config(ckpt)

        self.assertEqual(cfg.architectures, ["MiniMaxM3SparseForCausalLM"])
        self.assertEqual(cfg.num_hidden_layers, 60)
        self.assertIsNone(getattr(cfg, "vision_config", None))
        self._assert_text_dims(lambda k: getattr(cfg, k, None))
        self._assert_schedules(
            cfg.sparse_attention_config, cfg.moe_layer_freq, num_layers=60, num_dense=3
        )


def _reduced_model_config(num_layers: int = 5, num_dense: int = 3, num_experts: int = 8):
    """A small MiniMax-M3 ModelConfig that constructs cheaply but keeps the
    real hybrid structure (leading dense layers, trailing sparse+MoE layers)."""
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig

    text = _make_text_config(num_layers=num_layers, num_dense=num_dense)
    # Shrink the expensive axes; keep head/index geometry real so the attention
    # and index-branch wiring is exercised as shipped.
    text["num_local_experts"] = num_experts
    text["vocab_size"] = 1024
    text["torch_dtype"] = "bfloat16"
    cfg = PretrainedConfig.from_dict(text)
    return ModelConfig(pretrained_config=cfg)


class TestMiniMaxM3Registry(unittest.TestCase):
    """Registry + AutoModel construction for the MiniMax-M3 text tower."""

    def test_minimax_m3_registry_maps_architecture(self):
        """The text architecture string resolves to the registered class and is
        not shadowed by the legacy TRT model map."""
        from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3SparseForCausalLM
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("MiniMaxM3SparseForCausalLM", MODEL_CLASS_MAPPING)
        self.assertIs(MODEL_CLASS_MAPPING["MiniMaxM3SparseForCausalLM"], MiniMaxM3SparseForCausalLM)

        from tensorrt_llm.models import MODEL_MAP

        self.assertNotIn("MiniMaxM3SparseForCausalLM", MODEL_MAP)

    @unittest.skipUnless(
        torch.cuda.is_available(), "constructing the MiniMax-M3 model requires CUDA"
    )
    def test_minimax_m3_registry_constructs_text_model(self):
        """AutoModelForCausalLM resolves and constructs the registered text
        model, with the expected per-layer dense/sparse/MoE wiring."""
        from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
        from tensorrt_llm._torch.models.modeling_minimaxm3 import (
            MiniMaxM3MoE,
            MiniMaxM3SparseForCausalLM,
        )
        from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

        num_dense = 3
        model_config = _reduced_model_config(num_layers=5, num_dense=num_dense)
        model = AutoModelForCausalLM.from_config(model_config)
        self.assertIsInstance(model, MiniMaxM3SparseForCausalLM)

        layers = model.model.layers
        # Leading dense layers: dense attention + dense gated MLP, no index branch.
        for i in range(num_dense):
            attn = layers[i].self_attn
            self.assertFalse(attn.is_sparse_attention_layer, f"layer {i} attention should be dense")
            self.assertFalse(
                hasattr(attn, "index_q_proj"), f"layer {i} should have no index branch"
            )
            self.assertIsInstance(layers[i].mlp, GatedMLP, f"layer {i} MLP should be dense")
        # Trailing layers: sparse attention (index branch) + MoE.
        for i in range(num_dense, len(layers)):
            attn = layers[i].self_attn
            self.assertTrue(attn.is_sparse_attention_layer, f"layer {i} attention should be sparse")
            self.assertTrue(hasattr(attn, "index_q_proj"), f"layer {i} should have an index branch")
            self.assertTrue(hasattr(attn, "index_k_proj"))
            self.assertTrue(hasattr(attn, "index_q_norm"))
            self.assertTrue(hasattr(attn, "index_k_norm"))
            # Released checkpoint disables the index value/output branch.
            self.assertTrue(attn.disable_index_value)
            self.assertFalse(hasattr(attn, "index_v_proj"))
            self.assertFalse(hasattr(attn, "index_o_proj"))
            self.assertIsInstance(layers[i].mlp, MiniMaxM3MoE, f"layer {i} MLP should be MoE")
            self.assertEqual(layers[i].mlp.top_k, 4)
            self.assertAlmostEqual(layers[i].mlp.routed_scaling_factor, 2.0)

        # Per-head Gemma QK norm on head_dim.
        attn = layers[num_dense].self_attn
        self.assertTrue(attn.q_norm.use_gemma)
        self.assertTrue(attn.k_norm.use_gemma)
        self.assertEqual(attn.q_norm.weight.shape[0], attn.head_dim)
        self.assertTrue(attn.index_q_norm.use_gemma)


class TestMiniMaxM3Norm(unittest.TestCase):
    """Norm + partial-RoPE semantics used by MiniMax-M3 attention (CPU)."""

    def test_gemma_per_head_rms_norm_matches_reference(self):
        """The per-head Gemma RMSNorm the model uses matches (1 + w) * rms(x)."""
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        head_dim = 128
        norm = RMSNorm(hidden_size=head_dim, eps=1e-6, dtype=torch.float32, use_gemma=True)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(head_dim) * 0.1)
            x = torch.randn(7, head_dim, dtype=torch.float32)
            out = norm(x)
            ref_rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
            ref = (norm.weight + 1.0) * ref_rms
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_partial_rope_dim_from_config(self):
        """RoPE is partial: only rotary_dim (64) of each 128-dim head rotates."""
        from transformers import PretrainedConfig

        from tensorrt_llm._torch.attention_backend.interface import RopeParams

        cfg = PretrainedConfig.from_dict(_make_text_config())
        rope = RopeParams.from_config(cfg)
        self.assertEqual(rope.dim, 64)
        self.assertNotEqual(rope.dim, cfg.head_dim)


class TestMiniMaxM3WeightAccounting(unittest.TestCase):
    """Static text weight accounting over the unmodified checkpoint index.

    Classifies every safetensors key into the exact set of text modules this
    bring-up builds, the explicitly-excluded non-text (vision/projector) keys,
    and anything unaccounted (a failure). This proves dense-vs-sparse and
    dense-vs-MoE layer ids, the K-only index branch (disable-index-value), and
    that no text weight is silently dropped -- without loading 800 GB of tensors.
    """

    _NON_TEXT_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")

    def _expected_text_keys(self, num_layers, moe_freq, sparse_freq, num_experts):
        keys = set()
        keys.add("model.embed_tokens.weight")
        keys.add("model.norm.weight")
        keys.add("lm_head.weight")
        for layer in range(num_layers):
            p = f"model.layers.{layer}"
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                keys.add(f"{p}.self_attn.{proj}.weight")
            keys.add(f"{p}.self_attn.q_norm.weight")
            keys.add(f"{p}.self_attn.k_norm.weight")
            keys.add(f"{p}.input_layernorm.weight")
            keys.add(f"{p}.post_attention_layernorm.weight")
            if sparse_freq[layer]:
                for proj in ("index_q_proj", "index_k_proj"):
                    keys.add(f"{p}.self_attn.{proj}.weight")
                keys.add(f"{p}.self_attn.index_q_norm.weight")
                keys.add(f"{p}.self_attn.index_k_norm.weight")
            if moe_freq[layer]:
                m = f"{p}.block_sparse_moe"
                keys.add(f"{m}.gate.weight")
                keys.add(f"{m}.e_score_correction_bias")
                for e in range(num_experts):
                    for w in ("w1", "w2", "w3"):
                        keys.add(f"{m}.experts.{e}.{w}.weight")
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    keys.add(f"{m}.shared_experts.{proj}.weight")
            else:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    keys.add(f"{p}.mlp.{proj}.weight")
        return keys

    def test_checkpoint_text_weights_fully_accounted(self):
        root = llm_models_root()
        if root is None:
            self.skipTest("llm_models_root() is unavailable in this environment")
        ckpt = os.path.join(str(root), "MiniMax-M3")
        self.assertTrue(os.path.isdir(ckpt), f"MiniMax-M3 checkpoint not found at {ckpt}")

        with open(os.path.join(ckpt, "config.json")) as f:
            raw = json.load(f)
        text = raw["text_config"]
        num_layers = text["num_hidden_layers"]
        moe_freq = text["moe_layer_freq"]
        sparse_freq = text["sparse_attention_config"]["sparse_attention_freq"]
        num_experts = text["num_local_experts"]

        index_path = os.path.join(ckpt, "model.safetensors.index.json")
        with open(index_path) as f:
            all_keys = list(json.load(f)["weight_map"].keys())
        # Sanity: the index enumerates the checkpoint (fail loudly if empty).
        self.assertGreater(len(all_keys), 0)

        prefix = "language_model."
        expected_text = self._expected_text_keys(num_layers, moe_freq, sparse_freq, num_experts)

        seen_text = set()
        ignored_non_text = set()
        unaccounted = []
        for key in all_keys:
            if key.startswith(prefix):
                stripped = key[len(prefix) :]
                if stripped in expected_text:
                    seen_text.add(stripped)
                else:
                    unaccounted.append(key)
            elif key.startswith(self._NON_TEXT_PREFIXES):
                ignored_non_text.add(key)
            else:
                unaccounted.append(key)

        self.assertEqual(unaccounted, [], f"Unaccounted-for checkpoint keys: {unaccounted[:10]}")
        # Every expected text weight is present -- nothing silently missing.
        missing = expected_text - seen_text
        self.assertEqual(
            missing, set(), f"Expected text weights missing from checkpoint: {sorted(missing)[:10]}"
        )
        # Non-text weights exist and were excluded on purpose.
        self.assertGreater(len(ignored_non_text), 0)

        # Disable-index-value: the K-only index branch has no value/output proj.
        for key in all_keys:
            self.assertNotIn("index_v_proj", key)
            self.assertNotIn("index_o_proj", key)

    def test_checkpoint_dense_and_sparse_layer_ids(self):
        """Layers 0-2 are dense (no index/MoE keys); 3-59 are sparse+MoE."""
        root = llm_models_root()
        if root is None:
            self.skipTest("llm_models_root() is unavailable in this environment")
        ckpt = os.path.join(str(root), "MiniMax-M3")
        self.assertTrue(os.path.isdir(ckpt), f"MiniMax-M3 checkpoint not found at {ckpt}")
        with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
            all_keys = list(json.load(f)["weight_map"].keys())

        def layer_has(layer, needle):
            token = f"model.layers.{layer}."
            return any(token in k and needle in k for k in all_keys)

        # Dense leading layers.
        for layer in range(3):
            self.assertTrue(
                layer_has(layer, ".mlp.gate_proj"), f"layer {layer} should have a dense MLP"
            )
            self.assertFalse(
                layer_has(layer, ".block_sparse_moe."), f"layer {layer} should have no MoE"
            )
            self.assertFalse(
                layer_has(layer, ".index_q_proj"), f"layer {layer} should have no index branch"
            )
        # Sparse + MoE trailing layers.
        for layer in (3, 4, 59):
            self.assertTrue(
                layer_has(layer, ".block_sparse_moe.gate"), f"layer {layer} should have MoE"
            )
            self.assertTrue(
                layer_has(layer, ".index_q_proj"), f"layer {layer} should have an index branch"
            )
            self.assertFalse(
                layer_has(layer, ".mlp.gate_proj"), f"layer {layer} should have no dense MLP"
            )


if __name__ == "__main__":
    unittest.main()

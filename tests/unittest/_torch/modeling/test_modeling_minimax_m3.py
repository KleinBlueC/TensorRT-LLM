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


if __name__ == "__main__":
    unittest.main()

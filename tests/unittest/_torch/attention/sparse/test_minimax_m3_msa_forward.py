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
"""Model-owned MiniMax Sparse Attention (MSA) forward test.

Exercises ``MiniMaxM3Attention.forward_sparse`` end to end for a real sparse
decoder layer: project + per-head Gemma QK/index norm + explicit partial NeoX
RoPE -> scatter main K/V + index-K into the ``MiniMaxM3CacheManager`` (V2 main
pool + K-only index side pool) -> paged Triton MSA over the top-k selected
blocks -> ``o_proj``. The load-bearing check is **forward_sparse == contiguous
reference**: the reference lays the module's *own* projected/roped tensors into a
trivially contiguous cache and runs the same MSA kernel, so any bug in the
cache-slot mapping, the prefill/decode split, or the metadata extraction breaks
the equality. Covers both a prefill batch (with the top-k drop regime) and a
decode step reusing prefilled history (cache reuse).
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MiniMax-M3 MSA forward tests require CUDA"
)

import tensorrt_llm.bindings  # noqa: E402
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (  # noqa: E402
    MiniMaxM3CacheManager,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels import (  # noqa: E402
    minimax_m3_sparse_attention_decode,
    minimax_m3_sparse_attention_prefill,
)
from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention  # noqa: E402
from tensorrt_llm.llmapi.llm_args import KvCacheConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

# Reduced but structurally-real geometry: GQA 16/4, head_dim 128, K-only index
# branch with num_index_heads == num_kv_heads (no top-k index reduction).
HIDDEN = 2048
NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 128
INDEX_DIM = 128
NUM_INDEX_HEADS = 4
BLOCK_SIZE = 128
TOPK = 4  # drop regime begins at kv_len > TOPK * BLOCK_SIZE = 512
NUM_LAYERS = 4
SPARSE_LAYER = 3  # layers 0-2 dense, 3+ sparse


def _text_config():
    freq = [0, 0, 0, 1]
    return {
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": 1024,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "use_gemma_norm": True,
        "rope_theta": 5000000,
        "rotary_dim": 64,
        "partial_rotary_factor": 0.5,
        "qk_norm_type": "per_head",
        "use_qk_norm": True,
        "torch_dtype": "bfloat16",
        "moe_layer_freq": list(freq),
        "sparse_attention_config": {
            "sparse_index_dim": INDEX_DIM,
            "sparse_num_index_heads": NUM_INDEX_HEADS,
            "sparse_topk_blocks": TOPK,
            "sparse_block_size": BLOCK_SIZE,
            "sparse_score_type": "max",
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_disable_index_value": list(freq),
            "sparse_attention_freq": list(freq),
        },
    }


def _build_sparse_attention(device):
    from transformers import PretrainedConfig

    cfg = PretrainedConfig.from_dict(_text_config())
    model_config = ModelConfig(pretrained_config=cfg)
    attn = MiniMaxM3Attention(model_config=model_config, layer_idx=SPARSE_LAYER)
    attn = attn.to(device).eval()
    assert attn.is_sparse_attention_layer
    assert attn.head_dim == HEAD_DIM and attn.num_heads == NUM_Q_HEADS
    # Deterministic finite weights (default init leaves empty tensors). Both the
    # forward_sparse path and the reference use the SAME module, so the exact
    # values only need to be finite and reproducible.
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype) * 0.05)
    return attn


def _make_manager(max_seq_len, max_batch_size, max_tokens):
    kv_cache_config = KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False)
    mapping = Mapping(world_size=1, tp_size=1, rank=0, pp_size=1)
    return MiniMaxM3CacheManager(
        kv_cache_config,
        CacheType.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=BLOCK_SIZE,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=DataType.BF16,
        vocab_size=1024,
        index_head_dim=INDEX_DIM,
    )


def _ref_msa_prefill(attn, q, k, v, index_q, index_k, seq_lens, device):
    """Contiguous-cache MSA prefill over the module's own roped tensors."""
    num_seqs = len(seq_lens)
    max_len = max(seq_lens)
    max_slots = num_seqs * max_len
    k_c = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=k.dtype, device=device)
    v_c = torch.zeros_like(k_c)
    ik_c = torch.zeros(max_slots, 1, INDEX_DIM, dtype=index_k.dtype, device=device)
    r2t = torch.zeros(num_seqs, max_len, dtype=torch.int32, device=device)
    off = 0
    for r, length in enumerate(seq_lens):
        base = r * max_len
        k_c[base : base + length] = k[off : off + length]
        v_c[base : base + length] = v[off : off + length]
        ik_c[base : base + length, 0] = index_k[off : off + length]
        r2t[r, :length] = torch.arange(base, base + length, device=device, dtype=torch.int32)
        off += length
    cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seq_lens, device=device).cumsum(0)
    prefix = torch.zeros(num_seqs, dtype=torch.int32, device=device)
    slot_ids = torch.arange(num_seqs, dtype=torch.int32, device=device)
    o, _ = minimax_m3_sparse_attention_prefill(
        q,
        k_c,
        v_c,
        index_q,
        ik_c,
        r2t,
        cu,
        prefix,
        slot_ids,
        block_size=attn.msa_params.block_size,
        topk=attn.msa_params.topk_blocks,
        init_blocks=attn.msa_params.init_blocks,
        local_blocks=attn.msa_params.local_blocks,
    )
    return o


def _fake_metadata(manager, num_contexts, num_ctx_tokens, request_ids, query_lens, cached_lens):
    return SimpleNamespace(
        kv_cache_manager=manager,
        num_contexts=num_contexts,
        num_ctx_tokens=num_ctx_tokens,
        request_ids=list(request_ids),
        seq_lens=torch.tensor(query_lens, dtype=torch.int32),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=list(cached_lens)),
    )


@pytest.mark.parametrize("seq_lens", [[640, 256], [768, 512]])
def test_forward_sparse_prefill_matches_contiguous(seq_lens):
    device = torch.device("cuda")
    torch.manual_seed(1234)
    attn = _build_sparse_attention(device)
    manager = _make_manager(max_seq_len=1024, max_batch_size=len(seq_lens), max_tokens=8192)
    try:
        request_ids = list(range(len(seq_lens)))
        manager.add_dummy_requests(
            request_ids, token_nums=list(seq_lens), is_gen=False, prepare_resource=True
        )
        total = sum(seq_lens)
        hidden = torch.randn(total, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
        position_ids = torch.cat([torch.arange(length, device=device) for length in seq_lens]).to(
            torch.int32
        )

        metadata = _fake_metadata(
            manager, len(seq_lens), total, request_ids, list(seq_lens), [0] * len(seq_lens)
        )
        with torch.no_grad():
            out = attn.forward_sparse(position_ids, hidden, metadata)
            q, k, v, index_q, index_k = attn._project_and_rope_sparse(position_ids, hidden)
            ref_attn = _ref_msa_prefill(attn, q, k, v, index_q, index_k, list(seq_lens), device)
            ref_out = attn.o_proj(ref_attn.reshape(total, NUM_Q_HEADS * HEAD_DIM))

        assert out.shape == (total, HIDDEN)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=3e-2)
        # Prove the drop regime is exercised: req0 has >TOPK blocks.
        nblocks0 = (seq_lens[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        assert nblocks0 > TOPK
    finally:
        manager.shutdown()


def test_forward_sparse_decode_reuses_prefilled_cache():
    device = torch.device("cuda")
    torch.manual_seed(7)
    attn = _build_sparse_attention(device)
    hist = [640, 320]
    request_ids = list(range(len(hist)))
    seq_full = [h + 1 for h in hist]
    manager = _make_manager(max_seq_len=1024, max_batch_size=len(hist), max_tokens=8192)
    try:
        manager.add_dummy_requests(
            request_ids, token_nums=list(seq_full), is_gen=True, prepare_resource=True
        )
        # Full per-request hidden for positions 0..hist (history + the decode
        # token). Laid out per request so slicing the last token per request
        # yields the decode batch.
        hidden_by_req = [
            torch.randn(length, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
            for length in seq_full
        ]
        hidden_prefill = torch.cat([h[:-1] for h in hidden_by_req], dim=0)
        pos_prefill = torch.cat([torch.arange(length, device=device) for length in hist]).to(
            torch.int32
        )
        hidden_decode = torch.cat([h[-1:] for h in hidden_by_req], dim=0)
        pos_decode = torch.tensor(hist, dtype=torch.int32, device=device)

        with torch.no_grad():
            # 1) Prefill populates history (positions 0..hist-1) in the cache.
            meta_ctx = _fake_metadata(
                manager, len(hist), sum(hist), request_ids, list(hist), [0] * len(hist)
            )
            attn.forward_sparse(pos_prefill, hidden_prefill, meta_ctx)

            # 2) Decode one token per request; forward_sparse writes position hist
            # and attends over the full 0..hist history from the cache.
            meta_gen = _fake_metadata(manager, 0, 0, request_ids, [1] * len(hist), list(hist))
            out = attn.forward_sparse(pos_decode, hidden_decode, meta_gen)

            # Reference: contiguous decode over the module's own roped tensors for
            # the full 0..hist sequence (re-projected -> identical values).
            hidden_full = torch.cat(hidden_by_req, dim=0)
            pos_full = torch.cat([torch.arange(length, device=device) for length in seq_full]).to(
                torch.int32
            )
            q_f, k_f, v_f, iq_f, ik_f = attn._project_and_rope_sparse(pos_full, hidden_full)

            num_seqs = len(seq_full)
            max_len = max(seq_full)
            k_c = torch.zeros(
                num_seqs * max_len, NUM_KV_HEADS, HEAD_DIM, dtype=k_f.dtype, device=device
            )
            v_c = torch.zeros_like(k_c)
            ik_c = torch.zeros(num_seqs * max_len, 1, INDEX_DIM, dtype=ik_f.dtype, device=device)
            r2t = torch.zeros(num_seqs, max_len, dtype=torch.int32, device=device)
            q_dec = torch.empty(num_seqs, NUM_Q_HEADS, HEAD_DIM, dtype=q_f.dtype, device=device)
            iq_dec = torch.empty(
                num_seqs, NUM_INDEX_HEADS, INDEX_DIM, dtype=iq_f.dtype, device=device
            )
            off = 0
            for r, length in enumerate(seq_full):
                base = r * max_len
                k_c[base : base + length] = k_f[off : off + length]
                v_c[base : base + length] = v_f[off : off + length]
                ik_c[base : base + length, 0] = ik_f[off : off + length]
                r2t[r, :length] = torch.arange(
                    base, base + length, device=device, dtype=torch.int32
                )
                q_dec[r] = q_f[off + length - 1]
                iq_dec[r] = iq_f[off + length - 1]
                off += length
            seq_lens_t = torch.tensor(seq_full, dtype=torch.int32, device=device)
            slot_ids = torch.arange(num_seqs, dtype=torch.int32, device=device)
            max_nb = (max(seq_full) + BLOCK_SIZE - 1) // BLOCK_SIZE
            ref_attn, _ = minimax_m3_sparse_attention_decode(
                q_dec,
                k_c,
                v_c,
                iq_dec,
                ik_c,
                r2t,
                seq_lens_t,
                slot_ids,
                block_size=attn.msa_params.block_size,
                topk=attn.msa_params.topk_blocks,
                init_blocks=attn.msa_params.init_blocks,
                local_blocks=attn.msa_params.local_blocks,
                max_num_blocks=max_nb,
            )
            ref_out = attn.o_proj(ref_attn.reshape(num_seqs, NUM_Q_HEADS * HEAD_DIM))

        assert out.shape == (num_seqs, HIDDEN)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=3e-2)
    finally:
        manager.shutdown()

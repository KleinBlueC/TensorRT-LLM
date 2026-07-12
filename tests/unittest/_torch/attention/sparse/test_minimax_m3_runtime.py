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
"""Runtime seam tests for MiniMax-M3 sparse attention (MSA).

These exercise the *new* runtime piece -- a ``KVCacheManagerV2`` K-only index
side cache feeding the Triton MSA kernels -- on real V2-allocated paged buffers,
rather than the kernels in isolation (covered by
``test_minimax_m3_sparse_attention.py``).  The load-bearing check is
**paged == contiguous**: the same kernel over the same data must give the same
result whether the K/V/index-K live in the paged V2 pool (addressed through the
block table) or in a trivially contiguous buffer.  Any bug in the side-pool
allocation, the block-id -> flat-slot mapping, or the main-K/V read view breaks
that equality.  We also assert the top-k drop regime is actually exercised and
that the paged decode path runs under CUDA-graph capture/replay.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MiniMax-M3 MSA runtime tests require CUDA"
)

import tensorrt_llm.bindings  # noqa: E402
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (  # noqa: E402
    MiniMaxM3CacheManager, MiniMaxM3Params, flatten_slot_ids,
    minimax_m3_build_req_to_token, minimax_m3_paged_decode,
    minimax_m3_paged_prefill)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels import (  # noqa: E402
    minimax_m3_sparse_attention_decode, minimax_m3_sparse_attention_prefill)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

# Real MiniMax-M3 attention dims (checkpoint-scale head geometry, few layers).
NUM_KV_HEADS = 4
NUM_Q_HEADS = 64
HEAD_DIM = 128
INDEX_DIM = 128
NUM_INDEX_HEADS = 4  # == NUM_KV_HEADS: no top-k index reduction (K-only branch)
BLOCK_SIZE = 128  # MSA block == cache page here (keeps block<->page 1:1)
TOPK = 4  # small so the drop regime starts at seq_len > TOPK*BLOCK_SIZE = 512


def _params():
    return MiniMaxM3Params(
        index_head_dim=INDEX_DIM,
        num_index_heads=NUM_INDEX_HEADS,
        topk_blocks=TOPK,
        block_size=BLOCK_SIZE,
        init_blocks=0,
        local_blocks=1,
    )


def _make_manager(num_layers, max_seq_len, max_batch_size, max_tokens):
    kv_cache_config = KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False)
    mapping = Mapping(world_size=1, tp_size=1, rank=0, pp_size=1)
    return MiniMaxM3CacheManager(
        kv_cache_config,
        CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=BLOCK_SIZE,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=DataType.BF16,
        vocab_size=32000,
        index_head_dim=INDEX_DIM,
    )


def _block_ids_for(manager, request_ids, seq_lens):
    """Valid (non-padding) block ids per request, one per allocated page."""
    raw = manager.get_batch_cache_indices(request_ids)
    tpb = manager.tokens_per_block
    out = []
    for req_id, length in zip(request_ids, seq_lens):
        nblocks = (length + tpb - 1) // tpb
        ids = [b for b in raw[request_ids.index(req_id)] if b >= 0][:nblocks]
        assert len(ids) == nblocks, (
            f"request {req_id}: got {len(ids)} valid blocks, need {nblocks}"
        )
        out.append(ids)
    return out


def _contiguous_prefill(q, k, v, index_q, index_k, seq_lens, params, device):
    """Reference: same kernel over a trivially contiguous per-request cache.

    Lays each request's K/V/index-K at slots ``[r * max_len, ...]`` so the flat
    slot map is trivial, then runs the *same* prefill kernel.  Equals the paged
    run iff the paged block-id -> slot mapping and cache round-trip are correct.
    """
    num_seqs = len(seq_lens)
    max_len = max(seq_lens)
    max_slots = num_seqs * max_len
    k_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=k.dtype, device=device)
    v_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=v.dtype, device=device)
    index_k_cache = torch.zeros(max_slots, 1, INDEX_DIM, dtype=index_k.dtype, device=device)
    req_to_token = torch.zeros(num_seqs, max_len, dtype=torch.int32, device=device)
    off = 0
    for r, length in enumerate(seq_lens):
        base = r * max_len
        sl = slice(off, off + length)
        k_cache[base:base + length] = k[sl]
        v_cache[base:base + length] = v[sl]
        index_k_cache[base:base + length, 0] = index_k[sl]
        req_to_token[r, :length] = torch.arange(base, base + length, device=device, dtype=torch.int32)
        off += length
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(seq_lens, device=device).cumsum(0)
    prefix_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device)
    slot_ids = torch.arange(num_seqs, dtype=torch.int32, device=device)
    return minimax_m3_sparse_attention_prefill(
        q, k_cache, v_cache, index_q, index_k_cache, req_to_token, cu_seqlens,
        prefix_lens, slot_ids, block_size=params.block_size, topk=params.topk_blocks,
        init_blocks=params.init_blocks, local_blocks=params.local_blocks)


def test_cache_manager_index_pool_shape():
    """The K-only index side pool is sized to the primary pool's block count."""
    manager = _make_manager(num_layers=2, max_seq_len=1024, max_batch_size=2,
                            max_tokens=4096)
    try:
        buf = manager.get_buffers(0)  # [num_blocks, kv_factor, tpb, H, D]
        num_blocks = buf.shape[0]
        assert buf.shape[1] == 2, "main K/V pool must be 2-factor (SELF)"
        assert buf.shape[3] == NUM_KV_HEADS and buf.shape[4] == HEAD_DIM
        index_pool = manager.get_index_k_buffers(0)
        assert index_pool.shape == (num_blocks * manager.tokens_per_block, 1, INDEX_DIM)
        assert index_pool.dtype == torch.bfloat16
    finally:
        manager.shutdown()


def test_index_k_write_read_roundtrip():
    """Index-K scattered at block-table slots reads back exactly at those slots."""
    device = torch.device("cuda")
    manager = _make_manager(num_layers=1, max_seq_len=1024, max_batch_size=2,
                            max_tokens=4096)
    try:
        request_ids = [0, 1]
        seq_lens = [300, 200]  # spans >1 block each (tpb=128)
        manager.add_dummy_requests(request_ids, token_nums=seq_lens, is_gen=False,
                                  prepare_resource=True)
        block_ids = _block_ids_for(manager, request_ids, seq_lens)

        positions, req_of_token, values = [], [], []
        for r, length in enumerate(seq_lens):
            for p in range(length):
                positions.append(p)
                req_of_token.append(r)
        index_k = torch.randn(len(positions), INDEX_DIM, dtype=torch.bfloat16, device=device)
        slots = flatten_slot_ids(block_ids, positions, req_of_token,
                                 manager.tokens_per_block, device)
        manager.write_index_k(0, slots, index_k)

        pool = manager.get_index_k_buffers(0)
        torch.testing.assert_close(pool[slots, 0], index_k)
        # distinct tokens must map to distinct slots (no aliasing across requests)
        assert slots.unique().numel() == slots.numel()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("seq_lens", [[640, 256], [768, 512]])
def test_paged_prefill_matches_contiguous(seq_lens):
    """MSA prefill over the paged V2 pool == over a contiguous cache (drop regime)."""
    device = torch.device("cuda")
    torch.manual_seed(1234)
    total = sum(seq_lens)
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    index_q = torch.randn(total, NUM_INDEX_HEADS, INDEX_DIM, dtype=torch.bfloat16, device=device)
    index_k = torch.randn(total, INDEX_DIM, dtype=torch.bfloat16, device=device)
    params = _params()

    manager = _make_manager(num_layers=1, max_seq_len=1024, max_batch_size=len(seq_lens),
                            max_tokens=4096)
    try:
        request_ids = list(range(len(seq_lens)))
        manager.add_dummy_requests(request_ids, token_nums=list(seq_lens), is_gen=False,
                                  prepare_resource=True)
        block_ids = _block_ids_for(manager, request_ids, seq_lens)

        positions, req_of_token = [], []
        for r, length in enumerate(seq_lens):
            positions.extend(range(length))
            req_of_token.extend([r] * length)
        slots = flatten_slot_ids(block_ids, positions, req_of_token,
                                 manager.tokens_per_block, device)
        manager.write_main_kv(0, slots, k, v)
        manager.write_index_k(0, slots, index_k)

        req_to_token = minimax_m3_build_req_to_token(block_ids, seq_lens,
                                                     manager.tokens_per_block, device)
        cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.tensor(seq_lens, device=device).cumsum(0)
        prefix_lens = torch.zeros(len(seq_lens), dtype=torch.int32, device=device)
        slot_ids = torch.arange(len(seq_lens), dtype=torch.int32, device=device)
        paged_o, paged_topk = minimax_m3_paged_prefill(
            manager, 0, q, index_q, req_to_token, cu_seqlens, prefix_lens, slot_ids, params)

        ref_o, _ = _contiguous_prefill(q, k, v, index_q, index_k, list(seq_lens), params, device)
        torch.testing.assert_close(paged_o, ref_o, atol=2e-2, rtol=2e-2)

        # Drop regime: the longest query in a >TOPK-block request must not select
        # every eligible block (otherwise the test is dense-equivalent).
        last_q_of_req0 = seq_lens[0] - 1
        nblocks0 = (seq_lens[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        assert nblocks0 > TOPK, "test config must exceed TOPK blocks to prove drops"
        selected = paged_topk[:, last_q_of_req0, :]  # [num_kv_heads, topk]
        for head in range(NUM_KV_HEADS):
            valid = (selected[head] >= 0).sum().item()
            assert valid <= TOPK < nblocks0
    finally:
        manager.shutdown()


def test_paged_decode_matches_contiguous_and_cuda_graph():
    """MSA decode over the paged V2 pool == contiguous, and runs under CUDA graph."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    hist = [640, 320]  # cached history; decode one token at position == history
    seq_lens = [h + 1 for h in hist]
    total = sum(seq_lens)
    params = _params()

    # Per-request K/V/index-K for every cached + new position.
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    index_k = torch.randn(total, INDEX_DIM, dtype=torch.bfloat16, device=device)
    # One decode query + index-query per request.
    q = torch.randn(len(seq_lens), NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    index_q = torch.randn(len(seq_lens), NUM_INDEX_HEADS, INDEX_DIM, dtype=torch.bfloat16, device=device)

    manager = _make_manager(num_layers=1, max_seq_len=1024, max_batch_size=len(seq_lens),
                            max_tokens=4096)
    try:
        request_ids = list(range(len(seq_lens)))
        manager.add_dummy_requests(request_ids, token_nums=list(seq_lens), is_gen=True,
                                  prepare_resource=True)
        block_ids = _block_ids_for(manager, request_ids, seq_lens)

        positions, req_of_token = [], []
        for r, length in enumerate(seq_lens):
            positions.extend(range(length))
            req_of_token.extend([r] * length)
        slots = flatten_slot_ids(block_ids, positions, req_of_token,
                                 manager.tokens_per_block, device)
        manager.write_main_kv(0, slots, k, v)
        manager.write_index_k(0, slots, index_k)

        req_to_token = minimax_m3_build_req_to_token(block_ids, seq_lens,
                                                     manager.tokens_per_block, device)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        slot_ids = torch.arange(len(seq_lens), dtype=torch.int32, device=device)
        max_num_blocks = (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE

        paged_o, _ = minimax_m3_paged_decode(
            manager, 0, q, index_q, req_to_token, seq_lens_t, slot_ids, params,
            max_num_blocks=max_num_blocks)

        # Contiguous reference decode.
        num_seqs = len(seq_lens)
        max_len = max(seq_lens)
        k_c = torch.zeros(num_seqs * max_len, NUM_KV_HEADS, HEAD_DIM, dtype=k.dtype, device=device)
        v_c = torch.zeros_like(k_c)
        ik_c = torch.zeros(num_seqs * max_len, 1, INDEX_DIM, dtype=index_k.dtype, device=device)
        r2t_c = torch.zeros(num_seqs, max_len, dtype=torch.int32, device=device)
        off = 0
        for r, length in enumerate(seq_lens):
            base = r * max_len
            k_c[base:base + length] = k[off:off + length]
            v_c[base:base + length] = v[off:off + length]
            ik_c[base:base + length, 0] = index_k[off:off + length]
            r2t_c[r, :length] = torch.arange(base, base + length, device=device, dtype=torch.int32)
            off += length
        ref_o, _ = minimax_m3_sparse_attention_decode(
            q, k_c, v_c, index_q, ik_c, r2t_c, seq_lens_t, slot_ids,
            block_size=params.block_size, topk=params.topk_blocks,
            init_blocks=params.init_blocks, local_blocks=params.local_blocks,
            max_num_blocks=max_num_blocks)
        torch.testing.assert_close(paged_o, ref_o, atol=2e-2, rtol=2e-2)

        # CUDA-graph hard path: capture/replay the paged decode kernel dispatch.
        k_cache, v_cache = manager.get_main_kv_slot_buffers(0)
        index_k_cache = manager.get_index_k_buffers(0)
        graph_out = torch.empty_like(paged_o)

        def _run():
            o, _ = minimax_m3_sparse_attention_decode(
                q, k_cache, v_cache, index_q, index_k_cache, req_to_token, seq_lens_t,
                slot_ids, block_size=params.block_size, topk=params.topk_blocks,
                init_blocks=params.init_blocks, local_blocks=params.local_blocks,
                max_num_blocks=max_num_blocks)
            graph_out.copy_(o)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _run()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out, paged_o, atol=2e-2, rtol=2e-2)
    finally:
        manager.shutdown()

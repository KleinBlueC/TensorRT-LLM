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
"""Triton kernels for MiniMax-M3 sparse attention (MSA).

MiniMax-M3 trailing layers (3-59) run a low-dimensional "index" branch that
scores fixed-size KV blocks and selects the top-k blocks per query, then run the
main GQA attention **only** over the selected blocks. This mirrors the SGLang
reference sparse ops under
``python/sglang/srt/layers/attention/minimax_sparse_ops`` (``minimax_sparse.py``
orchestration, ``decode/topk_sparse.py`` sparse GQA, ``decode/flash_with_topk_idx``
index scoring). The semantics implemented here are the definition of correctness;
the kernels are clean-room OpenAI-Triton (parity-first: performance tuning such as
split-K / TMA that SGLang adds is intentionally out of scope for bring-up).

Contract (paged KV, decode step -- one query token per request):

* ``q``            ``[batch, num_q_heads, head_dim]`` (bf16/fp16)
* ``k_cache``/``v_cache`` ``[max_slots, num_kv_heads, head_dim]`` (paged main cache)
* ``index_q``     ``[batch, num_index_heads, index_dim]`` (bf16/fp16)
* ``index_k_cache`` ``[max_slots, 1, index_dim]`` (paged K-only index side cache)
* ``req_to_token`` ``[max_reqs, max_kv_len]`` maps ``(slot_id, pos) -> cache slot``
* ``slot_ids``    ``[batch]`` selects the ``req_to_token`` row for each request

For the released MiniMax-M3 checkpoint ``sparse_disable_index_value=1``: the index
branch produces **only** the top-k block ids (no ``index_o_proj`` value output),
and ``num_index_heads == num_key_value_heads`` (4 == 4) so the per-index-head
top-k already lines up with the KV heads (no ``topk_index_reduce`` is needed).

Block selection semantics (score_type="max"):

* ``score[h, b, blk]`` = max over the tokens of block ``blk`` of
  ``index_q[b, h, :] . index_k[slot, :] * idx_sm_scale`` (positions ``>= seq_len``
  masked to ``-inf``).
* ``init_blocks`` leading blocks and ``local_blocks`` trailing blocks are always
  retained (scores forced high). For MiniMax-M3: ``init_blocks=0``,
  ``local_blocks=1``, ``topk=16``, ``block_size=128``.
* Blocks start being dropped once ``ceil(seq_len / block_size) > topk`` (i.e.
  ``seq_len > 2048`` for the production config); below that MSA is
  dense-equivalent.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Scores used to force init/local blocks to the top of the selection. INIT ranks
# strictly above LOCAL so that, when both apply to the same block, the init
# guarantee wins (matches the SGLang naive reference constants).
_INIT_SCORE = 1e30
_LOCAL_SCORE = 1e29


@triton.jit
def _index_block_score_decode_kernel(
    q_ptr,  # [batch, num_index_heads, index_dim]
    k_cache_ptr,  # [max_slots, index_dim] (index head 0 squeezed out)
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    score_ptr,  # [num_index_heads, batch, max_num_blocks]
    seq_lens_ptr,  # [batch]
    slot_ids_ptr,  # [batch]
    max_slots,
    max_kv_len,
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_r2t_b,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    INDEX_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program computes the score of a single (batch, index_head, block).

    grid = (batch, num_index_heads, max_num_blocks). Blocks past the request's
    ``seq_len`` write ``-inf`` so the downstream vectorized top-k drops them.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + pid_b)
    seq_len = tl.minimum(seq_len, max_kv_len)
    sid = (tl.load(slot_ids_ptr + pid_b).to(tl.int64) + max_slots) % max_slots

    off_d = tl.arange(0, BLOCK_D)
    d_mask = off_d < INDEX_DIM
    q = tl.load(
        q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + off_d * stride_q_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    off_n = tl.arange(0, BLOCK_N)
    pos = pid_n * BLOCK_N + off_n
    pos_mask = pos < seq_len
    slots = tl.load(
        req_to_token_ptr + sid * stride_r2t_b + pos,
        mask=pos_mask,
        other=0,
    ).to(tl.int64)
    slots = (slots + max_slots) % max_slots

    k = tl.load(
        k_cache_ptr + slots[:, None] * stride_k_s + off_d[None, :] * stride_k_d,
        mask=pos_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # per-position score = index_q . index_k, then block score = max over tokens.
    scores = tl.sum(k * q[None, :], axis=1) * sm_scale  # [BLOCK_N]
    scores = tl.where(pos_mask, scores, float("-inf"))
    block_score = tl.max(scores, axis=0)
    tl.store(
        score_ptr + pid_h * stride_s_h + pid_b * stride_s_b + pid_n * stride_s_n,
        block_score,
    )


def _select_topk_blocks(
    block_scores: torch.Tensor,  # [num_index_heads, batch, max_num_blocks] (fp32)
    seq_lens: torch.Tensor,  # [batch]
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Vectorized, CUDA-graph-safe top-k block selection.

    Returns ``topk_idx`` ``[num_index_heads, batch, topk]`` (int32), left-packed
    with the selected block ids and ``-1`` padding, matching the SGLang naive
    reference set. Init/local blocks are forced to the top; blocks past a
    request's valid block count are masked to ``-inf`` (never selected, and when
    they are the only option they become ``-1``).
    """
    num_index_heads, batch, max_num_blocks = block_scores.shape
    device = block_scores.device
    block_ids = torch.arange(max_num_blocks, device=device)
    num_blocks = (seq_lens + block_size - 1) // block_size  # [batch]
    valid = block_ids[None, :] < num_blocks[:, None]  # [batch, max_num_blocks]

    scores = torch.where(valid[None], block_scores, float("-inf")).clone()
    if init_blocks > 0:
        is_init = valid & (block_ids[None, :] < init_blocks)
        scores = torch.where(is_init[None], torch.full_like(scores, _INIT_SCORE), scores)
    if local_blocks > 0:
        local_start = torch.clamp(num_blocks - local_blocks, min=0)  # [batch]
        is_local = valid & (block_ids[None, :] >= local_start[:, None])
        scores = torch.where(is_local[None], torch.full_like(scores, _LOCAL_SCORE), scores)

    k = min(topk, max_num_blocks)
    sel_scores, sel_idx = torch.topk(scores, k=k, dim=-1)  # descending
    # positions whose selected score is -inf correspond to padding blocks.
    sel_idx = torch.where(
        sel_scores > float("-inf"),
        sel_idx.to(torch.int32),
        torch.full_like(sel_idx, -1, dtype=torch.int32),
    )
    if k < topk:
        pad = torch.full((num_index_heads, batch, topk - k), -1, dtype=torch.int32, device=device)
        sel_idx = torch.cat([sel_idx, pad], dim=-1)
    return sel_idx.contiguous()


def minimax_m3_index_topk(
    index_q: torch.Tensor,  # [num_queries, num_index_heads, index_dim]
    index_k_cache: torch.Tensor,  # [max_slots, 1, index_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [num_queries] per-query causal K length
    slot_ids: torch.Tensor,  # [num_queries] req_to_token row per query
    block_size: int,
    topk: int,
    init_blocks: int = 0,
    local_blocks: int = 1,
    idx_sm_scale: Optional[float] = None,
    max_num_blocks: Optional[int] = None,
) -> torch.Tensor:
    """MiniMax-M3 index branch: select top-k KV blocks per query.

    Per-query: ``seq_lens[i]`` is the causal K length seen by query ``i`` and
    ``slot_ids[i]`` its ``req_to_token`` row. Decode passes one query per request
    (``seq_lens`` = full sequence length); prefill passes one query per token with
    ``seq_lens[i] = abs_pos(i) + 1`` (its causal prefix), so the same per-query
    kernel serves both regimes (MiniMax-M3 uses ``block_size_q == 1``).

    ``max_num_blocks`` may be supplied (``ceil(max_seqlen / block_size)``) to keep
    the launch grid independent of a device->host sync, which CUDA-graph capture
    requires; otherwise it is derived from ``seq_lens.max()``.
    """
    num_queries, num_index_heads, index_dim = index_q.shape
    max_slots = index_k_cache.shape[0]
    max_kv_len = req_to_token.shape[1]
    if idx_sm_scale is None:
        idx_sm_scale = index_dim**-0.5
    if max_num_blocks is None:
        max_seqlen = int(seq_lens.max().item())
        max_num_blocks = (max_seqlen + block_size - 1) // block_size
    max_num_blocks = max(1, max_num_blocks)

    index_k = index_k_cache.reshape(max_slots, index_dim)  # squeeze head 0
    block_scores = torch.empty(
        num_index_heads, num_queries, max_num_blocks, dtype=torch.float32, device=index_q.device
    )
    grid = (num_queries, num_index_heads, max_num_blocks)
    _index_block_score_decode_kernel[grid](
        index_q,
        index_k,
        req_to_token,
        block_scores,
        seq_lens,
        slot_ids,
        max_slots,
        max_kv_len,
        idx_sm_scale,
        index_q.stride(0),
        index_q.stride(1),
        index_q.stride(2),
        index_k.stride(0),
        index_k.stride(1),
        req_to_token.stride(0),
        block_scores.stride(0),
        block_scores.stride(1),
        block_scores.stride(2),
        INDEX_DIM=index_dim,
        BLOCK_D=triton.next_power_of_2(index_dim),
        BLOCK_N=block_size,
    )
    return _select_topk_blocks(block_scores, seq_lens, block_size, topk, init_blocks, local_blocks)


@triton.jit
def _sparse_gqa_decode_kernel(
    q_ptr,  # [batch, num_q_heads, head_dim]
    k_cache_ptr,  # [max_slots, num_kv_heads, head_dim]
    v_cache_ptr,  # [max_slots, num_kv_heads, head_dim]
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    idx_ptr,  # [num_kv_heads, batch, topk]
    o_ptr,  # [batch, num_q_heads, head_dim]
    seq_lens_ptr,
    slot_ids_ptr,
    max_slots,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_r2t_b,
    stride_i_h,
    stride_i_b,
    stride_i_t,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One program does the sparse GQA for a (batch, kv_head) over its blocks.

    Flash-attention online softmax over the selected blocks only. All ``gqa``
    query heads sharing this KV head reuse the same block selection.
    """
    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.minimum(tl.load(seq_lens_ptr + pid_b), max_kv_len)
    sid = (tl.load(slot_ids_ptr + pid_b).to(tl.int64) + max_slots) % max_slots

    off_h = tl.arange(0, BLOCK_H)
    off_d = tl.arange(0, BLOCK_D)
    off_n = tl.arange(0, BLOCK_N)
    h_mask = off_h < gqa_group_size
    d_mask = off_d < head_dim

    q = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + (pid_h + off_h)[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    idx_base = idx_ptr + pid_kh * stride_i_h + pid_b * stride_i_b
    # topk_idx is left-packed (valid ids first, -1 padding last), so counting the
    # valid entries lets us loop only over selected blocks. Looping over exactly
    # the valid blocks (rather than masking invalid ones) also avoids a NaN when
    # the selection is empty: the loop body simply never runs.
    off_t = tl.arange(0, BLOCK_T)
    topk_vals = tl.load(idx_base + off_t * stride_i_t, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_vals >= 0).to(tl.int32), axis=0)

    for t in tl.range(0, real_topk):
        blk = tl.load(idx_base + t * stride_i_t).to(tl.int32)
        c = blk * BLOCK_N
        pos = c + off_n
        pos_mask = pos < seq_len
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots
        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=pos_mask[None, :] & d_mask[:, None],
            other=0.0,
        )  # [BLOCK_D, BLOCK_N]
        v = tl.load(
            v_cache_ptr
            + slots[:, None] * stride_v_s
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )  # [BLOCK_N, BLOCK_D]
        qk = tl.dot(q, k) * sm_scale  # [BLOCK_H, BLOCK_N]
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc = acc * tl.exp(m_i - m_ij)[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

    # Empty selection (all -1) leaves m_i = lse_i = -inf; emit a clean zero.
    scale = tl.where(lse_i > float("-inf"), tl.exp(m_i - lse_i), tl.zeros_like(lse_i))
    acc = acc * scale[:, None]
    tl.store(
        o_ptr
        + pid_b * stride_o_b
        + (pid_h + off_h)[:, None] * stride_o_h
        + off_d[None, :] * stride_o_d,
        acc.to(o_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )


def minimax_m3_sparse_gqa(
    q: torch.Tensor,  # [batch, num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch]
    slot_ids: torch.Tensor,  # [batch]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads, batch, topk]
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """MiniMax-M3 sparse GQA (decode): main attention over the selected blocks."""
    batch, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    max_kv_len = req_to_token.shape[1]
    max_topk = topk_idx.shape[2]
    assert num_q_heads % num_kv_heads == 0
    assert topk_idx.shape[0] == num_kv_heads
    gqa_group_size = num_q_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    o = torch.empty_like(q)
    grid = (batch, num_kv_heads)
    _sparse_gqa_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        req_to_token,
        topk_idx,
        o,
        seq_lens,
        slot_ids,
        max_slots,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        req_to_token.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        BLOCK_H=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_N=block_size,
        BLOCK_T=triton.next_power_of_2(max_topk),
        num_warps=4,
    )
    return o


def minimax_m3_sparse_attention_decode(
    q: torch.Tensor,  # [batch, num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    index_q: torch.Tensor,  # [batch, num_index_heads, index_dim]
    index_k_cache: torch.Tensor,  # [max_slots, 1, index_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch]
    slot_ids: torch.Tensor,  # [batch]
    block_size: int,
    topk: int,
    init_blocks: int = 0,
    local_blocks: int = 1,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    max_num_blocks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full MiniMax-M3 sparse attention decode step (K-only index branch).

    Returns ``(o, topk_idx)`` where ``o`` is ``[batch, num_q_heads, head_dim]``
    and ``topk_idx`` is ``[num_kv_heads, batch, topk]``. ``num_index_heads`` must
    equal ``num_kv_heads`` for the released checkpoint, so no top-k reduction is
    applied.
    """
    num_kv_heads = k_cache.shape[1]
    num_index_heads = index_q.shape[1]
    assert num_index_heads == num_kv_heads, (
        f"MiniMax-M3 decode expects num_index_heads == num_kv_heads, got "
        f"{num_index_heads} != {num_kv_heads}"
    )
    topk_idx = minimax_m3_index_topk(
        index_q,
        index_k_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        block_size,
        topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        idx_sm_scale=idx_sm_scale,
        max_num_blocks=max_num_blocks,
    )
    o = minimax_m3_sparse_gqa(
        q,
        k_cache,
        v_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        block_size,
        topk_idx,
        sm_scale=sm_scale,
    )
    return o, topk_idx


def minimax_m3_sparse_attention_prefill(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim]
    index_q: torch.Tensor,  # [total_q, num_index_heads, index_dim]
    index_k_cache: torch.Tensor,  # [max_slots, 1, index_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    cu_seqlens: torch.Tensor,  # [batch + 1] extend-token cumulative counts
    prefix_lens: torch.Tensor,  # [batch] cached prefix length per request
    slot_ids: torch.Tensor,  # [batch] req_to_token row per request
    block_size: int,
    topk: int,
    init_blocks: int = 0,
    local_blocks: int = 1,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    max_num_blocks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full MiniMax-M3 sparse-attention prefill (per-token causal MSA).

    MiniMax-M3 uses ``block_size_q == 1``, so each of the ``total_q`` extend
    tokens runs its own causal MSA: the query at absolute position ``p`` attends
    to K positions ``[0, p + 1)``. That is exactly the per-query kernel with a
    per-query causal ``seq_lens = p + 1`` -- the position mask ``pos < seq_len``
    handles both block-level causality (blocks past ``p // block_size`` fall
    outside the query's block count) and intra-block causality (positions ``> p``
    are masked). This wrapper only lowers the varlen
    (``cu_seqlens``/``prefix_lens``/``slot_ids``) layout into that per-query
    metadata; no separate prefill kernel is needed.

    Returns ``(o, topk_idx)`` with ``o`` = ``[total_q, num_q_heads, head_dim]`` and
    ``topk_idx`` = ``[num_kv_heads, total_q, topk]``.
    """
    num_kv_heads = k_cache.shape[1]
    num_index_heads = index_q.shape[1]
    assert num_index_heads == num_kv_heads, (
        f"MiniMax-M3 prefill expects num_index_heads == num_kv_heads, got "
        f"{num_index_heads} != {num_kv_heads}"
    )
    device = q.device
    total_q = q.shape[0]
    batch = cu_seqlens.shape[0] - 1
    # Lower varlen -> per-query causal metadata: for query token i in request b at
    # local offset l, the absolute position is prefix_lens[b] + l and the causal K
    # length is that + 1.
    extend_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)  # [batch]
    req_id = torch.repeat_interleave(
        torch.arange(batch, device=device, dtype=torch.int64), extend_lens
    )  # [total_q]
    starts = cu_seqlens[:-1].to(torch.int64)  # [batch]
    local_idx = torch.arange(total_q, device=device, dtype=torch.int64) - starts[req_id]
    q_abs_pos = prefix_lens.to(torch.int64)[req_id] + local_idx
    q_seq_lens = (q_abs_pos + 1).to(torch.int32)  # per-query causal K range
    q_slot_ids = slot_ids.to(torch.int64)[req_id]  # [total_q]

    topk_idx = minimax_m3_index_topk(
        index_q,
        index_k_cache,
        req_to_token,
        q_seq_lens,
        q_slot_ids,
        block_size,
        topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        idx_sm_scale=idx_sm_scale,
        max_num_blocks=max_num_blocks,
    )
    o = minimax_m3_sparse_gqa(
        q,
        k_cache,
        v_cache,
        req_to_token,
        q_seq_lens,
        q_slot_ids,
        block_size,
        topk_idx,
        sm_scale=sm_scale,
    )
    return o, topk_idx

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
"""Runtime seam for MiniMax-M3 sparse attention (MSA).

MiniMax-M3 sparse layers (3-59) run the main GQA attention **only over the
top-k KV blocks** selected by a low-dimensional "index" branch. That needs two
cache contracts the standard runtime does not provide together:

1. the normal paged main K/V cache (every attention layer), and
2. a **K-only index side cache** (sparse layers only): one replicated index-K
   head, ``index_head_dim`` wide, written at the *same* token slots as the main
   cache and read by the block-selection kernel.

This module supplies (2) on top of ``KVCacheManagerV2`` (the task's mandated
cache manager) and the thin paged dispatch that feeds the MiniMax-M3 Triton MSA
kernels (``minimax_m3_kernels``) from those runtime buffers.

Design note (why a Python-owned side tensor).  DeepSeek Sparse Attention (DSA)
-- the only existing lightning-indexer sparse path in the tree -- allocates its
indexer K cache as a **separate C++ pool on KVCacheManager V1**
(``enable_indexer_k_cache=True``).  ``KVCacheManagerV2`` has no such C++
plumbing, and the project bring-up rule forbids new C++/CUDA for a
parity-first bring-up.  The plan's preferred direction is therefore realized
here: subclass ``KVCacheManagerV2`` for the main K/V pool and allocate a
**Python-owned bf16 K-only side tensor** per layer, sized to the same block
count and addressed by the *same* V2 block ids, so an index-K entry always
shares its token's main-cache slot.  V2 stays the single source of truth for
block allocation; the side tensor never invents its own scheduler.

Slot addressing.  ``KVCacheManagerV2.get_buffers(layer_idx)`` returns
``[num_blocks, kv_factor, tokens_per_block, num_kv_heads, head_dim]`` and
``get_batch_cache_indices(request_ids)`` returns per-request block ids already
divided by ``kv_factor`` (so they index ``num_blocks`` directly).  A token at
absolute position ``p`` of request ``r`` lives at flat slot
``block_ids[r][p // tokens_per_block] * tokens_per_block + p % tokens_per_block``
in both the main-K/V read view and the index side pool -- one ``req_to_token``
serves both.  The MSA kernels consume that flat ``[max_slots, ...]`` contract
(see ``minimax_m3_kernels``), so this module reshapes the main K/V pool into
that view for reads and writes new K/V/index-K into the aliased pool.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch

from ...pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from .minimax_m3_kernels import (
    minimax_m3_sparse_attention_decode,
    minimax_m3_sparse_attention_prefill,
)
from .params import SparseParams


@dataclass(frozen=True)
class MiniMaxM3Params(SparseParams):
    """Lowered per-backend MiniMax-M3 MSA parameters.

    Mirrors the released checkpoint's ``sparse_attention_config``: a K-only
    index branch (``disable_index_value``) with ``num_index_heads`` index heads
    (equal to ``num_key_value_heads`` for this checkpoint, so no top-k index
    reduction is needed), ``index_head_dim``-wide index vectors, and top-k
    selection of ``block_size``-token KV blocks with ``init_blocks`` leading and
    ``local_blocks`` trailing blocks always retained.
    """

    algorithm: Literal["minimax_m3"] = field(init=False, default="minimax_m3")
    index_head_dim: int = 128
    num_index_heads: int = 4
    topk_blocks: int = 16
    block_size: int = 128
    init_blocks: int = 0
    local_blocks: int = 1
    disable_index_value: bool = True

    @property
    def indices_block_size(self) -> int:
        # MiniMax-M3 selects whole KV blocks; block granularity == block_size.
        return self.block_size


def minimax_m3_build_req_to_token(
    block_ids_per_seq: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    tokens_per_block: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the ``req_to_token [num_seqs, max_kv_len]`` slot map for the kernels.

    ``req_to_token[r, p] = block_ids[r][p // tokens_per_block] * tokens_per_block
    + (p % tokens_per_block)`` for ``p < seq_lens[r]`` (0 elsewhere -- masked by
    the kernels' ``pos < seq_len`` guard).  ``block_ids_per_seq`` comes from
    ``KVCacheManagerV2.get_batch_cache_indices`` (already divided by
    ``kv_factor``).  Built on host/once-per-step, outside any CUDA-graph capture.
    """
    num_seqs = len(block_ids_per_seq)
    max_kv_len = max((int(s) for s in seq_lens), default=0)
    req_to_token = torch.zeros(num_seqs, max(1, max_kv_len), dtype=torch.int32, device=device)
    for r in range(num_seqs):
        length = int(seq_lens[r])
        if length <= 0:
            continue
        block_ids = torch.as_tensor(list(block_ids_per_seq[r]), dtype=torch.int64, device=device)
        pos = torch.arange(length, device=device, dtype=torch.int64)
        block = block_ids[pos // tokens_per_block]
        slot = block * tokens_per_block + (pos % tokens_per_block)
        req_to_token[r, :length] = slot.to(torch.int32)
    return req_to_token


def flatten_slot_ids(
    block_ids_per_seq: Sequence[Sequence[int]],
    token_positions: Sequence[int],
    req_ids_per_token: Sequence[int],
    tokens_per_block: int,
    device: torch.device,
) -> torch.Tensor:
    """Flat cache slots for a set of tokens being written this step.

    For token ``i`` belonging to request ``req_ids_per_token[i]`` at absolute
    position ``token_positions[i]``, the flat slot is
    ``block_ids[req][pos // tpb] * tpb + pos % tpb`` -- the same addressing as
    :func:`minimax_m3_build_req_to_token`, used to scatter new K/V/index-K.
    """
    slots: List[int] = []
    for pos, req in zip(token_positions, req_ids_per_token):
        block_ids = block_ids_per_seq[req]
        slots.append(
            int(block_ids[pos // tokens_per_block]) * tokens_per_block + pos % tokens_per_block
        )
    return torch.as_tensor(slots, dtype=torch.int64, device=device)


class MiniMaxM3CacheManager(KVCacheManagerV2):
    """``KVCacheManagerV2`` + a K-only index side pool for MiniMax-M3 MSA.

    The base manager owns the main K/V pool (2-factor ``SELF``) for every layer.
    Each layer additionally gets a Python-owned bf16 K-only index side pool
    (``[num_slots, 1, index_head_dim]``) sized to the same block count and keyed
    by the same V2 block ids, so index-K shares its token's main-cache slot.
    """

    def __init__(
        self,
        *args,
        index_head_dim: int = 128,
        index_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index_head_dim = index_head_dim
        self.index_dtype = index_dtype

        # num_blocks == the primary pool's block count (get_buffers dim 0). Valid
        # after construction: the pool is allocated in the base __init__.
        any_layer = next(iter(self.layer_offsets))
        num_blocks = int(self.get_buffers(any_layer).shape[0])
        self.index_num_blocks = num_blocks
        self.index_num_slots = num_blocks * self.tokens_per_block

        device = torch.device("cuda", torch.cuda.current_device())
        # One flat K-only side pool per local layer. Flat (num_slots, 1, dim) so
        # the kernels' [max_slots, 1, index_dim] contract is a zero-copy alias:
        # writes persist and reads see live memory.
        self._index_k_pool: Dict[int, torch.Tensor] = {
            layer_idx: torch.zeros(
                self.index_num_slots, 1, index_head_dim, dtype=index_dtype, device=device
            )
            for layer_idx in self.layer_offsets
        }

    def get_index_k_buffers(self, layer_idx: int) -> torch.Tensor:
        """The layer's K-only index side pool, ``[num_slots, 1, index_head_dim]``."""
        return self._index_k_pool[layer_idx]

    def get_main_kv_slot_buffers(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main K and V as flat ``[num_slots, num_kv_heads, head_dim]`` read views.

        ``get_buffers`` returns the native ``[num_blocks, kv_factor, tpb, H, D]``
        pool.  The K and V halves are non-contiguous strided slices, so
        ``reshape`` returns contiguous copies reflecting the current cache
        contents -- correct for the read-only kernel path (bring-up is
        parity-first; a native-layout kernel that avoids the copy is a
        post-bring-up perf change).
        """
        buf = self.get_buffers(layer_idx)  # [num_blocks, kv_factor, tpb, H, D]
        num_heads, head_dim = buf.shape[3], buf.shape[4]
        k = buf[:, 0].reshape(-1, num_heads, head_dim)
        v = buf[:, 1].reshape(-1, num_heads, head_dim)
        return k, v

    def write_main_kv(
        self,
        layer_idx: int,
        slots: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatter new main K/V into the aliased pool at flat ``slots``.

        ``slots`` ``[num_tokens]`` (from :func:`flatten_slot_ids`); ``k``/``v``
        ``[num_tokens, num_kv_heads, head_dim]``.
        """
        buf = self.get_buffers(layer_idx)  # native aliased pool
        tpb = self.tokens_per_block
        block = slots // tpb
        pos = slots % tpb
        # buf[:, 0] / buf[:, 1] are aliased views; advanced-indexing the view
        # writes through to the pool (avoids the mixed basic/advanced ambiguity
        # of buf[block, 0, pos]).
        buf[:, 0][block, pos] = k.to(buf.dtype)
        buf[:, 1][block, pos] = v.to(buf.dtype)

    def write_index_k(self, layer_idx: int, slots: torch.Tensor, index_k: torch.Tensor) -> None:
        """Scatter new index-K into the side pool at flat ``slots``.

        ``index_k`` ``[num_tokens, index_head_dim]`` (single replicated head).
        """
        pool = self._index_k_pool[layer_idx]
        pool[slots, 0] = index_k.to(pool.dtype)

    def shutdown(self):
        # Drop the Python side pools before the base frees the C++ pools.
        self._index_k_pool = {}
        super().shutdown()


def minimax_m3_paged_prefill(
    cache_manager: MiniMaxM3CacheManager,
    layer_idx: int,
    q: torch.Tensor,
    index_q: torch.Tensor,
    req_to_token: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    slot_ids: torch.Tensor,
    params: MiniMaxM3Params,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    max_num_blocks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MiniMax-M3 MSA prefill over the layer's runtime paged buffers."""
    k_cache, v_cache = cache_manager.get_main_kv_slot_buffers(layer_idx)
    index_k_cache = cache_manager.get_index_k_buffers(layer_idx)
    return minimax_m3_sparse_attention_prefill(
        q,
        k_cache,
        v_cache,
        index_q,
        index_k_cache,
        req_to_token,
        cu_seqlens,
        prefix_lens,
        slot_ids,
        block_size=params.block_size,
        topk=params.topk_blocks,
        init_blocks=params.init_blocks,
        local_blocks=params.local_blocks,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
        max_num_blocks=max_num_blocks,
    )


def minimax_m3_paged_decode(
    cache_manager: MiniMaxM3CacheManager,
    layer_idx: int,
    q: torch.Tensor,
    index_q: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_ids: torch.Tensor,
    params: MiniMaxM3Params,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    max_num_blocks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MiniMax-M3 MSA decode over the layer's runtime paged buffers."""
    k_cache, v_cache = cache_manager.get_main_kv_slot_buffers(layer_idx)
    index_k_cache = cache_manager.get_index_k_buffers(layer_idx)
    return minimax_m3_sparse_attention_decode(
        q,
        k_cache,
        v_cache,
        index_q,
        index_k_cache,
        req_to_token,
        seq_lens,
        slot_ids,
        block_size=params.block_size,
        topk=params.topk_blocks,
        init_blocks=params.init_blocks,
        local_blocks=params.local_blocks,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
        max_num_blocks=max_num_blocks,
    )

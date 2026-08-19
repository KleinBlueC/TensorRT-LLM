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
"""Inkling's KV cache manager: paged KV plus the short-conv state pool.

The short-conv state is **registered with V2 as SSM layers** rather than
allocated on the side. ``SsmLayerConfig`` is the framework's container for
per-request fixed-size state (``MambaHybridCacheManagerV2`` carries Mamba's
through it), and a short-conv window is ``[channels, kernel-1]`` per request:
it does not grow with the sequence, so it cannot be an attention buffer, where
``BufferConfig.size`` means bytes *per block*. V2 also sizes SSM layers with a
dedicated rule -- one dedicated block per request, never shared
(``_storage_manager._compute_slots_for_batch``) -- which is exactly the
semantics, and which the attention branch's capacity-driven, cross-request-shared
accounting would get wrong by two orders of magnitude.

What that buys: the pool's bytes enter V2's quota (they used to be a side
``torch`` allocation, counted only because the throwaway manager built during
size estimation also held one -- correct only while the two pools matched, which
nothing enforced); stable device addresses from the pool allocator; and
automatic coalescing of equal-sized buffers, so the k/v pair and the attn/mlp
pair do not each need their own allocation.

What it does not buy: slot allocation. V2 owns the memory, not the
request-to-slot mapping, so :class:`InklingConvStateCache` still assigns rows --
as Mamba's manager also does.

The SSM layers are **appended** after the attention layers, not interleaved.
DeepSeek-V4 must interleave because there one model layer's KV is split across
several cache layers; Inkling's is not, so appending leaves every attention
``layer_id`` in place and ``get_buffers`` / ``get_batch_cache_indices`` / the
metadata's row mapping all stay untouched.

There is deliberately no shared conv-state protocol: ``BaseMambaCacheManager``
mandates SSM state Inkling cannot back, and its one-tensor-per-layer accessor
cannot express four convs at two widths. Widen that hook if a second short-conv
model appears.
"""

import hashlib
from dataclasses import replace
from typing import List

import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BatchDesc,
    BufferConfig,
    DataRole,
    KVCacheDesc,
    LayerId,
    PageIndexMode,
    SsmLayerConfig,
)

from .....logger import logger
from ....pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, ReusableStateSnapshotMixin
from .conv_state import CONV_ROLES, InklingConvState, InklingConvStateCache


def _resolve_conv_dtype(pretrained_config) -> torch.dtype:
    """The compute dtype the short-conv pool holds.

    Not the manager's ``dtype`` argument: that is the KV cache dtype, a C++
    binding type ``torch.zeros`` rejects, and it is nvfp4/fp8 on quantized
    releases while this pool holds unquantized pre-conv activations.

    ``torch_dtype`` may be a ``torch.dtype`` or its name, so both are accepted.
    An unresolvable value raises rather than defaulting: silently falling back
    to bfloat16 turned an fp16 checkpoint into a wrong-dtype pool, surfacing far
    from its cause.
    """
    config = getattr(pretrained_config, "text_config", pretrained_config)
    dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if isinstance(resolved, torch.dtype):
            return resolved
    raise ValueError(
        f"Inkling short-conv pool needs the model's compute dtype, but "
        f"torch_dtype={dtype!r} on {type(config).__name__} is not a torch dtype"
    )


class _InklingConvGeometry:
    """Everything ``_build_cache_config`` needs to size the conv state.

    Split out because it has to be computed *before* ``super().__init__()`` --
    ``_build_cache_config`` runs inside it -- and every input is already an
    argument of the manager's constructor.
    """

    def __init__(self, pretrained_config, mapping, max_batch_size, *, spec_config=None):
        config = getattr(pretrained_config, "text_config", pretrained_config)
        self.config = config
        self.dtype = _resolve_conv_dtype(pretrained_config)
        self.kwin = config.sconv_kernel_size - 1
        # k/v width follows the attention kv-head split, so this takes the
        # attention TP, not the global one -- as V2 does for the paged pool.
        self.tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        # One row per resident sequence; pipeline stages each hold a
        # microbatch, hence * pp_size (Mamba's _max_resident_sequences).
        self.num_request_slots = max_batch_size * mapping.pp_size
        self.reserve_attention_dp_slot = bool(mapping.enable_attention_dp)
        self.max_draft_len = int(getattr(spec_config, "max_draft_len", 0) or 0)
        # Asked of the pool rather than re-derived: the slot layout is
        # InklingConvStateCache's to define, and the two counts must agree
        # exactly or the V2 buffer is indexed out of bounds by slots_for.
        self.num_reserved_slots = InklingConvStateCache.reserved_slot_count(
            reserve_attention_dp_slot=self.reserve_attention_dp_slot
        )

    def channels(self, global_layer_idx: int) -> List[int]:
        """Per-conv channel counts for one layer, in ``InklingConvState`` order."""
        kv_dim = (
            self.config.layer_num_kv_heads(global_layer_idx)
            * self.config.layer_head_dim(global_layer_idx)
        ) // self.tp_size
        hidden = self.config.hidden_size
        return [kv_dim, kv_dim, hidden, hidden]

    def bytes_per_slot(self, global_layer_idx: int) -> List[int]:
        """Bytes one request occupies in each of the layer's four conv states."""
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        return [c * self.kwin * itemsize for c in self.channels(global_layer_idx)]


class InklingHybridCacheManager(ReusableStateSnapshotMixin, KVCacheManagerV2):
    """Paged KV (V2, per-layer geometry) + the short-conv state pool.

    Folding the pool into the manager -- ``CppMambaHybridCacheManager``'s shape
    -- lets it reach the model through ``attn_metadata.kv_cache_manager`` and be
    released by the manager's own ``free_resources``, so conv rows and KV blocks
    cannot drift apart. Its bytes are declared to V2 in
    :meth:`_build_cache_config`; see the module docstring.
    """

    def __init__(self, *args, pretrained_config, mapping, max_batch_size, **kwargs):
        # Declared rather than dug out of **kwargs: a missing one then fails as
        # a TypeError naming the parameter, not a KeyError from in here.
        # Geometry is computed BEFORE super().__init__() because
        # _build_cache_config runs inside it and must size the SSM buffers.
        self._conv_geometry = _InklingConvGeometry(
            pretrained_config,
            mapping,
            max_batch_size,
            spec_config=kwargs.get("spec_config"),
        )
        # kv_cache_config is the base's first positional parameter. Kept
        # because prepare_expect_snapshot_points needs the snapshot interval and
        # the base keeps no reference of its own. ``.get``, not subscripting:
        # the latter is the bare-KeyError failure the comment above describes.
        self._kv_cache_config = args[0] if args else kwargs.get("kv_cache_config")
        # One warning per manager, not per request: the condition is a property
        # of the model, so a multimodal workload would emit it thousands of times.
        self._warned_multimodal_reuse = False
        super().__init__(
            *args,
            pretrained_config=pretrained_config,
            mapping=mapping,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        geo = self._conv_geometry
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            geo.tp_size,
            geo.num_request_slots,
            torch.device("cuda", torch.cuda.current_device()),
            geo.dtype,
            reserve_attention_dp_slot=geo.reserve_attention_dp_slot,
            max_draft_len=geo.max_draft_len,
            allocate=self._conv_state_buffer,
        )
        logger.info(
            f"Inkling short-conv state pool: {self._conv_cache.num_slots} rows "
            f"({geo.num_request_slots} request + {geo.num_reserved_slots} reserved), "
            f"{self._conv_cache.conv_state_bytes() / (1 << 20):.1f} MiB, "
            "backed by V2 SSM layers"
        )

    # ---- reusable snapshots -----------------------------------------------
    def prepare_expect_snapshot_points(self, requests) -> None:
        """Where this batch's requests may snapshot their short-conv window.

        Found by ``py_executor`` through ``hasattr`` and consumed by the
        scheduler, which will not end a context chunk anywhere else. That is the
        half a model cannot do for itself: a snapshot can only be taken where an
        iteration *ends*, so without a say in where chunks end, capture depends
        on the operator having picked a ``max_num_tokens`` smaller than the
        shared prefix. Measured at 8192, a 700-token 5-shot prompt prefilled in
        one chunk and 100 requests produced two snapshots.

        The interval is ``mamba_state_config.periodic_snapshot_interval``. The
        field is named for Mamba but means "tokens between recurrent-state
        snapshots", and a short-conv window is that kind of state, so this
        reuses it rather than adding a second knob for the same quantity.

        Coarse on purpose: a snapshot costs the whole model's conv window, so
        one per block over an 8k prompt would be hundreds of MiB for a single
        request. The cost is that reuse only lands on multiples of the interval.
        """
        state_config = getattr(self._kv_cache_config, "mamba_state_config", None)
        interval = getattr(state_config, "periodic_snapshot_interval", 0) or 0
        for request in requests:
            if not self.enable_block_reuse or not interval:
                request.expect_snapshot_points = []
                continue
            request.expect_snapshot_points = list(range(interval, request.prompt_len + 1, interval))

    def _augment_tokens_for_block_reuse(self, tokens, req, start=0, end=None):
        """Keep multimodal requests out of the shared radix tree.

        The base method rewrites each multimodal token span into cache-key
        tokens carrying the item's content digest, precisely because a
        placeholder token id is the same whatever image is behind it. It can
        only do that when the request carries ``multimodal_hashes``, and for a
        request that has none it returns the raw ids -- so two prompts holding
        DIFFERENT images look like the same prefix and one is served the other's
        keys and values.

        Inkling's input processor does not produce those hashes (the docs record
        multimodal-hash prefix caching as unsupported), so that is the case
        here, not a corner of it. Measured on MMMU over the same 32 items,
        `Inkling-small-NVFP4`: 81.3% with reuse off, 31.3% with reuse on
        (jobs 6310792 / 6310791). Nothing errors; the answers are simply wrong.

        Salting position 0 with the request id gives such a request a private
        chain in the tree: it matches nothing and nothing matches it, so reuse
        turns itself off exactly where it cannot be correct while text requests
        keep it. A blanket refusal for the model would be wrong -- Inkling is
        always multimodal-capable, and the text path is measured good.
        """
        augmented = super()._augment_tokens_for_block_reuse(tokens, req, start, end)
        if not self.enable_block_reuse:
            return augmented
        if getattr(req, "multimodal_hashes", None) is not None:
            return augmented
        if getattr(req, "py_multimodal_data", None) is None:
            return augmented
        # Only the chunk containing position 0 needs poisoning: the radix tree
        # chains a block's key through its parent, so a private first block
        # makes every continuation of this request private too.
        if start != 0 or not len(augmented):
            return augmented
        if not self._warned_multimodal_reuse:
            self._warned_multimodal_reuse = True
            logger.warning(
                "Inkling: KV cache block reuse is disabled for multimodal "
                "requests. Their image/video/audio spans carry no content "
                "digest, so a shared prefix would be matched on placeholder "
                "token ids alone and serve one item's KV for another's. Text "
                "requests are unaffected."
            )
        salt = hashlib.sha256(
            b"inkling-multimodal-no-digest-%d" % int(getattr(req, "request_id", 0))
        ).digest()
        poisoned = list(augmented)
        poisoned[0] = salt
        return poisoned

    def prepare_context(self, req):
        """Observation only: count how much prefix each request actually reused.

        ``get_kv_cache_stats`` cannot serve here -- measured, it reports
        reused/missed/alloc all zero for this configuration, so it reads the
        same whether reuse works or never runs. Only the FIRST context chunk is
        counted: a continuation chunk also arrives with
        ``context_current_position > 0`` (its own earlier chunks), and counting
        those made a reuse-disabled arm report 221 hits.
        """
        first = bool(getattr(req, "is_first_context_chunk", True))
        ok = super().prepare_context(req)
        if not first:
            return ok
        d = getattr(self, "_reuse_dbg", None)
        if d is None:
            d = self._reuse_dbg = {"n": 0, "hits": 0, "best": 0, "total": 0}
        d["n"] += 1
        pos = int(getattr(req, "context_current_position", 0) or 0)
        if pos > 0:
            d["hits"] += 1
            d["total"] += pos
            d["best"] = max(d["best"], pos)
        if d["n"] % 64 == 0:
            logger.info(
                f"Inkling prefix reuse: {d['hits']}/{d['n']} requests, "
                f"longest={d['best']} tokens, total={d['total']}"
            )
        return ok

    # ---- V2 configuration -------------------------------------------------
    def _conv_layer_id(self, local_layer_idx: int) -> LayerId:
        """Cache-layer id holding ``local_layer_idx``'s four conv states.

        Appended after the attention layers, so this is plain arithmetic with no
        mapping table -- and every attention id stays at ``0..N-1``, which is
        what keeps the paged-KV addressing untouched.
        """
        return LayerId(self.num_local_layers + local_layer_idx)

    def _build_cache_config(self, config):
        """Append one SSM layer per decoder layer for the short-conv state.

        The attention layers ``_build_base_config`` produced are preserved
        exactly -- this only extends the list, following
        ``MambaHybridCacheManagerV2``'s "preserve what the base built, then
        declare our own" shape (it replaces its Mamba layers; Inkling's layers
        need *both* KV and conv, and a ``LayerConfig`` is one or the other, so
        appending is the only way to have both).
        """
        geo = self._conv_geometry
        layers = list(config.layers)
        num_attention_layers = len(layers)
        # _conv_layer_id derives SSM ids arithmetically, so the base must have
        # emitted one attention layer per local decoder layer. A mismatch would
        # otherwise surface as a wrong pool address at first decode.
        assert num_attention_layers == self.num_local_layers, (
            num_attention_layers,
            self.num_local_layers,
        )
        for local_idx in range(num_attention_layers):
            layers.append(
                SsmLayerConfig(
                    layer_id=LayerId(num_attention_layers + local_idx),
                    buffers=[
                        BufferConfig(role=role, size=nbytes)
                        for role, nbytes in zip(
                            CONV_ROLES, geo.bytes_per_slot(self.pp_layers[local_idx])
                        )
                    ],
                )
            )

        # Non-request rows as zero-capacity requests, like Mamba: no attention
        # pages, one state slot each -- exactly their effect on the pool.
        dummies = [KVCacheDesc(capacity=0, history_length=0) for _ in range(geo.num_reserved_slots)]
        constraints = [
            replace(batch, kv_caches=[*batch.kv_caches, *dummies]) for batch in config.constraints
        ]
        # Length-independent floor. Attention pages scale with tokens so the
        # base's constraints bound them, but a conv row is fixed per resident
        # sequence; without this the pool can be sized below what the scheduler
        # admits, failing mid-run rather than at startup.
        constraints.append(
            BatchDesc(
                [
                    KVCacheDesc(capacity=0, history_length=0)
                    for _ in range(geo.num_request_slots + geo.num_reserved_slots)
                ]
            )
        )
        # KVCacheManagerConfig hard-asserts this whenever an SSM layer exists.
        # Harmless here -- Inkling refuses block reuse, so nothing ever commits
        # -- but the invariant is still required (as Mamba's manager does).
        return replace(config, layers=layers, constraints=constraints, commit_min_snapshot=True)

    def _get_pool_roles(self, pool_id: int):
        """Name a role that actually exists in ``pool_id``.

        The base returns ``Role.KEY`` unconditionally, but conv pools hold no
        KEY, so ``_build_pool_mapping_tensors`` would ``KeyError`` on them.
        ``MambaHybridCacheManagerV2`` overrides this for the same reason.

        Unlike Mamba's it *resolves* the role rather than naming one: the four
        states come in two widths and V2 coalesces by size, so which role
        represents a pool is its packing decision, not ours.
        """
        first_layer = int(self.impl.layer_grouping[pool_id][0])
        if first_layer < self.num_local_layers:
            return super()._get_pool_roles(pool_id)
        for role in CONV_ROLES:
            if (pool_id, role) in self._pool_layer_ids_by_role:
                return role, None
        raise RuntimeError(
            f"Inkling conv pool {pool_id} (first layer {first_layer}) holds none "
            f"of {[str(r) for r in CONV_ROLES]}; _build_cache_config and the pool "
            "packing disagree about what landed where."
        )

    def _conv_state_buffer(
        self, local_layer_idx: int, role: DataRole, state_shape: List[int]
    ) -> torch.Tensor:
        """A ``[num_slots, *state_shape]`` view of one conv state's pool memory.

        Mirrors ``MambaHybridCacheManagerV2._get_state_buffer``. The
        ``as_strided`` step is not cosmetic: V2 coalesces same-sized per-layer
        buffers inside one pool slot, so the raw page-indexed view has
        ``page_index_scale`` pages per logical slot. Callers index by slot.
        """
        layer_id = self._conv_layer_id(local_layer_idx)
        addr = self.impl.get_mem_pool_base_address(layer_id, role, PageIndexMode.SHARED)
        num_pages = self.impl.get_page_index_upper_bound(layer_id, role)
        raw = convert_to_torch_tensor(
            TensorWrapper(addr, self._conv_geometry.dtype, [num_pages] + list(state_shape))
        )
        scale = self.impl.get_page_index_scale(layer_id, role)
        num_slots = (num_pages + scale - 1) // scale
        return raw.as_strided(
            [num_slots] + list(state_shape),
            [raw.stride(0) * scale] + list(raw.stride()[1:]),
        )

    # ---- model-facing -----------------------------------------------------
    @property
    def conv_state_cache(self) -> InklingConvStateCache:
        """The short-conv state pool, for the metadata's per-step publication."""
        return self._conv_cache

    def get_conv_states(self, layer_idx: int) -> InklingConvState:
        """The four short-conv state buffers of ``layer_idx``.

        Named after ``BaseMambaCacheManager.get_conv_states`` deliberately, but
        it cannot *implement* that hook: one tensor per layer cannot express four
        convs at two widths. Widen the shared hook if a second model needs it.
        """
        return self._conv_cache.layer_state(layer_idx)

    def get_state_indices(self) -> torch.Tensor:
        """Pool rows of the current batch, in packed batch order."""
        return self._conv_cache.state_indices

    def free_conv_state(self, request_ids) -> None:
        self._conv_cache.free(list(request_ids))

    # ---- KVCacheManagerV2 -----------------------------------------------------
    def free_resources(self, request, *args, **kwargs):
        """Release the conv row with the request's KV blocks.

        Lets the engine's dummy-batch cleanup stay generic -- it already calls
        ``free_resources`` per request -- and a leaked row would later be reused
        with stale state by a real request whose id collides with a dummy's.
        """
        rid = getattr(request, "py_request_id", None)
        if rid is not None:
            self.free_conv_state([rid])
        return super().free_resources(request, *args, **kwargs)

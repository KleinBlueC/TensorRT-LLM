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

Lives with the model's attention package rather than under ``pyexecutor``,
matching ``sparse/minimax_m3/cache_manager.py``.

There is deliberately no shared conv-state protocol. ``BaseMambaCacheManager``
is the closest existing one, but it mandates SSM state and replay metadata
Inkling cannot back, and its one-tensor-per-layer accessor cannot express
Inkling's four convs per layer at two different widths. If a second short-conv
model appears, widen that hook rather than adding another beside it.
"""

from dataclasses import replace

import torch

from ....logger import logger
from ...._utils import TensorWrapper, convert_to_torch_tensor
from ....runtime.kv_cache_manager_v2 import (BatchDesc, BufferConfig, DataRole,
                                             KVCacheDesc, LayerId,
                                             PageIndexMode, SsmLayerConfig)
from ....runtime.kv_cache_manager_v2 import \
    KVCacheManagerConfig as KVCacheManagerConfigPy
from ...pyexecutor.kv_cache_manager_v2 import (KVCacheManagerV2,
                                              ReusableStateSnapshotMixin)
from .conv_state import InklingConvState, InklingConvStateCache


class InklingRole:
    """V2 buffer roles owned only by this manager.

    Four of them, not one, because Inkling runs four short convs per layer at
    two different widths: k/v follow the attention kv-head split and are
    TP-sharded, attn/mlp are hidden-width. V2 groups sub-pages by size on its
    own, so the two widths need no handling here -- but they do need distinct
    roles, because a role is the key a buffer is looked up by and
    ``SsmLayerConfig`` rejects duplicates within one layer.

    Roles are opaque string keys; new ones require no C++ change.
    """

    CONV_K = DataRole("inkling_conv_k")
    CONV_V = DataRole("inkling_conv_v")
    CONV_ATTN = DataRole("inkling_conv_attn")
    CONV_MLP = DataRole("inkling_conv_mlp")


def _resolve_conv_dtype(pretrained_config) -> torch.dtype:
    """The compute dtype the short-conv pool holds.

    Not the manager's ``dtype`` argument -- that is the KV cache dtype, a C++
    binding type ``torch.zeros`` rejects, and it is ``nvfp4``/``fp8`` on
    quantized releases while the conv pool holds unquantized pre-conv
    activations.

    HuggingFace configs carry ``torch_dtype`` as either a ``torch.dtype`` or its
    name (``"bfloat16"``), so both are accepted. An unresolvable value raises:
    the previous silent fall back to bfloat16 turned an fp16 checkpoint into a
    pool of the wrong dtype, which reaches the conv kernels as a dtype mismatch
    far from its cause.
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
    """Per-layer short-conv widths, and the byte size each one costs a slot.

    Split out of the pool because ``_build_cache_config`` runs inside
    ``super().__init__`` -- before the pool can exist -- and needs the same
    numbers the pool later allocates against. Deriving them twice is how the two
    silently disagree.
    """

    def __init__(self, pretrained_config, tp_size: int, dtype: torch.dtype):
        config = getattr(pretrained_config, "text_config", pretrained_config)
        self.dtype = dtype
        self.kwin = config.sconv_kernel_size - 1
        self.hidden = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.kv_dims = [
            (config.layer_num_kv_heads(i) * config.layer_head_dim(i)) // tp_size
            for i in range(self.num_layers)
        ]

    def _bytes(self, channels: int) -> int:
        return channels * self.kwin * self.dtype.itemsize

    def buffer_configs(self, layer_idx: int) -> list:
        """The four ``BufferConfig`` entries one layer's SSM cache layer holds.

        ``BufferConfig.size`` is bytes per slot here rather than bytes per
        block: an SSM layer is not paged, which is exactly why the conv state
        belongs in one instead of in ``_extra_buffers_per_layer``.
        """
        kv_bytes = self._bytes(self.kv_dims[layer_idx])
        hidden_bytes = self._bytes(self.hidden)
        return [
            BufferConfig(role=InklingRole.CONV_K, size=kv_bytes),
            BufferConfig(role=InklingRole.CONV_V, size=kv_bytes),
            BufferConfig(role=InklingRole.CONV_ATTN, size=hidden_bytes),
            BufferConfig(role=InklingRole.CONV_MLP, size=hidden_bytes),
        ]

    def state_shape(self, role, layer_idx: int) -> list:
        channels = (
            self.kv_dims[layer_idx]
            if role in (InklingRole.CONV_K, InklingRole.CONV_V)
            else self.hidden
        )
        return [channels, self.kwin]


class InklingHybridCacheManager(ReusableStateSnapshotMixin, KVCacheManagerV2):
    """Paged KV (V2, per-layer geometry) + the short-conv state pool.

    Folding the pool into the cache manager -- the shape
    ``CppMambaHybridCacheManager`` uses for mamba conv/SSM state -- lets it reach
    the model through the standard ``attn_metadata.kv_cache_manager`` field and
    be released by the manager's own ``free_resources``. The conv rows are then
    freed by the same call that frees the request's KV blocks, so the two views
    cannot drift apart.

    The conv state is declared to V2 as SSM cache layers (see
    :meth:`_build_cache_config`), so its bytes are inside V2's own quota. They
    used to be a plain torch allocation that no quota knew about, counted
    against the KV budget only because the throwaway estimation manager held one
    while ``configure_kv_cache_capacity`` read peak memory -- an identity that
    held exactly while the estimation pool and the serving pool were the same
    fixed size, enforced by nothing but a comment.
    """

    def __init__(self, *args, pretrained_config, mapping, max_batch_size, **kwargs):
        # The three arguments the pool needs are declared, not read back out of
        # ``**kwargs``. KVCacheManagerV2 takes ``mapping`` / ``max_batch_size``
        # keyword-only and absorbs ``pretrained_config`` into ``**kwargs``
        # without storing it, so subscripting kwargs worked only as long as
        # every caller passed all three by keyword: omitting one surfaced as a
        # bare KeyError from inside this constructor rather than as a TypeError
        # naming the parameter.
        # The conv pool's k/v width follows the attention kv-head split, so it
        # takes the attention TP, not the global one -- the same rule
        # KVCacheManagerV2 applies to the paged pool. Dividing by the global
        # tp_size would allocate narrow conv rows for full-width convs.
        attn_tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        # One row per sequence that can be resident at once. Pipeline stages
        # each hold a microbatch, so the bound is max_batch_size * pp_size --
        # the same count MambaHybridCacheManagerV2 calls
        # ``_max_resident_sequences``. The padding and attention-DP rows are
        # reserved on top of this by the pool itself.
        num_request_slots = max_batch_size * mapping.pp_size
        spec_config = kwargs.get("spec_config")
        max_draft_len = int(getattr(spec_config, "max_draft_len", 0) or 0)
        # Everything _build_cache_config needs has to exist BEFORE super(),
        # because super().__init__ is what calls it. Same ordering as
        # MambaHybridCacheManagerV2, which computes ssm_bytes/conv_bytes at
        # :2976 and only then calls super at :3015.
        self._conv_geometry = _InklingConvGeometry(
            pretrained_config, attn_tp_size, _resolve_conv_dtype(pretrained_config)
        )
        self._conv_num_slots = num_request_slots + 1 + int(mapping.enable_attention_dp)
        # kv_cache_config is the base's first positional parameter. Kept because
        # prepare_expect_snapshot_points needs the snapshot interval and the
        # base stores no reference of its own.
        #
        # ``.get``, not ``[...]``: subscripting is precisely the bare-KeyError
        # failure the comment above this one exists to describe. The base
        # declares this parameter as required, so a real manager always has it;
        # absent means a caller that stubbed the base out, and the hook below
        # reads it defensively rather than making construction fail here.
        self._kv_cache_config = args[0] if args else kwargs.get("kv_cache_config")
        super().__init__(
            *args,
            pretrained_config=pretrained_config,
            mapping=mapping,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            attn_tp_size,
            num_request_slots,
            torch.device("cuda", torch.cuda.current_device()),
            _resolve_conv_dtype(pretrained_config),
            reserve_attention_dp_slot=mapping.enable_attention_dp,
            max_draft_len=max_draft_len,
            layer_states=self._conv_states_from_v2(),
        )
        logger.info(
            f"Inkling short-conv state pool: {self._conv_cache.num_slots} rows "
            f"({num_request_slots} request + reserved), "
            f"{self._conv_cache.conv_state_bytes() / (1 << 20):.1f} MiB"
        )

    def prepare_context(self, req):
        """Observation only: count how much prefix each request actually reused.

        ``get_kv_cache_stats`` is not usable here -- measured, it returns
        reused/missed/alloc all zero for this configuration, so it reports the
        same thing whether reuse works or never runs. ``context_current_position``
        after the base call is the reused prefix length, which is the signal the
        earlier bespoke implementation used and the only one confirmed to move.
        """
        first = bool(getattr(req, "is_first_context_chunk", True))
        ok = super().prepare_context(req)
        pos = int(getattr(req, "context_current_position", 0) or 0)
        d = getattr(self, "_hit_dbg", None)
        if d is None:
            d = self._hit_dbg = {"n": 0, "chunks": 0, "hits": 0, "best": 0,
                                 "total": 0}
        d["chunks"] += 1
        # Only the FIRST context chunk can carry a reused prefix. A
        # continuation chunk also arrives with context_current_position > 0 --
        # its own earlier chunks -- and counting those made a reuse-OFF arm
        # report 221 hits, i.e. the counter measured chunked prefill, not reuse.
        if not first:
            return ok
        d["n"] += 1
        if pos > 0:
            d["hits"] += 1
            d["total"] += pos
            d["best"] = max(d["best"], pos)
        if d["n"] % 64 == 0:
            logger.info(
                f"Inkling prefix reuse: {d['hits']}/{d['n']} requests, "
                f"longest={d['best']} tokens, total={d['total']}")
        return ok

    # ---- reusable snapshots ------------------------------------------------
    def prepare_expect_snapshot_points(self, requests) -> None:
        """Where this batch's requests must snapshot their short-conv window.

        Picked up by ``py_executor`` through ``hasattr``, and consumed by the
        scheduler, which will not end a context chunk anywhere else. That is the
        half the bespoke implementation was missing: a snapshot can only be
        taken where an iteration *ends*, so without a say in where chunks end,
        capture depends on the operator having chosen a ``max_num_tokens``
        smaller than the shared prefix -- which nothing states and nothing
        checks. Measured: at 8192 a 700-token 5-shot prompt prefilled in one
        chunk and 100 requests produced two snapshots.

        The interval is ``mamba_state_config.periodic_snapshot_interval``. The
        field is named for Mamba but means "tokens between recurrent-state
        snapshots", and Inkling's short-conv window is that kind of state, so
        this reuses it rather than adding a second user-facing knob for the
        same quantity.

        Interval, not every block, because a snapshot costs the whole model's
        conv window (~2 MiB on the small checkpoint). One per 32-token block
        over an 8k prompt would be ~560 MiB for a single request. The cost of
        the coarser grid is that reuse only lands on multiples of the interval.
        """
        state_config = getattr(self._kv_cache_config, "mamba_state_config", None)
        interval = getattr(state_config, "periodic_snapshot_interval", 0) or 0
        for request in requests:
            if not self.enable_block_reuse or not interval:
                request.expect_snapshot_points = []
                continue
            request.expect_snapshot_points = list(
                range(interval, request.prompt_len + 1, interval)
            )
            if not getattr(self, "_logged_first_points", False):
                self._logged_first_points = True
                logger.info(
                    f"Inkling snapshot points: interval={interval} "
                    f"prompt_len={request.prompt_len} "
                    f"points={request.expect_snapshot_points}")

    # ---- V2 cache layout ---------------------------------------------------
    def _build_cache_config(
        self, config: KVCacheManagerConfigPy
    ) -> KVCacheManagerConfigPy:
        """Register the short-conv state as V2 SSM cache layers.

        **Appended, not substituted.** Mamba replaces its mamba layers
        (``layers[i] = SsmLayerConfig(...)``) because a Mamba layer is *either*
        attention or recurrent. Every Inkling layer is *both* -- paged K/V and
        four short convs -- and ``LayerConfig`` is a union, so one ``layer_id``
        cannot be both. Appending gives each model layer a second cache layer:

            conv_layer_id = num_attention_layers + local_layer_idx

        which is a one-line map rather than the forward/reverse tables
        DeepSeek-V4 needs, and leaves every K/V ``layer_id`` untouched.

        This is what moves the conv bytes inside V2's byte quota. Before it,
        the pool was a plain ``torch.zeros`` that no quota knew about, and it
        was counted only because the throwaway estimation manager happened to
        hold one while ``configure_kv_cache_capacity`` read peak memory -- an
        accounting identity that held only while the estimation pool and the
        serving pool were the same fixed size, enforced by nothing but a
        comment.
        """
        layers = list(config.layers)
        self._conv_layer_id_base = len(layers)
        for local_layer_idx, global_layer_idx in enumerate(self.pp_layers):
            layers.append(
                SsmLayerConfig(
                    layer_id=LayerId(self._conv_layer_id_base + local_layer_idx),
                    buffers=self._conv_geometry.buffer_configs(global_layer_idx),
                )
            )
        # An SSM slot is fixed-size per sequence, so its floor is independent of
        # sequence length and the base config's length-derived constraints
        # cannot express it. Zero-capacity requests cost no attention page but
        # reserve one conv slot each, which is exactly the shape of this bound.
        constraints = [
            *config.constraints,
            BatchDesc(
                [
                    KVCacheDesc(capacity=0, history_length=0)
                    for _ in range(self._conv_num_slots)
                ]
            ),
        ]
        logger.info(
            f"Inkling short-conv registered as V2 SSM cache layers "
            f"{self._conv_layer_id_base}..{len(layers) - 1} "
            f"({len(self.pp_layers)} layers x 4 convs); "
            f"attention layers 0..{self._conv_layer_id_base - 1} unchanged"
        )
        return replace(
            config,
            layers=layers,
            constraints=constraints,
            # Required, not chosen: _config.py asserts it whenever any SSM layer
            # is present. Harmless with reuse off -- no commits are attempted --
            # and the same line Mamba carries for the same reason.
            commit_min_snapshot=True,
        )

    def _get_pool_roles(self, pool_id):
        """Which roles the page-table index lanes carry for ``pool_id``.

        The base answers Role.KEY/Role.VALUE for every pool, because every pool
        it knows about is a paged attention pool. The conv pools hold no K/V --
        asking for them raises ``KeyError: (pool_id, 'key')`` out of
        ``_build_pool_mapping_tensors`` before the server can start.

        Reporting CONV_K with no second lane matches what Mamba does for its
        SSM pools: the page table is an attention structure, and these pools are
        addressed by slot, not through it.
        """
        layer_id = int(self.impl.layer_grouping[pool_id][0])
        if layer_id >= self._conv_layer_id_base:
            return InklingRole.CONV_K, None
        return super()._get_pool_roles(pool_id)

    def _conv_state_buffer(self, local_layer_idx: int, role, global_layer_idx: int):
        """One conv buffer as a ``[num_slots, channels, kwin]`` view of V2 memory.

        The ``as_strided`` is not cosmetic: V2 coalesces same-size per-layer
        buffers inside one slot, so the raw page-index view walks other layers'
        sub-pages. Striding by ``page_index_scale`` exposes only this layer's.
        Lifted from ``MambaHybridCacheManagerV2._get_state_buffer``.
        """
        layer_id = LayerId(self._conv_layer_id_base + local_layer_idx)
        addr = self.impl.get_mem_pool_base_address(
            layer_id, role, PageIndexMode.SHARED
        )
        num_pages = self.impl.get_page_index_upper_bound(layer_id, role)
        state_shape = self._conv_geometry.state_shape(role, global_layer_idx)
        raw = convert_to_torch_tensor(
            TensorWrapper(addr, self._conv_geometry.dtype, [num_pages] + state_shape)
        )
        scale = self.impl.get_page_index_scale(layer_id, role)
        num_slots = (num_pages + scale - 1) // scale
        return raw.as_strided(
            [num_slots] + state_shape,
            [raw.stride(0) * scale] + list(raw.stride()[1:]),
        )

    def _conv_states_from_v2(self) -> list:
        """The per-layer four-conv views, in local layer order."""
        return [
            InklingConvState(
                k=self._conv_state_buffer(local, InklingRole.CONV_K, glob),
                v=self._conv_state_buffer(local, InklingRole.CONV_V, glob),
                attn=self._conv_state_buffer(local, InklingRole.CONV_ATTN, glob),
                mlp=self._conv_state_buffer(local, InklingRole.CONV_MLP, glob),
            )
            for local, glob in enumerate(self.pp_layers)
        ]

    # ---- model-facing -----------------------------------------------------
    @property
    def conv_state_cache(self) -> InklingConvStateCache:
        """The short-conv state pool, for the metadata's per-step publication."""
        return self._conv_cache

    def get_conv_states(self, layer_idx: int) -> InklingConvState:
        """The four short-conv state buffers of ``layer_idx``.

        Named after ``BaseMambaCacheManager.get_conv_states`` on purpose: this
        is the same question asked of the same kind of manager. It cannot
        *implement* that hook, which returns one tensor per layer and cannot
        express Inkling's four convs at two widths -- widening the shared hook
        is the move if a second short-conv model appears.
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

        This is what lets the model engine's warmup/estimation dummy-batch
        cleanup drop its Inkling-specific branch: it already calls
        ``kv_cache_manager.free_resources(req)`` for every dummy request, and a
        leaked conv row would later be reused, with stale state, by a real
        request whose id collides with a dummy id.
        """
        rid = getattr(request, "py_request_id", None)
        if rid is not None:
            self.free_conv_state([rid])
        return super().free_resources(request, *args, **kwargs)

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
"""Per-request short-conv state for Inkling: the pool, the per-forward split,
and the helper that runs one short convolution through them.

This is runtime state, not model weights: it has the lifetime of the paged KV
cache, is owned by :class:`~.cache_manager.InklingHybridCacheManager`, and is
published each step from :meth:`~.metadata.InklingAttentionMetadata.prepare`.
It therefore lives beside the cache manager and the metadata rather than in
``models/modeling_inkling.py``, which owns only the weight-bearing modules.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

from ....._utils import prefer_pinned

# Per-request short-conv state of one decoder layer, carried across decode steps.
# Each field is a ``[num_req, channels, sconv_kernel_size - 1]`` window of the
# previous pre-conv inputs (oldest first): ``k``/``v`` for the attention k/v
# convs (TP-sharded), ``attn``/``mlp`` for the residual-stream convs (replicated).
InklingConvState = namedtuple("InklingConvState", ["k", "v", "attn", "mlp"])


class InklingRole:
    """V2 pool roles for the four short convolutions of a decoder layer.

    Four roles despite only two widths: a role is a buffer's identity, and
    k/v/attn/mlp merely happen to pair up in size today. V2 coalesces same-sized
    buffers on its own, so naming them apart is free.

    Lives here so the pool and its roles are declared together, and so this
    module and ``cache_manager`` do not import each other.
    """

    CONV_K = DataRole("inkling_conv_k")
    CONV_V = DataRole("inkling_conv_v")
    CONV_ATTN = DataRole("inkling_conv_attn")
    CONV_MLP = DataRole("inkling_conv_mlp")


# Order matches InklingConvState's fields so the two can be zipped.
CONV_ROLES = (InklingRole.CONV_K, InklingRole.CONV_V, InklingRole.CONV_ATTN, InklingRole.CONV_MLP)


class _ConvVerifyCapture:
    """What a speculative verify step must remember to undo itself.

    A short conv carries a window of its last ``kwin`` INPUTS, mutated in place.
    A verify step runs the conv over every drafted token, so it leaves the
    window advanced past the tokens the target went on to reject.

    The window after ``k`` tokens is a pure function of the window before the
    step and the inputs consumed, ``last kwin of (init ++ x[:k])``, so capturing
    those two is enough to reconstruct the accepted state for any ``k``. That is
    cheaper than saving one window per drafted position -- the ``x`` it is
    derived from is a factor of ``kwin`` smaller -- and needs no scatter kernel.

    These buffers are a private allocation, not V2 pool memory: they are scratch
    for one step rather than per-request state with the KV cache's lifetime, and
    V2 has no role for something that is overwritten every forward.
    """

    def __init__(self, max_batch, channels, kwin, steps, device, dtype):
        # The window as it stood before the verify step consumed anything.
        self.init = torch.zeros(max_batch, channels, kwin, device=device, dtype=dtype)
        # The verify step's pre-conv inputs, in token order per request.
        self.x = torch.zeros(max_batch, steps, channels, device=device, dtype=dtype)
        self.max_steps = steps
        # How many of those step slots the last save actually filled. The buffer
        # is sized for the TARGET's verify step (1 + max_draft_len), but the
        # draft chain's own generation steps are shorter, and both go through
        # here now that the chain has its own conv state.
        self.steps_used = steps

    def save(self, pool_buf, rows, x_gen, steps):
        """Record this verify step, before the conv mutates ``pool_buf``."""
        n = rows.shape[0]
        if steps > self.max_steps:
            raise ValueError(
                f"conv capture holds {self.max_steps} steps but {steps} were "
                "presented; it is sized from max_draft_len."
            )
        self.init[:n].copy_(pool_buf.index_select(0, rows))
        self.x[:n, :steps].copy_(x_gen.view(n, steps, -1))
        self.steps_used = steps

    def accepted_window(self, num_accepted, kwin):
        """The window each request should be left holding, given acceptances.

        ``num_accepted`` is per request and at least 1 -- the target's own token
        is never rejected -- so every request advances by something and none is
        rolled back to before the step.
        """
        n = num_accepted.shape[0]
        # Only the slots the last save filled: a shorter step than the buffer
        # holds would otherwise concatenate stale inputs after the real ones and
        # commit a window built from a previous batch.
        steps = self.steps_used
        num_accepted = num_accepted.clamp(max=steps)
        # [n, C, kwin + steps]: the window followed by the inputs it consumed.
        stream = torch.cat([self.init[:n], self.x[:n, :steps].transpose(1, 2)], dim=-1)
        offs = num_accepted.view(n, 1, 1) + torch.arange(
            kwin, device=stream.device, dtype=num_accepted.dtype
        ).view(1, 1, kwin)
        return stream.gather(2, offs.expand(n, stream.shape[1], kwin).to(torch.int64))


class InklingConvStateCache:
    """Runtime-owned per-request short-conv state pool for the whole decoder.

    Carries the four causal short-convs of every decoder layer per request
    across decode steps, with the same lifetime as the paged KV cache.

    Per layer it holds the four :class:`InklingConvState` buffers, each
    ``[num_slots, channels, kernel_size - 1]``. The k/v conv channels follow the
    fused-qkv k/v split (TP-sharded); the residual-stream convs are replicated.

    The buffers come from the ``allocate`` callback, which
    :class:`InklingHybridCacheManager` wires to V2 SSM-layer pool memory: this
    class owns the request-to-row mapping, V2 owns the bytes -- the same split
    ``MambaHybridCacheManagerV2`` uses.

    The row count is fixed at construction: ``num_request_slots`` real rows,
    plus one row shared by every CUDA-graph padding sentinel, plus one for the
    attention-DP idle dummy when enabled. This is the ``MambaCacheManager``
    layout (``_padding_slot`` / ``_attention_dp_dummy_slot``), fixed for two
    reasons. Padding rows must not consume real-request capacity -- the
    CUDA-graph runner keeps one sentinel per runtime draft length, so charging
    each a real row silently shrinks the servable batch. And every buffer,
    including the int32 ``state_indices``, keeps a stable device address and is
    mutated in place, so a captured graph replays cleanly; a pool that
    reallocated to grow would strand captured pointers, and replaying against a
    freed buffer does not raise -- it reads whatever the allocator handed out
    next.
    """

    @staticmethod
    def reserved_slot_count(*, reserve_attention_dp_slot: bool) -> int:
        """Rows sitting above the per-request ones.

        One row shared by every CUDA-graph padding sentinel, plus one for the
        attention-DP idle dummy when enabled.

        Static because the count is needed twice -- here to size the pool, and in
        ``_InklingConvGeometry`` to declare V2's min-slots floor. They must agree
        exactly (a buffer sized from a smaller count is indexed out of bounds by
        :meth:`slots_for`), and the layout is this class's to define, so the
        geometry asks rather than re-deriving.
        """
        return 1 + int(reserve_attention_dp_slot)

    def __init__(
        self,
        pretrained_config,
        tp_size: int,
        num_request_slots: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        reserve_attention_dp_slot: bool = False,
        max_draft_len: int = 0,
        num_layers: Optional[int] = None,
        layer_offset: int = 0,
        allocate: Optional[Callable[[int, object, List[int]], torch.Tensor]] = None,
    ):
        # Takes the pretrained config + tp_size rather than a ``ModelConfig`` so
        # the KV cache manager can build the pool from what it already has.
        # Accept either the text config or the top-level multimodal one.
        config = getattr(pretrained_config, "text_config", pretrained_config)
        kwin = config.sconv_kernel_size - 1
        if num_request_slots <= 0:
            raise ValueError(f"num_request_slots must be > 0, got {num_request_slots}")
        self.num_request_slots = num_request_slots
        # Reserved rows sit above the real ones so a real slot id is always a
        # valid index into the first ``num_request_slots`` rows.
        self._padding_slot = num_request_slots
        self._attention_dp_dummy_slot = num_request_slots + 1 if reserve_attention_dp_slot else None
        self._max_draft_len = max(0, int(max_draft_len))
        num_slots = num_request_slots + self.reserved_slot_count(
            reserve_attention_dp_slot=reserve_attention_dp_slot
        )
        self.num_slots = num_slots
        self.kwin = kwin

        # ``allocate`` hands back V2 pool memory when the cache manager built us,
        # which is what puts these bytes inside V2's quota. Standalone callers
        # (unit tests) fall back to a private allocation of the same shapes.
        #
        # Pool memory arrives UNINITIALIZED. Safe only because ``slots_for``
        # zeroes a row when it first hands it out, and every row that is read was
        # handed out first. A reader that skips claiming needs a zero-fill.
        if allocate is None:

            def allocate(_layer_idx, _role, state_shape):
                return torch.zeros(num_slots, *state_shape, device=device, dtype=dtype)

        # Speculative decoding replays the verify step's conv inputs after the
        # target has decided how many tokens it accepts, so those inputs are
        # captured per conv. Allocated up front, alongside the pool: the first
        # verify step can happen inside a captured CUDA graph, where allocating
        # is not an option.
        self.verify_steps = self._max_draft_len + 1

        def cap(channels):
            if self.verify_steps < 2:
                return None
            return _ConvVerifyCapture(num_slots, channels, kwin, self.verify_steps, device, dtype)

        # The draft chain's manager owns a pool for the chain's layers only, so
        # it is sized by ``num_layers`` and addressed by GLOBAL layer index --
        # the same index the draft KV cache is keyed by. Without the offset a
        # draft block at index 42 would index past a 3-row pool.
        self._layer_offset = layer_offset
        self._num_layers = num_layers if num_layers is not None else config.num_hidden_layers
        self._layers: List[InklingConvState] = []
        self._captures: List[InklingConvState] = []
        for i in range(self._num_layers):
            if layer_offset:
                # The draft chain's pool. Its rows are ADDRESSED by the global
                # layer index, but their WIDTH comes from the chain's own
                # geometry: asking the trunk's accessor at index 42 gets the
                # trunk's answer for a layer the trunk does not have, and the
                # chain's banded depths then get global widths (or the reverse).
                # The two disagree by exactly the banded/global head ratio,
                # which surfaces as a channel-width mismatch inside the verify
                # capture -- and only on a checkpoint where the two differ.
                kv_heads = config.mtp_depth_num_kv_heads(i)
                head_dim = config.mtp_depth_head_dim(i)
            else:
                kv_heads = config.layer_num_kv_heads(i)
                head_dim = config.layer_head_dim(i)
            kv_dim = (kv_heads * head_dim) // tp_size
            hidden = config.hidden_size
            bufs = [
                allocate(i, role, [channels, kwin])
                for role, channels in zip(CONV_ROLES, (kv_dim, kv_dim, hidden, hidden))
            ]
            for buf in bufs:
                if buf.shape[0] < num_slots:
                    raise RuntimeError(
                        f"Inkling conv pool layer {i}: allocator returned "
                        f"{buf.shape[0]} slots but the pool needs {num_slots} "
                        f"({num_request_slots} request + reserved). The V2 SSM "
                        "layer was sized from a different slot count than "
                        "InklingConvStateCache assumes."
                    )
            self._layers.append(InklingConvState(*(b[:num_slots] for b in bufs)))
            self._captures.append(
                InklingConvState(k=cap(kv_dim), v=cap(kv_dim), attn=cap(hidden), mlp=cap(hidden))
            )
        # Refreshed in place per forward so a captured graph aliases it and every
        # replay sees the current batch. Indexed by batch position, not by slot,
        # so it needs one entry per row the padded batch can carry.
        self.state_indices = torch.arange(num_slots, dtype=torch.int32, device=device)
        # Pinned host staging for that write: one async H2D copy per forward,
        # legal under graph capture. Kept in lock-step size with ``state_indices``.
        self.state_indices_cpu = torch.zeros(
            num_slots, dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self._slot_of = {}
        self._free = list(range(num_request_slots - 1, -1, -1))

    def conv_state_bytes(self) -> int:
        """Total device bytes this pool holds. Reported by the cache manager."""
        return sum(t.numel() * t.element_size() for st in self._layers for t in st)

    def layer_state(self, layer_idx: int) -> InklingConvState:
        """The four short-conv state buffers for ``layer_idx`` (pool views)."""
        return self._layers[layer_idx - self._layer_offset]

    def layer_capture(self, layer_idx: int) -> InklingConvState:
        """The four verify-step captures for ``layer_idx``; entries None if off."""
        return self._captures[layer_idx - self._layer_offset]

    def commit_after_verify(self, num_accepted: torch.Tensor, gen_rows: torch.Tensor) -> None:
        """Roll every conv window back to each request's last accepted token.

        Called once after the target has verified, with ``num_accepted`` per
        generation request and the pool rows they occupy. Without this the
        windows keep the rejected tokens: no shape is wrong and no kernel
        complains, the model simply continues from a history it never produced.
        """
        if self.verify_steps < 2:
            raise RuntimeError(
                "Inkling conv state was not built for speculative decoding; "
                "commit_after_verify has nothing captured to replay from."
            )
        num_accepted = num_accepted.to(torch.int64)
        for layer_caps, layer_state in zip(self._captures, self._layers):
            for cap, pool_buf in zip(layer_caps, layer_state):
                window = cap.accepted_window(num_accepted, self.kwin)
                pool_buf.index_copy_(0, gen_rows, window.to(pool_buf.dtype))

    def _reserved_slot_for(self, request_id: int) -> Optional[int]:
        """The reserved row ``request_id`` aliases, or None for a real request.

        Padding rows carry no state anyone reads back, so every sentinel shares
        one row. ``cuda_graph_runner`` caches one dummy id per runtime draft
        length (``CUDA_GRAPH_DUMMY_REQUEST_ID - draft_len``), so the whole
        descending range has to map here, not just the exact sentinel.
        """
        from ....pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
        from ....pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID

        if CUDA_GRAPH_DUMMY_REQUEST_ID - self._max_draft_len <= request_id:
            return self._padding_slot
        if request_id == ATTENTION_DP_DUMMY_REQUEST_ID:
            return self._attention_dp_dummy_slot
        return None

    def slots_for(self, request_ids: List[int]) -> List[int]:
        """Map request ids to their (stable) pool rows, allocating new ones.

        Fresh requests get a zero-initialised row; existing ones keep theirs so
        their conv windows persist across decode steps. Padding sentinels and the
        attention-DP dummy alias their reserved rows.

        Raises on exhaustion. The pool covers every sequence that can be resident
        at once, so running out means a request held a row past its
        ``free_resources``; the alternative to raising is handing it a row
        another request is still carrying state in, which corrupts silently.
        """
        slots = []
        for r in request_ids:
            slot = self._slot_of.get(r)
            if slot is None:
                slot = self._reserved_slot_for(r)
                if slot is None:
                    if not self._free:
                        raise RuntimeError(
                            "Inkling short-conv pool is out of rows "
                            f"({self.num_request_slots} request rows, all live). "
                            "Every resident sequence needs one row; a row is "
                            "returned only by the cache manager's free_resources."
                        )
                    slot = self._free.pop()
                self._slot_of[r] = slot
                for st in self._layers:
                    for t in st:
                        t[slot].zero_()
            slots.append(slot)
        return slots

    def write_state_indices(self, request_ids: List[int]) -> List[int]:
        """Resolve ``request_ids`` to pool rows and publish them into the stable
        ``state_indices`` CUDA buffer -- the eager, pre-capture slot write.

        Returns the resolved slots in packed batch order (contexts first). A
        captured decode graph aliases ``state_indices``, so this must run every
        forward from eager input-prep, not inside ``model.forward``.
        """
        slots = self.slots_for(request_ids)
        n = len(slots)
        if n > self.num_slots:
            raise RuntimeError(
                f"Inkling short-conv batch has {n} rows but the pool publishes "
                f"only {self.num_slots} state indices; the pool is sized to the "
                "padded scheduler batch, so this signals a batch-shape mismatch."
            )
        self.state_indices_cpu[:n].copy_(torch.tensor(slots, dtype=torch.int32))
        self.state_indices[:n].copy_(self.state_indices_cpu[:n], non_blocking=True)
        return slots

    def free(self, request_ids: List[int]):
        for r in request_ids:
            slot = self._slot_of.pop(r, None)
            # Reserved rows are permanent: returning one to the free list would
            # hand a real request a row the next padding batch overwrites.
            if slot is not None and slot < self.num_request_slots:
                self._free.append(slot)


def _context_num_cached(attn_metadata, num_contexts):
    """Per-context-request cached token counts, or None when unavailable.

    Non-zero only for a later chunk of a chunked prefill. Returns None when the
    metadata carries no KV cache params at all (the cache-free unit-test path),
    which callers read as "everything is fresh".
    """
    params = getattr(attn_metadata, "kv_cache_params", None)
    if params is None:
        return None
    per_seq = getattr(params, "num_cached_tokens_per_seq", None)
    if per_seq is None:
        return None
    return [int(c) for c in per_seq[:num_contexts]]


@dataclass
class InklingConvRuntime:
    """Per-forward short-conv plumbing for the pool path (all layers share it).

    Splits the packed ``[context tokens | one-token generation]`` batch at the
    context boundary so each of the four short-convs seeds the pool for context
    requests (varlen ``causal_conv1d_fn``) and updates it in place for generation
    requests (``causal_conv1d_update``), exactly like the paged attention split
    in :meth:`InklingAttention._attention`.
    """

    num_ctx_tokens: int
    ctx_indices: Optional[torch.Tensor]  # int32 pool slots, context requests
    gen_indices: Optional[torch.Tensor]  # int32 pool slots, generation requests
    query_start_loc: Optional[torch.Tensor]  # int32 [n_ctx+1] varlen offsets
    has_initial_state: Optional[torch.Tensor]  # bool [n_ctx]
    # Tokens per generation request. 1 for ordinary decode; under speculative
    # decoding the target verifies 1 + max_draft_len tokens per request in one
    # step, and the one-token assumption below stops holding.
    gen_tokens_per_seq: int = 1
    # Varlen offsets/flags for a multi-token generation step, built only when
    # gen_tokens_per_seq > 1.
    gen_query_start_loc: Optional[torch.Tensor] = None
    gen_has_initial_state: Optional[torch.Tensor] = None

    @classmethod
    def build(cls, attn_metadata, cache: InklingConvStateCache) -> "InklingConvRuntime":
        """Publish this batch's pool rows, then build the context/generation split.

        The split mirrors the attention split: context requests first (each with
        its full new-token span), then one-token generation requests. Called from
        ``InklingAttentionMetadata.prepare()``, so the host->device slot write
        lands outside the captured ``model.forward``.
        """
        slots = cache.write_state_indices(list(attn_metadata.request_ids))
        seq_lens = attn_metadata.seq_lens.tolist()
        num_contexts = attn_metadata.num_contexts
        state_indices = cache.state_indices
        device = state_indices.device
        num_ctx_tokens = sum(seq_lens[:num_contexts])
        ctx_indices = state_indices[:num_contexts] if num_contexts else None
        gen_indices = (
            state_indices[num_contexts : len(slots)] if num_contexts < len(slots) else None
        )
        query_start_loc = has_initial_state = None
        if num_contexts:
            cu = torch.zeros(num_contexts + 1, dtype=torch.int32, device=device)
            cu[1:] = torch.tensor(seq_lens[:num_contexts], dtype=torch.int32, device=device).cumsum(
                0
            )
            query_start_loc = cu
            # A context request that already has tokens in the KV cache -- a
            # later chunk of a chunked prefill -- also has the conv window those
            # tokens left behind: ``slots_for`` keeps its pool row and
            # ``causal_conv1d_fn`` wrote the trailing window there on the
            # preceding call. Declaring it here is what makes that state get
            # consumed rather than ignored (the ``Mamba2Metadata`` pattern).
            #
            # Only *sufficient* because ``_run_context`` routes such requests to
            # the chunked-context kernel, which reads the cached KV back. Alone,
            # this would leave attention losing the history while the convs kept
            # it -- a subtler wrong answer than before. The two land together.
            cached = _context_num_cached(attn_metadata, num_contexts)
            if cached is None:
                has_initial_state = torch.zeros(num_contexts, dtype=torch.bool, device=device)
            else:
                has_initial_state = torch.tensor(
                    [c > 0 for c in cached], dtype=torch.bool, device=device
                )

        # Speculative decoding verifies several tokens per generation request in
        # one step. The per-request token count is uniform (1 + max_draft_len),
        # so it divides out of the generation seq_lens.
        num_gen = len(slots) - num_contexts
        gen_tokens_per_seq = 1
        gen_query_start_loc = gen_has_initial_state = None
        if num_gen > 0:
            gen_lens = seq_lens[num_contexts : len(slots)]
            gen_tokens_per_seq = max(1, int(gen_lens[0]))
            if gen_tokens_per_seq > 1:
                if any(int(sl) != gen_tokens_per_seq for sl in gen_lens):
                    raise ValueError(
                        "Inkling short-conv expects a uniform token count per "
                        f"generation request; got {gen_lens}."
                    )
                gen_query_start_loc = torch.arange(
                    0,
                    (num_gen + 1) * gen_tokens_per_seq,
                    gen_tokens_per_seq,
                    dtype=torch.int32,
                    device=device,
                )
                # Unlike a fresh prefill, a generation request always has a
                # prior conv window in the pool: these tokens continue a stream.
                gen_has_initial_state = torch.ones(num_gen, dtype=torch.bool, device=device)
        return cls(
            num_ctx_tokens=num_ctx_tokens,
            ctx_indices=ctx_indices,
            gen_indices=gen_indices,
            query_start_loc=query_start_loc,
            has_initial_state=has_initial_state,
            gen_tokens_per_seq=gen_tokens_per_seq,
            gen_query_start_loc=gen_query_start_loc,
            gen_has_initial_state=gen_has_initial_state,
        )


def apply_short_conv(
    sconv,  # models.modeling_inkling.InklingShortConv (duck-typed to avoid a cycle)
    x: torch.Tensor,
    pool_buf: Optional[torch.Tensor],
    rt: Optional[InklingConvRuntime],
    capture: Optional[_ConvVerifyCapture] = None,
) -> torch.Tensor:
    """Run one short-conv over a (possibly mixed) batch through the state pool.

    ``rt is None`` -> stateless full-sequence causal conv (no pool registered).
    Otherwise the context slice seeds ``pool_buf`` (varlen prefill) and the
    generation slice updates it in place at ``rt.gen_indices`` (decode), then the
    two outputs are concatenated in packed order. ``pool_buf`` is this conv's
    ``[max_batch, channels, kernel-1]`` state buffer from
    :class:`InklingConvStateCache`.
    """
    if rt is None:
        return sconv(x)
    parts = []
    nctx = rt.num_ctx_tokens
    if nctx > 0:
        parts.append(
            sconv.forward(
                x[:nctx],
                conv_state=pool_buf,
                cache_indices=rt.ctx_indices,
                query_start_loc=rt.query_start_loc,
                has_initial_state=rt.has_initial_state,
                is_decode=False,
            )
        )
    if x.shape[0] > nctx:
        if rt.gen_tokens_per_seq > 1:
            if capture is not None:
                # Before the conv mutates the pool: the window as it stands now,
                # plus the inputs about to be consumed, are what the post-verify
                # commit replays from.
                capture.save(
                    pool_buf, rt.gen_indices.to(torch.int64), x[nctx:], rt.gen_tokens_per_seq
                )
            # Speculative decoding: several tokens per generation request in one
            # step. ``causal_conv1d_update`` is a single-token kernel -- it
            # requires one cache index per ROW, so it rejects this outright --
            # and even reshaped it would apply the same initial state to every
            # drafted token instead of advancing through them. The varlen path
            # walks the run in order, which is what continuing a stream means;
            # ``has_initial_state`` is True here rather than False as in prefill,
            # because these tokens continue a window already in the pool.
            parts.append(
                sconv.forward(
                    x[nctx:],
                    conv_state=pool_buf,
                    cache_indices=rt.gen_indices,
                    query_start_loc=rt.gen_query_start_loc,
                    has_initial_state=rt.gen_has_initial_state,
                    is_decode=False,
                )
            )
        else:
            parts.append(
                sconv.forward(
                    x[nctx:], conv_state=pool_buf, cache_indices=rt.gen_indices, is_decode=True
                )
            )
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

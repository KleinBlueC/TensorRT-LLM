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
"""Inkling attention backend: KV write, page tables, prefill/decode dispatch."""

import os
from typing import Optional

import torch

from ...interface import AttentionForwardArgs, merge_attention_forward_args
from ...trtllm import TrtllmAttention
from .kernels import (
    build_page_table,
    inkling_chunked_prefill_attention,
    inkling_decode_attention,
    inkling_prefill_attention,
    write_kv_cache_hnd,
)
from .metadata import InklingAttentionMetadata
from .params import InklingBackendForwardArgs


def _kv_capacity_report(mgr, request_id) -> str:
    """What the manager thinks this request's KV cache holds.

    The page list alone cannot say whether a short write is a capacity
    shortfall or blocks that were never materialised: blocks follow capacity
    (``div_up(capacity, tokens_per_block)``), so printing both settles it.
    """
    kv_cache = getattr(mgr, "kv_cache_map", {}).get(request_id)
    if kv_cache is None:
        return "no kv_cache entry"
    return (
        f"capacity={getattr(kv_cache, 'capacity', '?')} "
        f"history={getattr(kv_cache, 'history_length', '?')} "
        f"num_blocks={getattr(kv_cache, 'num_blocks', '?')} "
        f"extra_kv_tokens={getattr(mgr, 'num_extra_kv_tokens', '?')}"
    )


def check_verify_write_room(
    base: int,
    steps: int,
    page_size: int,
    block_ids,
    *,
    request_id=None,
    cache_layer=None,
    detail: str = "",
) -> None:
    """Refuse a verify step whose KV writes would land outside the request's pages.

    A verify step writes positions ``base .. base + steps - 1``, so it needs the
    page holding the LAST of them; the manager has to have reserved that during
    the context phase. Both ways of getting this wrong have happened, and both
    were silent:

    * a base past the end -- the manager reserved for a different drafted length
      than the step presents -- surfaces as a bare ``IndexError: list index out
      of range`` from inside ``write_kv_cache_hnd``, naming neither the request
      nor the position;
    * a NEGATIVE base indexes from the END of the page list, so the write
      succeeds and quietly rewrites the start of the request's own history. That
      one produced fluent output that differed from greedy, and cost four rounds
      to find.

    Public because the tests exercise it directly: the arithmetic is the
    contract, and asserting it through a full attention forward would need a
    GPU to say something that is true on paper.
    """
    where = f"request={request_id} layer={cache_layer} " if request_id is not None else ""
    if base < 0:
        raise RuntimeError(
            f"Inkling verify step has a negative KV write base: {where}"
            f"base={base} steps={steps}. Torch indexes negatively from the end "
            "of the page list, so this would silently rewrite the start of the "
            f"request's own history rather than fail. {detail}"
        )
    valid = sum(1 for b in block_ids if int(b) >= 0)
    need = (base + steps - 1) // page_size + 1
    if need > valid:
        raise RuntimeError(
            f"Inkling verify step has no KV page for the last drafted position: {where}"
            f"base={base} steps={steps} last_pos={base + steps - 1} "
            f"page_size={page_size} needs {need} pages, has {valid} valid of "
            f"{len(block_ids)}. The request's KV capacity was grown for a "
            f"different drafted length than this step presents. {detail}"
        )


def _verify_write_base(attn_metadata, num_cached, num_gen, steps):
    """Where a verify step's ``steps`` tokens start in each request's cache.

    Prefers ``kv_lens_cuda - steps`` over ``num_cached_tokens_per_seq``. The two
    agree except under the overlap scheduler, where only the first is right.

    ``kv_lens_cuda`` is provisioned optimistically for the drafted tokens and
    then corrected in-forward by ``model_engine._preprocess_inputs`` (it adds
    ``accepted - provisioned``, a negative number), so by the time attention
    runs it holds the true KV length. The CPU list gets no such correction: its
    rewind lives in ``mtp.py``'s ``change_attn_metadata``, which under overlap
    has not run for the previous step yet. Measured 26 against a true 23.

    Taking the optimistic base does not corrupt the step that takes it -- reads
    and writes both derive from ``base`` and stay consistent -- it skips over
    the previous step's REJECTED draft positions and leaves them in the cache,
    so every later step attends over tokens the model never emitted.

    Falls back to the CPU list when there is no ``kv_lens_cuda`` (the CPU-only
    unit tests build metadata without one), which is also the pre-existing
    behaviour, so nothing outside the overlap path changes.

    The ``max(0, ...)`` clamp is kept from that pre-existing code and is not
    decoration: generation-step warmup presents tiny dummy sequences whose base
    underflows, and torch indexes a negative offset from the end of the page.
    """
    kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
    if kv_lens is None:
        return [max(0, int(x)) for x in list(num_cached)[:num_gen]]
    # One D2H copy per verify step. Affordable here specifically: Inkling
    # refuses CUDA graphs together with speculation, so this path is already
    # eager, and the block-id bookkeeping below is host-side regardless.
    start = attn_metadata.num_contexts
    lens = kv_lens[start : start + num_gen].tolist()
    return [max(0, int(x) - steps) for x in lens]


def _batch_cache_indices(mgr, request_ids, cache_layer):
    """``mgr.get_batch_cache_indices`` with the layer-not-in-this-manager case named.

    A draft block is addressed by the GLOBAL layer index (trunk + depth), which
    only the SEPARATE draft KV cache manager is keyed by: its layer mask is
    ``[False]*trunk + [True]*depths``. If the chain ends up running against the
    target's manager instead, that index means nothing there and the manager
    raises a bare ``KeyError: 42`` from inside its own pool lookup, minutes into
    a multi-GPU run, naming neither the layer's origin nor the mismatch.
    """
    try:
        return mgr.get_batch_cache_indices(request_ids, cache_layer)
    except KeyError as exc:
        raise RuntimeError(
            f"Inkling layer {cache_layer} has no slot in the KV cache manager in "
            f"play ({type(mgr).__name__}). Draft-chain layers are addressed by "
            "the global layer index and need the separate draft KV cache "
            "manager; the chain cannot share the target's cache because its "
            "layers are not the target's."
        ) from exc


def _needs_chunked_context(num_cached) -> bool:
    """True when the context path must read cached KV back from the pages.

    Any request carrying cached tokens needs it: a later chunk of a chunked
    prefill. The packed kernel cannot serve those -- it sees only the tokens of
    its own call.

    ``INKLING_FORCE_CHUNKED_ATTN=1`` forces the chunked path even for all-fresh
    requests. Test-only, default off. It exists because comparing the two
    kernels any other way is impossible on this model: its sampled output is not
    reproducible run to run, or even between two generate() calls in one
    process, unless the autotuner is off. The knob makes both kernels reachable
    in one process against one set of weights (jobs 6046191 / 6046341).
    """
    if os.environ.get("INKLING_FORCE_CHUNKED_ATTN", "0") == "1":
        return True
    return any(int(c) > 0 for c in num_cached)


class InklingTritonAttention(TrtllmAttention):
    """Runs Inkling's Triton attention over the paged KV cache.

    Standard :meth:`AttentionBackend.forward`. Inkling's two private inputs
    (``rel_logits``, ``allow_mixed``) ride
    ``AttentionForwardArgs.sparse_backend_args``, the registered slot for
    exactly this; ``params.py`` covers why neither widening the shared dataclass
    nor reusing T5's ``relative_attention_bias`` works.

    Subclasses ``TrtllmAttention``, not ``AttentionBackend``:
    :class:`InklingAttentionMetadata` extends the Trtllm metadata to reuse its
    ``prepare()``, and the model engine gates warmup on that same subclass check.

    ``sm_scale`` / ``rel_extent`` / ``window_left`` are assigned by
    ``InklingAttention`` after construction -- ``create_attention()`` has a fixed
    kwarg list. ``layer_idx`` is the *global* index, which is what
    ``KVCacheManagerV2`` expects.
    """

    Metadata = InklingAttentionMetadata

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: InklingAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch prefill / decode over the paged cache, supporting mixed
        context+generation batches.

        The runtime packs context requests first, then one-token generation
        requests, so slicing at that boundary feeds the prefill and paged-decode
        kernels respectively; the pure cases fall out as single slices.

        Overriding ``forward`` rather than delegating to
        ``TrtllmAttention.forward`` is what keeps Inkling off the sparse
        prediction hooks that method calls.
        """
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        args = forward_args.sparse_backend_args
        if not isinstance(args, InklingBackendForwardArgs):
            raise TypeError(
                "InklingTritonAttention.forward needs its rel_logits bias in "
                "forward_args.sparse_backend_args as an "
                f"InklingBackendForwardArgs, got {type(args).__name__}. Build it "
                "with inkling_forward_args(); the shared AttentionForwardArgs "
                "has no field that can carry a per-query-token bias."
            )
        rel_logits = args.rel_logits
        allow_mixed = args.allow_mixed
        attn_metadata = metadata
        # KVCacheManagerV2 takes the global layer index and maps it through
        # ``layer_offsets`` itself.
        cache_layer = self.layer_idx
        kv = attn_metadata.kv_cache_manager.get_buffers(cache_layer, kv_layout="HND")
        # kv: [num_pages, 2, num_kv_heads, page_size, head_dim]
        k_cache, v_cache = kv[:, 0], kv[:, 1]
        page_size = kv.shape[3]
        mgr = attn_metadata.kv_cache_manager
        request_ids = attn_metadata.request_ids
        num_cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        seq_lens = attn_metadata.seq_lens.tolist()
        num_contexts = attn_metadata.num_contexts
        num_seqs = len(seq_lens)
        ctx_tokens = sum(seq_lens[:num_contexts])

        # A mixed context+generation batch needs the per-request short-conv state
        # pool: the stateless path would convolve across the context/generation
        # boundary. Refuse it unless the pool path is active.
        if 0 < num_contexts < num_seqs and not allow_mixed:
            raise NotImplementedError(
                "InklingAttention: mixed context+generation batch needs the "
                "short-conv state pool; the stateless short-conv path convolves "
                "across the context/generation boundary of the packed batch. "
                "Set allow_mixed=True on InklingBackendForwardArgs only when the "
                "pool path is active (conv_rt is not None)."
            )

        outs = []
        if num_contexts > 0:
            outs.append(
                self._run_context(
                    q[:ctx_tokens],
                    k[:ctx_tokens],
                    v[:ctx_tokens],
                    rel_logits[:ctx_tokens],
                    seq_lens[:num_contexts],
                    num_cached[:num_contexts],
                    request_ids[:num_contexts],
                    mgr,
                    cache_layer,
                    k_cache,
                    v_cache,
                    page_size,
                )
            )
        if num_contexts < num_seqs:
            outs.append(
                self._run_generation(
                    q[ctx_tokens:],
                    k[ctx_tokens:],
                    v[ctx_tokens:],
                    rel_logits[ctx_tokens:],
                    num_cached[num_contexts:],
                    request_ids[num_contexts:],
                    mgr,
                    cache_layer,
                    k_cache,
                    v_cache,
                    page_size,
                    attn_metadata,
                )
            )
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

    def _run_context(
        self,
        q,
        k,
        v,
        rel_logits,
        seq_lens,
        num_cached,
        request_ids,
        mgr,
        cache_layer,
        k_cache,
        v_cache,
        page_size,
    ):
        device = q.device
        # Persist new K/V to the paged cache for later generation reuse.
        block_ids = mgr.get_batch_cache_indices(request_ids, cache_layer)
        off = 0
        for i, sl in enumerate(seq_lens):
            write_kv_cache_hnd(
                k_cache,
                v_cache,
                k[off : off + sl],
                v[off : off + sl],
                block_ids[i],
                int(num_cached[i]),
                page_size,
            )
            off += sl
        cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)
        max_seqlen = max(seq_lens)
        # A request with cached history -- a later chunk of a chunked prefill --
        # must attend to tokens it did not bring with it, which the packed
        # kernel cannot do: it takes no paged-KV argument. Route those to the
        # chunked-context kernel, which reads every key from the page table. No
        # gather is needed because the write above already put this chunk's K/V
        # into the same pages.
        #
        # All-fresh keeps the packed kernel: it is the common case and skips the
        # page indirection. The two must agree exactly at num_cached == 0, which
        # test_chunked_prefill_parity pins.
        if _needs_chunked_context(num_cached):
            max_total = max(int(c) + int(sl) for c, sl in zip(num_cached, seq_lens))
            max_pages = (max_total + page_size - 1) // page_size
            page_table = build_page_table(block_ids, max_pages, device)
            num_cached_dev = torch.tensor(
                [int(c) for c in num_cached], dtype=torch.int32, device=device
            )
            return inkling_chunked_prefill_attention(
                q,
                k_cache,
                v_cache,
                cu,
                num_cached_dev,
                page_table,
                page_size,
                max_seqlen,
                self.sm_scale,
                rel_logits,
                self.rel_extent,
                self.window_left,
            )
        return inkling_prefill_attention(
            q, k, v, cu, max_seqlen, self.sm_scale, rel_logits, self.rel_extent, self.window_left
        )

    def _run_verify(
        self,
        q,
        k,
        v,
        rel_logits,
        num_cached,
        request_ids,
        mgr,
        cache_layer,
        k_cache,
        v_cache,
        page_size,
        attn_metadata,
        steps,
    ):
        """Generation attention for a speculative verify step.

        A verify step presents ``steps = 1 + max_draft_len`` query tokens per
        request instead of one. The decode path cannot serve that -- it writes a
        single KV entry per request and reads a page table sized for one new
        position -- which is the out-of-bounds access this replaces.

        Nor can the context path, which is why this is not simply routed there:
        ``inkling_prefill_attention`` attends only within the tokens it is
        given, and ``inkling_chunked_prefill_attention`` would read the prefix
        but has no notion of the run's own causality. So the drafted run is
        walked one position at a time, each step writing its KV and then
        attending over cache-so-far. Causality within the run comes out of the
        ordering rather than a mask: position t attends to the prefix plus
        positions 0..t, which is what a linear draft chain means.

        The cost is ``steps`` decode launches over a growing cache; a fused
        verify kernel would read the cache once instead, and is the obvious
        optimisation once this is known to be right.
        """
        if getattr(attn_metadata, "is_cuda_graph", False):
            raise RuntimeError(
                "Inkling speculative verify attention runs eagerly (it walks "
                "the drafted positions one at a time) and cannot be captured. "
                "Disable CUDA graphs when using speculative decoding with "
                "Inkling, or wait for the fused verify kernel."
            )
        num_gen = len(request_ids)
        # Where this step's tokens go. ``num_cached_tokens_per_seq`` is what the
        # framework fills with the request's history for a speculative
        # generation batch (model_engine's _prepare_tp_inputs appends
        # ``past_seen_token_num`` on that branch), so position t belongs at
        # ``num_cached[i] + t``.
        #
        # An earlier reading of this field as 0 came from warmup batches, where
        # 0 is correct; deriving the base from the sequence lengths instead
        # produced a NEGATIVE offset, which torch happily indexes from the end
        # of the page. That is worse than the thing it replaced.
        #
        # The draft chain sees this field AFTER the framework rewinds it by the
        # rejected drafts (``mtp.py``: ``num_cached_tokens_per_seq[i] -=
        # runtime_draft_len + 1 - num_accepted``). On a real sequence that is the
        # correct post-rewind count; on the tiny dummy sequences of
        # generation-step warmup it underflows below zero. ``mtp.py`` clamps
        # ``kv_lens_cuda`` in the same place and for the same reason, and says so
        # in a comment -- it just does not clamp this CPU list, which is the one
        # Inkling reads. So clamp it here, on the same grounds.
        #
        # ...and under the OVERLAP SCHEDULER that rewind has not happened yet
        # when this runs, so the list is optimistic by ``max_draft_len``: it
        # counts every drafted token of the previous step as accepted. Measured
        # 26 against a true 23 (job 6320803). Writing at 26 does not corrupt
        # this step -- reads and writes both use ``base`` and stay consistent --
        # it leaves positions 23..25, the previous step's REJECTED drafts, in
        # the cache unoverwritten, and every later step attends over tokens the
        # model never emitted. That is the overlap defect.
        #
        # ``kv_lens_cuda`` does not have the problem: the engine corrects it
        # in-forward (``_preprocess_inputs``, by ``accepted - provisioned``,
        # which is negative), so by the time this runs it holds the true length.
        # Derive the base from it instead -- ``kv_len - steps`` -- which is the
        # same 23 in both modes. The clamp stays for the warmup sequences.
        base = _verify_write_base(attn_metadata, num_cached, num_gen, steps)
        block_ids = _batch_cache_indices(mgr, request_ids, cache_layer)
        # A verify step writes positions ``base .. base + steps - 1``, so it
        # needs the page holding the LAST of them. The manager grows a
        # generation request's capacity by "1 + drafted tokens" per step, which
        # covers exactly that -- but only while the drafted length it was asked
        # for is the one this step presents. When the two disagree the write
        # runs off the end of the block list and torch reports a bare "list
        # index out of range" from inside the kernel helper, naming neither the
        # request, the position, nor the size of the shortfall.
        rids = list(request_ids)
        for i in range(num_gen):
            check_verify_write_room(
                base[i],
                steps,
                page_size,
                block_ids[i],
                request_id=rids[i],
                cache_layer=cache_layer,
                detail=_kv_capacity_report(mgr, rids[i]),
            )
        max_pages = max(len(b) for b in block_ids)
        page_table = build_page_table(block_ids, max_pages, q.device)
        # [num_gen, steps, ...]: the packed batch is request-major, so a request's
        # drafted tokens are contiguous and this view is free.
        qv = q.view(num_gen, steps, *q.shape[1:])
        kv_ = k.view(num_gen, steps, *k.shape[1:])
        vv = v.view(num_gen, steps, *v.shape[1:])
        rv = rel_logits.view(num_gen, steps, *rel_logits.shape[1:])
        out = None
        for t in range(steps):
            for i in range(num_gen):
                write_kv_cache_hnd(
                    k_cache,
                    v_cache,
                    kv_[i, t : t + 1].contiguous(),
                    vv[i, t : t + 1].contiguous(),
                    block_ids[i],
                    base[i] + t,
                    page_size,
                )
            seq_lens = torch.tensor(
                [base[i] + t + 1 for i in range(num_gen)],
                dtype=torch.int32,
                device=q.device,
            )
            # The per-step slices are strided views of the packed batch; the
            # Triton kernels assert contiguity (and the fused-qkv v slice has
            # bitten this backend before).
            step_out = inkling_decode_attention(
                qv[:, t].contiguous(),
                k_cache,
                v_cache,
                seq_lens,
                page_table,
                page_size,
                self.sm_scale,
                rv[:, t].contiguous(),
                self.rel_extent,
                self.window_left,
            )
            if out is None:
                out = torch.empty(
                    (num_gen, steps, *step_out.shape[1:]),
                    dtype=step_out.dtype,
                    device=step_out.device,
                )
            out[:, t] = step_out
        return out.reshape(num_gen * steps, *out.shape[2:])

    def _run_generation(
        self,
        q,
        k,
        v,
        rel_logits,
        num_cached,
        request_ids,
        mgr,
        cache_layer,
        k_cache,
        v_cache,
        page_size,
        attn_metadata,
    ):
        device = q.device
        # A speculative verify step presents 1 + max_draft_len query tokens per
        # request instead of one, which every assumption below breaks. Route it
        # before any of them are made.
        num_gen = len(request_ids)
        steps = q.shape[0] // num_gen if num_gen else 1
        if steps > 1:
            return self._run_verify(
                q,
                k,
                v,
                rel_logits,
                num_cached,
                request_ids,
                mgr,
                cache_layer,
                k_cache,
                v_cache,
                page_size,
                attn_metadata,
                steps,
            )
        # --- Runtime CUDA-graph-safe path. ---------------------------------
        # Zero host->device copy: slice the base metadata's graph-stable
        # ``kv_lens_cuda`` and ``kv_cache_block_offsets``, then persist the new
        # K/V with an in-graph scatter whose indices are derived on-GPU. Padding
        # rows carry dummy request slots, so the scatter never touches a real
        # request's page.
        num_req = q.shape[0]
        # Guard on the metadata *type*, explicitly: the previous version relied
        # on a ``getattr(..., 0)`` default to route a foreign metadata object
        # here, which is not a stated check.
        if (
            isinstance(attn_metadata, InklingAttentionMetadata)
            and attn_metadata.num_generations == num_req
        ):
            sl = attn_metadata.ink_gen_seq_lens(num_req)
            pt = attn_metadata.ink_gen_page_table(cache_layer)[:num_req]
            page_div = attn_metadata.ink_page_div
            pos = (sl - 1).long()  # write slot = total_kv_len - 1 = num_cached
            page_row = torch.div(pos, page_size, rounding_mode="floor")
            offs = pos - page_row * page_size
            # Divide after the gather: [num_req] elements, not [num_req, blocks].
            pages = pt.gather(1, page_row.unsqueeze(1)).squeeze(1).long()
            if page_div > 1:
                pages = torch.div(pages, page_div, rounding_mode="floor")
            # Paired advanced indices select one (page, slot) per request ->
            # [num_req, num_kv_heads, head_dim], matching the new k/v.
            k_cache[pages, :, offs, :] = k.to(k_cache.dtype)
            v_cache[pages, :, offs, :] = v.to(v_cache.dtype)
            return inkling_decode_attention(
                q,
                k_cache,
                v_cache,
                sl,
                pt,
                page_size,
                self.sm_scale,
                rel_logits,
                self.rel_extent,
                self.window_left,
                page_div=page_div,
            )
        # Eager fallback (never captured): the decode metadata was not published,
        # so build it here from the host block table, like the context path. This
        # path is illegal under CUDA graph, and the usual cause is an
        # ``attn_backend`` override that swapped the metadata type -- say so.
        if getattr(attn_metadata, "is_cuda_graph", False):
            raise RuntimeError(
                "Inkling decode metadata is unusable for a CUDA-graph batch: "
                f"attn_metadata is {type(attn_metadata).__name__} with "
                f"num_generations="
                f"{getattr(attn_metadata, 'num_generations', None)}, expected an "
                f"InklingAttentionMetadata with {num_req}. "
                "Inkling's backend is selected through sparse/registry.py under "
                "the TRTLLM backend family; remove any attn_backend override "
                "from --extra_llm_api_options / LLM(attn_backend=...) so the "
                "default applies."
            )
        num_req = len(request_ids)
        block_ids = mgr.get_batch_cache_indices(request_ids, cache_layer)
        for i in range(num_req):
            write_kv_cache_hnd(
                k_cache,
                v_cache,
                k[i : i + 1],
                v[i : i + 1],
                block_ids[i],
                int(num_cached[i]),
                page_size,
            )
        total = [int(num_cached[i]) + 1 for i in range(num_req)]
        decode_seq_lens = torch.tensor(total, dtype=torch.int32, device=device)
        max_pages = max(len(b) for b in block_ids)
        decode_page_table = build_page_table(block_ids, max_pages, device)
        # No ``page_div`` here (it defaults to 1): unlike the graph path's
        # ``kv_cache_block_offsets`` slice, ``get_batch_cache_indices`` already
        # divides by ``kv_factor`` itself (see
        # ``_get_batch_cache_indices_by_pool_id``), so these are block indices.
        return inkling_decode_attention(
            q,
            k_cache,
            v_cache,
            decode_seq_lens,
            decode_page_table,
            page_size,
            self.sm_scale,
            rel_logits,
            self.rel_extent,
            self.window_left,
        )

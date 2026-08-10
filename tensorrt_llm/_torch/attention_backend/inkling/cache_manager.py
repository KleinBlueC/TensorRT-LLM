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

import torch

from tensorrt_llm.logger import logger

from ...pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2


class InklingHybridCacheManager(KVCacheManagerV2):
    """Paged KV (V2, per-layer geometry) + the short-conv state pool.

    Folding the pool into the cache manager -- the shape
    ``CppMambaHybridCacheManager`` uses for mamba conv/SSM state -- lets it reach
    the model through the standard ``attn_metadata.kv_cache_manager`` field and
    be released by the manager's own ``free_resources``. The conv rows are then
    freed by the same call that frees the request's KV blocks, so the two views
    cannot drift apart.

    The cost is that the pool is also allocated for the throwaway manager built
    during KV-cache size estimation, and freed along with it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Imported here, not at module scope: modeling_inkling imports from
        # _torch.attention_backend and _torch.modules, and a top-level import
        # would close a cycle back through pyexecutor at model-load time.
        from ...models.modeling_inkling import InklingConvStateCache

        pretrained_config = kwargs["pretrained_config"]
        mapping = kwargs["mapping"]
        max_batch_size = kwargs["max_batch_size"]
        # Not kwargs["dtype"] -- that is the KV cache dtype, a C++ binding type
        # torch.zeros rejects. The conv pool holds pre-conv activations, so it
        # takes the model's compute dtype from the (text) config.
        text_config = getattr(pretrained_config, "text_config", pretrained_config)
        conv_dtype = getattr(text_config, "torch_dtype", None)
        if not isinstance(conv_dtype, torch.dtype):
            conv_dtype = torch.bfloat16
        # The conv pool's k/v width follows the attention kv-head split, so it
        # takes the attention TP, not the global one -- the same rule
        # KVCacheManagerV2 applies to the paged pool. Dividing by the global
        # tp_size would allocate narrow conv rows for full-width convs.
        attn_tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        # +1 row for the CUDA-graph padding / dummy-request slot (the mamba
        # pattern): a padded decode batch admits up to max_batch_size real
        # requests plus a shared dummy row.
        # Under speculative decoding the target verifies 1 + max_draft_len
        # tokens per request in one step, and the conv windows have to be
        # rolled back to whatever prefix is accepted. The capture buffers that
        # makes possible are sized here, up front, because the first verify step
        # can land inside a captured CUDA graph.
        spec_config = kwargs.get("spec_config")
        verify_steps = 1
        if spec_config is not None:
            verify_steps = int(getattr(spec_config, "max_draft_len", 0) or 0) + 1
            # The first verify step of a request needs KV room for all
            # ``verify_steps`` positions it writes, and the context phase is
            # what has to have reserved it: capacity there is
            # ``prompt + num_extra_kv_tokens``, and the generic one-engine
            # reserve is ``max_draft_len - 1``. Measured on a real run
            # (job 6026096): prompt 669, capacity 671, first verify step writing
            # positions 669..672 -- two short. It only surfaces when that last
            # position lands on a page boundary, which is why short-prompt runs
            # never caught it and a 5-shot GSM8K prompt did.
            #
            # From the second step on the scheduler's per-step growth
            # (+1 + draft) takes over and the margin is 6, so this is a
            # context-phase reservation, not a per-step one.
            self.num_extra_kv_tokens = max(self.num_extra_kv_tokens, verify_steps)
        # A draft manager covers only the chain's layers, and addresses them by
        # the GLOBAL layer index its KV layer offsets already use. Sizing its
        # conv pool by the trunk's layer count would allocate 42 rows to hold 3
        # and then index past them.
        is_draft = bool(kwargs.get("is_draft"))
        num_conv_layers = None
        layer_offset = 0
        if is_draft:
            num_conv_layers = int(kwargs.get("num_layers") or 0) or None
            text_layers = int(getattr(text_config, "num_hidden_layers", 0))
            layer_offset = text_layers
        self._conv_cache = InklingConvStateCache(
            pretrained_config,
            attn_tp_size,
            max_batch_size + 1,
            torch.device("cuda", torch.cuda.current_device()),
            conv_dtype,
            verify_steps=verify_steps,
            num_layers=num_conv_layers,
            layer_offset=layer_offset,
        )
        self._last_conv_rt = None

    # ---- model-facing -----------------------------------------------------
    def prepare_conv_runtime(self, attn_metadata):
        from ...models.modeling_inkling import InklingConvRuntime

        rt = InklingConvRuntime.build(attn_metadata, self._conv_cache)
        # Retained for the post-verify conv commit, which runs from the spec
        # worker after the forward context has exited and so cannot rebuild it.
        self._last_conv_rt = rt
        return self._conv_cache, rt

    def commit_conv_state_after_verify(self, num_accepted) -> None:
        """Roll the conv windows back to each request's last accepted token.

        The pool rows come from the runtime built for the verify step, so this
        commits against the same rows the forward advanced -- not whatever the
        next batch happens to occupy.
        """
        rt = self._last_conv_rt
        if rt is None or rt.gen_indices is None or rt.gen_tokens_per_seq < 2:
            return
        # Diagnostic switch. Speculative decoding is currently lossy on Inkling
        # and this commit is one of the few places that could cause it; a run
        # with it disabled says whether it contributes at all, which no amount
        # of reading the code establishes.
        import os

        if os.environ.get("INKLING_DISABLE_CONV_COMMIT") == "1":
            logger.info_once("Inkling conv commit DISABLED by env", key="ink_conv_commit_off")
            return
        logger.info_once(
            f"Inkling conv commit active (steps={rt.gen_tokens_per_seq}, "
            f"rows={int(rt.gen_indices.shape[0])})",
            key="ink_conv_commit_on",
        )
        rows = rt.gen_indices.to(torch.int64)
        import os

        if os.environ.get("INKLING_PROBE_KV") == "1":
            # How many drafted tokens the target actually accepted. Always the
            # full chain would mean verification is not rejecting anything,
            # which is a different bug from the drafts simply being bad.
            n = getattr(self, "_probe_accept_n", 0)
            if n < 8:
                self._probe_accept_n = n + 1
                print(
                    f"[probe accept #{n}] steps={rt.gen_tokens_per_seq} "
                    f"rows={int(rows.shape[0])} "
                    f"num_accepted={num_accepted[-rows.shape[0] :].tolist()}",
                    flush=True,
                )
        self._conv_cache.commit_after_verify(num_accepted[-rows.shape[0] :], rows)

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

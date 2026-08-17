# Inkling prefix cache reuse — design proposal

Status: **proposal, not implemented.** It needs a decision from the
KVCacheManagerV2 owners before any code lands, because it adds state to the
reuse contract rather than using what is already there.

This document exists because the paged-KV prefill work solved half of the
problem and it is not obvious from the outside which half. Chunked prefill
shipped; block reuse is still refused by
`reject_unsupported_inkling_kv_cache_features`, and the reason changed.

## What the paged prefill already solved

`inkling_prefill_attention` reads K/V from the page table using absolute
positions (`num_cached + local_row`), so a context request carrying cached
history attends to that history correctly. That is one code path, not a special
case: a fresh context is the same call with `num_cached == 0`.

So for a reused prefix, **attention would already be right**. Measured against
an independent fp32 dense oracle, splitting a prompt costs no more than not
splitting it (max_abs 2.2e-3 at a split vs a 6.5e-3 unsplit floor; cos
0.9999978), and end-to-end on GSM8K 1319 the chunked arm is indistinguishable
from the baseline against a same-code noise floor.

## Why reuse is still refused

The four depthwise short convolutions per layer hold a `kernel_size - 1` window
as per-request state that lives **outside** the KV cache, in
`InklingConvStateCache`.

Chunked prefill and block reuse differ in exactly one way, and it is decisive:

| | where the conv window comes from | what is needed |
|---|---|---|
| chunked prefill | the *same request's* earlier chunk wrote it into that request's own pool row | declare `has_initial_state` — done |
| block reuse | the prefix belongs to a *different request*, whose pool row went back to the free list at `free_resources` | **there is no window to restore** |

A reused prefix would therefore restore half of what the request needs:
attention resumes from real history, the convolutions restart from zeros.

## The cheap option, and why it is closed

"Recompute the window from the last `kernel_size - 1` tokens of the reused
prefix" is not implementable. The convs consume **activations**, not KV:

* the k/v convs run inside `_project`, on `qkv_proj(hidden_states)`;
* the attn and mlp convs run on the residual stream (`InklingDecoderLayer`).

For a reused prefix those activations were never computed — skipping that
computation is the entire point of prefix reuse. Only K/V is in the cache, and
the conv window is a different tensor at a different point in the layer.
Recomputing the last few tokens means re-running them through the whole stack,
and each layer's input depends on the previous layer's conv output, so the
problem recurses one layer down. Record this option as closed.

## The option that works, and what it costs

Snapshot the conv window at reusable block boundaries and restore it on a hit.

**Size.** Per snapshot point, for the shipped geometry (66 layers, hidden 6144,
kv_dim 1024 at TP=1, kernel 4, bf16):

```
66 layers x (2*kv_dim + 2*hidden) x (kernel-1) x 2 B
  = 66 x (2048 + 12288) x 3 x 2  ~= 5.7 MB
```

TP divides the k/v part only; the residual-stream convs are replicated, so it
does not shrink proportionally. This is per *snapshot point*, not per request:
how many are kept is the policy question below.

**What the owners have to decide.** These are the reasons this is a design
review and not a patch:

1. **Where snapshots are taken.** Every block boundary is the simplest rule and
   the most expensive. Only at boundaries that are actually reuse candidates is
   cheaper but couples the conv pool to the reuse policy.
2. **Lifetime and eviction.** The snapshot must be evicted exactly with the
   blocks it belongs to, or a later hit restores a window that no longer
   matches the KV.
3. **The reuse key.** Today the key covers token ids. A snapshot is derived
   state; it does not change what the prefix *is*, so it probably does not
   belong in the key — but a hit must be refused when the snapshot is missing,
   which is a new failure mode the manager has to express.
4. **Partial reuse.** `enable_partial_reuse` can accept a prefix shorter than a
   full block. There is no conv snapshot at a sub-block position, so either
   partial reuse is refused for Inkling or snapshots become finer-grained.
5. **Where it lives.** `InklingConvStateCache` is model-side today. Reuse is
   manager-side. One of the two has to grow a dependency on the other.

## Suggested acceptance criteria

Mirroring what the paged-prefill work used, with one correction it learned the
hard way.

**Layer level.** Reuse-on vs reuse-off, identical attention *and* conv output
for a prefix hit, against a dense oracle rather than against another kernel —
using a Triton kernel to check a Triton kernel shares any misreading of the
contract. Plus negative controls: a missing or stale snapshot must change the
result, and a wrong page order must change the result. Without those, a passing
parity test can mean the input simply did not exercise the path.

**Engine level.** GSM8K with reuse on vs off, paired per sample, **with a
same-code noise floor arm**. Do not gate on an accuracy delta alone and do not
gate on output text identity:

* Inkling's batched MoE and autotuner are non-deterministic. Two runs of
  *identical code* reproduce only ~13–22% of outputs verbatim and differ by
  0.15–0.23 pt of accuracy. Any threshold not expressed relative to that is
  meaningless.
* Judge the flip split with an **exact McNemar** test, not by eye. During the
  paged-prefill campaign a 12/4 split against a 9/7 floor (p=0.077) looked like
  a regression; it did not survive its own repeat, and every comparison that
  isolated a *single* change was symmetric (p 0.45–0.80).
* An earlier attempt at this feature recommended `enable_autotuner=False` to
  suppress that variance. That is not necessary and it changes the thing being
  measured — a noise-floor arm costs one more job and keeps the production
  configuration under test.

**Do not** gate on service-level throughput without repeats: run-to-run spread
on that benchmark is 3.6–5.6%, which is wider than most effects worth arguing
about.

## Until then

`reject_unsupported_inkling_kv_cache_features` refuses `enable_block_reuse=True`
for Inkling, so the configuration fails loudly instead of silently producing
wrong output. Its message names the conv-window lifecycle, not the attention
path — the attention reason no longer exists, and leaving it in would send the
next reader to fix a kernel that is already fixed.

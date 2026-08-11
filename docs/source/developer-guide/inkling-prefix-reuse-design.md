# Inkling prefix cache reuse — design proposal

Status: **proposal, not implemented.** It needs a decision from the
KVCacheManagerV2 owners before any code lands, because it adds state to the
reuse contract rather than using what is already there.

## Why Inkling cannot just turn reuse on

Inkling carries four depthwise short convolutions per layer. Their
`kernel_size - 1` window is per-request state that lives **outside** the KV
cache, in `InklingConvStateCache`. A reused prefix therefore restores half of
what the request needs: attention resumes from real history, the convolutions
restart from zeros.

The attention half is already solved. The chunked-context prefill kernel added
on this branch reads K/V back from the page table using absolute positions, so
a context request with `num_cached > 0` attends to its prefix correctly — that
is the same code path chunked prefill uses, and it is measured bit-identical
to one-shot prefill at the layer level (job 6048348: `max_abs 0.0`, `cos 1.0`).

What remains is the conv window, and it cannot be reconstructed.

## The option that does not work

The obvious cheap fix — "recompute the window from the last `kernel_size - 1`
tokens of the reused prefix" — is not implementable. The convs consume
**activations**, not KV:

* the k/v convs run inside `_project`, on `qkv_proj(hidden_states)`
  (`modeling_inkling.py`, `_project`);
* the attn and mlp convs run on the residual stream (`InklingDecoderLayer`).

For a reused prefix those activations were never computed — skipping that
computation is the entire point of prefix reuse. Only K/V is in the cache, and
the conv window is a different tensor at a different point in the layer.
Recomputing the last 3 tokens means re-running them through the whole stack,
and each layer's input depends on the previous layer's conv output, so the
problem recurses one layer down. This option should be recorded as closed.

## The option that works, and what it costs

Snapshot the conv window at reusable block boundaries and restore it on a hit.

**Size.** Per snapshot point, for the shipped geometry (66 layers,
hidden 6144, kv_dim 1024 at TP=1, kernel 4, bf16):

```
66 layers x (2*kv_dim + 2*hidden) x (kernel-1) x 2 B
  = 66 x (2048 + 12288) x 3 x 2  ~= 5.7 MB
```

TP divides the k/v part only; the residual-stream convs are replicated, so it
does not shrink proportionally. Note this is per *snapshot point*, not per
request: how many are kept is the policy question below.

**What the owners have to decide.** These are the reasons this is a design
review and not a patch:

1. **Where snapshots are taken.** Every block boundary is the simplest rule and
   the most expensive. Only at block boundaries that are actually reuse
   candidates is cheaper but couples the conv pool to the reuse policy.
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

Whatever shape is chosen, these are what should gate it, mirroring what the
chunked prefill work on this branch used:

* layer-level: reuse-on vs reuse-off bit-identical attention *and* conv output
  for a prefix hit, the way `test_inkling_chunked_prefill.py` does it, with a
  negative control that a missing/stale snapshot changes the result;
* engine-level: GSM8K with reuse on vs off, at n >= 500, with
  `enable_autotuner=False` — without that the model's own variance is 2.36
  logprob and swamps the comparison (job 6046341).

## Until then

`reject_unsupported_inkling_kv_cache_features` refuses
`enable_block_reuse=True` for Inkling, so the configuration fails loudly
instead of silently producing wrong output.

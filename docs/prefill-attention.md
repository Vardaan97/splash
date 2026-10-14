# Experimental prefill attention schedules (PF3)

| Flag | Default | Meaning | Promotion / sunset |
| --- | --- | --- | --- |
| `SPLASH_ATTN_PIPE` | unset | `qk2` preissues the next attended page's QK; `prob2` ping-pongs probabilities and rescale flags and removes the third page barrier; `rows4` uses four rows but retains the original eight-row plan's split count | Apple10 only after U0–U9 and ≥1.2× at 32K–131K in shared and independent QSA; then replace the flag with a measured device rule and remove the flag in PR text |
| `SPLASH_ATTN_STATS` | 0 | B1 CPU count of attended/visible pages per tile from the final attention layer's completed bitmap, logged to stderr; no GPU counter or extra GPU synchronization | Diagnostic only; remove after collecting real-text sparsity data |
| `SPLASH_KERNEL_SELFTEST` | 0 | `1` explicitly tests all schedules and fails on mismatch; requesting a schedule always runs its device-local test even when this is unset | Shared universal gate hook; retain until a common device self-test replaces it |

`splash --kernel-selftest [METALLIB]` performs U2 without loading a model.
Apple9 reports unavailable and always uses the original kernels. Existing kernel names and bodies are untouched.
The fixed-order reduction and per-page arithmetic remain unchanged; `rows4` changes only tile geometry, with the original split count.
A failed automatic self-test forces fallback. An explicit self-test fails the command. No numerical tolerance can promote PF3.

Run `prefill-attention-variants METALLIB --full` with and without shader validation before any M5 timing.
It compares output bytes and repeat bytes over rows 1/4/7/8/9/16/17/32/128/129/2048,
histories 0/2K/4K/32K/64K/128K/200704, all three GQA layouts, int8/BF16 KV,
shared/independent/page/empty masks and a poisoned nonzero bitmap row offset.
`attention-sweep --compare-metallib` explicitly forces the second arm to the baseline,
so it cannot accidentally compare two enabled schedules. Merge PF3 before WP9.

Retain composite GDN states on a fixed prefill grid

Part of #387

## Problem

After an agent edits or prunes an old tool result, the cached recurrent states can all lie after the shared prefix. Attention KV alone cannot restore GDN state, so the next request prefills the conversation again. The issue's 174,378-token example took 139.7 s to its first token after such an edit.

## Fix

Opt in with `SPLASH_GDN_CHECKPOINT_TOKENS=8192` or `16384`. Retain each crossed prefill-grid boundary as an ordinary composite cache state, including GDN recurrence/convolution, PLE history, MTP carried streams and draft context. Reuse the existing deepest matching state lookup, governor admission, eviction and lossless SSD offload. Unset/0 preserves the current policy.

Strengthen the replay oracle so its bitwise reference has matching GDN scan partitions. Add full-window engine tests with refused captures and both spacings. Gate the measurement against a separate base binary, every growing-turn output and a fresh cold edited reference; reject truncated generation streams.

## Effect

fp32 state is bitwise for any GDN scan partition. bf16 state rounds at each capture split: bf16 + grid checkpoints is class N (numerics-changing, opt-in only), with no greedy identity guarantee.

An edit with 105,000 common tokens can restore at 98,304 under either proposed spacing. For 174,378 input tokens, that reduces prefill work from 174,378 to 76,074 rows (56.4%) when the point and its KV remain cached.

TODO(M5): measured edited TTFT base→candidate, repetitions/order, 8K/16K, exact common-token count and restored rows; initial-cold/growing-turn overhead; output identity counts; resident/streamed PLE settings; allocation/RSS high-water marks.

One fp32 Flash-Next/MTP composite point is 113.28 MiB. Near the 262,144-token window, 31/15 grid points cost 3.43/1.66 GiB before tail/rolling states. Prefer 16K spacing on a 128 GB host with a large full-window target. Grid states remain evictable and multiple edited branches compete for the same budget.

This covers prefill points retained during the process. Decode-only spans are not backfilled, and persistent write-behind keeps its newest-point policy. Historical grid coverage after restart is not promised. Long decode-only spans and durable historical retention remain outside this change.

## Tests

- PASS on M4 Pro: `make -j2 check` at `1011be15044450d778a80018316e3eb9d4fed836`, including full-window capture refusal at both grids, engine/cache/governor, PLE/MTP kernels, state/disk restore, and native build checks. Python: 1,265 tests, 27 skipped; Ruff passed.
- Local light CPU: 3 mocked-stream identity regressions pass; two persistent-gate CPU simulations also pass. Ruff and shell/Python syntax checks pass; whitespace checks pass. No GPU work on the development Mac.
- TODO(M5): `m5-gates-D.sh` exact restore oracle, all KV/state combinations, MTP off, greedy base/candidate/cold identity, 8K/16K alternating-order TTFT A/B, informational class-N bf16 output comparison and persistent restart/deep-edit scenario.
- TODO(M5 integration): matched D+#410 versus #253+#410 builds, streamed PLE activation and first resumed MTP output identity; near-full-window model memory stress and pressure/cancellation fallback.

The fp32 and candidate-off greedy identity gates must pass before acceptance; bf16 grid identity is informational; no M5 speed or identity result is claimed yet.

Under memory pressure, refused optional captures can truncate prefill commands at grid boundaries. This known cost of the 8K grid remains unmeasured near the full-window ceiling.

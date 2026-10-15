# Experimental decode expert gather

`SPLASH_MOE_GATHER_FOLD` is off by default. Eligible plans use staged M8,
up to 32 rows, and matching gate/up formats. An eligible
`SPLASH_EXPERT_TILE=1` plan takes precedence and remains N-det.

- `1`: fold grouped-input gathering into each gate/up tile. Exactly-eight-row
  plans already read input in place and keep the baseline kernel.
- `2`: additionally prefetch the next K32 input slice into registers before
  running the current matmul. This also applies to in-place eight-row plans.

Both modes reuse space in the existing 9,216-byte stage allocation and retain
K32 matmul order, half-rounded weights, bf16 gate/up rounding, and the separate
down/combine passes. Threadgroup-input matmul needs device-specific bitwise
qualification; source order alone is not proof. No performance gain is claimed.

On the M4, `kernels/check-wp5-m4.sh gather` compares both modes against the
baseline, including live intermediate values, all 43 staged formats, rows
1–32, uniform and mixed legal lane bounds, with SV off/on. `exhaustive` tests
the Cartesian product of every legal per-lane lastRows for each firstRow.
`air` accepts `WP5_BASELINE_METALLIB` and compares every baseline kernel.

The M4-built `gguf-moe-benchmark-e512 <metallib> [rounds] [gate/up format]
[down format] [shared format]` defaults to decode replay at H=2560, I=640,
E=512, top-10, with 1/4/8/16/32 rows. It reports full-layer medians and separate
dispatch replays. Synthetic cache reuse means replay is not a DRAM-bandwidth
measurement. M5 runs belong only to the main-session queue, after SLOT-2.
Qualification receipts and remaining gates are in `kernels/WP5-STATUS.md`.

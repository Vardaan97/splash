# WP5 — expert GEMV

2026-10-15. Base `e8c386d6`, branch `flashnext-expert-gemv`.
Session deadline: 03:33 UTC (09:03 IST). No M5 builds/GPU work or large I/O;
no sub-agents, pushes or posts. M4 slot: `~/.flashnext-build.lock-slot5`.

## Step 0 — G port

Ported the uncommitted G candidate from `~/src/splash-expert-tile`, including
its native build targets, fp64 interval tests and microbenchmark. Preserved
perf-all's m64 kernels and other test targets while resolving append conflicts.
`SPLASH_EXPERT_TILE=1` remains opt-in N-det; no bitwise or performance claim.
Source diff whitespace check passed. Commit: `d7447588`.
M4 focused SV test passed (`expert_tile_test: PASS`, fp64 interval policy).
Full M4 `make check` exited 0 under slot 5, including both MoE suites with SV.
Python: 1,287 tests, 27 skipped. Remote receipt: `~/wp5-logs/G-check.log`.
The first AIR baseline (reused perf-all library) differed in six unrelated
hyper kernels and is unsuitable as an exact e8c386d6 oracle. A clean baseline
worktree/build is queued; no AIR pass claimed yet.

## Step 1 — bitwise candidates

Gather-fold prototype and e512 memcmp/replay harness built on M4; validation
is running. Default off (`SPLASH_MOE_GATHER_FOLD=1|2`). Mode 2 prefetches
the next input slice in registers and also covers in-place B1 plans.
It reuses the second weights stage for a gathered K32 input slice and keeps
the compiled 9,216-byte allocation (both baseline and candidate measured
through `staticThreadgroupMemoryLength`). Only staged M8, rows <=32, fused gate/up,
non-in-place plans qualify. Exactly 8 total rows already skip gather in base.

Still required: indirect-grid reduction where safe,
load-ahead variant without exceeding the existing 9,216-byte stage.
Proof: e512 memcmp rows 1–32 and all valid firstRow/lastRows, 43 staged formats,
SV off/on; unchanged kernel AIR identity. No bitwise claim before receipts.

## Main-session M5 queue

SLOT-2 limiter capture first; then e512 benchmark 1/4/8 rows per format.
GO requires ≥15% per-layer at 8 rows plus bitwise. Only then d4 512/2K ABBA
`decode-profile --cycles 64`, ≥2% outside twice spread and 14/14 greedy bytes.
No M5 work launched by this session.

# WP5 — expert GEMV

2026-10-15. Base `e8c386d6`, branch `flashnext-expert-gemv`, source worktree
`~/src/splash-expert-gemv`. Deadline 03:33 UTC / 09:03 IST; exhaustive runner
stops at 03:28 UTC. M4 builds/tests use slot 5 and nice 19. No M5 jobs,
sub-agents, pushes or posts. No performance gain or M5 bitwise claim.

## Step 0 — G complete on M4

`d7447588` ports G from the original uncommitted expert-tile tree, preserving
perf-all's m64 kernels and build targets. `SPLASH_EXPERT_TILE=1` remains opt-in
N-det. Focused fp64-interval test with SV and full `make check` passed (exit 0);
both MoE suites passed. Python: 1,287 tests, 27 skipped.
Fresh e8c386d6 baseline AIR: all 1,847 existing kernels identical; two G kernels added.
Receipts on M4: `~/wp5-logs/G-check.log`, `G-check.exit`, `G-air-fresh.log`.

## Step 1a — gather fold and input prefetch, default off

`SPLASH_MOE_GATHER_FOLD=1`: gather each K32 input slice inside gate/up.
Mode `2`: additionally prefetch the next input slice into registers while the
current matmul runs; applies to in-place B1 plans too. Eligible: staged M8,
rows 1–32, matching gate/up formats, no eligible G override. Mode 1 leaves
exactly-eight-row plans alone: baseline already skips their gather.
K32 accumulation order, half weights, bf16 gate/up rounding and down/combine
are retained. Candidate lives in `moe_gguf_gather.metal`, linked last to keep
strict AIR identity. No additional threadgroup allocation.

| M4 gate | Result |
|---|---|
| Compiled TG bytes, SV off | 9,216 baseline and candidate |
| Instrumented TG bytes, SV on | 18,432 baseline and candidate (validation overhead) |
| All 43 staged formats, rows 1–32, every legal firstRow, uniform/mixed lastRows; routing + live intermediate + output memcmp | PASS, 16,512 cases, both modes vs baseline, SV off and on |
| Same matrix with reused maximum workspace (stale scratch coverage) | PASS, SV off and on |
| Strict AIR vs fresh e8c386d6 | PASS, all 1,847 existing kernels identical; only G's two and gather's one added |
| Every Cartesian lastRows combination, all firstRow, B1–B4, all formats | Running; 761,788 cases per SV mode; not yet a pass |
| M5 identity / performance / greedy | Not run; main-session queue only |

Receipts: M4 `~/wp5-logs/{gather-off-final,gather-sv-final,gather-off-reuse,
gather-sv-reuse,gather-air-isolated}.log`; exhaustive logs and JSON exits
use `exhaustive-off` / `exhaustive-sv`. The first SV attempt stopped at an
incorrect memory assertion on the instrumented baseline, before numerics;
the cap is now applied to uninstrumented pipelines, with SV sizes reported.

Reproduce on M4: `kernels/check-wp5-m4.sh g|gather|exhaustive|check|air`.
The air mode needs `WP5_BASELINE_METALLIB` from a fresh exact baseline build.
The initial reused perf-all library was not that oracle: six untouched
`gguf_decode_q80_m{8,16,32}_{hc,hcmix}` kernels had a different hyper parameter
ABI. Fresh baseline lives at `~/src/splash-expert-gemv-base` on M4; old failed
comparisons are retained, not normalized away.

## Remaining work

Single gate/up+down indirect dispatch is not implemented. Down depends on all
intermediate columns. A whole-expert threadgroup can avoid a global barrier,
but serializes column tiles and needs enough live expert tiles for occupancy;
SLOT-2/live-tile measurements must qualify any selection rule. Do not add a
spinning grid barrier. Prior weight pf2/pf4 variants were slower (DESIGN §3.2);
mode 2 here prefetches input slices instead.

## Main-session M5 queue

1. SLOT-2 limiter capture first (`~/Developer/gpu-profile/READY.md`); inspect
   that captured dispatches are decode shapes, not the harness prefill warm-up.
2. Run device memcmp with SV off/on. Then fresh-process 0/2/2/0 replay:
   `SPLASH_MOE_GATHER_FOLD=<mode> build/engine-tests/gguf-moe-benchmark-e512
   build/splash.metallib 40 iq4xs iq4nl q80` (repeat per format).
   Harness reports layer medians plus replay at 1/4/8/16/32 rows. Synthetic
   reuse is not a DRAM-bandwidth measurement. GO: ≥15% at 8 rows and bitwise.
3. Only on GO: d4 512/2K `decode-profile --cycles 64` ABBA, ≥2% outside twice
   spread, and 14/14 greedy bytes. Follow the standing thermal/burn-in protocol.

No M5 run or artifact promotion was performed by this session.

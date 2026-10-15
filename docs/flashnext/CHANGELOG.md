# Changelog — Flash-Next fork branches

Newest entry first. One entry per update to `docs/flashnext/BRANCHES.md`; every number in an entry cites a receipt
listed in that file. Hardware: M5 Max 128 GB (Apple10, 40 GPU cores) for measurements, M4 Pro for builds and tests.

## 2026-10-15

Published / prepared:
- `flashnext-perf-all` `46c3972114` (base #253 `174e62e48c`), `flashnext-max-stack` `2591fd56d5`,
  `flashnext-expert-gemv` `301acb096b`, `flashnext-head-q6k` `deed1af13b`, `speed/w2-5` `54b781a557`.
- Held: `flashnext-max-gsq` `40d010520a`. Upcoming: `ple-ssd-b4-f1`/`-f2`, `flashnext-hostgap` `fb1684dc`, `flashnext-split-rule`.

Measured on M5 Max:
- perf-all: flags-off identity PASS; device self-test PASS with PF1 `moe_m64` PASS and **PF3 `qk2/prob2/rows4` differing
  on Apple10 → KEEP-OFF**; WP2-k64 and WP1 in-model identity PASS [R2–R5].
- E2 gate: dispatch-floor saving 0.618 µs per dispatch < 2 µs prerequisite → `SPLASH_DECODE_DISPATCH=2` KEEP-OFF [R6].
- WP2 sweep: tiles marginal (0–1 %); production split policy suboptimal on Apple10 for `2560x6144`, `2560x5120`,
  `10240x2560` and the fused qkv projection (raw medians in BRANCHES.md) → new `flashnext-split-rule` work item [R7].
- max-stack + D vs #410+A+B: 4K cold prefill −8.4 % (noise 2.1 %), 4K cache-hit TTFT −6.3 % (noise 1.1 %), decode
  inconclusive (drift 10–12 %); whole-tree, unattributed [R9].
- Leave-one-out decode matrix (R = 2): all-off faster than all-on by 0.97 % (B1) and 2.59 % (B3); F costs 0.9–1.7 %;
  other flags neutral within noise [R10].
- SLOT-1 B1 profile: 33.35 ms fused GPU cycle, 1,343 dispatches, Q6_K head 4.82 ms (5 evaluations) [R16].
- B draft head: replay all-full N=248320 (3.93 ms of 31.5 ms at B1) [R17]; fresh-request canary shows the 32K head
  loaded and used (7 × N=32768 per cycle) [R18]; sustained-use ABBA PENDING.

Measured on M4 Pro:
- expert-gemv 761,788 bitwise cases, `make check` PASS [R11]; head-q6k 4,736 memcmp PASS, no speedup claimed [R12];
  w2-5 full check PASS at `6cf1aa44`, focused suites at `54b781a5` [R13]; b4-fix F1/F2 synthetic B1/B4 byte-identical [R14];
  hostgap `make check` PASS, identity PASS [R15].

Pending at close of entry: perf-all timed steps (WP1 profile running), prefill attribution, all M5 gates for
expert-gemv / head-q6k / hostgap / w2-5 / ple-ssd-b4, PF3 Apple10 fix.

## Template for the next entry

```
## YYYY-MM-DD
Published / prepared: <branch> <head> (<base>) …
Measured on M5 Max: <claim> <number> (noise <x %>) [R<n>] …
Measured on M4 Pro: …
Pending at close of entry: …
```

# Small projection experiments

These Q8_0 decode variants are opt-in. Existing kernel bodies and flag-unset plans stay unchanged.

| Flag | Default | Scope | Sunset rule |
| --- | --- | --- | --- |
| `SPLASH_SMALL_PROJ` | off | Reserved for a measured Apple10 selection rule; currently no change | Remove tiles if the sweep saves less than 1 ms/cycle; replace the disabled rule only after device identity and performance gates |
| `SPLASH_SMALL_PROJ_TILE` | unset | Force `base`, `k64`, `k128`, `sg1`, or `k64sg1` on eligible Apple10 Q8_0 plain projections | Keep for experiments until a validated rule replaces forced selection; never enable by default from a unit pass |

Flags are captured at Linear construction. Apple9 always uses the baseline policy, including with flags set. Explicit test plans still exercise each variant and report family, case count, and mismatches. An Apple9 variant mismatch is reported as unavailable; guards and counters remain hard failures. Apple10 mismatches fail the binary.

Deep tiles preserve group order and rotation boundaries. Two KS-wide stages alternate and load the next payload before the current matmul. K128 reuses the completed stage for its arrival word after a threadgroup barrier to fit 32 KiB. K=320's K128 request selects K64 on the host. The one-simdgroup tile is a control variant.

Before timing: shared kernel gate U0–U5, including device self-test and M5 kernel memcmp with and without shader validation. Sweep 40 rounds at L1 and L4, comparing unchanged splits. Require hc-down >=1.5x, hc-up >=1.25x, predicted saving >=1 ms/cycle, and the chosen variants no slower at L4. Fit a rule from those receipts, then rebuild and validate its Apple9 no-change branch on M4. No rule is inferred from unmeasured performance.

In-model: CYCLES=64, >=2% GPU ms/cycle saving outside twice the repeated-off spread, exact identity, B4 no regression, short and long contexts, resident and streamed oracles, and reference-model no-op gates. No M5 results are recorded by implementation sessions.

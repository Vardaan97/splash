# Overlap independent Flash-Next decode projections behind a switch

## Problem

Flash-Next encodes QSA query and key projections after the attention input
projection through a serial compute encoder, although all three read the
same normalized hidden rows and write separate outputs. Dispatch gaps are
suspected in decode; the current per-kernel replay cannot measure those gaps.

## Fix

Add explicit independent groups to CommandGraph and honor them in direct and
pre-encoded commands. Commands containing a group use concurrent compute
encoding with full buffer barriers at every remaining boundary. Events keep
their existing encoder and command-buffer boundaries.

`SPLASH_DECODE_DISPATCH=1` enables the QSA group in Flash-Next decode. A main
projection that needs several dispatches keeps its ordering and only its
indexer pair overlaps. The default stays serial. The profiler gains real token-file input, streamed
PLE loading and a mode without isolated replay for Metal System Trace.
Kernels, reduction orders,
weights, scratch allocation and prefill are unchanged.

## Effect

No measured speed claim yet. This is a narrow overlap prototype, not the
whole proposed 1.5–4 ms fusion programme. It removes one or two serialization
boundaries per attention pass, without changing dispatch count.

- Attribution/trace only, M5 B1 cycle, 512 / 32K / 128K: **[pending]**
- Attribution/trace only, M5 B4 cycle, 512 / 32K / 128K: **[pending]**
- Metal System Trace within-command idle / overlapped time: **[pending]**
- Untraced real-prompt ABBA spread and category effects: **[pending]**
- Greedy token identity and memory: **[pending M5 gates]**

## Tests

CPU graph tests cover independent entry/exit boundaries, adjacent groups,
invalid/overlapping ranges and event crossing. Metal tests cover dependent
producer/consumer/overwrite, disjoint views of one allocation, direct and
pre-encoded submission, and bitwise equality of nonzero indexer projections
beside float/Q8_0 input projections at all four decode widths and staged
K splits 1/2/4/8, with differently poisoned scratch.

M4 `make -j2 check`: **PASS**. Additional enabled MTP runtime with Metal
shader validation: **PASS**. Enhanced decode-profile executable build:
**PASS**. Exact SHA and logs in STATUS.md.
M5 real-model oracle, token-exact native ABBA and real-prompt category gates:
**pending**. No GPU work was run on the M5 during preparation.

Local draft only. No issue or PR has been opened; agree the larger fusion
scope in an issue before proposing it upstream.

`m5-profile-E.sh` is attribution-only. Eight-cycle replay/process arms cannot resolve the expected 0.2–0.5% effect and do not support a speed claim, even after the 15-minute burn-in. Oracle identity defaults on.

The unrelated decode-profile tooling is already isolated in commit `4fa810d`; keep it in a separate tooling PR when preparing upstream changes. No rebase is performed here.

Before default-on promotion, plan shuffle verification that permutes dispatches inside each annotated group and compares exact outputs, then shadow verification that runs serial and annotated commands with independent scratch and compares every declared output bitwise. Exercise direct/pre-encoded paths, event boundaries, B1–B4 and production shapes under shader validation. These modes are planned, not implemented.

# Propose prompt-lookup continuations in MTP lanes

## Problem

Copy-heavy edits and tool arguments can already accept nearly the entire
seven-token MTP chain, yet still pay for its serial head steps. MTP lanes
currently cannot use the prompt/output continuations already in the request.

## Fix

Stack on C (`891a0fc`). Add incremental request-local exact n-gram lookup
and decayed shadow acceptance. Behind `SPLASH_MTP_LOOKUP=1`, C's cost planner
can select lookup proposals as a zero-head-step lane with live verify rows.
A Metal mixing operator preserves lookup lanes when another lane drafts.
Eligible lookup can also graft a low-softmax-confidence head tail after a
matching prefix; the head chain remains available as fallback. Verification,
one-hot proposal probabilities and retention caps are unchanged.

## Effect

Measured ABBA category effects: **[pending M5]**. Mixed batches may keep another lane's head steps. Grafting currently
improves proposal choice only; it does not skip head computation. Off by default.

## Tests

M4 `make -j2 check` passed at `ebc6254`: native CPU, shader-validated Metal,
source/build checks and 1,272 Python tests (27 skips). See STATUS.md for logs
and exact validated revisions. CPU
matcher/tracker/planner tests and Metal confidence/mixing tests accompany
Runtime mixed/all-lookup, target-oracle-output and lane/ID-reuse coverage.
`m5-gates-F.sh` supplies C/F/F/C HTTP category and mixed-request identity,
F-off control, exact native transcript token-vector identity, and off/on serial-target oracles across KV/state precisions.
Real-model identity and speed remain pending the main session's M5 window.

Lookup indexing is capped at 256 new tokens per sync and 4096 keys per map (two maps per request), including the first cycle. Older unindexed rows are skipped; eviction can reduce hit rate without changing verified output. Reserve a conservative 2 MiB host allowance per eligible lane. Sampled and thinking-cutoff lanes skip both planner and shadow indexing. The M5 gate requires 15 minutes of discarded load and supports either ABBA starting arm; alternate `ORDER=C-first` and `ORDER=F-first` across windows. Width-4 identity is recorded as informational pending diagnosis of upstream concurrent nondeterminism.

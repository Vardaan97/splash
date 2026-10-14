# Combined Flash-Next test stack

Local `flashnext-max-stack` starts at `0a08372e` (#253 `e5ce29f`, #410,
A/B/C), with reviewed D `84f059d4`, E `293a3add`, F `e503457f` and native
grid integration `7a483b91`. The October 14 follow-up keeps C modular and
restores the pre-C common rule default. No speed claim accompanies these
unmeasured fold-ins.

Source `dev/scripts/flashnext-max.env` for **max** (C off). Then call
`flashnext_profile max-c` for the otherwise identical C-on profile. Set
`FLASHNEXT_MAX_PROFILE=max-c` before sourcing to select it directly. All new
behaviors remain opt-in; A/B retain their inherited defaults. Use fp32 GDN
state for strict identity and choose streamed cache size explicitly.

B = bitwise tensor/output scheduling change; X = greedy exact, proposals and
seeded streams can differ; N = numerics change. B4 output identity is checked
where off/off repeats; existing B4 nondeterminism remains informational.

| Flag | Default | Expected effect (unmeasured unless inherited) | Identity | Test |
| --- | --- | --- | --- | --- |
| `SPLASH_MTP` | 1 | Enable inherited MTP | X | mtp-runtime/oracle |
| `SPLASH_MTP_DEPTH` | rule | cost opts into C; lane uses individual rule depths | X; lane B1 B with width-1 threshold .30 | planner, runtime B1 greedy/seeded and B4 live rows |
| `SPLASH_MTP_WORTHWHILE` | .30,.35,.40,.45 | lane thresholds by width; one value applies to all widths | X | planner parsing and mixed depths 2/3/7/4 |
| `SPLASH_MTP_ADAPTIVE` | 1 | Disable with 0 for fixed depth | X | mtp-runtime |
| `SPLASH_MTP_DRAFTS` | 7 adaptive, 3 fixed | Cap proposals, sweep 3/4/5/7 | X | planner/runtime |
| `SPLASH_MTP_DRAFT_VOCAB` | 32768 (1 alias) | 49152/65536 extend coverage; 0 uses full head | X | gguf-planner/context/runtime and 144-prompt sweep |
| `SPLASH_GDN_CHECKPOINT_TOKENS` | 0 | D improves deep-edit reuse; max uses 16384 | B fp32; N bf16 | engine grids/oracle |
| `SPLASH_DECODE_DISPATCH` | 0 | 1 narrow E; 2 audited E2 sites including draft | B | shader-validated backend/runtime/projection/Qwen4/MoE |
| `SPLASH_DECODE_DISPATCH_SITES` | mode mask (1 or 57) | Restrict sites for bisection | B | E2 identity gates |
| `SPLASH_DISPATCH_VERIFY` | unset | serial or shuffle:0..7 verifies marked independence | B | metal-backend fork/join/event tests, runtime shuffle |
| `SPLASH_HOST_WINS` | off | 1/all both trims, eh/kv individually | B | projection and runtime; fp32/bf16 oracle gates |
| `SPLASH_PLE_PREFETCH` | 0 | Anchor + midpoint prefix read-ahead during streamed draft | B | synthetic streamed runtime, event/backend, streamed oracle |
| `SPLASH_GRAPH_CACHE` | off | 1 prepared topology LRU; verify compares fresh preparation | B | 200-cycle backend payload/grid test, runtime/oracle |
| `SPLASH_MTP_LOOKUP` | 0 | F full lookup and graft in rule/lane/cost | X | mixed/all lookup runtime, matcher/planner |
| `SPLASH_MTP_LOOKUP_BUCKETS` | 0 | Confidence by matched length <6/<12/<24/24+ | X | matcher bucket/262K bound, runtime |
| `SPLASH_MTP_LOOKUP_SCAN` | 256 | 1..8192 newest tokens per sync; clamps input | X | 262K bounded work/node test |
| `SPLASH_MTP_LOOKUP_REPROBE` | 0 | Confident seven-token cost lookup reprobe after 300 rounds | X | planner/runtime, greedy M5 gate |
| `SPLASH_MTP_LOG` | 0 | Existing plan/cost/lookup diagnostics | B | diagnostic only |

## Audits and limits

The head trims port net changes from `32ec82b`, `7f14071`, `6bd352c`, keeping
B's `draftHead` logits selection. EH streams share one projection where the
existing bitwise batch-width projection tests permit it. KV-only head verify
still stores KV and QSA keys; it drops only unread attention output.

E2 uses one design: G1 (mask 1) attention/indexer projections; G4 (8) float
packed columns beside a **single**, unrotated, no-epilogue quantized dispatch;
G5 (16) single-dispatch PLE key/value; G6 (32) disjoint per-lane MTP input and
carry slices. Multi-dispatch shared scratch remains ordered. QSA project/pool
across lanes needs pooled-page alias proof; QSA select needs scratch renaming.
Context-head projections and EH use shared scratch and stay ordered. No
speculative independence annotations were added. Shuffle operates on whole
marked runs in both submission paths, respecting event boundaries. E2's
15-minute M5 dispatch-floor script is prepared, not run: promotion requires
at least 2 us saved per boundary and later identity/performance gates.

PLE prefetch uses one midpoint signal (one extra command-buffer boundary),
an immutable token snapshot before graft/cutoff, and copied hash/history
inputs. The GPU never waits on host I/O. The ticket drains its callback before
staging or teardown; synchronous staging remains authoritative. Prefetch
failures do not waive staging checks. Resident PLE is unaffected.

The graph cache conservatively caches **prepared dispatch topology**, not
Runtime graph construction. Exact pipeline, grid, group, binding, view and
byte-layout keys avoid omitted shape dimensions. All fresh payloads are
rebound, including positions, parity, live rows and lookup. Thus position
patches cannot become stale. 64-entry LRU, at most 32 MiB counted key/metadata;
cache entries hold weak allocation references and do not pin released state/KV
memory; actual submitted commands retain their live allocations as before. The
bounded host metadata sits within the existing 512 MiB runtime reserve. Verify
re-resolves pipelines/grids/retained allocations and compares. Skipping fresh
Runtime graph construction is deliberately deferred until every baked
parameter can be audited; no 0.2–0.4 ms saving is claimed for this narrower
implementation.

F's rule/lane integration chooses eligible lookup continuations that cover
the planned head depth; graft retains the head fallback. Cost mode retains
its planner and optional stale reprobe. Buckets extend matched suffixes
backwards, capped at 8192 comparisons, with four-observation priors
.75/.88/.93/.96. Maps remain at most 8192 nodes/request (~2 MiB allowance).
Sampled/cutoff lanes skip lookup; no whole-prompt indexing allocation.

B's retained `out-rank2.npy` has a top-32K set exactly equal to the original
bitmap. `dev/tools/data/qwen4-draft-ranking.npy.gz` retains it for repeatable
larger sets; the sizes manifest records its SHA256. Original tokenizer/corpus
revisions were not recorded, so this does not invent their provenance. Sets
are sorted top-K of the documented weighted ranking (`draft_vocabulary.py`),
with added tokens first. Request context excludes the **selected** set;
miss-based full-vocabulary fallback stays request-local and language-neutral
(>3% exponentially decayed uncovered emissions). Larger sets add head image
memory. No language classifier or new fallback threshold was introduced.

D's fp32 oracle remains bitwise; bf16 rounds at extra checkpoint boundaries.
Optional checkpoint reservation refusal can split prefill and cost latency.
F lookup mixing precedes verify-input/PLE hashing, composing with E barriers.

## Deferred host proposals

Prefill co-batching needs an arrival trace before changing scheduler policy:
`Scheduler::prefillBudget` already batches ready same-priority requests within
a shared row budget. Instrument the ready list before `nextPrefill`, drain
transport before planning if peers have not arrived, then consider a flag
only for short leaders within the 2048-row budget. Require two-lane prefill
KV/GDN/first-logit bitwise equivalence to separate commands before promotion;
different matrix tiles make that a material gate. No new prefill flag yet.

The optional draft attention window is deferred: paged verify plans expose
committed prefix length, not a host-side first-page offset; merely shortening
that length would attend the wrong prefix. A kernel ABI/window change is
outside the host-only authorization for this item.

M4 receipts and exact SHAs are in the handoff STATUS. M5 scripts in that
handoff directory build/download nothing and await an exclusive window.

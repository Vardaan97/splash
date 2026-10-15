# Flash-Next fork branches — index and measured status

Fork: `github.com/Vardaan97/splash`. This file is `docs/flashnext/BRANCHES.md` on every published branch. Last update: 2026-10-15 (see `CHANGELOG`).

## Purpose

The fork carries experiments on top of upstream Splash PR #253 (Qwen3.8-Flash-Next GGUF) and its follow-ups
(#410 streamed PLE, #423 hyper-norm, #429 draft vocabulary). Every experiment is behind an environment flag that is
**off by default**; a build of any branch with no flags set must produce byte-identical outputs to its base.
The goal is faster decode, prefill and TTFT for Flash-Next on Apple silicon, with every claim traced to a receipt.

Hardware: **M5 Max, 128 GB (Apple10 GPU family, 40 cores)** for all timing and identity runs;
**M4 Pro** (Apple9) as the build/test host (`make check`, kernel memcmp matrices). Model unless stated: `Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS`.

Labels: **PASS/FAIL** = measured, receipt listed under *Receipts*; **PENDING** = not yet measured on the M5.

## Branches

| Branch | Head | Base | Contents (flags, all default off) | M4 Pro status | M5 Max status |
|---|---|---|---|---|---|
| `flashnext-perf-all` | `46c3972114` | #253 `174e62e48c` (#410 included) | Unified experiment tree (107 commits): B/C/D/E/F, host wins, PLE prefetch, graph cache ported from the max-stack history, plus kernel packages WP1 `SPLASH_HYPER_FUSION`, WP2 `SPLASH_SMALL_PROJ_TILE`, PF1 `SPLASH_MOE_PREFILL_TILE`, PF2 `SPLASH_PREFILL_SLICES`, PF3 `SPLASH_ATTN_PIPE`, PF6 `SPLASH_PREFILL_SMALL_SPLIT`, PF7 `SPLASH_PREFILL_QSA_COMPACT`. Flag table: `docs/flashnext-perf-all.md`; baseline env `dev/scripts/flashnext-perf-all.env` | 63 checks PASS [R1] | flags-off identity PASS; device self-test PASS (PF1 m64 PASS, **PF3 differs → KEEP-OFF**); WP2-k64 identity PASS; WP1 identity PASS; dispatch floor PASS (E2 KEEP-OFF); WP2 shape sweep PASS. Timed WP1 profile, greedy-14 window and burn-in: PENDING [R2–R8] |
| `flashnext-max-stack` | `2591fd56d5` | older #253 base `e5ce29f8` + #410 (72 commits past the #410 merge-base) | Combined stack: `SPLASH_MTP_DRAFT_VOCAB`, `SPLASH_MTP_DEPTH`, `SPLASH_GDN_CHECKPOINT_TOKENS` (D), `SPLASH_DECODE_DISPATCH` (E/E2), `SPLASH_MTP_LOOKUP*` (F), `SPLASH_HOST_WINS`, `SPLASH_PLE_PREFETCH`, `SPLASH_GRAPH_CACHE`; profile env `dev/scripts/flashnext-max.env`; `docs/flashnext-max-stack.md` | PENDING at this head (not in today's receipts) | identity PASS (flags-off and full-stack bitwise on 8 transcripts); 4K cold prefill −8.4 %, 4K cache-hit TTFT −6.3 % (whole tree vs #410+A+B); decode inconclusive; leave-one-out matrix: no flag helps synthetic decode [R9, R10] |
| `flashnext-expert-gemv` | `301acb096b` | perf-all `e8c386d6` (on `174e62e`) | MoE expert GEMV: `SPLASH_EXPERT_TILE=1` (G split tile), `SPLASH_MOE_GATHER_FOLD=1/2` (gather fold, input prefetch), `SPLASH_MOE_WHOLE_EXPERT=1` (single indirect grid) | `make check` PASS (1,287 Python tests); gather 761,788 cases bitwise, shader validation off and on; whole-expert 761,788 cases (SV off), 16,512 (SV on); 1,847 existing kernels AIR-identical [R11] | PENDING (identity, replay, timing) |
| `flashnext-head-q6k` | `deed1af13b` | perf-all `e8c386d6` | Q6_K output-head kernel variants `SPLASH_HEAD_Q6K=1` (alias `c128d0`) and head-shape attribution `SPLASH_PROFILE_HEAD_SHAPES=1` | 4,736 full-buffer memcmp PASS with GPU validation; no speedup established (contended host) [R12] | attribution run only (see §Measured, B draft head); timing PENDING |
| `speed/w2-5` | `54b781a557` | perf-all `e8c386d6` | TTFT bundle: persistent-cache quota/free-space floor, PF5 `SPLASH_PLE_ADMISSION_PREFETCH=1`, `SPLASH_PREFILL_COBATCH=1`, deep-edit and co-batch identity tests, SIMD attribution switch | full `make check` PASS at `6cf1aa44`; focused suites PASS at `54b781a5` [R13] | PENDING (deep-edit TTFT, co-batch identity, B4 burst) |

Upcoming (not yet published):

| Branch | Head | Base | Contents | Status |
|---|---|---|---|---|
| `flashnext-max-gsq` | `40d010520a` | max-stack | `SPLASH_PROMPT_LOOKUP=0` opt-out of the always-on history drafter; benchmark receipt validator | **held** (commit text under review before publication) |
| `ple-ssd-b4-f1` / `ple-ssd-b4-f2` | `cd5100a908` / `a3ba576bf9` | #410 `bac17ac889` | F1: residency set on every pre-encoded command buffer; F2: pre-encode the streamed verify on the main queue | M4 build + tests PASS, synthetic B1/B4 outputs byte-identical across baseline/F1/F2 [R14]; M5 B4 streamed-vs-resident runbook PENDING |
| `flashnext-hostgap` | `fb1684dc` | perf-all `46c3972114` | `SPLASH_HOSTGAP_OVERLAP=1`: encode verify while the MTP draft runs (one command in flight); `/status.host_gap` counters | `make check` PASS (1,287 Python tests), synthetic B1/B4 identity PASS [R15]; M5 ABBA PENDING; expected saving is an estimate (0–0.4 ms/cycle), not measured |
| `flashnext-split-rule` | — | perf-all | `SPLASH_Q80_SPLIT_RULE`: Apple10 split policy from the WP2 sweep finding | in progress; no commit yet |

## Measured on M5 Max (2026-10-15)

**perf-all kernel packages** (`review-one-by-one-3`, Balanced-PLE8 root, flags-off reference):
- Identity PASS: flags-off greedy bytes == reference [R2]; device self-test [R3]; WP2 k64 small-projection tile in-model [R4]; WP1 hyper fusion [R5].
- PF1 `moe_m64` self-test PASS on Apple10 [R3]. **PF3** attention variants `qk2`, `prob2`, `rows4` **differ on Apple10**; the runtime forces the fallback. PF3 stays off and must not be offered upstream until bitwise on Apple10 [R3].
- E2 prerequisite not met: 900-s dispatch-floor microbenchmark, concurrent-vs-serial saving **0.618 µs per dispatch < 2 µs** gate → `SPLASH_DECODE_DISPATCH=2` KEEP-OFF on Apple10 [R6].
- WP2 shape sweep (q80, 40-round medians): k64/sg1 tiles within 0–1 % of base. The production **split choice is suboptimal on Apple10**: `2560x6144` S8 (production) 0.0399 ms vs S4 0.0366 (L1) and 0.0640 vs 0.0494 (L4); `2560x5120` 0.0355 vs 0.0321 (L1), 0.0594 vs 0.0447 (L4); `10240x2560` S2 0.0546 vs S1 0.0531 (L1), 0.0621 vs 0.0534 (L4); fused qkv `12288+512+512x2560` S2 0.0751 vs S1 0.0665 (L1), 0.0940 vs 0.0673 (L4). Follow-up: `flashnext-split-rule` [R7].

**max-stack `2591fd56` with D vs #410+A+B `3de5853`** (native ABBA, rounds B,C,C,B, warm pacing) [R9]:

| Metric | Base | Max+D | Change | Noise | Verdict |
|---|---:|---:|---:|---:|---|
| 4K cold prefill GPU ms | 7234 | 6624 | −8.4 % | 2.1 % | PASS |
| 4K cache-hit TTFT ms | 2332 | 2185 | −6.3 % | 1.1 % | PASS |
| Decode B1 GPU ms/step | 36.33 | 36.90 | +1.6 % | 10.5 % | inconclusive (drift) |
| Decode B2 / B3 / B4 | 40.65 / 49.19 / 51.79 | 40.99 / 49.30 / 51.56 | +0.9 / +0.2 / −0.4 % | 10–12 % | inconclusive |

The prefill/TTFT gains are **whole-tree and unattributed** (the run was resident, so PLE prefetch was inactive; D captured once, no restore). Attribution per commit is PENDING.

**Leave-one-out decode matrix** (synthetic decode, bracketed reference, R = 2, 23/44 launches clean; reference B1 42.18, B2 47.58, B3 57.91, B4 61.15 ms/step) [R10]:

| Arm | B1 | B2 | B3 | B4 | 95 % CI excludes 0 |
|---|---:|---:|---:|---:|---|
| all off | −0.97 % | −1.35 % | −2.59 % | −2.21 % | B1, B3 |
| minus F (lookup) | −0.93 % | −1.67 % | −1.47 % | −1.51 % | B2 |
| minus host wins / graph cache / prefetch / D | within ±1.4 % | | | | none |
| minus E (1 round) | −0.42 % | | | | n/a |

Negative = faster without the feature. No max-stack flag is a measurable decode win on this fixture; the stack's measured value is prefill/TTFT.

**SLOT-1 decode profile (B1, xctrace + replay)** [R16]: cycle 33.35 ms fused GPU / 35.12 wall, 1,343 dispatches; separate-dispatch replay 34.11 ms (dispatch overhead 0.76 ms). Families: q80_m8 7.19 ms (21 %), MoE gate/up 6.54 (19 %), Q6_K output head 4.82 (14 %, 5 evaluations), fused_m8 4.74 (14 %), MoE down 3.68 (11 %), hyper norm/combine/mix 1.53, f32 1.18, verify GDN 1.13. Prefill 512 rows 249 ms: q80_a 37 %, MoE gate/up 26.5 %, MoE down 13.5 %, GDN scan 6.2 %.

**B draft vocabulary (32K head)**: replay attribution on the `flashnext-head-q6k` runtime showed every head evaluation at N=248320 (B1 4.08/cycle × 0.965 ms = 3.93 ms of 31.5 ms; B4 4.41/cycle), zero N=32768 [R17]. Fresh-request canary: `draft-head-check: requested=32768 configured=32768 loaded=32768 bytes=69632000` and 7 × N=32768 dispatches per cycle beside 1 × N=248320 [R18]. So the head loads and is used on a fresh request; the all-full replay came after 64 fused cycles on the same request (request-local fallback). Sustained-use ABBA: PENDING (fixture ended before profiling).

**Context (not fork code)**: persistent-cache restart TTFT −98.7 % with identical outputs, Balanced-PLE8, main's persistent cache measured on our build [R19]. Grid-edit (D) TTFT −0.02 % — the scenario did not exercise D [R19].

## Pending

- perf-all: WP1 timed profile (running at time of writing), greedy-14 window, burn-in + native ABBA for WP1/WP2/PF1/PF2/PF6.
- max-stack: per-commit attribution of −8.4 % / −6.3 % (base vs max all-off vs max all-on); streamed decode matrix; E2 race gate; real-prose window.
- expert-gemv, head-q6k, hostgap, w2-5: all M5 identity/timing gates.
- B draft head: sustained-use ABBA with a long-answer prompt; `/status` exposure of `mtp.vocabulary_cycles`.
- ple-ssd-b4: M5 stage 1/2 (B4 streamed vs resident), then the fix lands in #410.
- PF3: Apple10-bitwise rewrite before any timing.

## How to reproduce

In-tree (verified with `git ls-files` on `flashnext-perf-all` `46c3972114`): baseline env `dev/scripts/flashnext-perf-all.env`, stack env `dev/scripts/flashnext-max.env`, decode profiler `dev/benchmarks/decode_profile.mm`, dispatch floor `dev/benchmarks/dispatch_floor_benchmark.mm`, shape sweep `dev/benchmarks/flashnext-shapes.sh`, projection/MoE/attention microbenchmarks `dev/benchmarks/gguf_projection_benchmark.mm`, `gguf_moe_benchmark.mm`, `attention_sweep.mm`, ABBA/regression drivers `dev/benchmarks/abba.py`, `backend_regression.py`, `http_regression.py`, flag tables `docs/flashnext-perf-all.md`, `docs/flashnext-max-stack.md`. Protocol for decode comparisons (bracketed reference, quiet mode, taint/rerun): [R20]. The M5 queue runner (`m5-block.py`) and the receipt archive live outside the repository.

## Receipts

Paths are relative to the maintainer's local receipt archive (outside the repository; available on request).

- R1 `approvals-20261015/FORK-PUSHES.md` §1 (M4 63 checks)
- R2 `max-stack/perf-all/m5/review-one-by-one-3/perf-off.json`
- R3 `…/review-one-by-one-3/perf-device-selftest.json`, `perf-device-selftest-*/kernel-selftest-sv1-pf3.json`
- R4 `…/review-one-by-one-3/perf-WP2-k64-identity.json`
- R5 `…/review-one-by-one-3/perf-WP1-identity.json`
- R6 `…/review-one-by-one-3/perf-dispatch-*/dispatch-gate.json` (`saved_us 0.617806`, `e2_prerequisite_pass false`)
- R7 `…/review-one-by-one-3/perf-WP2-sweep-*/shape-sweep.log`
- R8 `…/review-one-by-one-3/perf-WP1-profile.json` (RUNNING)
- R9 `max-stack/RESULTS-MAXD-20261015.md`; raw `max-stack/m5/block-2591fd56-maxd/d-native-*/backend-regression.json`
- R10 `max-stack/RESULTS-ARMS-20261015.md`; raw `max-stack/m5/block-2591fd56-arms1d/d-arms-*/analysis-R2.json`
- R11 `kernels/WP5-STATUS.md`
- R12 `speed/w2-6/STATUS.md` and `speed/w2-6/receipts/`
- R13 `speed/w2-5/STATUS.md`, `speed/w2-5/m4-receipts/validation.json`
- R14 `splash-ple-ssd/b4-fix/STATUS.md`, `b4-fix/m4-logs/b4-identity-runner.log`
- R15 `speed/w3-hostgap/STATUS.md`, `speed/w3-hostgap/receipts/m4-check-receipt.json`
- R16 `gpu-profile/runs/slot1-20261015-072525/decode-profile.txt`
- R17 `speed/w2-6/m5-20261015-110934/shapes-off.log`
- R18 `speed/b-draft-head/m5-20261015-115545/canary.log`
- R19 `pr-def/STATUS-D-ORACLE.md` (06:35 entry)
- R20 `max-stack/DECODE-PROTOCOL.md`

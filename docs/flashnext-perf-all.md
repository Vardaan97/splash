# Unified Flash-Next experiments

`flashnext-perf-all` is based on #253 `174e62e`, with #410 (`bac17ac`),
A (`3217e3d`) and B (`d1934c3`) applied in stack order, the complete
`2591fd56` max-stack history, WP1/WP2 and PF1/PF2/PF6/PF3/PF7.
The earlier `e9f5018` snapshot is only a history prefix: all twelve missing
follow-ups, including the E2 scratch, prefetch notification and weak cache
ownership fixes, are restored before any build. D's test-only `4063ec1a`
oracle diagnosis is carried without changing its production capture code.

Source `dev/scripts/flashnext-perf-all.env` for the baseline. Every optional
package remains off. C is carried for F's planner structures, but `rule` is
the default and cost timing/selection runs only with an explicit opt-in.
The exact old-to-new mapping and M4 receipt live in
`~/Developer/max-stack/STATUS-PERF-ALL.md`. No M5 performance claim is made.

| Package | Existing flag / selection | Default | Identity / focused check |
| --- | --- | --- | --- |
| C | `SPLASH_MTP_DEPTH=cost` | rule | X; planner/runtime, greedy tokens |
| lane | `SPLASH_MTP_DEPTH=lane`, `SPLASH_MTP_WORTHWHILE` | rule; .30/.35/.40/.45 | B1 B at .30; B4 X; planner/runtime |
| B size | `SPLASH_MTP_DRAFT_VOCAB=0/32768/49152/65536` | 32768 | X; vocabulary/context tests |
| D | `SPLASH_GDN_CHECKPOINT_TOKENS=8192/16384` | 0 | Same-partition fp32 B; repartition tolerance / bf16 N; engine/GDN oracle |
| E / E2 | `SPLASH_DECODE_DISPATCH=1/2`, site mask / serial or seeded shuffle | 0 | B; shader-validated backend/projection/runtime/Qwen4/MoE |
| head trims | `SPLASH_HOST_WINS=eh/kv/1` | off | B; projection/runtime |
| PLE prefetch | `SPLASH_PLE_PREFETCH=1` | 0 | B; streamed runtime |
| prepared cache | `SPLASH_GRAPH_CACHE=1/verify` | off | B; backend payload/reclaim/runtime; fresh Runtime graph remains |
| F | `SPLASH_MTP_LOOKUP=1`, buckets / scan / reprobe | 0; buckets 0, scan 256, reprobe 0 | X; lookup/planner/runtime; rule/lane supported, reprobe cost-only |
| WP1 | `SPLASH_HYPER_FUSION=1/2/4/5/7` | 0 | B; hyper/producer memcmp, projection/MoE/runtime |
| WP2 | `SPLASH_SMALL_PROJ_TILE=base/k64/k128/sg1/k64sg1`; `SPLASH_SMALL_PROJ=1` | unset / 0 | B; forced projection matrix/floor; automatic rule remains disabled |
| PF1 | `SPLASH_MOE_PREFILL_TILE=m64/m32/auto` | unset | B; self-test and live-route MoE matrix; m64 requires device check |
| PF2 | `SPLASH_PREFILL_SLICES=2..48` | 0 | B; segment/cancel/state/KV/first-token/runtime/engine |
| PF6 | `SPLASH_PREFILL_SMALL_SPLIT=33..256` | 0 | N-det; fixed-group projection/guards; paired quality still required |
| PF3 | `SPLASH_ATTN_PIPE=qk2/prob2/rows4`, `SPLASH_ATTN_STATS=1` | unset / 0 | B; attention variants; Apple9 fallback is not Apple10 proof |
| PF7 | `SPLASH_PREFILL_QSA_COMPACT=validate`; capture path | unset | N; compact tolerance/repeat plus mandatory long-context quality. `=1` still refuses promotion |
| CACHE | `--persistent-cache`, `FN2_CACHE_SIZE=16G` in copied launcher | off; FN2 size 0 | B output; disk hit/restart/deep-edit replay, D isolated with `--no-grid` |

The stale B4 planner fixture expected 2/3/7/4 from rates .5/.7/.99/.8
at threshold .45, which actually produces 2/2/7/3 (B1 .30 produces 2/3/7/5). Restoring `2b853657` fixes the fixture;
the extra regression retains those old rates and independently checks the
reach rule across every width threshold and caps 1/2/3/4/5/7. No production
threshold is weakened to make the test pass.

Integration keeps WP1 hyper-combine and HOST_WINS KV-only attention in one
function; PF2 slices preserve the caller-profile lookup/prefetch fixture;
WP2 and PF6 policies are captured in one Linear constructor; PF3 and PF7
have distinct test binaries and combined workspace bounds. Device self-test
callbacks register through Metal's layer-neutral registry. All original
shader modules precede added modules. PF3/PF7 retain their device/quality
refusal gates, and D stays off pending the instrumented M5 review.

The copied `dev/scripts/splash-fn2` never replaces the installed launcher,
uses this worktree's prebuilt binary, and persists only when explicitly
requested. `dev/gates/m5-cache-ttft.py --no-grid` reuses the existing CACHE
replay with memory/disk ABBA; it starts servers only in an authorized M5
window. Optional prefill co-batching and host-only attention-window work
remain the documented design/deferred items from max-stack.

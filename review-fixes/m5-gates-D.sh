#!/bin/bash
# Run only inside the main session's reserved M5 GPU window, with prebuilt inputs.
set -euo pipefail
: "${SPLASH_M5_WINDOW:?Set to 1 in the reserved M5 window}"
[ "$SPLASH_M5_WINDOW" = 1 ]
case "$(sysctl -n machdep.cpu.brand_string)" in *M5*) ;; *) echo 'M5 only' >&2; exit 2 ;; esac
: "${CANDIDATE_REPO:?Prebuilt D checkout (D plus #410 when streaming is needed)}"
: "${BASELINE_REPO:?Prebuilt #253 checkout (same #410 overlay when streaming)}"
: "${MODEL_ROOT:?Installed Flash-Next model root}"
MODEL=${MODEL:-Splash-128}
[ -d "$MODEL_ROOT/mtp" ] || { echo 'MTP head required for the carried-stream gate' >&2; exit 2; }
export SPLASH_MTP=1
CANDIDATE_BINARY=${CANDIDATE_BINARY:-$CANDIDATE_REPO/build/splash}
BASELINE_BINARY=${BASELINE_BINARY:-$BASELINE_REPO/build/splash}
PYTHON=${PYTHON:-$CANDIDATE_REPO/.venv/bin/python}
ORACLE=${ORACLE:-$CANDIDATE_REPO/build/engine-tests/model-runtime-oracle}
METALLIB=${METALLIB:-$CANDIDATE_REPO/build/splash.metallib}
WARMUP=${WARMUP:-900}
[[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 900 ]] || { echo "WARMUP must be >=900 seconds" >&2; exit 2; }
GATE_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS=${RESULTS:-$GATE_DIR/m5-$(date -u +%Y%m%d-%H%M%S)}
mkdir "$RESULTS"
git -C "$CANDIDATE_REPO" merge-base --is-ancestor 1011be15044450d778a80018316e3eb9d4fed836 HEAD
git -C "$BASELINE_REPO" merge-base --is-ancestor e5ce29f80affff9ba98722f35913f77a3bac9b78 HEAD
if git -C "$BASELINE_REPO" merge-base --is-ancestor 26e34b01f0b154a36f1012c222b60890f67e0f87 HEAD 2>/dev/null; then
  echo 'Baseline contains D; supply a separate #253 baseline' >&2
  exit 2
fi
for task_repo in "$CANDIDATE_REPO" "$BASELINE_REPO"; do
  git -C "$task_repo" rev-parse HEAD
  git -C "$task_repo" status --short
done > "$RESULTS/commits.txt"
sysctl hw.memsize machdep.cpu.brand_string > "$RESULTS/host.txt"
shasum -a 256 "$CANDIDATE_BINARY" "$BASELINE_BINARY" "$ORACLE" "$METALLIB" > "$RESULTS/binaries.sha256"
# D alone contains no streaming implementation. Supply matched, prebuilt
# D+#410 and #253+#410 inputs to exercise this path; never silently skip it.
PLE_SSD_BYTES=${PLE_SSD_BYTES:-0}
oracle_args=()
server_args=()
if [ "$PLE_SSD_BYTES" != 0 ]; then
  oracle_args=(--ple-ssd-cache "$PLE_SSD_BYTES")
  server_args=(--server-arg=--ple-ssd-cache --server-arg="$PLE_SSD_BYTES")
fi
# Exact complete-cell restore; fp32 mid-command splits must be bitwise.
# bf16 uses matching partitions because extra splits are class N. Run every KV/state pairing.
for task_state in fp32 bf16; do
  for task_kv in int8 bf16; do
    "$ORACLE" "$METALLIB" "$MODEL_ROOT" --ane-ffn-share 0 \
      --gdn-checkpoints-only --state-format "$task_state" --kv-format "$task_kv" \
      "${oracle_args[@]}" > "$RESULTS/oracle-$task_state-$task_kv.log" 2>&1
  done
done
SPLASH_MTP=0 "$ORACLE" "$METALLIB" "$MODEL_ROOT" --ane-ffn-share 0 \
  --gdn-checkpoints-only "${oracle_args[@]}" > "$RESULTS/oracle-no-mtp.log" 2>&1
# Alternate case order at each spacing. Each experiment runs five growing
# turns, a deep edit around 105K, and a fresh cold edited reference.
task_failures=0
for task_grid in 8192 16384; do
  for task_order in off-on on-off; do
    if ! "$PYTHON" "$GATE_DIR/measure_m5.py" --repo "$CANDIDATE_REPO" \
      --binary "$CANDIDATE_BINARY" --baseline-repo "$BASELINE_REPO" \
      --baseline-binary "$BASELINE_BINARY" --model-root "$MODEL_ROOT" --model "$MODEL" \
      --warmup "$WARMUP" --target-tokens 170000 --edit-depth 105000 --grid "$task_grid" --order "$task_order" \
      --output "$RESULTS/grid-$task_grid-$task_order" "${server_args[@]}"; then
      task_failures=$((task_failures + 1))
    fi
  done
done
# bf16 + grid is class N: record identity diagnostically, never gate it.
if ! "$PYTHON" "$GATE_DIR/measure_m5.py" --repo "$CANDIDATE_REPO" \
  --binary "$CANDIDATE_BINARY" --baseline-repo "$BASELINE_REPO" \
  --baseline-binary "$BASELINE_BINARY" --model-root "$MODEL_ROOT" --model "$MODEL" \
  --warmup "$WARMUP" --target-tokens 30000 --edit-depth 12000 \
  --informational-identity --grid 8192 --state-format bf16 --output "$RESULTS/bf16-state" "${server_args[@]}"; then
  echo "INFO bf16 diagnostic failed; inspect receipt (not an identity gate)" >&2
fi
# Restart/SSD interaction with the retained grid and a complete KV chain.
if ! "$PYTHON" "$GATE_DIR/m5-persistent-D.py" --repo "$CANDIDATE_REPO" \
  --binary "$CANDIDATE_BINARY" --model-root "$MODEL_ROOT" --model "$MODEL" \
  --output "$RESULTS/persistent" "${server_args[@]}"; then
  task_failures=$((task_failures + 1))
fi
if [ "$task_failures" != 0 ]; then
  printf 'FAIL %s D experiments; receipts: %s\n' "$task_failures" "$RESULTS" >&2
  exit 1
fi
printf 'PASS all D gates; receipts: %s\n' "$RESULTS"

#!/bin/bash
# Run only in the main session's exclusive M5 GPU window. Never auto-build.
set -euo pipefail
: "${E_M5_WINDOW:?Set E_M5_WINDOW=1 in the authorized M5 window}"
[[ "$E_M5_WINDOW" == 1 ]] || exit 2
: "${MODEL_ROOT:?Installed, pinned Flash-Next model root required}"
BASE=${BASE:-$HOME/src/splash-stack-e5c}
CAND=${CAND:-$HOME/src/splash-dispatch-e}
OUT=${OUT:-$HOME/Developer/pr-def/E/m5/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
system_profiler SPHardwareDataType > "$OUT/hardware.txt"
for repo in "$BASE" "$CAND"; do
  git -C "$repo" rev-parse HEAD
  test -x "$repo/build/engine-tests/decode-profile"
  test -x "$repo/build/engine-tests/model-runtime-oracle"
  shasum -a 256 "$repo/build/engine-tests/decode-profile" "$repo/build/splash.metallib"
done > "$OUT/shas.txt"
shasum -a 256 "$MODEL_ROOT/model.json" > "$OUT/model-manifest.sha256"
pmset -g custom > "$OUT/power-mode.txt"
printf 'adaptive=%s drafts=%s depth=%s vocab=%s\n' \
  "${SPLASH_MTP_ADAPTIVE:-default}" "${SPLASH_MTP_DRAFTS:-default}" \
  "${SPLASH_MTP_DEPTH:-default}" "${SPLASH_MTP_DRAFT_VOCAB:-default}" > "$OUT/draft-env.txt"
# The enhanced profiler is identical in both arms; the switch alone changes
# command ordering. Compare candidate-off against the base separately using
# m5-real-E.sh. The old base profiler cannot stream PLE or read token files.
: "${PROMPT_DIR:?Directory containing 512.tokens, 32768.tokens and 131072.tokens}"
PLE_CACHE_BYTES=${PLE_CACHE_BYTES:-2147483648}
WARMUP=${WARMUP:-900}
[[ $WARMUP =~ ^[0-9]+$ && $WARMUP -ge 900 ]] || { echo "WARMUP must be >=900 seconds" >&2; exit 2; }
# Discard sustained load before any timed arm; these are attribution-only profiles.
warm_until=$((SECONDS + WARMUP))
warm_round=0
while (( SECONDS < warm_until )); do
  env SPLASH_DECODE_DISPATCH=0 "$CAND/build/engine-tests/decode-profile" \
    "$CAND/build/splash.metallib" "$MODEL_ROOT" --prompt-file "$PROMPT_DIR/512.tokens" \
    --ple-ssd-cache "$PLE_CACHE_BYTES" --cycles "${CYCLES:-8}" --profile-dispatches 0 \
    --kv-format int8 --state-format fp32 > "$OUT/warm-$warm_round.txt" 2>&1
  warm_round=$((warm_round + 1))
done
for ctx in 512 32768 131072; do
  prompt="$PROMPT_DIR/$ctx.tokens"
  test -f "$prompt"
  actual=$(wc -w < "$prompt" | tr -d ' ')
  [[ "$actual" == "$ctx" ]] || { echo "wrong token count: $prompt ($actual)" >&2; exit 2; }
  shasum -a 256 "$prompt" >> "$OUT/prompts.sha256"
  args=("$CAND/build/splash.metallib" "$MODEL_ROOT" --prompt-file "$prompt"
        --ple-ssd-cache "$PLE_CACHE_BYTES" --cycles "${CYCLES:-8}"
        --kv-format int8 --state-format fp32)
  # B1..B4 are all exercised, including the requested B1/B4 matrix.
  for arm in off1 on1 on2 off2; do
    enabled=1
    case "$arm" in off*) enabled=0;; esac
    env SPLASH_DECODE_DISPATCH=$enabled "$CAND/build/engine-tests/decode-profile" \
      "${args[@]}" --profile-dispatches 0 > "$OUT/$ctx-$arm.txt" 2>&1
  done
  # Isolated kernel attribution is a separate pass, not the idle-gap metric.
  env SPLASH_DECODE_DISPATCH=0 "$CAND/build/engine-tests/decode-profile" \
    "${args[@]}" --profile-dispatches 1 > "$OUT/$ctx-attribution.txt" 2>&1
  if [[ ${TRACE:-0} == 1 ]]; then
    for enabled in 0 1; do
      xcrun xctrace record --env "SPLASH_DECODE_DISPATCH=$enabled" \
        --template "${TRACE_TEMPLATE:-Metal System Trace}" \
        --output "$OUT/$ctx-$enabled.trace" \
        --target-stdout "$OUT/$ctx-$enabled-traced.txt" --launch -- \
        "$CAND/build/engine-tests/decode-profile" "${args[@]}" --profile-dispatches 0
      xcrun xctrace export --input "$OUT/$ctx-$enabled.trace" --toc \
        --output "$OUT/$ctx-$enabled-toc.xml"
    done
  fi
done
# Exact TOKENS-line comparison supplements the oracle's internal checks.
# Default streamed PLE fits the intended large model; add 0 to ORACLE_CACHES
# only for a model that also fits resident. Never silently skip a failed arm.
if [[ ${ORACLE:-1} == 1 ]]; then
  for state in fp32 bf16; do
    for cache in ${ORACLE_CACHES:-2147483648}; do
      args=(--state-format "$state")
      if [[ $cache != 0 ]]; then args+=(--ple-ssd-cache "$cache"); fi
      for arm in base1 off on1 on2 base2; do
        repo=$CAND; enabled=1
        case "$arm" in base*) repo=$BASE; enabled=0;; off) enabled=0;; esac
        log="$OUT/oracle-$arm-$state-$cache.txt"
        env SPLASH_DECODE_DISPATCH=$enabled "$repo/build/engine-tests/model-runtime-oracle" \
          "$repo/build/splash.metallib" "$MODEL_ROOT" "${args[@]}" > "$log" 2>&1
        grep '^TOKENS' "$log" > "$log.tokens"
        if [[ $arm != base1 ]]; then
          cmp "$OUT/oracle-base1-$state-$cache.txt.tokens" "$log.tokens"
        fi
      done
    done
  done
fi
printf 'Results: %s\n' "$OUT"

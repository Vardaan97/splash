#!/bin/bash
# Run only in the main session's exclusive M5 GPU window. No builds here.
set -euo pipefail
: "${F_M5_WINDOW:?Set F_M5_WINDOW=1 in the exclusive M5 window}"
[[ "$F_M5_WINDOW" == 1 ]] || exit 2
: "${MODEL_ROOT:?Pinned installed model root required}"
: "${MODEL:?Served model ID required}"
BASE=${BASE:-$HOME/src/splash-mtp-depth-e5c}
CAND=${CAND:-$HOME/src/splash-mtp-lookup}
PYTHON=${PYTHON:-$BASE/.venv/bin/python}
OUT=${OUT:-$HOME/Developer/pr-def/F/m5/$(date +%Y%m%d-%H%M%S)}
test -x "$PYTHON"
test -x "$CAND/build/engine-tests/model-runtime-oracle"
mkdir -p "$OUT"
[[ $(git -C "$BASE" rev-parse HEAD) == $(git -C "$CAND" rev-parse 891a0fc) ]]
for repo in "$BASE" "$CAND"; do
  test -x "$repo/build/splash"
  test -x "$repo/build/engine-tests/backend-benchmark"
  git -C "$repo" rev-parse HEAD
  git -C "$repo" status --porcelain
  shasum -a 256 "$repo/build/splash" "$repo/build/splash.metallib"
done > "$OUT/builds.txt"
system_profiler SPHardwareDataType > "$OUT/hardware.txt"
cd "$CAND"
"${PYTHON:-$CAND/.venv/bin/python}" "$HOME/Developer/pr-def/F/http-gates-F.py" \
  --binary "$CAND/build/splash" --baseline-binary "$BASE/build/splash" \
  --model-root "$MODEL_ROOT" --model "$MODEL" --out "$OUT" \
  --warmup "${WARMUP:-900}" --order "${ORDER:-C-first}" --samples "${SAMPLES:-6}" --max-cache-disk 0 --kv-format int8
# Exact token vectors come from the explicit --progress JSONL journal.
# Stdout remains the separate summary document.
for arm in C1 F1 F2 C2; do
  repo=$CAND; enabled=1
  case "$arm" in C*) repo=$BASE; enabled=0;; esac
  env SPLASH_MTP_LOOKUP=$enabled SPLASH_MTP_ADAPTIVE=1 SPLASH_MTP_DEPTH=cost SPLASH_MTP_DRAFTS=7 \
    "$repo/build/engine-tests/backend-benchmark" "$repo/build/splash.metallib" "$MODEL_ROOT" \
    --scenario transcripts --samples 1 --ane-ffn-share 0 --progress "$OUT/native-$arm.jsonl" \
    > "$OUT/native-$arm.json" 2> "$OUT/native-$arm.log"
done
"${PYTHON:-$CAND/.venv/bin/python}" - "$OUT" <<'PY_NATIVE'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
reference = None
for arm in ("C1", "F1", "F2", "C2"):
    # backend_benchmark.mm Progress::complete uses kind=request for transcripts;
    # phase appears only on begin events, not on completion events.
    events = [json.loads(line) for line in (root / f"native-{arm}.jsonl").read_text().splitlines() if line.strip()]
    completed = [m for m in events if m.get("event") == "complete" and m.get("kind") == "request"]
    vectors = {(m["scenario"], m["sample"]): m["output_tokens"] for m in completed}
    assert len(vectors) == len(completed), (arm, "duplicate native transcript cases")
    assert len(vectors) == 8, (arm, "missing native transcript cases")
    assert all(vectors.values()), (arm, "empty native transcript")
    if reference is None:
        reference = vectors
    else:
        assert vectors == reference, (arm, "greedy native token vectors changed")
print("Native greedy token vectors: PASS")
PY_NATIVE
# Off/on target-vs-serial oracle, both KV and state precisions. No real models on M4.
for enabled in 0 1; do
  for kv in int8 bf16; do
    for state in fp32 bf16; do
      env SPLASH_MTP_LOOKUP=$enabled "$CAND/build/engine-tests/model-runtime-oracle" \
        "$CAND/build/splash.metallib" "$MODEL_ROOT" --kv-format "$kv" --state-format "$state" \
        > "$OUT/oracle-$enabled-$kv-$state.log" 2>&1
    done
  done
done
printf 'F gates passed: %s\n' "$OUT"

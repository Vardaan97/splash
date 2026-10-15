#!/bin/bash
# M4-only validation. Acquire only this task's slot; never move another lock.
set -euo pipefail
case "$(sysctl -n machdep.cpu.brand_string)" in
  *'Apple M4'*) ;;
  *) echo 'WP5 builds/tests must run on the M4' >&2; exit 2 ;;
esac
mode=${1:-gather}
case "$mode" in g|gather|exhaustive|whole|whole-exhaustive|check|air) ;; *) echo 'mode: g|gather|exhaustive|whole|whole-exhaustive|check|air' >&2; exit 2 ;; esac
lock="$HOME/.flashnext-build.lock-slot5"
mkdir "$lock" || { echo "slot occupied: $lock" >&2; exit 73; }
owner="wp5-$$"
printf '%s\n' "$owner" > "$lock/owner"
release() {
  if [[ $(cat "$lock/owner" 2>/dev/null) == "$owner" ]]; then
    rm "$lock/owner"
    rmdir "$lock"
  fi
}
trap release EXIT
cd "$(dirname "$0")/.."
export PATH="$HOME/src/metal-toolchain/bin:$HOME/src/metal-toolchain/Metal.xctoolchain/usr/bin:$HOME/.local/bin:$PATH"
export XCRUN="$HOME/src/metal-toolchain/bin/xcrun"
unset SPLASH_EXPERT_TILE SPLASH_MOE_GATHER_FOLD SPLASH_MOE_WHOLE_EXPERT
case "$mode" in
  g)
    nice -n 19 make -j4 build/engine-tests/expert-tile-test build/engine-tests/expert-tile-benchmark
    nice -n 19 env MTL_SHADER_VALIDATION=1 build/engine-tests/expert-tile-test build/splash.metallib
    ;;
  gather|exhaustive|whole|whole-exhaustive)
    nice -n 19 make -j4 build/engine-tests/gguf-moe-e512 build/engine-tests/gguf-moe-benchmark-e512
    option=--gather-fold
    case "$mode" in
      exhaustive) option=--gather-fold-exhaustive ;;
      whole) option=--whole-expert ;;
      whole-exhaustive) option=--whole-expert-exhaustive ;;
    esac
    nice -n 19 env -u MTL_SHADER_VALIDATION build/engine-tests/gguf-moe-e512 build/splash.metallib "$option"
    nice -n 19 env MTL_SHADER_VALIDATION=1 build/engine-tests/gguf-moe-e512 build/splash.metallib "$option"
    ;;
  check) nice -n 19 make -j4 check ;;
  air)
    : "${WP5_BASELINE_METALLIB:?set the exact e8c386d6 baseline library}"
    nice -n 19 python3 dev/tools/kernel_identity.py "$WP5_BASELINE_METALLIB" build/splash.metallib --match '*' --allow-new
    ;;
esac

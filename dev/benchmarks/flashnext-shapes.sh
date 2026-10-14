#!/bin/bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
bench=${BENCH:-$root/build/engine-tests/gguf-projection-benchmark}
lib=${LIB:-$root/build/splash.metallib}
rounds=${ROUNDS:-40}
variants=${VARIANTS:-base,k64,k128,sg1,k64sg1}
for shape in '320 10240' '10240 320' '2560 6144' '2560 5120' '10240 2560' '2560 2560' '16384 2560' '12288+512+512 2560'; do
  read -r n k <<< "$shape"
  fmt=q80
  [[ $n != *+* ]] || fmt=q80+q80+q80
  "$bench" "$lib" "$fmt" "$n" "$k" none "$rounds" "variants=$variants"
  [[ ${COOLWAIT:-0} != 1 ]] || $HOME/Developer/overnight-1014/queue/coolwait.sh 55
done

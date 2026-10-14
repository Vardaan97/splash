#pragma once
#include "metal/abi/PagedAttention.h"
#include "metal/abi/Qwen4.h"
#define QWEN4_QSA_NO_BLOCK 0xffffffffu
#define QWEN4_QSA_COMPACT_BLOCKS (SPLASH_TARGET_KV_BLOCK_TOKENS / QWEN4_QSA_BLOCK)
// One cell gathers eight selected blocks. A split owns up to 64 cells.
#define SPLASH_PREFILL_COMPACT_CELLS_PER_SPLIT 64u
#define SPLASH_PREFILL_COMPACT_SPLITS 9u
#define SPLASH_PREFILL_COMPACT_THREADS 256u
struct SplashPrefillCompactParams {
  SplashPrefillAttentionParams attention;
  Qwen4QsaMaskParams mask;
  uint32_t list_offset; // uint words after the statistics
  uint32_t list_stride; // count plus padded ids per query tile
};

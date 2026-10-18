#pragma once

#include "metal/abi/BlockIndexer.h"

#include <cstdint>

namespace splash::model {

// A row's indexer bitmap and score stride follow the served window.
constexpr uint32_t qwen4IndexerMaskWords(uint32_t contextTokens) {
  return (contextTokens / BLOCK_INDEXER_BLOCK + BLOCK_INDEXER_TILE_SLACK + 31) / 32;
}

} // namespace splash::model

#pragma once

#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct DraftContextLookupLane {
  uint32_t tokens[SPLASH_DRAFT_PROPOSAL_TOKENS];
  uint32_t count;
};

struct DraftContextLookupParams {
  DraftContextLookupLane lane[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftContextLookupParams) == 128,
              "Context lookup parameters are 128 bytes on both sides");

// Shared by the CPU merge oracle and the device dispatch. The untouched
// suffix keeps the neural proposal; the target verifies the entire prefix.
inline uint32_t draft_context_lookup_token(uint32_t position, uint32_t count,
                                           uint32_t lookup, uint32_t neural) {
  return position < count ? lookup : neural;
}

#include "metal/abi/DraftContextLookup.h"

using namespace metal;

// Replace only a greedy lane's planned lookup prefix. The usual target
// acceptance remains authoritative; sampled lanes have count zero.
kernel void draft_context_lookup(device uint *proposals [[buffer(0)]],
                                  constant DraftContextLookupParams &params [[buffer(1)]],
                                  uint lane [[threadgroup_position_in_grid]],
                                  uint position [[thread_position_in_threadgroup]]) {
  const uint index = lane * SPLASH_DRAFT_PROPOSAL_TOKENS + position;
  proposals[index] = draft_context_lookup_token(position, params.lane[lane].count,
                                                 params.lane[lane].tokens[position], proposals[index]);
}

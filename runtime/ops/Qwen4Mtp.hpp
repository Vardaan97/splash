#pragma once

// Flash-Next MTP head and draft vocabulary row operations.
#include "metal/CommandGraph.hpp"
#include "metal/abi/Qwen4Mtp.h"
#include "ops/HyperConnection.hpp"
#include "ops/Normalization.hpp"
#include "ops/Weights.hpp"

#include <cstdint>

namespace splash::ops {

struct Qwen4Mtp final {
  // The MTP head (metal/abi/Qwen4Mtp.h Qwen4MtpParams). A draft step's tokens:
  // each lane's anchor and its first limit - 1 proposals, then repeats.
  static void addMtpTokens(metal::CommandGraph &graph, metal::MetalBuffer anchors, metal::MetalBuffer proposals,
                           metal::MetalBuffer tokens, uint32_t lanes, uint32_t laneRows, uint32_t proposalsPerLane,
                           uint32_t limit);
  // One lane's (or sequence's) `rows` eh-projection inputs, rows rowOffset..
  // of each stream's `totalRows` rows of `output`.
  static void addMtpInput(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer carried,
                          metal::MetalBuffer history, const NormWeights &embedNorm, const NormWeights &hiddenNorm,
                          metal::MetalBuffer output, HyperShape shape, uint32_t rows, uint32_t limit,
                          uint32_t rowOffset, uint32_t totalRows);
  // fp32 streams [rows][streams][hidden] from the eh projection's
  // stream-major bf16 rows [streams][rows][hidden].
  static void addMtpWiden(metal::CommandGraph &graph, metal::MetalBuffer projected, metal::MetalBuffer streams,
                          HyperShape shape, uint32_t rows, uint32_t stride);
  // Each lane's argmax of its logits row `row` (`vocabulary` logits per row)
  // as its proposal `row` and, `last`, every later one. With `tokens`, the
  // logits are a draft vocabulary's, ascending token ids of which `tokens`
  // holds (uint32 each), and the proposal is the argmax's token.
  static void addMtpArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                           metal::MetalBuffer candidates, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                           uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last,
                           metal::MetalBuffer tokens = {});
  // The same over the draft vocabulary (`tokens`, `vocabulary` logits) and,
  // beside it, each lane's context rows: `contextLogits` (logitsSlot floats
  // per lane, `capacity` per row), the state lanes' `contextTokens`
  // (`capacity` each) and each lane's Qwen4MtpContextLane in `lanesInfo`.
  // Ties go toward the lower token.
  static void addMtpArgmaxContext(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                                  metal::MetalBuffer candidates, metal::MetalBuffer tokens,
                                  metal::MetalBuffer contextLogits, metal::MetalBuffer contextTokens,
                                  metal::MetalBuffer lanesInfo, uint32_t lanes, uint32_t laneRows,
                                  uint32_t vocabulary, uint32_t capacity, uint32_t logitsSlot,
                                  uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last);
  // Rows of `source` (one quantized tensor) copied to rows of `target` (one
  // of the same format and width): `count` (source row, destination row)
  // uint32 pairs in `pairs`.
  static void addCopyHeadRows(metal::CommandGraph &graph, const QuantizedSegment &source,
                              const QuantizedSegment &target, metal::MetalBuffer pairs, uint32_t count);
  // Row retained - 1 of `streams` (retained a GPU count) or, without
  // `retained`, row `rows` - 1.
  static void addMtpCarry(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer carried,
                          metal::MetalBuffer retained, HyperShape shape, uint32_t rows);


};

} // namespace splash::ops

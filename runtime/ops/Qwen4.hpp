#pragma once

// Qwen3.8-Flash-Next's QSA indexer, which picks the blocks its
// full-attention layers attend (kernels/shared/qwen4_qsa.metal). Its
// projections run through ops::Linear; model/Qwen4ExpTarget.cpp composes
// them.

#include "metal/CommandGraph.hpp"
#include "ops/HyperConnection.hpp"
#include "metal/abi/Qwen4.h"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// Where a layer's QSA index sits for a dispatch's lanes: lane l's pooled
// keys through index table tables[l], an entry per logical page
// (SplashKvPage), `offset` bytes into each entry's extent (metal/abi/
// Qwen4.h). The index sits in the KV pages' extents
// (kv::PageStorage::indexOffset), so a lane's index table is its page table.
struct QsaIndex final {
  std::span<const metal::MetalBuffer> tables;
  uint32_t offset = 0;
  // The MTP head (metal/abi/Qwen4.h Qwen4MtpParams). A draft step's tokens:
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
                          HyperShape shape, uint32_t rows);
  // Each lane's argmax of its logits row `row` (`vocabulary` logits per row)
  // as its proposal `row` and, `last`, every later one.
  static void addMtpArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                           metal::MetalBuffer candidates, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                           uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last);
  // Row retained - 1 of `streams` (retained a GPU count) or, without
  // `retained`, row `rows` - 1.
  static void addMtpCarry(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer carried,
                          metal::MetalBuffer retained, HyperShape shape, uint32_t rows);

};

// QSA of a dispatch's lanes (a verify step's, or a prefill sequence as one):
// `rows` query rows of each lane, lane l's from logical position
// firstPositions[l], which attend sparsely past topBlocks complete blocks
// of the index; bitmaps of maskWords words and scores of scoreStride per
// row, which cover each row's complete blocks and its tail.
struct QsaRows final {
  uint32_t rows = 0;
  std::span<const uint32_t> firstPositions;
  QsaIndex index;
  uint32_t topBlocks = 0;
  uint32_t maskWords = 0;
  uint32_t scoreStride = 0;
  // The MTP head (metal/abi/Qwen4.h Qwen4MtpParams). A draft step's tokens:
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
                          HyperShape shape, uint32_t rows);
  // Each lane's argmax of its logits row `row` (`vocabulary` logits per row)
  // as its proposal `row` and, `last`, every later one.
  static void addMtpArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                           metal::MetalBuffer candidates, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                           uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last);
  // Row retained - 1 of `streams` (retained a GPU count) or, without
  // `retained`, row `rows` - 1.
  static void addMtpCarry(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer carried,
                          metal::MetalBuffer retained, HyperShape shape, uint32_t rows);

};

// The QSA blocks a dispatch's lanes' rows complete, which the pool pools:
// lane l's counts[l] Qwen4QsaBlock records from record l * laneRecords of
// `records`, from the rows' raw keys (`keys`, bf16 rows of QWEN4_QSA_DIMENSION
// as the project writes them) and the tokens before lane l's rows from
// pending[l], its pending keys (QWEN4_QSA_PENDING bf16 rows).
struct QsaBlocks final {
  metal::MetalBuffer records;
  std::span<const uint32_t> counts;
  uint32_t laneRecords = 0;
  metal::MetalBuffer keys;
  std::span<const metal::MetalBuffer> pending;
  // The MTP head (metal/abi/Qwen4.h Qwen4MtpParams). A draft step's tokens:
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
                          HyperShape shape, uint32_t rows);
  // Each lane's argmax of its logits row `row` (`vocabulary` logits per row)
  // as its proposal `row` and, `last`, every later one.
  static void addMtpArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                           metal::MetalBuffer candidates, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                           uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last);
  // Row retained - 1 of `streams` (retained a GPU count) or, without
  // `retained`, row `rows` - 1.
  static void addMtpCarry(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer carried,
                          metal::MetalBuffer retained, HyperShape shape, uint32_t rows);

};

struct Qwen4 final {
  // The complete blocks of QWEN4_QSA_BLOCK tokens before the last of `rows`
  // rows from `firstPosition`, its own included.
  [[nodiscard]] static constexpr uint64_t qsaCompleteBlocks(uint64_t firstPosition, uint32_t rows) noexcept {
    return (firstPosition + rows) / QWEN4_QSA_BLOCK;
  }
  // Whether any row of `rows` from `firstPosition` attends sparsely: has
  // more than topBlocks complete blocks.
  [[nodiscard]] static bool qsaSparse(uint32_t firstPosition, uint32_t rows, uint32_t topBlocks) noexcept {
    return qsaCompleteBlocks(firstPosition, rows) > topBlocks;
  }
  // The indexer's query heads, normalized and rotated (bf16 rows of
  // QWEN4_QSA_HEADS heads), and the rows' raw keys (bf16).
  static void addQsaProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                            const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                            metal::MetalBuffer queries, metal::MetalBuffer keys, const QsaRows &rows);
  // The pooled keys of the blocks the lanes' rows complete, into the index.
  static void addQsaPool(metal::CommandGraph &graph, const QsaBlocks &blocks, const NormWeights &keyNorm,
                         metal::MetalBuffer inverseFrequencies, const QsaRows &rows);
  // The rows' scores of their complete blocks' pooled keys and, from them,
  // their bitmaps. A dispatch of one lane scores its rows in batches of the
  // score scratch's scoreRows rows; one of several lanes scores every lane's
  // rows at once, which the scratch must hold.
  static void addQsaSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer scores,
                           uint32_t scoreRows, metal::MetalBuffer mask, const QsaRows &rows);
  // A sequence's pending keys of one layer after its first `retained` rows
  // from `firstPosition`, of which `keys` holds the raw keys: those of the
  // block they leave open, from `keys` and `pendingIn`, its pending keys
  // before them.
  static void addQsaPending(metal::CommandGraph &graph, metal::MetalBuffer keys, metal::MetalBuffer pendingIn,
                            metal::MetalBuffer pendingOut, uint32_t firstPosition, uint32_t retained);
  // Each verify lane's pending keys of each of `layers` layers after the rows
  // it retained (`retained`, a count per lane the GPU wrote), from the step's
  // keys of every layer (`layerRows` rows apart, each lane's `laneRows` rows
  // from firstPositions[l] after the lanes before it) and pendingIn[l], the
  // lane's pending keys before the step; a lane's layers follow each other in
  // its pending keys.
  static void addQsaCommit(metal::CommandGraph &graph, metal::MetalBuffer keys, uint32_t layers, uint32_t layerRows,
                           uint32_t laneRows, std::span<const uint32_t> firstPositions, metal::MetalBuffer retained,
                           std::span<const metal::MetalBuffer> pendingIn,
                           std::span<const metal::MetalBuffer> pendingOut);
  // The MTP head (metal/abi/Qwen4.h Qwen4MtpParams). A draft step's tokens:
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
                          HyperShape shape, uint32_t rows);
  // Each lane's argmax of its logits row `row` (`vocabulary` logits per row)
  // as its proposal `row` and, `last`, every later one.
  static void addMtpArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                           metal::MetalBuffer candidates, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                           uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last);
  // Row retained - 1 of `streams` (retained a GPU count) or, without
  // `retained`, row `rows` - 1.
  static void addMtpCarry(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer carried,
                          metal::MetalBuffer retained, HyperShape shape, uint32_t rows);

};

} // namespace splash::ops

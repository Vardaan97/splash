#include "ops/Qwen4Mtp.hpp"
#include "ops/HyperConnection.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/QuantFormat.h"
#include "metal/abi/Qwen4Mtp.h"
#include "metal/abi/RoPE.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {
constexpr uint32_t kThreads = HYPER_CONNECTION_ROW_THREADS;
uint32_t groups(uint32_t width) { return (width + kThreads - 1) / kThreads; }
void requireShape(HyperShape shape) {
  if (!shape.hidden || !shape.streams || shape.streams > HYPER_CONNECTION_MAXIMUM_STREAMS)
    throw std::invalid_argument("invalid hyper-connection shape");
}
}

void Qwen4Mtp::addMtpTokens(metal::CommandGraph &graph, metal::MetalBuffer anchors, metal::MetalBuffer proposals,
                         metal::MetalBuffer tokens, uint32_t lanes, uint32_t laneRows, uint32_t proposalsPerLane,
                         uint32_t limit) {
  if (!lanes || !limit || limit > laneRows || limit > proposalsPerLane + 1)
    throw std::invalid_argument("invalid MTP step");
  const uint32_t rows = lanes * laneRows;
  const Qwen4MtpParams params{rows, 0, 0, laneRows, limit, 0, proposalsPerLane, 0, 0, 0, 0, 0, 0, {0, 0, 0}};
  graph.add("qwen4_mtp_tokens", {std::move(anchors), std::move(proposals), std::move(tokens)}, params,
            {groups(rows), 1, 1}, {kThreads, 1, 1});
}

void Qwen4Mtp::addMtpInput(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer carried,
                        metal::MetalBuffer history, const NormWeights &embedNorm, const NormWeights &hiddenNorm,
                        metal::MetalBuffer output, HyperShape shape, uint32_t rows, uint32_t limit,
                        uint32_t rowOffset, uint32_t totalRows) {
  requireShape(shape);
  if (!rows || !limit || limit > rows || rowOffset + rows > totalRows) throw std::invalid_argument("invalid MTP input");
  const Qwen4MtpParams params{rows,      shape.hidden, shape.streams, totalRows, limit, 0, 0, 0,
                              rowOffset, 0,            0,             0,         0,     {0, 0, 0}};
  // A single row reads no history: it binds the carried streams instead.
  graph.add("qwen4_mtp_input",
            {std::move(embedding), carried, history ? std::move(history) : carried, embedNorm.buffer,
             hiddenNorm.buffer, std::move(output)},
            params, {rows, 1, 1}, {kThreads, 1, 1});
}

void Qwen4Mtp::addMtpWiden(metal::CommandGraph &graph, metal::MetalBuffer projected, metal::MetalBuffer streams,
                        HyperShape shape, uint32_t rows, uint32_t stride) {
  requireShape(shape);
  if (!rows || stride < rows) throw std::invalid_argument("invalid MTP widen");
  const Qwen4MtpParams params{rows, shape.hidden, shape.streams, stride, 0, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0, 0}};
  graph.add("qwen4_mtp_widen", {std::move(projected), std::move(streams)}, params,
            {groups(rows * shape.width()), 1, 1}, {kThreads, 1, 1});
}

void Qwen4Mtp::addMtpArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                         metal::MetalBuffer candidates, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                         uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last,
                         metal::MetalBuffer tokens) {
  if (!lanes || !vocabulary || !candidatesPerProposal || row >= laneRows || row >= proposalsPerLane)
    throw std::invalid_argument("invalid MTP argmax");
  if (tokens && tokens.sizeBytes() < uint64_t{vocabulary} * sizeof(uint32_t))
    throw std::invalid_argument("MTP draft vocabulary is shorter than its logits");
  const Qwen4MtpParams params{lanes * laneRows, 0, 0, laneRows, 0, vocabulary, proposalsPerLane,
                              candidatesPerProposal, row, last ? 1u : 0u, 0, 0, 0, {0, 0, 0}};
  if (tokens)
    graph.add("qwen4_mtp_argmax_mapped",
              {std::move(logits), std::move(proposals), std::move(candidates), std::move(tokens)}, params,
              {lanes, 1, 1}, {1024, 1, 1});
  else
    graph.add("qwen4_mtp_argmax", {std::move(logits), std::move(proposals), std::move(candidates)}, params,
              {lanes, 1, 1}, {1024, 1, 1});
}

void Qwen4Mtp::addMtpArgmaxContext(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                                metal::MetalBuffer candidates, metal::MetalBuffer tokens,
                                metal::MetalBuffer contextLogits, metal::MetalBuffer contextTokens,
                                metal::MetalBuffer lanesInfo, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                                uint32_t capacity, uint32_t logitsSlot, uint32_t proposalsPerLane,
                                uint32_t candidatesPerProposal, uint32_t row, bool last) {
  if (!lanes || !vocabulary || !capacity || !candidatesPerProposal || row >= laneRows || row >= proposalsPerLane ||
      logitsSlot < uint64_t{laneRows} * capacity || tokens.sizeBytes() < uint64_t{vocabulary} * sizeof(uint32_t) ||
      contextLogits.sizeBytes() < (uint64_t{lanes - 1} * logitsSlot + uint64_t{laneRows} * capacity) * sizeof(float) ||
      lanesInfo.sizeBytes() < uint64_t{lanes} * sizeof(Qwen4MtpContextLane))
    throw std::invalid_argument("invalid MTP context argmax");
  const Qwen4MtpParams params{lanes * laneRows, 0, 0, laneRows, 0, vocabulary, proposalsPerLane,
                              candidatesPerProposal, row, last ? 1u : 0u, 0, capacity, logitsSlot, {0, 0, 0}};
  graph.add("qwen4_mtp_argmax_context",
            {std::move(logits), std::move(proposals), std::move(candidates), std::move(tokens),
             std::move(contextLogits), std::move(contextTokens), std::move(lanesInfo)},
            params, {lanes, 1, 1}, {1024, 1, 1});
}

void Qwen4Mtp::addCopyHeadRows(metal::CommandGraph &graph, const QuantizedSegment &source,
                            const QuantizedSegment &target, metal::MetalBuffer pairs, uint32_t count) {
  if (!count) return;
  if (source.isFloat() || source.formatId != target.formatId || source.inputSize != target.inputSize ||
      pairs.sizeBytes() < uint64_t{count} * 2 * sizeof(uint32_t))
    throw std::invalid_argument("invalid head row copy");
  const QuantFormat &format = source.format();
  const uint32_t groups = source.inputSize / 32;
  const Qwen4HeadRowsParams params{count, groups, format.meta_groups, format.plane0_bytes, format.plane1_bytes,
                                   format.meta_bytes, {0, 0}};
  graph.add("qwen4_copy_head_rows",
            {source.plane0, source.plane1Slot(), source.meta, target.plane0, target.plane1Slot(), target.meta,
             std::move(pairs)},
            params, {count, (groups + 63) / 64, 1}, {1, 64, 1});
}

void Qwen4Mtp::addMtpCarry(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer carried,
                        metal::MetalBuffer retained, HyperShape shape, uint32_t rows) {
  requireShape(shape);
  if (!rows) throw std::invalid_argument("invalid MTP carry");
  const bool counted = static_cast<bool>(retained);
  const Qwen4MtpParams params{rows, shape.hidden, shape.streams, rows, rows, 0, 0, 0, 0, 0, counted ? 1u : 0u, 0, 0,
                              {0, 0, 0}};
  graph.add("qwen4_mtp_carry", {streams, std::move(carried), counted ? std::move(retained) : streams}, params,
            {groups(shape.width()), 1, 1}, {kThreads, 1, 1});
}



} // namespace splash::ops

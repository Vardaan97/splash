#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::model {

// The history captured for a draft's speculative PLE row hints, with the
// current anchor appended before the proposals arrive.
struct Qwen4PlePrefetchInput final {
  uint64_t position;
  std::vector<uint32_t> history;
};

inline Qwen4PlePrefetchInput qwen4PlePrefetchInput(uint64_t position, std::span<const uint32_t> history,
                                                uint32_t anchor, uint32_t reach) {
  if (position > history.size() || !reach)
    throw std::invalid_argument("invalid PLE prefetch history");
  const size_t end = static_cast<size_t>(position);
  const size_t begin = end - std::min<size_t>(end, reach);
  std::vector<uint32_t> suffix(history.begin() + begin, history.begin() + end);
  suffix.push_back(anchor);
  // A proposal reaches back at most ngram - 1 tokens. The anchor plus
  // up to ngram prior tokens cover that reach, so shifting both position
  // and history to this suffix preserves every hash, even at sequence start.
  const uint64_t localPosition = suffix.size();
  return {localPosition, std::move(suffix)};
}

} // namespace splash::model

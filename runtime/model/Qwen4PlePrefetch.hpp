#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
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
  const size_t end = std::min<size_t>(history.size(), position);
  const size_t begin = end - std::min<size_t>(position, reach);
  std::vector<uint32_t> suffix(history.begin() + begin, history.begin() + end);
  suffix.push_back(anchor);
  return {position + 1, std::move(suffix)};
}

} // namespace splash::model

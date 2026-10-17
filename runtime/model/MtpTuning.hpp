#pragma once

#include <cstdlib>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>

namespace splash::model {

// Experimental MTP switches are opt-in; only an explicit 1 enables them.
inline bool mtpSwitchFromEnvironment(const char *name) {
  const char *value = std::getenv(name);
  return value && std::string_view(value) == "1";
}

// Zero depths mean the legacy batch-wide policy. An exhausted lane has an
// empty [firstRow, lastRow) expert range, while its proposal tail stays filled.
inline std::array<uint32_t, 4> mtpDraftLaneRows(std::span<const uint32_t> depths, uint32_t step, bool oneRow) {
  std::array<uint32_t, 4> rows{};
  for (uint32_t lane = 0; lane < depths.size(); ++lane)
    rows[lane] = depths[lane] && depths[lane] < step ? step - 1 : oneRow ? step : 0;
  return rows;
}

// Select mapped lanes before whole-head fallbacks reuse their logits buffer.
inline std::array<uint32_t, 4> mtpDraftHeadLanes(uint32_t vocabularyMask, uint32_t lanes) {
  std::array<uint32_t, 4> order{};
  uint32_t count = 0;
  for (bool mapped : {true, false})
    for (uint32_t lane = 0; lane < lanes; ++lane)
      if (bool(vocabularyMask & (1u << lane)) == mapped) order[count++] = lane;
  return order;
}

// Invalid thresholds leave the experiment off.
inline float mtpThresholdFromEnvironment() {
  const char *value = std::getenv("SPLASH_LIVE_VERIFY_THRESHOLD");
  if (!value || !*value) return 0;
  char *end = nullptr;
  const float threshold = std::strtof(value, &end);
  return end != value && !*end && std::isfinite(threshold) && threshold >= 0 && threshold <= 1 ? threshold : 0;
}

// Only a leading prefix counts. Validate the whole drafted chain before
// trimming so even a non-finite confidence past the cut keeps the bound.
inline uint32_t mtpLiveRows(std::span<const float> confidence, uint32_t bound, float threshold) {
  if (!(threshold > 0) || bound <= 1 || confidence.size() < bound - 1) return bound;
  for (uint32_t j = 0; j < bound - 1; ++j)
    if (!std::isfinite(confidence[j]) || confidence[j] < 0 || confidence[j] > 1) return bound;
  double product = 1;
  uint32_t live = 1;
  for (uint32_t j = 0; j < bound - 1; ++j) {
    product *= confidence[j];
    if (product < threshold) break;
    ++live;
  }
  return live;
}

} // namespace splash::model

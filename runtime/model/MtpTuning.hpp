#pragma once

#include <cstdlib>
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

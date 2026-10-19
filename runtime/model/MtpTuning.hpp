#pragma once

#include <cstdlib>
#include <array>
#include <string_view>
#include <cmath>
#include <cstdint>
#include <span>

namespace splash::model {

// Experimental MTP switches are opt-in; only an explicit 1 enables them.
inline bool mtpSwitchFromEnvironment(const char *name) {
  const char *value = std::getenv(name);
  return value && std::string_view(value) == "1";
}

// An exhausted lane has an empty [firstRow, lastRow) expert range.
// Zero means that an active lane keeps the existing draft routing bound.
inline std::array<uint32_t, 4> mtpDraftLaneRows(std::span<const uint32_t> depths, uint32_t step) {
  std::array<uint32_t, 4> rows{};
  for (uint32_t lane = 0; lane < depths.size(); ++lane)
    rows[lane] = depths[lane] && depths[lane] < step ? step - 1 : 0;
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

} // namespace splash::model

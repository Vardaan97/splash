#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::ops {

// Parsed once by startup, then copied into immutable operator plans.
struct DeterminismPolicy final {
  bool moe = false, dense = false, prefillDense = false;
  bool floatTile = false, attention = false, align = false;

  [[nodiscard]] bool enabled() const noexcept {
    return moe || dense || prefillDense || floatTile || attention || align;
  }
  [[nodiscard]] std::string parts() const {
    std::string result;
    const auto add = [&](bool on, const char *name) {
      if (!on) return;
      if (!result.empty()) result += ',';
      result += name;
    };
    add(moe, "moe"); add(dense, "dense"); add(prefillDense, "prefill");
    add(floatTile, "float"); add(attention, "attention"); add(align, "align");
    return result.empty() ? "0" : result;
  }
  [[nodiscard]] static DeterminismPolicy parse(std::string_view value) {
    if (value.empty() || value == "0") return {};
    if (value == "1" || value == "all") return {true, true, true, true, true, true};
    DeterminismPolicy p;
    for (;;) {
      const auto comma = value.find(',');
      const auto part = value.substr(0, comma);
      if (part == "moe") p.moe = true;
      else if (part == "dense") p.dense = true;
      else if (part == "prefill") p.prefillDense = true;
      else if (part == "float") p.floatTile = true;
      else if (part == "attention") p.attention = true;
      else if (part == "align") p.align = true;
      else throw std::invalid_argument("SPLASH_DETERMINISTIC requires 0, 1, all, or a comma list of moe,dense,prefill,float,attention,align");
      if (comma == std::string_view::npos) return p;
      value.remove_prefix(comma + 1);
    }
  }
  void requireCompatible(bool fp32State, bool aneFfn) const {
    if (!enabled()) return;
    if (!fp32State) throw std::invalid_argument("Deterministic mode requires --state-format fp32");
    if (aneFfn) throw std::invalid_argument("Deterministic mode requires --disable-ane (ANE FFN is enabled)");
  }
};

} // namespace splash::ops

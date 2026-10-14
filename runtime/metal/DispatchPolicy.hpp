#pragma once
#include <cstdlib>
#include <string_view>

namespace splash::metal {
// E2 sites: G1=1, G4=8, G5=16, G6=32. Other sites await scratch proofs.
inline bool decodeDispatchSite(unsigned site) {
  static const unsigned sites = [] {
    const char *mode = std::getenv("SPLASH_DECODE_DISPATCH");
    unsigned enabled = !mode ? 0 : std::string_view(mode) == "2" ? 57U
                                 : std::string_view(mode) == "1" ? 1U : 0U;
    const char *mask = std::getenv("SPLASH_DECODE_DISPATCH_SITES");
    return mask ? enabled & static_cast<unsigned>(std::strtoul(mask, nullptr, 0)) : enabled;
  }();
  return (sites & site) != 0;
}
} // namespace splash::metal

#include "model/PrefillPolicy.hpp"
#include <charconv>
#include <cstdlib>
#include <string_view>
namespace splash::model {
uint32_t prefillSlices() noexcept {
  static const uint32_t count = [] {
    const char *env = std::getenv("SPLASH_PREFILL_SLICES");
    if (!env) return 0u;
    const std::string_view text(env);
    uint32_t value = 0;
    const auto p = std::from_chars(text.data(), text.data() + text.size(), value);
    return p.ec == std::errc{} && p.ptr == text.data() + text.size() && value >= 2 && value <= 48 ? value : 0u;
  }();
  return count;
}
}

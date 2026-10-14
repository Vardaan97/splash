#include "ops/PrefillPolicy.hpp"
#include <charconv>
#include <cstdlib>
#include <string_view>
namespace splash::ops {
uint32_t smallPrefillRows() noexcept {
  static const uint32_t rows = [] {
    const char *env = std::getenv("SPLASH_PREFILL_SMALL_SPLIT");
    if (!env) return 0u;
    const std::string_view text(env);
    uint32_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
                   value >= 33 && value <= 256 ? value : 0u;
  }();
  return rows;
}
}

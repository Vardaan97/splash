#include "metal/MoePrefillPolicy.hpp"
#include <cstdlib>
#include <string_view>
namespace splash::metal {
MoePrefillChoice moePrefillChoice() {
  static const auto choice = [] {
    const char *v = std::getenv("SPLASH_MOE_PREFILL_TILE");
    if (!v) return MoePrefillChoice::Off;
    const std::string_view value(v);
    if (value == "m64") return MoePrefillChoice::M64;
    if (value == "auto") return MoePrefillChoice::Auto;
    if (value == "m32") return MoePrefillChoice::M32;
    return MoePrefillChoice::Off;
  }();
  return choice;
}
bool moePrefillSelfTestRequested() {
  static const bool requested = [] {
    const char *v = std::getenv("SPLASH_KERNEL_SELFTEST");
    return v && std::string_view(v) == "1";
  }();
  // An enabled m64 path may never bypass its device test.
  return requested || moePrefillChoice() == MoePrefillChoice::M64 ||
         moePrefillChoice() == MoePrefillChoice::Auto;
}
}

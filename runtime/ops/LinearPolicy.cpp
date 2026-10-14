#include "Linear.hpp"
#include "ops/PrefillPolicy.hpp"

#include <cstdlib>
#include <cstring>

namespace splash::ops {

Linear::Linear(const DeviceCapabilities &device) noexcept
    : smallPrefillRows_(smallPrefillRows()), family_(gpuFamilyClass(device.appleGpuFamily)),
      gpuCores_(plannedGpuCores(device)) {
  const char *enabled = std::getenv("SPLASH_SMALL_PROJ");
  smallProjectionEnabled_ = enabled && std::strcmp(enabled, "1") == 0;
  if (const char *forced = std::getenv("SPLASH_SMALL_PROJ_TILE")) {
    smallProjectionVariant_ = 6;
    constexpr const char *names[] = {"base", "k64", "k128", "sg1", "k64sg1"};
    for (uint32_t i = 0; i < 5; ++i)
      if (std::strcmp(forced, names[i]) == 0) smallProjectionVariant_ = i + 1;
  }
}

} // namespace splash::ops

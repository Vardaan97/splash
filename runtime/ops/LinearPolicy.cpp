#include "Linear.hpp"
#include "ops/PrefillPolicy.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace splash::ops {

Linear::Linear(const DeviceCapabilities &device) noexcept
    : smallPrefillRows_(smallPrefillRows()), family_(gpuFamilyClass(device.appleGpuFamily)),
      gpuCores_(plannedGpuCores(device)) {
  if (const char *head = std::getenv("SPLASH_HEAD_Q6K")) {
    // Explicit experiment only; unknown values leave the baseline selected.
    for (uint32_t columns : {32u, 64u, 128u, 256u})
      for (uint32_t depth : {0u, 1u, 2u, 4u, 8u}) {
        if (columns * depth > 512 || (columns == 256 && depth)) continue;
        char name[24];
        std::snprintf(name, sizeof(name), "c%ud%u", columns, depth);
        if (std::strcmp(name, head) == 0 || (std::strcmp(head, "1") == 0 && columns == 128 && depth == 0)) {
          headQ6kColumns_ = columns;
          headQ6kDepth_ = depth;
        }
      }
  }
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

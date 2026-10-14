#pragma once
#include <cstdint>
namespace splash::ops {
// Read once when the process first constructs Linear; invalid values disable.
uint32_t smallPrefillRows() noexcept;
}

#pragma once

#include <cstdlib>
#include <string_view>

namespace splash::model {

// Experimental MTP switches are opt-in; only an explicit 1 enables them.
inline bool mtpSwitchFromEnvironment(const char *name) {
  const char *value = std::getenv(name);
  return value && std::string_view(value) == "1";
}

} // namespace splash::model

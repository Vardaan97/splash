#include "metal/KernelSelfTest.hpp"
#include <array>
#include <stdexcept>
namespace splash::metal {
namespace {
struct Registry { std::array<KernelSelfTest, 4> tests{}; size_t count = 0; };
Registry &registry() { static Registry value; return value; }
} // namespace
bool registerKernelSelfTest(KernelSelfTest test) {
  auto &r = registry();
  if (!test || r.count == r.tests.size()) throw std::invalid_argument("invalid kernel self-test registration");
  r.tests[r.count++] = test;
  return true;
}
void runKernelSelfTests(MetalBackend &backend) {
  const auto &r = registry();
  for (size_t i = 0; i < r.count; ++i) r.tests[i](backend);
}
} // namespace splash::metal

#pragma once
namespace splash::metal {
class MetalBackend;
using KernelSelfTest = void (*)(MetalBackend &);
// Policies register at startup; each callback remains responsible for its
// opt-in flag and device gate. The backend does not depend on ops policies.
bool registerKernelSelfTest(KernelSelfTest test);
void runKernelSelfTests(MetalBackend &backend);
} // namespace splash::metal

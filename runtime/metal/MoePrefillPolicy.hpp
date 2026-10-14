#pragma once
#include <cstdint>
namespace splash::metal {
class MetalBackend;
// Environment is read once, outside graph construction and execution.
enum class MoePrefillChoice : uint8_t { Off, M32, M64, Auto };
MoePrefillChoice moePrefillChoice();
bool moePrefillSelfTestRequested();
bool moePrefillSelfTest(MetalBackend &backend);
}

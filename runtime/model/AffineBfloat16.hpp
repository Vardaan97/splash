#pragma once

#include <cstdint>
#include <span>

namespace splash::model {

// Pure CPU preparation, no backend or allocation. Every 128 input bytes
// hold one group of 64 BF16 values; output is 36 (af4g64) or 68 (af8g64)
// bytes per group. Rejects partial groups, wrong sizes and non-finite inputs.
void quantizeAffineBfloat16(std::span<const uint8_t> weights, uint32_t format, std::span<uint8_t> native);
// Used by stored-row BF16 preparation without changing any input bits.
void validateFiniteBfloat16(std::span<const uint8_t> weights);

} // namespace splash::model

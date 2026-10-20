#include "model/AffineBfloat16.hpp"

#include "metal/abi/QuantFormat.h"
#include "model/Bfloat16.hpp"
#include "model/GgufFile.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

namespace splash::model {
namespace {

// The BF16 nearest a finite value, ties to even.
uint16_t nearestBfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// A group of 64 BF16 weights, with the original four-bit arithmetic and
// packing preserved, in float, into a native af4g64
// block: the range runs from the minimum to the maximum or 0, whichever is
// greater, the end of the range farther from zero is the bias, the scale is
// adjusted so that 0 falls on a code unless that code is 0, each code is
// round((w - bias) / scale), halves away from zero, clamped to [0, 15], and
// scale and bias are stored as BF16. The loops have no early exit, so they
// vectorize; the sign of a zero minimum or maximum, which the order of a
// vector reduction may change, does not reach any output.
void quantizeGroup4(const uint8_t *weights, uint8_t *block) {
  std::array<float, 64> w;
  bool finite = true;
  for (size_t i = 0; i < w.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, weights + 2 * i, 2);
    w[i] = widenBfloat16(bits);
    finite &= std::isfinite(w[i]);
  }
  if (!finite) throw GgufError("non-finite weight in a BF16 projection");
  float minimum = w[0], maximum = w[0];
  for (float value : w) {
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  maximum = std::max(0.0F, maximum);
  const bool minimumEdge = std::fabs(minimum) > std::fabs(maximum);
  float step = std::max((maximum - minimum) / 15.0F, 1e-7F);
  if (!minimumEdge) step = -step;
  const float edge = minimumEdge ? minimum : maximum;
  const float q0 = std::round(edge / step);
  float offset = 0.0F;
  if (q0 != 0.0F) {
    step = edge / q0;
    offset = edge;
  }
  std::array<uint8_t, 64> code;
  for (size_t i = 0; i < w.size(); ++i)
    code[i] = static_cast<uint8_t>(std::clamp(std::round((w[i] - offset) / step), 0.0F, 15.0F));
  const uint16_t scale = nearestBfloat16(step), bias = nearestBfloat16(offset);
  std::memcpy(block, &scale, 2);
  std::memcpy(block + 2, &bias, 2);
  for (size_t i = 0; i < w.size(); i += 2) block[4 + i / 2] = static_cast<uint8_t>(code[i] | code[i + 1] << 4);
}

// The eight-bit recipe uses 255 intervals and one unsigned byte per code.
// Metadata is BF16; all arithmetic before that rounding is float32.
void quantizeGroup8(const uint8_t *weights, uint8_t *block) {
  std::array<float, 64> w;
  for (size_t i = 0; i < w.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, weights + 2 * i, 2);
    w[i] = widenBfloat16(bits);
    if (!std::isfinite(w[i])) throw GgufError("non-finite weight in a BF16 projection");
  }
  const float minimum = *std::min_element(w.begin(), w.end());
  const float maximum = std::max(0.0F, *std::max_element(w.begin(), w.end()));
  const bool minimumEdge = std::fabs(minimum) > std::fabs(maximum);
  float step = std::max((maximum - minimum) / 255.0F, 1e-7F);
  if (!minimumEdge) step = -step;
  const float edge = minimumEdge ? minimum : maximum;
  const float q0 = std::round(edge / step);
  if (!std::isfinite(step) || !std::isfinite(q0))
    throw GgufError("BF16 projection range exceeds affine float32 arithmetic");
  float offset = 0.0F;
  if (q0 != 0.0F) {
    step = edge / q0;
    offset = edge;
  }
  const uint16_t scale = nearestBfloat16(step), bias = nearestBfloat16(offset);
  std::memcpy(block, &scale, 2);
  std::memcpy(block + 2, &bias, 2);
  for (size_t i = 0; i < w.size(); ++i)
    block[4 + i] = static_cast<uint8_t>(std::clamp(std::round((w[i] - offset) / step), 0.0F, 255.0F));
}

} // namespace

void validateFiniteBfloat16(std::span<const uint8_t> weights) {
  if (weights.size() % 2) throw GgufError("a BF16 projection contains a partial value");
  for (size_t at = 0; at < weights.size(); at += 2) {
    uint16_t bits;
    std::memcpy(&bits, weights.data() + at, 2);
    if ((bits & 0x7F80) == 0x7F80) throw GgufError("non-finite weight in a BF16 projection");
  }
}

void quantizeAffineBfloat16(std::span<const uint8_t> weights, uint32_t format, std::span<uint8_t> native) {
  if (format != GGUF_FMT_AF4G64 && format != GGUF_FMT_AF8G64)
    throw GgufError("a BF16 weight is quantized into af4g64 or af8g64");
  const size_t block = kQuantFormats[format].block_bytes;
  if (weights.size() % 128 || native.size() != weights.size() / 128 * block)
    throw GgufError("a BF16 weight is quantized into whole affine groups of 64");
  for (size_t group = 0; group < weights.size() / 128; ++group) {
    if (format == GGUF_FMT_AF4G64) quantizeGroup4(weights.data() + group * 128, native.data() + group * block);
    else quantizeGroup8(weights.data() + group * 128, native.data() + group * block);
  }
}

} // namespace splash::model

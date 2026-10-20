#include "TestChecks.hpp"
#include "TestFiles.hpp"

#include <algorithm>
#include "model/AffineBfloat16.hpp"
#include "metal/abi/QuantFormat.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using splash::test::require;
using splash::test::rejects;
using splash::model::quantizeAffineBfloat16;
using splash::model::validateFiniteBfloat16;

namespace {

std::vector<uint8_t> bytes(std::span<const uint16_t> bits) {
  std::vector<uint8_t> result(bits.size_bytes());
  std::memcpy(result.data(), bits.data(), result.size());
  return result;
}

// Literal native-block fixtures: metadata bits and independently chosen
// codes. These check byte order, unsigned codes and adjacent-nibble packing.
void fixture(uint32_t format, std::array<uint16_t, 64> weights, uint16_t scale, uint16_t bias,
             std::array<uint8_t, 64> codes) {
  const auto input = bytes(weights);
  std::vector<uint8_t> expected(kQuantFormats[format].block_bytes, 0xCC), actual(expected.size(), 0xFF);
  std::memcpy(expected.data(), &scale, 2);
  std::memcpy(expected.data() + 2, &bias, 2);
  for (size_t i = 0; i < 64; ++i) {
    if (format == GGUF_FMT_AF8G64) expected[4 + i] = codes[i];
    else if (i % 2 == 0) expected[4 + i / 2] = uint8_t(codes[i] | codes[i + 1] << 4);
  }
  quantizeAffineBfloat16(input, format, actual);
  require(actual == expected, std::string(kQuantFormats[format].name) + " native fixture differs");
}

void testNativeFixtures() {
  std::array<uint16_t, 64> weights{};
  std::array<uint8_t, 64> codes{};
  for (const auto format : {GGUF_FMT_AF4G64, GGUF_FMT_AF8G64}) {
    // Zero and signed zero have exactly the same default recipe.
    fixture(format, weights, 0xB3D7, 0, codes);
    weights.fill(0x8000);
    fixture(format, weights, 0xB3D7, 0, codes);
    weights.fill(0x3F40); // +0.75, positive constant
    fixture(format, weights, 0xB3D7, 0x3F40, codes);
    weights.fill(0xBF40); // -0.75, negative constant includes zero in its range
    fixture(format, weights, format == GGUF_FMT_AF4G64 ? 0x3D4D : 0x3B41, 0xBF40, codes);
    weights.fill(0x3080); // 2^-30, too small to reach a code at the minimum step
    fixture(format, weights, 0xB3D7, 0, codes);
    weights.fill(0);
  }
  // 0, 2.5, 8.5, 15: exact step -1, ties round away from zero.
  for (size_t i = 0; i < weights.size(); ++i) {
    constexpr uint16_t values[]{0, 0x4020, 0x4108, 0x4170};
    constexpr uint8_t expected[]{15, 13, 7, 0};
    weights[i] = values[i % 4]; codes[i] = expected[i % 4];
  }
  fixture(GGUF_FMT_AF4G64, weights, 0xBF80, 0x4170, codes);
  for (auto &weight : weights) weight ^= 0x8000;
  // Negative edge at -15: codes 15, 13, 7, 0, step +1.
  fixture(GGUF_FMT_AF4G64, weights, 0x3F80, 0xC170, codes);
  for (size_t i = 0; i < weights.size(); ++i) {
    constexpr uint16_t values[]{0, 0x42FE, 0x4300, 0x437F}; // 0,127,128,255
    constexpr uint8_t expected[]{255, 128, 127, 0};
    weights[i] = values[i % 4]; codes[i] = expected[i % 4];
  }
  fixture(GGUF_FMT_AF8G64, weights, 0xBF80, 0x437F, codes);
  for (auto &weight : weights) weight ^= 0x8000;
  fixture(GGUF_FMT_AF8G64, weights, 0x3F80, 0xC37F, codes);
  // Asymmetric signed range [-128, 127]: zero is an exact interior code.
  for (size_t i = 0; i < weights.size(); ++i) {
    constexpr uint16_t values[]{0xC300, 0xBF80, 0, 0x42FE};
    constexpr uint8_t expected[]{0, 127, 128, 255};
    weights[i] = values[i % 4]; codes[i] = expected[i % 4];
  }
  fixture(GGUF_FMT_AF8G64, weights, 0x3F80, 0xC300, codes);
  // An interior half with step -1, endpoints 0 and 255.
  weights[0] = 0; weights[1] = 0x437F; weights[2] = 0x3F00; weights[3] = 0x3FC0;
  codes[0] = 255; codes[1] = 0; codes[2] = 255; codes[3] = 254;
  for (size_t i = 4; i < 64; ++i) { weights[i] = 0; codes[i] = 255; }
  fixture(GGUF_FMT_AF8G64, weights, 0xBF80, 0x437F, codes);
}

void testBoundaries() {
  // A run larger than a 4 MiB load task, with every group distinguished.
  constexpr size_t groups = 32771;
  std::vector<uint16_t> weights(groups * 64);
  for (size_t group = 0; group < groups; ++group)
    std::fill_n(weights.begin() + group * 64, 64, uint16_t(0x3E80 + group % 128));
  const auto input = bytes(weights);
  for (const auto format : {GGUF_FMT_AF4G64, GGUF_FMT_AF8G64}) {
    const size_t block = kQuantFormats[format].block_bytes;
    std::vector<uint8_t> whole(groups * block), pieces(whole.size(), 0xFF);
    quantizeAffineBfloat16(input, format, whole);
    // Includes an odd group count, a whole-row boundary and the load-task boundary.
    constexpr std::array<size_t, 6> cuts{0, 1, 4, 8191, 32768, groups};
    for (size_t i = 1; i < cuts.size(); ++i)
      quantizeAffineBfloat16(std::span(input).subspan(cuts[i-1] * 128, (cuts[i] - cuts[i-1]) * 128), format,
                            std::span(pieces).subspan(cuts[i-1] * block, (cuts[i] - cuts[i-1]) * block));
    require(whole == pieces, "affine preparation depends on chunk boundaries");
    for (size_t group = 0; group < groups; ++group) {
      uint16_t bias;
      std::memcpy(&bias, whole.data() + group * block + 2, 2);
      require(bias == weights[group * 64], "a multirow group read the wrong offset");
    }
  }
}

void testRefusalsAndBitValidation() {
  std::vector<uint8_t> input(128), output(68);
  for (const auto format : {GGUF_FMT_AF4G64, GGUF_FMT_AF8G64}) {
    output.resize(kQuantFormats[format].block_bytes);
    for (uint16_t bad : {0x7F80, 0xFF80, 0x7FC1, 0xFFC1}) {
      std::memcpy(input.data() + 126, &bad, 2);
      rejects([&] { quantizeAffineBfloat16(input, format, output); }, "non-finite", "non-finite input accepted");
      rejects([&] { validateFiniteBfloat16(input); }, "non-finite", "stored non-finite BF16 accepted");
    }
    std::fill(input.begin(), input.end(), 0);
    rejects([&] { quantizeAffineBfloat16(std::span(input).first(126), format, output); }, "whole affine groups",
            "partial BF16 group accepted");
    rejects([&] { quantizeAffineBfloat16(input, format, std::span(output).first(output.size()-1)); },
            "whole affine groups", "partial native block accepted");
  }
  rejects([&] { quantizeAffineBfloat16(input, GGUF_FMT_BF16, output); }, "af4g64 or af8g64", "wrong format accepted");
  output.resize(kQuantFormats[GGUF_FMT_AF8G64].block_bytes);
  auto huge = bytes(std::array<uint16_t, 64>{});
  for (size_t at = 0; at < huge.size(); at += 2) {
    const uint16_t value = 0x7F7F;
    std::memcpy(huge.data() + at, &value, 2);
  }
  rejects([&] { quantizeAffineBfloat16(huge, GGUF_FMT_AF8G64, output); }, "affine float32 arithmetic",
          "an overflowing affine zero-code adjustment was accepted");
  huge[1] = 0xFF;
  rejects([&] { quantizeAffineBfloat16(huge, GGUF_FMT_AF8G64, output); }, "affine float32 arithmetic",
          "an overflowing affine range was accepted");
  // BF16 values that half cannot preserve, including signed zero, tiny and
  // huge finite values. Validation must leave every stored bit alone.
  const std::array<uint16_t, 8> bits{0, 0x8000, 1, 0x8001, 0x0080, 0x7F7F, 0xFF7F, 0x3F81};
  auto stored = bytes(bits);
  const auto original = stored;
  validateFiniteBfloat16(stored);
  require(stored == original, "BF16 validation changed stored bits");
  rejects([&] { validateFiniteBfloat16(std::span(stored).first(3)); }, "partial value", "odd BF16 bytes accepted");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 1) {
      require(argc == 5 && std::string_view(argv[1]) == "--fixtures", "expected --fixtures INPUT AF4 AF8");
      const auto input = splash::test::readFile(argv[2]);
      for (const auto [format, path] : {std::pair{GGUF_FMT_AF4G64, argv[3]}, std::pair{GGUF_FMT_AF8G64, argv[4]}}) {
        const auto expected = splash::test::readFile(path);
        std::vector<uint8_t> actual(expected.size(), 0xFF);
        quantizeAffineBfloat16(input, format, actual);
        require(actual == expected, "affine quantizer differs from independent float32 fixture bytes");
      }
    }
    testNativeFixtures();
    testBoundaries();
    testRefusalsAndBitValidation();
    std::cout << "affine BF16 CPU tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

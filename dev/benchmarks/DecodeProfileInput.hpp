#pragma once

#include <charconv>
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::bench {

inline uint64_t profileDecodeTokens(uint32_t cycles, uint32_t rows, bool replay) {
  // The first lane participates at all four widths, in timed and replayed
  // rounds; leave one additional step of headroom.
  return uint64_t{cycles} * 4 * rows * (replay ? 2 : 1) + rows;
}

inline uint64_t profileCacheBytes(std::string_view text) {
  uint64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      (value && value < 67108864))
    throw std::invalid_argument("PLE cache takes 0 or at least 67108864 bytes");
  return value;
}

// Already chat-templated token IDs, separated by whitespace. No retokenizing,
// repetition or truncation: the file defines the exact measured request.
inline std::vector<uint32_t> profilePrompt(std::istream &input) {
  std::vector<uint32_t> tokens;
  std::string word;
  while (input >> word) {
    uint32_t value = 0;
    const auto result = std::from_chars(word.data(), word.data() + word.size(), value);
    if (result.ec != std::errc{} || result.ptr != word.data() + word.size())
      throw std::invalid_argument("prompt file takes whitespace-separated uint32 token IDs");
    if (tokens.size() == 1048576)
      throw std::invalid_argument("prompt file exceeds 1048576 tokens");
    tokens.push_back(value);
  }
  if (input.bad() || tokens.empty()) throw std::invalid_argument("empty or unreadable prompt file");
  return tokens;
}

} // namespace splash::bench

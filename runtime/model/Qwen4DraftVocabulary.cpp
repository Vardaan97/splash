#include "model/Qwen4DraftVocabulary.hpp"

#include <bit>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::model {
namespace {

static_assert(kQwen4DraftVocabularySource % 64 == 0, "the bitmap's words cover the vocabulary");

std::vector<uint32_t> draftTokens(uint32_t size, const uint64_t *words) {
    std::vector<uint32_t> result;
    result.reserve(size);
    for (uint32_t word = 0; word < kQwen4DraftVocabularySource / 64; ++word)
      for (uint64_t bits = words[word]; bits; bits &= bits - 1)
        result.push_back(word * 64 + static_cast<uint32_t>(std::countr_zero(bits)));
    if (result.size() != size)
      throw std::logic_error("the draft vocabulary's bitmap does not hold its tokens");
    return result;

}

} // namespace

std::span<const uint32_t> qwen4DraftVocabulary(uint32_t tokens) {
  if (!tokens) return {};
  static const auto small = draftTokens(32768, detail::kQwen4DraftVocabularyWords);
  static const auto medium = draftTokens(49152, detail::kQwen4DraftVocabularyWords49152);
  static const auto large = draftTokens(65536, detail::kQwen4DraftVocabularyWords65536);
  if (tokens == 32768) return small;
  if (tokens == 49152) return medium;
  if (tokens == 65536) return large;
  throw std::invalid_argument("no draft vocabulary has that size");
}

bool inQwen4DraftVocabulary(uint32_t token) noexcept {
  return qwen4DraftContains(token, 32768);
}

bool qwen4DraftContains(uint32_t token, uint32_t tokens) noexcept {
  const auto *words = tokens == 32768 ? detail::kQwen4DraftVocabularyWords
                    : tokens == 49152 ? detail::kQwen4DraftVocabularyWords49152
                    : tokens == 65536 ? detail::kQwen4DraftVocabularyWords65536 : nullptr;
  return words && token < kQwen4DraftVocabularySource && ((words[token / 64] >> (token % 64)) & 1);
}

uint32_t qwen4DraftTokensFor(uint32_t vocabulary) {
  const char *setting = std::getenv("SPLASH_MTP_DRAFT_VOCAB");
  if (vocabulary != kQwen4DraftVocabularySource) return 0;
  if (!setting || !*setting || std::string_view(setting) == "1") return 32768;
  const std::string_view size(setting);
  if (size == "0") return 0;
  if (size == "32768") return 32768;
  if (size == "49152") return 49152;
  if (size == "65536") return 65536;
  throw std::invalid_argument("SPLASH_MTP_DRAFT_VOCAB must be 0, 1, 32768, 49152 or 65536");
}

} // namespace splash::model

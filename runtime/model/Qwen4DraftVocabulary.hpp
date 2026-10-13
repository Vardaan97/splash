#pragma once

#include <cstdint>
#include <span>

namespace splash::model {

// Qwen3.8's draft vocabulary: the kQwen4DraftVocabularyTokens tokens of its
// tokenizer that text decodes to most often, ranked offline over a mixed
// corpus of reasoning, chat, code, English and 16 other languages' web text
// and math (dev/tools/draft_vocabulary.py), the chat template's, thinking and
// tool-call tokens among them. Qwen3.8-Flash-Next's MTP head drafts its
// argmax over them alone (Qwen4ExpLayout::mtpDraftTokens): it reads their rows
// of the vocabulary projection, an eighth of it, at each draft step. A draft
// is a proposal the verify judges with the whole vocabulary. Greedy output
// remains target-controlled; sampled batches use the whole head instead.
// A token outside this set and the context is never proposed by this head.
inline constexpr uint32_t kQwen4DraftVocabularyTokens = 32768;
// The vocabulary it is of: Qwen3.8's tokenizer.
inline constexpr uint32_t kQwen4DraftVocabularySource = 248320;
// Beside it each request drafts from up to kQwen4DraftContextTokens more:
// the tokens outside it that its own context holds (its prompt and output,
// in the order they first appear), whose rows of the vocabulary projection
// its state lane keeps. A request's context names its topic's words, which
// no fixed list ranks: on eight real agentic coding sessions they lift the
// decode tokens the head can draft from 91 % to 99 %.
inline constexpr uint32_t kQwen4DraftContextTokens = 4096;

// The draft vocabulary's tokens, ascending, when `tokens` is its size; none
// for 0. Throws for any other size.
[[nodiscard]] std::span<const uint32_t> qwen4DraftVocabulary(uint32_t tokens);
// Whether `token` is one of the draft vocabulary's.
[[nodiscard]] bool inQwen4DraftVocabulary(uint32_t token) noexcept;

// The draft vocabulary an MTP head over `vocabulary` tokens drafts from:
// kQwen4DraftVocabularyTokens for Qwen3.8's tokenizer unless
// SPLASH_MTP_DRAFT_VOCAB=0, which drafts from every token; else 0.
[[nodiscard]] uint32_t qwen4DraftTokensFor(uint32_t vocabulary);

namespace detail {
extern const uint64_t kQwen4DraftVocabularyWords[kQwen4DraftVocabularySource / 64];
} // namespace detail

} // namespace splash::model

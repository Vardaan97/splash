#include "model/DraftContextLookup.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::model {
namespace {
uint32_t setting(const char *name, uint32_t fallback, uint32_t low, uint32_t high) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  const std::string_view value(raw);
  uint32_t parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
      parsed < low || parsed > high)
    throw std::invalid_argument(std::string(name) + " is outside its supported range");
  return parsed;
}
} // namespace

DraftContextLookupConfig DraftContextLookupConfig::fromEnvironment() {
  DraftContextLookupConfig config;
  const char *enabled = std::getenv("SPLASH_DRAFT_CONTEXT_LOOKUP");
  config.enabled = enabled && std::string_view(enabled) == "1";
  if (!config.enabled) return config;
  config.minimum = setting("SPLASH_DRAFT_CONTEXT_LOOKUP_MIN", 3, 3, 8);
  config.maximum = setting("SPLASH_DRAFT_CONTEXT_LOOKUP_MAX", 8, config.minimum, 8);
  config.window = setting("SPLASH_DRAFT_CONTEXT_LOOKUP_WINDOW", 32768, 8, 262144);
  return config;
}

DraftContextLookup::DraftContextLookup(DraftContextLookupConfig config) : config_(config) {
  if (config.minimum < 3 || config.maximum > 8 || config.minimum > config.maximum ||
      config.window < 8 || config.window > 262144)
    throw std::invalid_argument("invalid draft context lookup configuration");
}

size_t DraftContextLookup::Hash::operator()(const Key &key) const noexcept {
  size_t hash = 14695981039346656037ULL;
  for (const uint32_t token : key.tokens) hash = (hash ^ token) * 1099511628211ULL;
  return hash;
}

DraftContextLookup::Key DraftContextLookup::keyAt(std::span<const uint32_t> history,
                                                 size_t end, uint32_t length) {
  Key key;
  std::copy_n(history.begin() + (end - length), length, key.tokens.begin());
  return key;
}

void DraftContextLookup::reset() {
  for (auto &index : latest_) index.clear();
  indexed_ = 0;
}

size_t DraftContextLookup::sync(std::span<const uint32_t> history) {
  if (history.size() < indexed_) reset();
  // A large first prompt or resume indexes only the retained window. Normal
  // decode visits at most eight new ends, independently of history length.
  const size_t begin = std::max(indexed_, history.size() > config_.window
                                              ? history.size() - config_.window : 0);
  if (begin > indexed_) reset();
  for (size_t end = begin + 1; end <= history.size(); ++end) {
    for (uint32_t length = config_.minimum; length <= config_.maximum; ++length) {
      auto &index = latest_[length - 1];
      if (end > config_.window && end - config_.window >= length) {
        const size_t expired = end - config_.window;
        const auto old = index.find(keyAt(history, expired, length));
        if (old != index.end() && old->second == expired) index.erase(old);
      }
      if (end >= length) index[keyAt(history, end, length)] = end;
    }
  }
  indexed_ = history.size();
  return history.size() - begin;
}

DraftContextLookupProposal DraftContextLookup::propose(std::span<const uint32_t> history,
                                                       uint32_t anchor) const {
  if (indexed_ != history.size()) throw std::logic_error("context lookup history is not synced");
  for (uint32_t length = config_.maximum; length >= config_.minimum; --length) {
    if (history.size() + 1 < length) continue;
    Key key;
    std::copy_n(history.end() - (length - 1), length - 1, key.tokens.begin());
    key.tokens[length - 1] = anchor;
    const auto found = latest_[length - 1].find(key);
    if (found == latest_[length - 1].end()) continue;
    DraftContextLookupProposal proposal;
    proposal.matched = length;
    proposal.continuation = found->second;
    proposal.count = static_cast<uint32_t>(std::min<size_t>(
        SPLASH_DRAFT_PROPOSAL_TOKENS, history.size() + 1 - proposal.continuation));
    for (uint32_t i = 0; i < proposal.count; ++i) {
      const size_t position = proposal.continuation + i;
      proposal.tokens[i] = position < history.size() ? history[position] : anchor;
    }
    return proposal;
  }
  return {};
}

size_t DraftContextLookup::indexedKeys() const noexcept {
  size_t count = 0;
  for (const auto &index : latest_) count += index.size();
  return count;
}

DraftContextLookupLane contextLookupLane(const DraftContextLookupProposal &proposal,
                                         uint32_t maximumRows, bool sampled,
                                         bool thinkingCutoff) {
  if (!maximumRows || maximumRows > SPLASH_DRAFT_QUERY_ROWS ||
      proposal.count > SPLASH_DRAFT_PROPOSAL_TOKENS)
    throw std::invalid_argument("invalid context lookup merge");
  DraftContextLookupLane lane{};
  if (sampled || thinkingCutoff) return lane;
  lane.count = std::min(proposal.count, maximumRows - 1);
  std::copy_n(proposal.tokens.begin(), lane.count, lane.tokens);
  return lane;
}

} // namespace splash::model

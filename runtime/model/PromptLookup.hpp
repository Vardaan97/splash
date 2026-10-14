#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace splash::model {

// Exact request-local n-grams, longest first: the newest full continuation,
// or the newest partial match when none is full. Tokens remain in Runtime.
// Each committed token updates at most eight keys; proposals never enter the index.
class PromptLookup final {
  struct Key {
    std::array<uint32_t, 4> tokens{};
    uint32_t size = 0;
    bool operator==(const Key &) const = default;
  };
  struct Hash {
    size_t operator()(const Key &key) const noexcept {
      size_t h = key.size;
      for (uint32_t i = 0; i < key.size; ++i) h = (h ^ key.tokens[i]) * 1099511628211ULL;
      return h;
    }
  };
  std::unordered_map<Key, size_t, Hash> latest_;
  std::unordered_map<Key, size_t, Hash> complete_;
  size_t scanned_ = 0;
  void insert(std::unordered_map<Key, size_t, Hash> &index, const Key &key, size_t next) {
    if (const auto found = index.find(key); found != index.end()) {
      found->second = next;
      return;
    }
    // Arbitrary eviction affects hit rate only; proposals are still verified.
    if (index.size() == maximumKeysPerIndex) index.erase(index.begin());
    index.emplace(key, next);
  }
  static Key suffix(std::span<const uint32_t> history, uint32_t anchor, uint32_t k) {
    Key key; key.size = k;
    for (uint32_t i = 0; i + 1 < k; ++i) key.tokens[i] = history[history.size() - (k - 1) + i];
    key.tokens[k - 1] = anchor;
    return key;
  }
public:
  // Hard bounds per request: at most 256 newly indexed tokens per sync and
  // 8192 total map nodes. Older unindexed prompt rows are deliberately skipped.
  // Reserve a conservative 2 MiB host allowance per lookup lane (including
  // buckets/allocator overhead); there is no whole-prompt-sized allocation.
  static constexpr size_t maximumTokensPerSync = 256;
  static constexpr size_t maximumKeysPerIndex = 4096;
  [[nodiscard]] size_t indexedKeys() const noexcept { return latest_.size() + complete_.size(); }
  [[nodiscard]] static bool usableLane(bool sampled, bool thinkingCutoff) noexcept {
    return !sampled && !thinkingCutoff;
  }
  size_t sync(std::span<const uint32_t> history, bool sampled = false, bool thinkingCutoff = false) {
    if (!usableLane(sampled, thinkingCutoff)) return 0;
    if (history.size() < scanned_) { latest_.clear(); complete_.clear(); scanned_ = 0; }
    const size_t begin = std::max(scanned_, history.size() > maximumTokensPerSync
                                              ? history.size() - maximumTokensPerSync : 0);
    for (size_t i = begin; i < history.size(); ++i) {
      // Index an occurrence only once it has a committed successor.
      if (i)
        for (uint32_t k = 1; k <= std::min<size_t>(4, i); ++k)
          insert(latest_, suffix(history.first(i - 1), history[i - 1], k), i);
      // Keep the newest occurrence with a full seven-token continuation
      // too, so a recent suffix does not hide a useful earlier copy span.
      if (i >= 7) {
        const size_t next = i + 1 - 7;
        for (uint32_t k = 1; k <= std::min<size_t>(4, next); ++k)
          insert(complete_, suffix(history.first(next - 1), history[next - 1], k), next);
      }
      scanned_ = i + 1;
    }
    return history.size() - begin;
  }
  [[nodiscard]] std::vector<uint32_t> propose(std::span<const uint32_t> history, uint32_t anchor,
                                               uint32_t maximum = 7) const {
    maximum = std::min(maximum, 7u);
    for (uint32_t k = std::min<size_t>(4, history.size() + 1); k; --k) {
      const auto found = latest_.find(suffix(history, anchor, k));
      if (found == latest_.end()) continue;
      size_t next = found->second;
      if (history.size() - next < maximum) {
        const auto complete = complete_.find(found->first);
        if (complete != complete_.end()) next = complete->second;
      }
      const size_t count = std::min<size_t>(maximum, history.size() - next);
      return {history.begin() + next, history.begin() + next + count};
    }
    return {};
  }
};

// Shadow first-token accuracy on emitted target tokens, even while the head
// drafts. No optimistic prior: stale evidence decays on every emitted token,
// including tokens without a match. Suspension keeps this request-local state.
struct LookupAcceptance final {
  float accepted = 0.0F, judged = 0.0F;
  void record(std::span<const uint32_t> proposal, uint32_t selected) noexcept {
    accepted *= 0.98F; judged *= 0.98F;
    if (!proposal.empty()) { accepted += proposal.front() == selected; judged += 1.0F; }
  }
  [[nodiscard]] float rate() const noexcept { return judged ? accepted / judged : 0.0F; }
  [[nodiscard]] bool eligible(float threshold = 0.85F) const noexcept { return judged >= 4.0F && rate() >= threshold; }
};
} // namespace splash::model

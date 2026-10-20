#pragma once

#include "metal/abi/DraftContextLookup.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>

namespace splash::model {

struct DraftContextLookupConfig final {
  bool enabled = false;
  uint32_t minimum = 3;
  uint32_t maximum = 8;
  uint32_t window = 32768;
  [[nodiscard]] static DraftContextLookupConfig fromEnvironment();
};

struct DraftContextLookupProposal final {
  std::array<uint32_t, SPLASH_DRAFT_PROPOSAL_TOKENS> tokens{};
  uint32_t count = 0;
  uint32_t matched = 0;
  size_t continuation = 0;
};

// Exact keys, newest occurrence, longest suffix first. The request owns the
// history; the index owns at most window keys for each configured length.
// Only committed history enters the index; the pending anchor ends the query.
class DraftContextLookup final {
public:
  explicit DraftContextLookup(DraftContextLookupConfig config);
  // History must append between calls. Reset after replacing it, including
  // a same-length replacement on resume. Returns newly indexed token ends.
  size_t sync(std::span<const uint32_t> history);
  void reset();
  [[nodiscard]] DraftContextLookupProposal propose(std::span<const uint32_t> history,
                                                   uint32_t anchor) const;
  [[nodiscard]] size_t indexedKeys() const noexcept;

private:
  struct Key final {
    std::array<uint32_t, 8> tokens{};
    bool operator==(const Key &) const = default;
  };
  struct Hash final {
    size_t operator()(const Key &key) const noexcept;
  };
  static Key keyAt(std::span<const uint32_t> history, size_t end, uint32_t length);
  DraftContextLookupConfig config_;
  std::array<std::unordered_map<Key, size_t, Hash>, 8> latest_;
  size_t indexed_ = 0;
};

// Plans one lane's prefix replacement. A sampled proposal's probability
// table must stay paired with its tokens; thinking cutoffs own their prefix.
[[nodiscard]] DraftContextLookupLane contextLookupLane(const DraftContextLookupProposal &proposal,
                                                        uint32_t maximumRows, bool sampled,
                                                        bool thinkingCutoff);

} // namespace splash::model

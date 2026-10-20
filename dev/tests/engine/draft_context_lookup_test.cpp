#include "TestChecks.hpp"
#include "model/DraftContextLookup.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace splash::model;
using splash::test::require;
using splash::test::rejects;

namespace {

// Independent bounded reverse scan: the production index must choose the
// longest suffix, then its newest occurrence, even if it has fewer tokens.
DraftContextLookupProposal reference(std::span<const uint32_t> history, uint32_t anchor,
                                      DraftContextLookupConfig config) {
  const size_t total = history.size() + 1;
  const auto token = [&](size_t position) { return position < history.size() ? history[position] : anchor; };
  const size_t oldest = history.size() > config.window ? history.size() - config.window + 1 : 1;
  for (uint32_t length = config.maximum; length >= config.minimum; --length) {
    if (total < length) continue;
    for (size_t end = history.size(); end >= std::max<size_t>(oldest, length); --end) {
      bool match = true;
      for (uint32_t i = 0; i < length; ++i)
        match &= token(end - length + i) == token(total - length + i);
      if (!match) continue;
      DraftContextLookupProposal result;
      result.matched = length;
      result.continuation = end;
      result.count = std::min<size_t>(7, total - end);
      for (uint32_t i = 0; i < result.count; ++i) result.tokens[i] = token(end + i);
      return result;
    }
  }
  return {};
}

void compare(DraftContextLookup &index, std::span<const uint32_t> history, uint32_t anchor,
             DraftContextLookupConfig config) {
  const auto got = index.propose(history, anchor);
  const auto expected = reference(history, anchor, config);
  require(got.tokens == expected.tokens && got.count == expected.count &&
              got.matched == expected.matched && got.continuation == expected.continuation,
          "index differs from independent longest/newest scan");
  require(index.indexedKeys() <= size_t{config.maximum - config.minimum + 1} * config.window,
          "index exceeded its memory bound");
}

void examples() {
  const DraftContextLookupConfig config{true, 3, 8, 32768};
  DraftContextLookup index(config);
  std::vector<uint32_t> history{1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 90, 1, 2};
  require(index.sync(history) == history.size(), "first sync did not index the prompt");
  auto proposal = index.propose(history, 3);
  require(proposal.matched == 3 && proposal.count == 7 && proposal.tokens[0] == 10 &&
              proposal.tokens[6] == 16, "copy span or pending anchor is wrong");
  require(index.sync(history) == 0, "unchanged history was rescanned");
  // A newer short occurrence hides a full earlier continuation by design.
  history.insert(history.end(), {3, 20, 1, 2});
  require(index.sync(history) == 4, "appended generation was not incremental");
  proposal = index.propose(history, 3);
  require(proposal.continuation == 14 && proposal.tokens[0] == 20 && proposal.count == 4,
          "newest earlier occurrence lost to an older full continuation");
  compare(index, history, 3, config);
  // The longest match beats a newer shorter one.
  history = {8, 1, 2, 3, 42, 43, 1, 2, 3, 99, 8, 1, 2};
  index.reset(); index.sync(history);
  proposal = index.propose(history, 3);
  require(proposal.matched == 4 && proposal.continuation == 4 && proposal.tokens[0] == 42,
          "newest short match hid the longest match");
  // Overlap is legal, but never extrapolates beyond known tokens + anchor.
  history.assign(20, 7);
  index.reset(); index.sync(history);
  proposal = index.propose(history, 7);
  require(proposal.matched == 8 && proposal.continuation == 20 && proposal.count == 1 &&
              proposal.tokens[0] == 7, "overlapping match read unknown future tokens");
  for (size_t n = 0; n < 4; ++n) {
    history.assign(n, 7); index.sync(history); compare(index, history, 7, config);
  }
  history = {0, UINT32_MAX, 3, 40, 0, UINT32_MAX};
  index.reset(); index.sync(history); compare(index, history, 3, config);
  require(index.propose(history, 3).tokens[0] == 40, "token ids were truncated");
  history = {5, 6, 7, 40, 5, 6}; // Same-size replacement must reset.
  index.reset(); index.sync(history); compare(index, history, 7, config);
  rejects([&] { static_cast<void>(index.propose(std::span(history).first(2), 7)); }, "not synced",
          "unsynchronized lookup was allowed");
}

void randomizedIndex() {
  std::mt19937 random(0x1c001);
  for (const uint32_t window : {8u, 17u, 64u, 32768u}) {
    for (const uint32_t minimum : {3u, 5u, 8u}) {
      const DraftContextLookupConfig config{true, minimum, 8, window};
      DraftContextLookup index(config);
      std::vector<uint32_t> history;
      for (uint32_t cycle = 0; cycle < 700; ++cycle) {
        const size_t old = history.size();
        const size_t appended = cycle % 59 == 0 ? window + 3 : random() % 9;
        for (size_t i = 0; i < appended; ++i) history.push_back(random() % 5);
        const size_t indexed = index.sync(history);
        require(indexed == std::min<size_t>(appended, window), "sync rescanned unchanged history");
        if (appended == 0) require(history.size() == old, "empty append changed history");
        for (uint32_t anchor = 0; anchor < 5; ++anchor) compare(index, history, anchor, config);
        if (cycle % 101 == 100) {
          history.resize(history.size() / 3);
          index.sync(history); compare(index, history, 0, config);
        }
      }
    }
  }
}

void longHistory() {
  DraftContextLookupConfig config{true, 3, 8, 32768};
  DraftContextLookup index(config);
  std::vector<uint32_t> history(23486);
  for (size_t i = 0; i < history.size(); ++i) history[i] = static_cast<uint32_t>(i + 100);
  const std::array<uint32_t, 8> pattern{1, 2, 3, 4, 5, 6, 7, 8};
  std::copy(pattern.begin(), pattern.end(), history.begin() + 20);
  std::copy_n(pattern.begin(), 7, history.end() - 7);
  index.sync(history);
  require(index.propose(history, 8).continuation == 28, "old prompt match was lost at 23K context");
  history.push_back(8);
  require(index.sync(history) == 1, "one token at 23K context rescanned the prompt");
  compare(index, history, 9, config);
}

void eviction() {
  const DraftContextLookupConfig config{true, 3, 8, 8};
  DraftContextLookup index(config);
  std::vector<uint32_t> history;
  for (uint32_t token = 1; token < 100; ++token) {
    history.push_back(token);
    require(index.sync(history) == 1, "incremental unique history rescanned tokens");
    require(index.indexedKeys() <= 48, "unique n-grams exceeded the node bound");
  }
  history.insert(history.end(), {1, 2});
  index.sync(history);
  require(index.propose(history, 3).count == 0, "evicted old occurrence remained searchable");
  compare(index, history, 3, config);
}

void merge() {
  DraftContextLookupProposal proposal;
  proposal.count = 7;
  for (uint32_t i = 0; i < 7; ++i) proposal.tokens[i] = 10 + i;
  for (uint32_t width = 1; width <= 4; ++width) {
    for (uint32_t count = 0; count <= 7; ++count) {
      proposal.count = count;
      for (uint32_t maximum = 1; maximum <= 8; ++maximum) {
        DraftContextLookupParams params{};
        for (uint32_t lane = 0; lane < width; ++lane)
          params.lane[lane] = contextLookupLane(proposal, maximum, lane == 1, lane == 2);
        for (uint32_t lane = 0; lane < 4; ++lane) {
          const uint32_t expected = lane >= width || lane == 1 || lane == 2 ? 0 : std::min(count, maximum - 1);
          require(params.lane[lane].count == expected, "merge eligibility or row limit is wrong");
          for (uint32_t pos = 0; pos < 7; ++pos) {
            const uint32_t neural = 200 + lane * 7 + pos;
            const auto token = draft_context_lookup_token(pos, params.lane[lane].count,
                                                          params.lane[lane].tokens[pos], neural);
            require(token == (pos < expected ? 10 + pos : neural), "merge touched another lane or suffix");
          }
        }
      }
    }
  }
  // Every rejection position and verify limit: only the matching prefix can
  // be accepted; the target correction closes the block. Lookup never lets
  // an incorrect proposal reach output, even when the neural tail is wrong.
  proposal.count = 7;
  const auto plan = contextLookupLane(proposal, 8, false, false);
  for (uint32_t rows = 1; rows <= 8; ++rows) {
    for (uint32_t reject = 0; reject < 8; ++reject) {
      std::vector<uint32_t> selected;
      uint32_t accepted = 0;
      for (uint32_t pos = 0; pos < rows; ++pos) {
        const uint32_t target = pos == reject ? 99 : 10 + pos;
        selected.push_back(target);
        if (pos == rows - 1) break;
        const uint32_t drafted = draft_context_lookup_token(pos, plan.count, plan.tokens[pos], 200 + pos);
        if (drafted != target) break;
        ++accepted;
      }
      require(accepted == std::min(reject, rows - 1), "merge bypassed target rejection or verify limit");
      require(selected.size() == accepted + 1, "target correction was not retained");
    }
  }
  rejects([&] { static_cast<void>(contextLookupLane(proposal, 0, false, false)); }, "invalid", "zero rows were accepted");
}

// Restore the process environment even if a parser assertion throws.
struct Environment final {
  const char *name;
  std::string previous;
  bool existed;
  explicit Environment(const char *key) : name(key), existed(std::getenv(key) != nullptr) {
    if (existed) previous = std::getenv(key);
    unsetenv(name);
  }
  ~Environment() { if (existed) setenv(name, previous.c_str(), 1); else unsetenv(name); }
};

void configuration() {
  Environment enabled("SPLASH_DRAFT_CONTEXT_LOOKUP"), low("SPLASH_DRAFT_CONTEXT_LOOKUP_MIN"),
      high("SPLASH_DRAFT_CONTEXT_LOOKUP_MAX"), window("SPLASH_DRAFT_CONTEXT_LOOKUP_WINDOW");
  require(!DraftContextLookupConfig::fromEnvironment().enabled, "lookup is not default off");
  setenv(low.name, "bad", 1);
  require(!DraftContextLookupConfig::fromEnvironment().enabled, "disabled lookup parsed tuning knobs");
  for (const char *value : {"0", "true", "2", ""}) {
    setenv(enabled.name, value, 1);
    require(!DraftContextLookupConfig::fromEnvironment().enabled, "lookup enabled without exact opt-in");
  }
  setenv(enabled.name, "1", 1); unsetenv(low.name);
  auto config = DraftContextLookupConfig::fromEnvironment();
  require(config.enabled && config.minimum == 3 && config.maximum == 8 && config.window == 32768,
          "lookup defaults changed");
  for (const char *value : {"2", "9", "", "3junk", "-3", "9999999999999999999"}) {
    setenv(low.name, value, 1);
    rejects([] { static_cast<void>(DraftContextLookupConfig::fromEnvironment()); }, low.name, "malformed minimum accepted");
  }
  setenv(low.name, "6", 1); setenv(high.name, "5", 1);
  rejects([] { static_cast<void>(DraftContextLookupConfig::fromEnvironment()); }, high.name, "reversed range accepted");
  setenv(high.name, "7", 1); setenv(window.name, "64", 1);
  config = DraftContextLookupConfig::fromEnvironment();
  require(config.minimum == 6 && config.maximum == 7 && config.window == 64, "valid knobs ignored");
  setenv(window.name, "262145", 1);
  rejects([] { static_cast<void>(DraftContextLookupConfig::fromEnvironment()); }, window.name, "unbounded window accepted");
}
}

int main() {
  try {
    examples(); randomizedIndex(); longHistory(); eviction(); merge(); configuration();
    std::cout << "draft context lookup: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

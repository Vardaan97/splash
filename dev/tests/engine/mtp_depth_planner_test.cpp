// The MTP depth planner (model/MtpDepthPlanner.hpp): per-position
// acceptance, the cycle cost model fitted to measured cycles, and the depths
// it plans per lane, down to none.
//   mtp-depth-planner
#include "TestChecks.hpp"
#include "model/MtpDepthPlanner.hpp"
#include "model/PromptLookup.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace splash::model;
using splash::test::require;

namespace {

// Acceptance after `cycles` cycles drafting `drafted` where each proposal is
// kept with chance `rate` while the ones before were.
MtpAcceptance acceptanceAt(double rate, uint32_t drafted = 7, uint32_t cycles = 200, uint32_t seed = 1) {
  std::mt19937 rng(seed);
  std::bernoulli_distribution keep(rate);
  MtpAcceptance acceptance;
  for (uint32_t cycle = 0; cycle < cycles; ++cycle) {
    uint32_t kept = 0;
    while (kept < drafted && keep(rng)) ++kept;
    acceptance.record(drafted, kept);
  }
  return acceptance;
}

uint32_t loneDepth(const MtpAcceptance &acceptance, const MtpCostModel &costs, bool measured) {
  const std::array<std::array<double, kMtpPlannerPositions + 1>, 1> expected{acceptance.expectedTokens()};
  const MtpDepthPlan plan = planMtpDepths(expected, costs, measured, true, 7);
  require(plan.steps == plan.depths[0], "a lone lane's batch drafts its depth");
  return plan.depths[0];
}

void acceptance() {
  const MtpAcceptance fresh;
  require(std::fabs(fresh.rate(1) - 0.8F) < 1e-6F && std::fabs(fresh.rate(7) - 0.8F) < 1e-6F,
          "acceptance: a fresh request's positions take the prior rate");
  MtpAcceptance a;
  for (int i = 0; i < 50; ++i) a.record(3, 1); // keeps the first, rejects the second
  require(a.rate(1) > 0.85F && a.rate(2) < 0.15F, "acceptance: per position, leaning on the pooled rate");
  require(std::fabs(a.rate(3) - a.pooledRate()) < 1e-6F, "acceptance: an unjudged position takes the pooled rate");
  const auto tokens = acceptanceAt(0.7).expectedTokens();
  bool increasing = tokens[0] == 1.0;
  for (uint32_t d = 1; d <= kMtpPlannerPositions; ++d) increasing = increasing && tokens[d] > tokens[d - 1];
  require(increasing && std::fabs(tokens[1] - 1.7) < 0.1, "acceptance: expected tokens grow with depth");
}

void prior() {
  const MtpCostModel costs;
  // At the prior's ratios (a + 0.2 per proposal with the draft vocabulary).
  require(loneDepth(acceptanceAt(0.97), costs, false) >= 6, "prior: a near-certain lane drafts six or seven");
  const uint32_t threeInFour = loneDepth(acceptanceAt(0.75), costs, false);
  require(threeInFour >= 2 && threeInFour <= 5, "prior: a lane keeping three in four drafts two to five");
  require(loneDepth(acceptanceAt(0.02), costs, false) == 0, "prior: a lane that keeps almost none drafts none");
}

void fit() {
  // A machine whose cycles cost a + b steps + c lane-steps, a by width.
  const double a[4] = {0.020, 0.030, 0.038, 0.046}, bWhole = 0.0025, bDraft = 0.0016, c = 0.0007, routed = 0.00014;
  MtpCostModel costs;
  std::mt19937 rng(3);
  std::uniform_int_distribution<uint32_t> depth(0, 7);
  std::normal_distribution<double> noise(0.0, 0.0005);
  for (int cycle = 0; cycle < 3000; ++cycle) {
    const uint32_t lanes = 1 + cycle % 4;
    uint32_t steps = 0, laneSteps = 0;
    std::array<uint32_t, 4> depths{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t d = depth(rng);
      depths[lane] = d;
      steps = std::max(steps, d);
      laneSteps += d;
    }
    const bool vocabulary = cycle % 5 != 0;
    const uint32_t rows = mtpDraftRoutedRows(std::span(depths).first(lanes));
    const double seconds = a[lanes - 1] + (vocabulary ? bDraft : bWhole) * steps + c * laneSteps + routed * rows + noise(rng);
    costs.observe({lanes, steps, laneSteps, vocabulary, rows}, seconds);
  }
  for (uint32_t lanes = 2; lanes <= 4; ++lanes) {
    const auto theta = costs.coefficients(lanes, true);
    require(std::fabs(theta[0] - a[lanes - 1]) < 0.1 * a[lanes - 1] && std::fabs(theta[1] - bWhole) < 0.2 * bWhole &&
                std::fabs(theta[2] - bDraft) < 0.2 * bDraft && std::fabs(theta[3] - c) < 0.2 * c && std::fabs(theta[4] - routed) < 0.2 * routed,
            "cost model: fits each width's base, step and proposal costs from varied cycles (B" +
                std::to_string(lanes) + ")");
  }
  // Even a lone lane distinguishes the nonlinear draft-row feature.
  const MtpCostModel::Cycle lone{1, 7, 7, true, 35};
  require(std::fabs(costs.cost(lone, true) - (a[0] + 7 * (bDraft + c) + 35 * routed)) < 0.001,
          "cost model: fits a lone lane's total routed work");
  require(costs.coefficients(1, false) == MtpCostModel{}.coefficients(1, false),
          "cost model: the prior ignores what was measured");
  // Measured costs move the plan: with a cheap base a lane keeping three in
  // four drafts fewer than at the prior's ratios.
  MtpCostModel steep;
  for (int cycle = 0; cycle < 400; ++cycle) {
    const uint32_t d = cycle % 8;
    steep.observe({1, d, d, true, mtpDraftRoutedRows(std::span(&d, 1))}, 0.010 + 0.004 * d);
  }
  require(loneDepth(acceptanceAt(0.75), steep, true) < loneDepth(acceptanceAt(0.75), steep, false),
          "plan: a dearer proposal relative to the base drafts shallower, measured only");
}

void routedWork() {
  const std::array<uint32_t, 4> concentrated{7, 7, 0, 0}, spread{7, 3, 2, 2};
  require(mtpDraftRoutedRows(concentrated) == 70 && mtpDraftRoutedRows(spread) == 86,
          "draft routes: equal verify proposals can route different draft rows");
  const MtpCostModel::Cycle a{4, 7, 14, true, mtpDraftRoutedRows(concentrated)};
  const MtpCostModel::Cycle b{4, 7, 14, true, mtpDraftRoutedRows(spread)};
  MtpCostModel costs;
  require(costs.cost(b, false) > costs.cost(a, false), "prior distinguishes draft routed work");
  for (int i = 0; i < 400; ++i) {
    costs.observe(a, 0.04);
    costs.observe(b, 0.06);
  }
  require(costs.cost(b, true) > costs.cost(a, true) + 0.015,
          "fit distinguishes equal-depth, equal-proposal plans by actual draft routes");
}

void lanes() {
  const MtpCostModel costs;
  const std::array<std::array<double, kMtpPlannerPositions + 1>, 2> expected{acceptanceAt(0.95).expectedTokens(),
                                                                               acceptanceAt(0.2).expectedTokens()};
  const MtpDepthPlan plan = planMtpDepths(expected, costs, false, true, 7);
  require(plan.depths[0] >= 5 && plan.depths[1] <= 1 && plan.steps == plan.depths[0],
          "plan: each lane its own depth, the batch as deep as its deepest");
  const MtpDepthPlan capped = planMtpDepths(expected, costs, false, true, 2);
  require(capped.steps <= 2 && capped.depths[0] <= 2, "plan: at most the steps it may draft");
}

void lookup() {
  PromptLookup matcher;
  std::vector<uint32_t> history{1, 2, 3, 4, 5, 6, 1, 2, 3};
  matcher.sync(history);
  require(matcher.propose(history, 4, 3) == std::vector<uint32_t>({5, 6, 1}), "lookup longest exact suffix");
  history.insert(history.end(), {4, 8, 9, 1, 2, 3});
  matcher.sync(history);
  require(matcher.propose(history, 4, 2) == std::vector<uint32_t>({8, 9}), "lookup latest occurrence and incremental sync");
  require(matcher.propose(history, 99).empty(), "lookup missing suffix");
  history = {7};
  matcher.sync(history);
  require(matcher.propose(history, 4).empty(), "lookup reset drops old history");
  PromptLookup repeated;
  const std::vector<uint32_t> zeros(24, 0);
  repeated.sync(zeros);
  require(repeated.propose(zeros, 0).size() == 7, "recent short suffix does not hide a complete earlier span");
  PromptLookup bounded;
  std::vector<uint32_t> large(262144);
  for (size_t i = 0; i < large.size(); ++i) large[i] = i;
  require(bounded.sync(large) == PromptLookup::maximumTokensPerSync,
          "first decode indexes only the bounded tail of a full-window prompt");
  require(bounded.sync(large) == 0, "unchanged history incurs no indexing work");
  for (uint32_t batch = 0; batch < 40; ++batch) {
    for (uint32_t i = 0; i < 300; ++i) large.push_back(large.size());
    require(bounded.sync(large) == PromptLookup::maximumTokensPerSync,
            "each sync caps newly indexed rows even after a large append");
    require(bounded.indexedKeys() <= 2 * PromptLookup::maximumKeysPerIndex,
            "lookup memory stays bounded across long requests");
  }
  large = {7};
  bounded.sync(large);
  require(bounded.indexedKeys() == 0, "shrink clears capped indexes");
  require(PromptLookup::usableLane(false, false) && !PromptLookup::usableLane(true, false) &&
              !PromptLookup::usableLane(false, true), "sampled and cutoff lanes cannot index lookup");
  PromptLookup excluded;
  const std::vector<uint32_t> fullPrompt(262144, 0);
  require(excluded.sync(fullPrompt, true, false) == 0 && excluded.indexedKeys() == 0,
          "sampled full-window lane allocates no lookup index");
  require(excluded.sync(fullPrompt, false, true) == 0 && excluded.indexedKeys() == 0,
          "thinking-cutoff full-window lane allocates no lookup index");
  require(excluded.sync(fullPrompt, false, false) == PromptLookup::maximumTokensPerSync &&
              excluded.indexedKeys() > 0, "greedy eligible lane still indexes the bounded tail");
  LookupAcceptance tracker;
  require(!tracker.eligible(), "lookup has no optimistic prior");
  const std::array<uint32_t, 1> proposed{3};
  for (int i = 0; i < 8; ++i) tracker.record(proposed, 3);
  require(tracker.eligible(), "shadow successes earn lookup");
  for (int i = 0; i < 5; ++i) tracker.record(proposed, 9);
  require(!tracker.eligible(), "shadow errors disable lookup");
  for (int i = 0; i < 500; ++i) tracker.record({}, 3);
  require(!tracker.eligible(0), "missing matches expire evidence");
  const MtpCostModel costs({1.0, 0.01, 0.01, 0.001, 0.001});
  const std::array<std::array<double, 8>, 2> expected{MtpAcceptance{}.expectedTokens(), MtpAcceptance{}.expectedTokens()};
  const std::array<uint32_t, 2> counts{7, 0};
  const std::array<double, 2> tokens{8, 0};
  const auto plan = planMtpDepths(expected, costs, false, true, 7, counts, tokens);
  require(plan.lookup[0] && !plan.lookup[1] && plan.depths[0] == 0 && plan.depths[1] > 0,
          "planner mixes zero-step lookup and head lanes");
}

void cadence() {
  uint32_t idle = 0;
  require(probeMtpDepth(0, idle, 2) == 0 && probeMtpDepth(0, idle, 2) == 1 && idle == 0,
          "cadence: injected interval reaches the probe in two cycles");
  require(exploreMtpDepth(3, 7, 2, 2) == 4 && exploreMtpDepth(3, 7, 4, 2) == 2 &&
              exploreMtpDepth(3, 7, 4, 0) == 3, "cadence: exploration alternates and can be disabled");
  const MtpCostModel custom({1.0, 0.8, 0.7, 0.6, 0.05});
  require(custom.coefficients(1, false)[1] == 0.8, "prior: injectable component cost ratios");
}

} // namespace

int main() {
  try {
    unsetenv("SPLASH_MTP_DEPTH");
    require(mtpDepthPolicyFromEnvironment() == MtpDepthPolicy::Rule, "unset depth uses rule");
    setenv("SPLASH_MTP_DEPTH", "cost", 1);
    require(mtpDepthPolicyFromEnvironment() == MtpDepthPolicy::Cost, "cost is opt-in");
    setenv("SPLASH_MTP_DEPTH", "lane", 1);
    require(mtpDepthPolicyFromEnvironment() == MtpDepthPolicy::Lane, "lane is opt-in");
    unsetenv("SPLASH_MTP_WORTHWHILE");
    const auto thresholds = mtpWorthwhileFromEnvironment();
    for (float r : {0.1F, 0.6F, 0.8F, 0.95F})
      require(mtpRuleDepth(r, 7) == mtpRuleDepth(r, 7, thresholds[0]), "B1 lane equals rule");
    const std::array<float, 4> rates{0.5F, 0.78F, 0.99F, 0.84F};
    const std::array<uint32_t, 4> wanted{2, 3, 7, 4};
    for (size_t i = 0; i < rates.size(); ++i)
      require(mtpRuleDepth(rates[i], 7, thresholds[3]) == wanted[i], "mixed B4 lane rule depths");
    setenv("SPLASH_MTP_WORTHWHILE", "0.3,0.35,0.4,0.45", 1);
    require(mtpWorthwhileFromEnvironment() == thresholds, "width thresholds parse");
    setenv("SPLASH_MTP_WORTHWHILE", "0.4", 1);
    require(mtpWorthwhileFromEnvironment()[0] == 0.4F && mtpWorthwhileFromEnvironment()[3] == 0.4F,
            "scalar threshold applies at every width");
    setenv("SPLASH_MTP_WORTHWHILE", "0.3,0.4", 1);
    bool rejected = false;
    try { (void)mtpWorthwhileFromEnvironment(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "reject partial threshold list");
    unsetenv("SPLASH_MTP_WORTHWHILE");
    unsetenv("SPLASH_MTP_DEPTH");
    acceptance();
    prior();
    fit();
    lanes();
    routedWork();
    cadence();
    lookup();
    require(lookupMatchBucket(5) == 0 && lookupMatchBucket(6) == 1 && lookupMatchBucket(12) == 2 &&
            lookupMatchBucket(24) == 3, "lookup match bucket boundaries");
    PromptLookup bounded;
    std::vector<uint32_t> longPrompt(262144);
    for (size_t i = 0; i < longPrompt.size(); ++i) longPrompt[i] = i % 512;
    require(bounded.sync(longPrompt, false, false, 8192) == 8192 && bounded.indexedKeys() <= 8192,
            "configurable scan bounds work and nodes at 262K");
    uint32_t matched = 0;
    (void)bounded.propose(longPrompt, 0, 7, &matched);
    require(matched >= 24, "lookup extends a repeated suffix to its true match bucket");
  } catch (const std::exception &error) {
    std::cerr << "MTP depth planner tests FAILED: " << error.what() << '\n';
    return 1;
  }
  std::cout << "MTP depth planner tests passed\n";
  return 0;
}

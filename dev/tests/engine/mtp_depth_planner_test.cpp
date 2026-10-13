// The MTP depth planner (model/MtpDepthPlanner.hpp): per-position
// acceptance, the cycle cost model fitted to measured cycles, and the depths
// it plans per lane, down to none.
//   mtp-depth-planner
#include "TestChecks.hpp"
#include "model/MtpDepthPlanner.hpp"

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
  const double a[4] = {0.020, 0.030, 0.038, 0.046}, bWhole = 0.0025, bDraft = 0.0016, c = 0.0014;
  MtpCostModel costs;
  std::mt19937 rng(3);
  std::uniform_int_distribution<uint32_t> depth(0, 7);
  std::normal_distribution<double> noise(0.0, 0.0005);
  for (int cycle = 0; cycle < 3000; ++cycle) {
    const uint32_t lanes = 1 + cycle % 4;
    uint32_t steps = 0, laneSteps = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t d = depth(rng);
      steps = std::max(steps, d);
      laneSteps += d;
    }
    const bool vocabulary = cycle % 5 != 0;
    const double seconds = a[lanes - 1] + (vocabulary ? bDraft : bWhole) * steps + c * laneSteps + noise(rng);
    costs.observe({lanes, steps, laneSteps, vocabulary}, seconds);
  }
  for (uint32_t lanes = 2; lanes <= 4; ++lanes) {
    const auto theta = costs.coefficients(lanes, true);
    require(std::fabs(theta[0] - a[lanes - 1]) < 0.1 * a[lanes - 1] && std::fabs(theta[1] - bWhole) < 0.2 * bWhole &&
                std::fabs(theta[2] - bDraft) < 0.2 * bDraft && std::fabs(theta[3] - c) < 0.2 * c,
            "cost model: fits each width's base, step and proposal costs from varied cycles (B" +
                std::to_string(lanes) + ")");
  }
  // A lone lane: only the cost of a step with its proposal is seen.
  const auto one = costs.coefficients(1, true);
  require(std::fabs(one[0] - a[0]) < 0.05 * a[0] && std::fabs(one[2] + one[3] - bDraft - c) < 0.1 * (bDraft + c),
          "cost model: fits a lone lane's base and cost per proposal");
  require(costs.coefficients(1, false) == MtpCostModel{}.coefficients(1, false),
          "cost model: the prior ignores what was measured");
  // Measured costs move the plan: with a cheap base a lane keeping three in
  // four drafts fewer than at the prior's ratios.
  MtpCostModel steep;
  for (int cycle = 0; cycle < 400; ++cycle) {
    const uint32_t d = cycle % 8;
    steep.observe({1, d, d, true}, 0.010 + 0.004 * d);
  }
  require(loneDepth(acceptanceAt(0.75), steep, true) < loneDepth(acceptanceAt(0.75), steep, false),
          "plan: a dearer proposal relative to the base drafts shallower, measured only");
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

} // namespace

int main() {
  try {
    acceptance();
    prior();
    fit();
    lanes();
  } catch (const std::exception &error) {
    std::cerr << "MTP depth planner tests FAILED: " << error.what() << '\n';
    return 1;
  }
  std::cout << "MTP depth planner tests passed\n";
  return 0;
}

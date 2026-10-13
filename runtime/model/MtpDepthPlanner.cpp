#include "model/MtpDepthPlanner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::model {
namespace {

// The counts survive a judgment by this much: the pooled ones about the
// last five cycles (as mtpStepsFor's), a position's about the last ten
// judgments of it, which lean on the pooled rate as on two more.
constexpr float kPooledDecay = 0.8F;
constexpr float kAcceptanceDecay = 0.9F;
constexpr float kPooledWeight = 2.0F;

// The prior cost of a cycle relative to its base a = 1 (one live row per
// lane, nothing drafted), from Flash-Next on an M5 Max: a draft step of the
// whole vocabulary about 0.15 (its head alone 0.05), of the draft
// vocabulary 0.10, and each lane's proposal, whose row its draft step and
// the verify route to experts, 0.10.
constexpr std::array<double, 4> kPriorRatios = {1.0, 0.15, 0.10, 0.10};
// How far the fit may stray from the prior, as standard deviations relative
// to the cycle's scale, and a cycle's measurement noise.
constexpr std::array<double, 4> kPriorDeviation = {0.5, 0.05, 0.05, 0.05};
constexpr double kNoise = 0.05;
// A measurement's weight decays by this much per later one (~200 cycles).
constexpr double kForgetting = 0.995;

std::array<double, 4> features(const MtpCostModel::Cycle &cycle) {
  return {1.0, cycle.draftVocabulary ? 0.0 : double(cycle.steps), cycle.draftVocabulary ? double(cycle.steps) : 0.0,
          double(cycle.laneSteps)};
}

// Solves the symmetric positive definite system m x = v (4 x 4).
std::array<double, 4> solve(std::array<std::array<double, 4>, 4> m, std::array<double, 4> v) {
  for (int k = 0; k < 4; ++k) {
    const double pivot = m[k][k];
    if (!(pivot > 0.0)) return v; // never: the prior keeps it definite
    for (int i = k + 1; i < 4; ++i) {
      const double f = m[i][k] / pivot;
      for (int j = k; j < 4; ++j) m[i][j] -= f * m[k][j];
      v[i] -= f * v[k];
    }
  }
  std::array<double, 4> x{};
  for (int i = 3; i >= 0; --i) {
    double sum = v[i];
    for (int j = i + 1; j < 4; ++j) sum -= m[i][j] * x[j];
    x[i] = sum / m[i][i];
  }
  return x;
}

} // namespace

void MtpAcceptance::record(uint32_t drafted, uint32_t kept) noexcept {
  drafted = std::min(drafted, kMtpPlannerPositions);
  if (!drafted) return;
  kept = std::min(kept, drafted);
  const uint32_t judgedPositions = kept < drafted ? kept + 1 : kept;
  for (uint32_t k = 0; k < judgedPositions; ++k) {
    accepted[k] = kAcceptanceDecay * accepted[k] + (k < kept ? 1.0F : 0.0F);
    judged[k] = kAcceptanceDecay * judged[k] + 1.0F;
  }
  pooledAccepted = kPooledDecay * pooledAccepted + float(kept);
  pooledJudged = kPooledDecay * pooledJudged + float(judgedPositions);
}

float MtpAcceptance::rate(uint32_t position) const noexcept {
  if (!position || position > kMtpPlannerPositions) return 0.0F;
  const uint32_t k = position - 1;
  return (accepted[k] + kPooledWeight * pooledRate()) / (judged[k] + kPooledWeight);
}

std::array<double, kMtpPlannerPositions + 1> MtpAcceptance::expectedTokens() const noexcept {
  std::array<double, kMtpPlannerPositions + 1> tokens{};
  double reach = 1.0;
  tokens[0] = 1.0;
  for (uint32_t d = 1; d <= kMtpPlannerPositions; ++d) {
    reach *= rate(d);
    tokens[d] = tokens[d - 1] + reach;
  }
  return tokens;
}

void MtpCostModel::observe(const Cycle &cycle, double seconds) noexcept {
  if (!cycle.lanes || cycle.lanes > kMtpPlannerLanes || !(seconds > 0.0) || !std::isfinite(seconds)) return;
  Fit &fit = fits_[cycle.lanes - 1];
  const auto x = features(cycle);
  if (!fit.count) {
    // The scale of this width's cycles: its first one, by the prior's ratios.
    double ratio = 0.0;
    for (int i = 0; i < 4; ++i) ratio += kPriorRatios[i] * x[i];
    fit.scale = seconds / ratio;
  }
  const double weight = 1.0 / (kNoise * fit.scale * kNoise * fit.scale);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) fit.gram[i][j] = kForgetting * fit.gram[i][j] + weight * x[i] * x[j];
    fit.moments[i] = kForgetting * fit.moments[i] + weight * x[i] * seconds;
  }
  ++fit.count;
  // Ridge toward the prior: (P0 + G) theta = P0 theta0 + m.
  std::array<std::array<double, 4>, 4> m = fit.gram;
  std::array<double, 4> v = fit.moments;
  for (int i = 0; i < 4; ++i) {
    const double deviation = kPriorDeviation[i] * fit.scale;
    const double precision = 1.0 / (deviation * deviation);
    m[i][i] += precision;
    v[i] += precision * kPriorRatios[i] * fit.scale;
  }
  fit.theta = solve(m, v);
  // A cost never falls below nothing.
  for (double &value : fit.theta) value = std::max(value, 0.0);
}

std::array<double, 4> MtpCostModel::coefficients(uint32_t lanes, bool measured) const noexcept {
  if (!measured || !lanes || lanes > kMtpPlannerLanes || !fits_[lanes - 1].count) return kPriorRatios;
  return fits_[lanes - 1].theta;
}

uint64_t MtpCostModel::observations(uint32_t lanes) const noexcept {
  return lanes && lanes <= kMtpPlannerLanes ? fits_[lanes - 1].count : 0;
}

double MtpCostModel::cost(const Cycle &cycle, bool measured) const noexcept {
  const auto theta = coefficients(cycle.lanes, measured);
  const auto x = features(cycle);
  double total = 0.0;
  for (int i = 0; i < 4; ++i) total += theta[i] * x[i];
  return total;
}

MtpDepthPlan planMtpDepths(std::span<const std::array<double, kMtpPlannerPositions + 1>> expected,
                           const MtpCostModel &costs, bool measured, bool draftVocabulary, uint32_t maximumSteps) {
  const uint32_t lanes = static_cast<uint32_t>(expected.size());
  if (!lanes || lanes > kMtpPlannerLanes) throw std::invalid_argument("invalid MTP depth plan batch");
  maximumSteps = std::min(maximumSteps, kMtpPlannerPositions);
  // Every combination of depths: at most 8^4.
  MtpDepthPlan best;
  double bestRate = -1.0;
  std::array<uint32_t, kMtpPlannerLanes> depths{};
  uint64_t combinations = 1;
  for (uint32_t lane = 0; lane < lanes; ++lane) combinations *= maximumSteps + 1;
  for (uint64_t index = 0; index < combinations; ++index) {
    uint64_t rest = index;
    uint32_t steps = 0, laneSteps = 0;
    double tokens = 0.0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      depths[lane] = static_cast<uint32_t>(rest % (maximumSteps + 1));
      rest /= maximumSteps + 1;
      steps = std::max(steps, depths[lane]);
      laneSteps += depths[lane];
      tokens += expected[lane][depths[lane]];
    }
    const double seconds = costs.cost({lanes, steps, laneSteps, draftVocabulary}, measured);
    const double rate = seconds > 0.0 ? tokens / seconds : 0.0;
    // Ties toward the earlier combination: shallower lanes first.
    if (rate > bestRate * (1.0 + 1e-9)) {
      bestRate = rate;
      best.steps = steps;
      best.depths = depths;
    }
  }
  return best;
}

} // namespace splash::model

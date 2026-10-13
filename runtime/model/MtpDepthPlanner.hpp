#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace splash::model {

// How deep an MTP head drafts (Runtime's Qwen3.8-Flash-Next decode): each
// lane by the depth that maximises the batch's expected tokens per second of
// cycle, from each lane's acceptance at every draft position and a cost
// model of the cycle the machine measures, down to drafting nothing.
inline constexpr uint32_t kMtpPlannerPositions = 7;
inline constexpr uint32_t kMtpPlannerLanes = 4;

// Component parameters: production uses these defaults, tests can shorten
// probe/exploration intervals and choose an explicit prior. Zero disables
// the respective interval.
struct MtpDepthParameters final {
  uint32_t probeCycles = 8;
  uint64_t exploreCycles = 32;
  std::array<double, 4> priorRatios{1.0, 0.15, 0.10, 0.10};
};
[[nodiscard]] uint32_t probeMtpDepth(uint32_t depth, uint32_t &idleCycles, uint32_t interval) noexcept;
[[nodiscard]] uint32_t exploreMtpDepth(uint32_t depth, uint32_t maximum, uint64_t cycle,
                                     uint64_t interval) noexcept;

// A request's decayed counts of the proposals the target judged and kept at
// each draft position (a cycle judges its proposals up to the first it
// rejects), and over all positions. A position with few judgments leans on
// the pooled rate.
struct MtpAcceptance final {
  std::array<float, kMtpPlannerPositions> accepted{};
  std::array<float, kMtpPlannerPositions> judged{};
  // The prior: a rate of 0.8 worth five proposals.
  float pooledAccepted = 4.0F;
  float pooledJudged = 5.0F;

  // A cycle drafted `drafted` proposals and the target kept `kept` of them.
  void record(uint32_t drafted, uint32_t kept) noexcept;
  [[nodiscard]] float pooledRate() const noexcept { return pooledAccepted / pooledJudged; }
  // The chance the target keeps proposal `position` (1-based) once it kept
  // the ones before.
  [[nodiscard]] float rate(uint32_t position) const noexcept;
  // The tokens a cycle that drafts d proposals emits on average, d = 0..7:
  // the target's own token and each proposal while all before it are kept.
  [[nodiscard]] std::array<double, kMtpPlannerPositions + 1> expectedTokens() const noexcept;
};

// The cost of a cycle of `lanes` lanes drafting `steps` steps (from the
// draft vocabulary or the whole one) whose lanes draft `laneSteps`
// proposals in all, which each route their row to experts in the draft and
// the verify: a + b steps + c laneSteps, per batch width. Fit to the
// cycles' measured seconds by ridge regression toward a prior with
// forgetting; `measured` false asks for the prior's ratios alone, which no
// measurement moves. Sampled batches use the upstream rule instead.
class MtpCostModel final {
public:
  explicit MtpCostModel(std::array<double, 4> prior = MtpDepthParameters{}.priorRatios);
  struct Cycle {
    uint32_t lanes = 1;
    uint32_t steps = 0;
    uint32_t laneSteps = 0;
    bool draftVocabulary = true;
  };
  void observe(const Cycle &cycle, double seconds) noexcept;
  [[nodiscard]] double cost(const Cycle &cycle, bool measured) const noexcept;
  // The fitted (or prior) a, b of whole-vocabulary steps, b of draft
  // vocabulary steps and c, in seconds (prior: relative to a of 1).
  [[nodiscard]] std::array<double, 4> coefficients(uint32_t lanes, bool measured) const noexcept;
  [[nodiscard]] uint64_t observations(uint32_t lanes) const noexcept;

private:
  struct Fit {
    std::array<std::array<double, 4>, 4> gram{};
    std::array<double, 4> moments{};
    double scale = 0.0;
    uint64_t count = 0;
    std::array<double, 4> theta{};
  };
  std::array<double, 4> priorRatios_;
  std::array<Fit, kMtpPlannerLanes> fits_{};
};

struct MtpDepthPlan final {
  uint32_t steps = 0;
  std::array<uint32_t, kMtpPlannerLanes> depths{};
};

// The depths of a batch's lanes, each at most maximumSteps, maximising
// sum_l expected[l][depth_l] / cost: the batch drafts as many steps as its
// deepest lane.
[[nodiscard]] MtpDepthPlan planMtpDepths(std::span<const std::array<double, kMtpPlannerPositions + 1>> expected,
                                         const MtpCostModel &costs, bool measured, bool draftVocabulary,
                                         uint32_t maximumSteps);

} // namespace splash::model

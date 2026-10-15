// Focused split-tile checks reuse the production MoE test's independent
// fp64 reference and projectionBound policy, including nonlinear intervals.
#define GGUF_MOE_HIDDEN 2560
#define GGUF_MOE_INTERMEDIATE 640
#define GGUF_MOE_EXPERTS 16
#define GGUF_MOE_TOP_K 10
#define main gguf_moe_full_main
#include "gguf_moe_test.mm"
#undef main
#include <cstdlib>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) return 2;
    try {
      MetalBackend backend(argv[1]);
      Buffers b;
      b.input = activations(32 * kHidden);
      b.residual = activations(32 * kHidden);
      b.moe.input = bfloatBuffer(backend, b.input, "input");
      b.moe.residual = bfloatBuffer(backend, b.residual, "residual");
      b.moe.output = zeros(backend, 32 * kHidden * 2, "output");
      const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate};
      for (const Fmt down : {Q51, Q80})
        for (const Fmt sharedGateUp : {Q5K, Q80}) {
          Model m;
          m.router = floating(backend, kExperts, kHidden, 0.05f);
          m.sharedGate = floating(backend, 1, kHidden, 0.05f);
          m.formats = {Q5K, Q5K, down, sharedGateUp, sharedGateUp, Q80};
          for (uint32_t p = 0; p < 3; ++p) {
            const uint32_t N = p == 2 ? kHidden : kIntermediate, K = p == 2 ? kIntermediate : kHidden;
            m.routed[p] = quantized(backend, m.formats[p], kExperts * N, K);
            m.shared[p] = quantized(backend, m.formats[p + 3], N, K);
          }
          m.weights = {m.router.segment, m.sharedGate.segment,
                       {m.routed[0].segment, m.shared[0].segment},
                       {m.routed[1].segment, m.shared[1].segment},
                       {m.routed[2].segment, m.shared[2].segment}};
          std::map<std::pair<uint32_t, uint32_t>, GateUp> products;
          Stats stats;
          for (const char *flag : {"0", "1"}) {
            setenv("SPLASH_EXPERT_TILE", flag, 1);
            for (uint32_t rows = 1; rows <= 8; ++rows) {
              const auto plan = MoE::prefillPlan(shape, rows, MoeConfig{});
              CommandGraph graph;
              allocate(backend, b, plan);
              MoE::add(graph, b.moe, m.weights, plan);
              const auto count = std::count_if(graph.dispatches().begin(), graph.dispatches().end(), [](const auto &d) {
                return d.pipelineName.starts_with("moe_expert_lane");
              });
              require(count == (*flag == '1' ? 2 : 0), "opt-in dispatch selection");
              runPlan(backend, m, b, plan, true, true, products, stats, "split rows=" + std::to_string(rows));
            }
            for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
              const auto plan = MoE::decodePlan(shape, lanes, MoeConfig{});
              const auto full = runPlan(backend, m, b, plan, true, true, products, stats, "split B" + std::to_string(lanes));
              for (uint32_t rows = 1; rows <= 8; ++rows)
                checkLiveRows(backend, m, b, plan, 0, std::vector<uint32_t>(lanes, rows), full, "split live rows");
            }
          }
        }
      unsetenv("SPLASH_EXPERT_TILE");
      std::cout << "expert_tile_test: PASS (existing fp64 interval policy)\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
  }
  return 0;
}

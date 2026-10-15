// GPU time of one sparse MoE layer at the 35B shape (hidden 2048, 256 experts, top 8, intermediate 512) in GGUF
// (by default the 35B UD-Q4_K_M's Q4_K gate/up and Q5_K down experts, a Q8_0 shared expert, F32 router), on the
// device's plans and the other GGUF tile: a 2048-row prefill chunk, decode B1-B4 and prefill chunks of their rows,
// prefill chunks of 64 to 512 rows, then every dispatch of the long chunk, B1 and B4 replayed as its own command. The
// numbers behind the MoE plans of ops/MoE.cpp:
//   gguf-moe-benchmark <metallib> [rounds] [gate/up format] [down format]
// Printed are medians of GPU ms per layer over `rounds` (20) commands. Every row routes to 8 experts of a pool of 24
// per request lane (decode, short chunks) or of all 256 (chunks of 64 rows or more): expert e scores 4 x[e], so the
// first 256 inputs pick the routes. The weights exceed the system cache, so each layer streams its experts from DRAM.
#include "../tests/engine/GgufFormatReference.hpp"
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/MoE.hpp"
#include "metal/MoePrefillPolicy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::BlockMoeWeights;
using splash::ops::MoE;
using splash::ops::MoeBuffers;
using splash::ops::MoeConfig;
using splash::ops::MoeScratchField;
using splash::ops::kMoeScratchFields;
using splash::ops::MoeGgufTile;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using splash::ops::QuantizedSegment;
using namespace gguf_reference;

float bf16(double value) { return float(__bf16(float(value))); }

MetalBuffer upload(MetalBackend &backend, const void *data, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

MetalBuffer zeros(MetalBackend &backend, uint64_t bytes, const char *label) {
  if (!bytes) return {};
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

MetalBuffer bfloatBuffer(MetalBackend &backend, const std::vector<float> &values, const char *label) {
  std::vector<__bf16> bits(values.begin(), values.end());
  return upload(backend, bits.data(), bits.size() * 2, label);
}

// The segments of a repacked [rows, K] tensor in format f and of [rows, K] floats.
QuantizedSegment planeSegment(MetalBackend &backend, Fmt f, const Packed &planes, uint32_t rows, uint32_t K) {
  return QuantizedSegment::planes(
      f, rows, K, upload(backend, planes.w0.data(), planes.w0.size(), "gguf-plane0"),
      kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1.data(), planes.w1.size(), "gguf-plane1")
                                    : MetalBuffer{},
      upload(backend, planes.meta.data(), planes.meta.size(), "gguf-meta"));
}
QuantizedSegment floatSegment(MetalBackend &backend, const std::vector<float> &values, uint32_t rows,
                              uint32_t K) {
  return QuantizedSegment::floats(rows, K, upload(backend, values.data(), values.size() * sizeof(float),
                                                  "gguf-floats"));
}

void allocate(MetalBackend &backend, MoeBuffers &m, const MoePlan &plan) {
  const auto &w = plan.workspace();
  for (const MoeScratchField &field : kMoeScratchFields)
    m.scratch.*field.buffer = zeros(backend, w.*field.bytes, "moe-scratch");
}

int timing(MetalBackend &backend, uint32_t rounds, Fmt gateUpFormat, Fmt downFormat, Fmt sharedFormat,
           const std::string &tile = "", bool flash = false, bool zipf = false) {
  constexpr uint32_t kRowsMax = 2048;
  const uint32_t H = flash ? 2560 : 2048, I = flash ? 640 : 512,
                 E = flash ? 512 : 256, top = flash ? 10 : 8;
  for (const auto &[format, k] : {std::pair{gateUpFormat, H}, std::pair{downFormat, I},
                                  std::pair{sharedFormat, H}, std::pair{sharedFormat, I}})
    if (k % quant_column_unit(kQuantFormats[format].block_elements))
      throw std::invalid_argument("format does not fit the projection K");
  // gate and up hold most of the routed expert weights.
  const MoeShape ggufShape{H, E, top, I, uint32_t(gateUpFormat),
                           quant_register_format(gateUpFormat) && quant_register_format(downFormat) &&
                               quant_register_format(sharedFormat)};
  std::mt19937 local(9);
  // The routing fixture's F32 router, and weights at a model's magnitudes in
  // every format as the tests draw them (scaleRange; an MLX affine format
  // takes bf16 scales and a bias).
  const auto planes = [&](Fmt f, uint32_t rows, uint32_t k) {
    const std::vector<uint8_t> native = makeNative(f, rows, k, local);
    return planeSegment(backend, f, repack(f, native, rows, k, nullptr), rows, k);
  };
  std::vector<float> router(uint64_t{E} * H, 0.0f), sharedGate(H, 0.0f);
  for (uint32_t e = 0; e < E; ++e) router[uint64_t{e} * H + e] = 4.0f;
  const QuantizedSegment routerSegment = floatSegment(backend, router, E, H);
  const QuantizedSegment sharedGateSegment = floatSegment(backend, sharedGate, 1, H);
  const BlockMoeWeights gguf{routerSegment, sharedGateSegment,
                             {planes(gateUpFormat, E * I, H), planes(sharedFormat, I, H)},
                             {planes(gateUpFormat, E * I, H), planes(sharedFormat, I, H)},
                             {planes(downFormat, E * H, I), planes(sharedFormat, H, I)}};
  // Rows route to 8 of a pool of 24 experts per lane of 8 rows (decode) or of all 256 (prefill).
  const auto input = [&](uint32_t rows, uint32_t pool) {
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<float> x(uint64_t{rows} * H);
    std::vector<uint32_t> experts(E);
    std::iota(experts.begin(), experts.end(), 0u);
    for (uint32_t r = 0; r < rows; ++r) {
      if (r % 8 == 0) std::shuffle(experts.begin(), experts.end(), local);
      std::vector<uint32_t> candidates(experts.begin(), experts.begin() + pool);
      std::shuffle(candidates.begin(), candidates.end(), local);
      if (zipf) {
        std::vector<double> weights(E);
        for (uint32_t e = 0; e < E; ++e) weights[e] = 1.0 / std::pow(e + 1.0, 0.8);
        candidates.clear();
        for (uint32_t rank = 0; rank < top; ++rank) {
          std::discrete_distribution<uint32_t> choose(weights.begin(), weights.end());
          const uint32_t e = choose(local);
          candidates.push_back(e); weights[e] = 0;
        }
      }
      for (uint32_t k = 0; k < H; ++k) x[uint64_t{r} * H + k] = bf16(k < E ? 0.05f * unit(local) : unit(local));
      for (uint32_t rank = 0; rank < top; ++rank) x[uint64_t{r} * H + candidates[rank]] = bf16(1.0f - 0.05f * rank);
    }
    return x;
  };
  MoeBuffers b;
  b.residual = zeros(backend, uint64_t{kRowsMax} * H * 2, "residual");
  b.output = zeros(backend, uint64_t{kRowsMax} * H * 2, "output");
  const splash::ops::ExecutionPlans plans(backend.capabilities());
  const auto time = [&](const BlockMoeWeights &weights, const MoePlan &plan) {
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b, weights, plan);
    std::vector<double> samples;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double seconds = backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
      if (i) samples.push_back(seconds * 1e3);   // the first round warms the pipelines
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  const MoeGgufTile device =
      splash::ops::moeGgufTile(splash::ops::gpuFamilyClass(backend.capabilities().appleGpuFamily), ggufShape);
  const MoeGgufTile other = device == MoeGgufTile::Register ? MoeGgufTile::Staged : MoeGgufTile::Register;
  printf("%s, GPU family %u, %u cores: median GPU ms per MoE layer of %u rounds\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, rounds);
  if (tile == "decode") {
    for (uint32_t rows : {1u, 4u, 8u, 16u, 32u}) {
      b.input = bfloatBuffer(backend, input(rows, 24), "input");
      const auto plan = MoE::prefillPlan(ggufShape, rows, MoeConfig{});
      const double layer = time(gguf, plan);
      CommandGraph graph;
      MoE::add(graph, b, gguf, plan);
      const uint32_t live = *static_cast<const uint32_t *>(b.scratch.tileCount.contents());
      const char *fold = std::getenv("SPLASH_MOE_GATHER_FOLD");
      const char *whole = std::getenv("SPLASH_MOE_WHOLE_EXPERT");
      printf("WP5 rows=%u gate_up=%s down=%s shared=%s gather_fold=%s whole_expert=%s layer_ms=%.6f live_tiles=%u dispatches=%zu\n",
             rows, fmtName(gateUpFormat), fmtName(downFormat), fmtName(sharedFormat), fold ? fold : "0", whole ? whole : "0",
             layer, live, graph.dispatches().size());
      for (const auto &[name, seconds] : splash::benchmark::replayDispatches(backend, graph.dispatches()))
        printf("WP5_REPLAY rows=%u kernel=%s ms=%.6f\n", rows, name.c_str(), seconds * 1e3);
    }
    return 0;
  }
  if (!tile.empty()) {
    if (!splash::metal::moePrefillSelfTest(backend)) return 1;
    using splash::ops::MoeExpertTile;
    for (uint32_t rows : {512u, 1024u, 2048u}) {
      b.input = bfloatBuffer(backend, input(rows, E), "input");
      const auto plan32 = MoE::prefillPlan(ggufShape, rows, MoeConfig{MoeExpertTile::M32});
      const auto plan = MoE::prefillPlan(ggufShape, rows,
                                       MoeConfig{tile == "m64" ? MoeExpertTile::M64 : MoeExpertTile::M32});
      const auto snapshot = [&](const MoePlan &p) {
        allocate(backend, b, p);
        CommandGraph graph;
        MoE::add(graph, b, gguf, p);
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        std::vector<uint16_t> values;
        const auto append = [&](const MetalBuffer &buffer, uint64_t offset, uint64_t count) {
          const auto *v = static_cast<const uint16_t *>(buffer.contents()) + offset;
          values.insert(values.end(), v, v + count);
        };
        append(b.output, 0, uint64_t{rows} * H);
        const auto *routeRows = static_cast<const uint32_t *>(b.scratch.routeRows.contents());
        for (uint32_t route = 0; route < rows * (top + 1); ++route) {
          append(b.scratch.expertIntermediate, uint64_t{routeRows[route]} * I, I);
          append(b.scratch.expertOutput, uint64_t{routeRows[route]} * H, H);
        }
        return values;
      };
      const auto reference = snapshot(plan32);
      if (reference != snapshot(plan)) throw std::runtime_error("PF1 live-route memcmp failed before timing");
      const double ms = time(gguf, plan);
      const uint32_t liveTiles = *static_cast<const uint32_t *>(b.scratch.tileCount.contents());
      CommandGraph graph;
      MoE::add(graph, b, gguf, plan);
      std::map<std::string, std::vector<double>> samples;
      for (uint32_t round = 0; round < rounds; ++round)
        for (const auto &[name, seconds] : splash::benchmark::replayDispatches(backend, graph.dispatches()))
          if (name.starts_with("moe_expert_gguf")) samples[name].push_back(seconds * 1e3);
      for (auto &[name, times] : samples) {
        std::sort(times.begin(), times.end());
        const double passMs = times[times.size() / 2];
        const bool pair = name.ends_with("_gate_up");
        const double flops = 2.0 * rows * (top + 1) * H * I * (pair ? 2 : 1);
        double weightBytes = 0;
        const auto *descriptors = static_cast<const MoeTileDescriptor *>(b.scratch.tileDescriptors.contents());
        for (uint32_t t = 0; t < liveTiles; ++t) {
          const auto &f = kQuantFormats[descriptors[t].expert == E ? sharedFormat : (pair ? gateUpFormat : downFormat)];
          weightBytes += double(H) * I / 32 *
                         (f.plane0_bytes + f.plane1_bytes + double(f.meta_bytes) / f.meta_groups) * (pair ? 2 : 1);
        }
        printf("PF1 shape=%s routing=%s tile=%s rows=%u pass=%s ms=%.6f layer_ms=%.6f live_tiles=%u tflops=%.3f staged_weight_gbps=%.3f identity=PASS\n",
               flash ? "flashnext" : "35b", zipf ? "zipf" : "uniform", tile.c_str(), rows,
               pair ? "gate_up" : "down", passMs, ms, liveTiles, flops / (passMs * 1e9), weightBytes / (passMs * 1e6));
      }
    }
    return 0;
  }
  // Prefill chunks first: they also bring the GPU clocks up for the short decode layers.
  b.input = bfloatBuffer(backend, input(kRowsMax, E), "input");
  printf("  prefill %u rows: %.3f\n", kRowsMax, time(gguf, plans.moePrefill(ggufShape, kRowsMax)));
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    b.input = bfloatBuffer(backend, input(lanes * 8, 24), "input");
    MoeConfig config = plans.moeDecode(ggufShape, lanes).configuration();
    config.ggufTile = other;
    const double g = time(gguf, plans.moeDecode(ggufShape, lanes));
    // The register tile takes the formats of quant_register_format only.
    const double o = other == MoeGgufTile::Register && !ggufShape.registerFormats
                         ? std::nan("")
                         : time(gguf, MoE::decodePlan(ggufShape, lanes, config));
    const double gp = time(gguf, plans.moePrefill(ggufShape, lanes * 8));
    printf("  decode B%u: %s %.3f  %s %.3f  | prefill chunk of %u rows: %.3f\n", lanes,
           device == MoeGgufTile::Register ? "register" : "staged", g,
           other == MoeGgufTile::Register ? "register" : "staged", o, lanes * 8, gp);
  }
  for (const uint32_t rows : {64u, 128u, 256u, 512u}) {
    b.input = bfloatBuffer(backend, input(rows, E), "input");
    printf("  prefill chunk of %u rows: %.3f\n", rows, time(gguf, plans.moePrefill(ggufShape, rows)));
  }
  // Where the time goes: every dispatch replayed as its own command.
  for (const auto &[label, plan] : {std::pair{"prefill", plans.moePrefill(ggufShape, kRowsMax)},
                                    std::pair{"B1", plans.moeDecode(ggufShape, 1)},
                                    std::pair{"B4", plans.moeDecode(ggufShape, 4)}}) {
    b.input = bfloatBuffer(backend, input(plan.rows(), plan.rows() == kRowsMax ? E : 24), "input");
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b, gguf, plan);
    std::map<std::string, double> spent;
    for (uint32_t i = 0; i < rounds; ++i)
      for (const auto &[name, seconds] : splash::benchmark::replayDispatches(backend, graph.dispatches()))
        spent[name] += seconds * 1e3 / rounds;
    printf("  %s:", label);
    for (const auto &[name, ms] : spent) printf(" %s %.3f", name.c_str(), ms);
    printf("\n");
  }
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
#ifdef GGUF_MOE_BENCHMARK_E512
    constexpr const char *defaultTile = "decode", *defaultShape = "flashnext";
    constexpr Fmt defaultDown = Q51;
#else
    constexpr const char *defaultTile = "", *defaultShape = "35b";
    constexpr Fmt defaultDown = Q5K;
#endif
    const Fmt gateUp = argc > 3 ? fmtNamed(argv[3]) : Q4K, down = argc > 4 ? fmtNamed(argv[4]) : defaultDown,
              shared = argc > 5 ? fmtNamed(argv[5]) : Q80;
    if (argc < 2 || argc > 9 || gateUp == FMT_COUNT || down == FMT_COUNT || shared == FMT_COUNT) {
      std::cerr << "usage: gguf-moe-benchmark <metallib> [rounds] [gate/up format] [down format] [shared format] [m32|m64|decode] [35b|flashnext] [uniform|zipf]\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const std::string tile = argc > 6 ? argv[6] : defaultTile;
      const std::string shape = argc > 7 ? argv[7] : defaultShape;
      const std::string routing = argc > 8 ? argv[8] : "uniform";
      const uint32_t rounds = argc > 2 ? std::stoul(argv[2]) : 20;
      if (!rounds || (!tile.empty() && tile != "m32" && tile != "m64" && tile != "decode") ||
          (shape != "35b" && shape != "flashnext") || (routing != "uniform" && routing != "zipf"))
        throw std::invalid_argument("expected rounds>0, tile m32|m64|decode, shape 35b|flashnext, routing uniform|zipf");
      return timing(backend, rounds, gateUp, down, shared, tile, shape == "flashnext", routing == "zipf");
    } catch (const std::exception &error) {
      std::cerr << "gguf-moe-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}

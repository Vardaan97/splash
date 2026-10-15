// Synthetic Flash-Next expert pass comparison. Timing is meaningful only on
// the target GPU; correctness is covered separately by expert-tile-test.
#include "../tests/engine/GgufFormatReference.hpp"
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/MoE.hpp"

#import <Metal/Metal.h>

#include <cstdlib>
#include <span>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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


int timing(MetalBackend &backend, uint32_t rounds, Fmt downFormat) {
  constexpr uint32_t H = 2560, I = 640, E = 128, Top = 10;
  std::mt19937 rng(19);
  const auto planes = [&](Fmt f, uint32_t n, uint32_t k) {
    const auto native = makeNative(f, n, k, rng);
    return planeSegment(backend, f, repack(f, native, n, k, nullptr), n, k);
  };
  std::vector<float> router(uint64_t(E) * H, 0), shared(H, 0);
  for (uint32_t e = 0; e < E; ++e) router[uint64_t(e) * H + e] = 4;
  const BlockMoeWeights w{floatSegment(backend, router, E, H), floatSegment(backend, shared, 1, H),
    {planes(Q5K, E * I, H), planes(Q80, I, H)},
    {planes(Q5K, E * I, H), planes(Q80, I, H)},
    {planes(downFormat, E * H, I), planes(Q80, H, I)}};
  const MoeShape shape{H, E, Top, I};
  printf("device=%s family=%u cores=%u H=%u I=%u experts=%u top=%u rounds=%u cache=rotating-expert-bank\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, H, I, E, Top, rounds);
  printf("down,lanes,live_rows,pool,arm,expert_ms,gate_up_ms,down_ms,logical_GBps,weight_bytes,tiles\n");
  for (uint32_t pool : {10u, 24u})
    for (uint32_t lanes = 1; lanes <= 4; ++lanes)
      for (uint32_t live = 1; live <= 8; ++live) {
        std::vector<float> x(uint64_t(lanes) * 8 * H);
        std::uniform_real_distribution<float> unit(-1, 1);
        for (uint32_t row = 0; row < lanes * 8; ++row) {
          for (uint32_t k = 0; k < H; ++k) x[uint64_t(row) * H + k] = bf16(k < E ? 0.01f * unit(rng) : unit(rng));
          for (uint32_t rank = 0; rank < Top; ++rank) {
            const uint32_t e = (row / 8) * 24 + ((row % 8) * 3 + rank) % pool;
            x[uint64_t(row) * H + e] = bf16(1.0f - 0.03f * rank);
          }
        }
        const auto plan = MoE::decodePlan(shape, lanes, MoeConfig{});
        MoeBuffers b;
        b.input = bfloatBuffer(backend, x, "input");
        b.residual = zeros(backend, x.size() * 2, "residual");
        b.output = zeros(backend, x.size() * 2, "output");
        allocate(backend, b, plan);
        for (const char *arm : {"0", "1", "1", "0"}) {
          setenv("SPLASH_EXPERT_TILE", arm, 1);
          CommandGraph graph;
          MoE::add(graph, b, w, plan, 0, std::vector<uint32_t>(lanes, live));
          (void)backend.submitCommandAsync(graph.dispatches()).wait();
          std::vector<splash::metal::ComputeDispatch> experts, prepare;
          for (const auto &d : graph.dispatches())
            if (d.pipelineName.starts_with("moe_expert_")) experts.push_back(d);
            else if (experts.empty()) prepare.push_back(d);
          const uint32_t tiles = *static_cast<const uint32_t *>(b.scratch.tileCount.contents());
          const auto *desc = static_cast<const MoeTileDescriptor *>(b.scratch.tileDescriptors.contents());
          uint64_t bytes = 0;
          for (uint32_t t = 0; t < tiles; ++t)
            bytes += 2 * uint64_t(I) * rowBytes(desc[t].expert == E ? Q80 : Q5K, H) +
                     uint64_t(H) * rowBytes(desc[t].expert == E ? Q80 : downFormat, I);
          const auto median = [&](std::span<const splash::metal::ComputeDispatch> dispatches) {
            std::vector<double> samples;
            for (uint32_t n = 0; n <= rounds; ++n) {
              // Rotate the selected expert set through the weight bank. The
              // router/group/gather preparation is outside the timed command.
              auto *bits = static_cast<__bf16 *>(b.input.contents());
              const uint32_t shift = (n * 31) % E;
              for (uint32_t row = 0; row < lanes * 8; ++row)
                for (uint32_t e = 0; e < E; ++e)
                  bits[uint64_t(row) * H + (e + shift) % E] = __bf16(x[uint64_t(row) * H + e]);
              (void)backend.submitCommandAsync(prepare).wait();
              const double ms = backend.submitCommandAsync(dispatches).wait().gpuSeconds * 1e3;
              if (n) samples.push_back(ms);
            }
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2];
          };
          const double total = median(experts), gate = median(std::span(experts).first(1)), down = median(std::span(experts).last(1));
          printf("%s,%u,%u,%u,%s,%.6f,%.6f,%.6f,%.3f,%llu,%u\n", fmtName(downFormat), lanes, live, pool,
                 *arm == '1' ? "candidate" : "current", total, gate, down, bytes / (total * 1e6),
                 (unsigned long long)bytes, tiles);
          fflush(stdout);
        }
      }
  unsetenv("SPLASH_EXPERT_TILE");
  return 0;
}
} // namespace
int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2 || argc > 4) return 2;
    try {
      MetalBackend backend(argv[1]);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError *error = nil;
      id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
      if (!library) throw std::runtime_error("cannot inspect expert pipeline library");
      for (NSString *name in @[@"moe_expert_gguf_m8_gate_up", @"moe_expert_gguf_m8_a",
                               @"moe_expert_lane_gate_up", @"moe_expert_lane_a"]) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) throw std::runtime_error("cannot inspect expert pipeline");
        printf("pipeline=%s static_tg_bytes=%lu max_threads=%lu simd_width=%lu\n", name.UTF8String,
               (unsigned long)pipeline.staticThreadgroupMemoryLength,
               (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
               (unsigned long)pipeline.threadExecutionWidth);
      }
      const Fmt down = argc > 3 ? fmtNamed(argv[3]) : Q51;
      if (down != Q51 && down != Q80) return 2;
      const uint32_t rounds = argc > 2 ? std::stoul(argv[2]) : 20;
      if (!rounds) return 2;
      return timing(backend, rounds, down);
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
  }
}

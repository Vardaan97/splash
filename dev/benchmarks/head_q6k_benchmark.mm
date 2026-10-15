// Identity matrix and DRAM-cold logits benchmark. No model files required.
// head-q6k METALLIB --test
// head-q6k METALLIB --bench N [rounds] [c64d4|all]
#include "../tests/engine/LinearNumerics.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/BackendInstrumentation.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Linear.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace splash;
using namespace splash::metal;
namespace {
struct Variant { uint32_t columns, depth; std::string name() const {
  return "gguf_head_q6k_c" + std::to_string(columns) + "_d" + std::to_string(depth);
} std::string flag() const { return "c" + std::to_string(columns) + "d" + std::to_string(depth); } };
std::vector<Variant> variants() {
  std::vector<Variant> out;
  for (uint32_t c : {32u, 64u, 128u, 256u}) for (uint32_t d : {0u, 1u, 2u, 4u, 8u})
    if (c * d <= 512 && (c != 256 || !d)) out.push_back({c, d});
  return out;
}
MetalBuffer buffer(MetalBackend &backend, uint64_t size) {
  auto b = backend.allocateBuffer(size, BufferStorage::Shared, "Q6K head benchmark");
  std::memset(b.contents(), 0, size); return b;
}
struct Weights {
  MetalBuffer p0, p1, meta;
  uint64_t bytes() const { return p0.sizeBytes() + p1.sizeBytes() + meta.sizeBytes(); }
};
Weights weights(MetalBackend &backend, uint32_t n, uint32_t k, std::mt19937 &rng) {
  Weights w{buffer(backend, uint64_t(n) * k / 2), buffer(backend, uint64_t(n) * k / 4),
            buffer(backend, uint64_t(n) * k / 256 * 20)};
  for (auto *b : {&w.p0, &w.p1, &w.meta}) {
    auto *words = static_cast<uint32_t *>(b->contents());
    for (uint64_t i = 0; i < b->sizeBytes() / 4; ++i) words[i] = rng();
  }
  auto *meta = static_cast<uint32_t *>(w.meta.contents());
  // Finite signed block scales, exact binary fractions; all 6-bit codes and
  // signed int8 subscales occur. The final halfword is reserved padding.
  for (uint64_t i = 4; i < w.meta.sizeBytes() / 4; i += 5)
    meta[i] = (rng() & 1 ? 0x8000u : 0u) | (0x1800u + (rng() & 0x7ffu));
  return w;
}
void add(CommandGraph &g, const Weights &w, MetalBuffer input, MetalBuffer output,
         uint32_t n, uint32_t k, uint32_t stride, uint32_t offset, bool spread, const Variant *v) {
  const uint32_t columns = v ? v->columns : 64;
  g.add(v ? v->name() : "gguf_decode_q6k_m8_a_f32",
        {input, w.p0, w.p1, w.meta, output, output, output, output},
        GgufDecodeParams{k, 1, stride, offset, uint32_t(spread)}, {n / columns, 1, 1}, {columns, 1, 1});
}
void inputRows(MetalBuffer &b, uint32_t k, uint32_t rows, std::mt19937 &rng) {
  auto *x = static_cast<uint16_t *>(b.contents());
  std::uniform_real_distribution<float> unit(-2.f, 2.f);
  for (uint32_t r = 0; r < 8; ++r) for (uint32_t i = 0; i < k; ++i)
    x[r * k + i] = r < rows ? test::floatToBf16(unit(rng) * (i % 257 == 0 ? 1e5f : 1.f)) : 0x7fc0;
}
void hostSelection(MetalBackend &backend) {
  using namespace splash::ops;
  std::mt19937 rng(2616);
  constexpr uint32_t n = 32768, k = 2560;
  const auto w = weights(backend, n, k, rng);
  Projection p(n, k, BlockWeights{{QuantizedSegment::planes(GGUF_FMT_Q6K, n, k, w.p0, w.p1, w.meta)}});
  p.destination = FloatOutput::Float32;
  auto input = buffer(backend, 8 * k * 2), output = buffer(backend, 8ull * n * 4);
  inputRows(input, k, 8, rng);
  const auto plan = Linear::plan({{n, k}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                 {.tile = LinearTile::GgufStaged, .spread = true}, FloatOutput::Float32);
  auto reference = buffer(backend, 8ull * n * 4);
  CommandGraph base; add(base, w, input, reference, n, k, n, 0, true, nullptr);
  static_cast<void>(backend.submitCommandAsync(base.dispatches()).wait());
  const auto encode = [&] {
    Linear linear(backend.capabilities()); CommandGraph graph;
    (void)linear.add(graph, {.input = input, .output = output}, p, plan);
    return graph;
  };
  for (const char *setting : {"", "0", "typo"}) {
    setenv("SPLASH_HEAD_Q6K", setting, 1);
    auto graph = encode();
    if (graph.dispatches().size() != 1 || graph.dispatches()[0].pipelineName != "gguf_decode_q6k_m8_a_f32")
      throw std::runtime_error("head flag default-off selection");
  }
  for (const auto &v : variants()) {
    setenv("SPLASH_HEAD_Q6K", v.flag().c_str(), 1);
    auto graph = encode();
    if (graph.dispatches().size() != 1 || graph.dispatches()[0].pipelineName != v.name() ||
        graph.dispatches()[0].threadgroups.x != n / v.columns ||
        graph.dispatches()[0].threadsPerThreadgroup.x != v.columns)
      throw std::runtime_error("head flag/grid selection " + v.flag());
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    if (std::memcmp(reference.contents(), output.contents(), 8ull * n * 4))
      throw std::runtime_error("host-selected head memcmp " + v.flag());
  }
  setenv("SPLASH_HEAD_Q6K", "1", 1);
  {
    auto graph = encode();
    if (graph.dispatches()[0].pipelineName != "gguf_head_q6k_c128_d0")
      throw std::runtime_error("head flag=1 alias");
  }
  Linear linear(backend.capabilities());
  const auto fallback = [&](const Projection &projection, const LinearPlan &other, const char *expected) {
    auto in = buffer(backend, uint64_t(other.storageRows()) * k * 2);
    auto out = buffer(backend, uint64_t(other.storageRows()) * projection.outputSize * 4);
    const auto size = other.scratchSize();
    LinearBuffers b{.input = in, .output = out};
    if (size.partials) b.scratch.partials = buffer(backend, size.partials);
    if (size.counters) b.scratch.counters = buffer(backend, size.counters);
    CommandGraph graph;
    (void)linear.add(graph, b, projection, other);
    if (graph.dispatches().size() != 1 || graph.dispatches()[0].pipelineName != expected)
      throw std::runtime_error("head eligibility fallback");
  };
  fallback(p, Linear::plan({{n, k}, 16, LinearPhase::Decode, LinearEpilogue::None}, {}, FloatOutput::Float32),
           "gguf_decode_q6k_m16_a_f32");
  fallback(p, Linear::plan({{n, k}, 8, LinearPhase::Decode, LinearEpilogue::None},
                          {.splits = 2}, FloatOutput::Float32), "gguf_decode_q6k_m8_a_f32");
  Projection bf16 = p; bf16.destination = FloatOutput::BFloat16;
  fallback(bf16, Linear::plan({{n, k}, 8, LinearPhase::Decode, LinearEpilogue::None}, {}, FloatOutput::BFloat16),
           "gguf_decode_q6k_m8_a");
  auto small = p.leadingRows(backend, 4096); small.destination = FloatOutput::Float32;
  fallback(small, Linear::plan({{4096, k}, 8, LinearPhase::Decode, LinearEpilogue::None}, {}, FloatOutput::Float32),
           "gguf_decode_q6k_m8_a_f32");
  Projection padded(n, k, BlockWeights{{QuantizedSegment::planes(GGUF_FMT_Q6K, n - 64, k, w.p0, w.p1, w.meta)}});
  padded.destination = FloatOutput::Float32;
  fallback(padded, plan, "gguf_decode_q6k_m8_a_f32");
  setenv("SPLASH_PROFILE_HEAD_SHAPES", "1", 1);
  for (const char *setting : {"0", "1"}) {
    setenv("SPLASH_HEAD_Q6K", setting, 1);
    auto graph = encode();
    BackendInstrumentation::setDispatchProfiling(backend, true);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    BackendInstrumentation::setDispatchProfiling(backend, false);
    const auto timings = BackendInstrumentation::takeDispatchProfile(backend);
    const std::string expected = graph.dispatches()[0].pipelineName +
        "[N=32768,K=2560,S=1,spread=1,offset=0,bytes=69468160]";
    if (timings.size() != 1 || timings.front().pipelineName != expected)
      throw std::runtime_error("profile head-shape metadata");
  }
  unsetenv("SPLASH_PROFILE_HEAD_SHAPES");
  std::puts("baseline/candidate instrumented head-shape attribution PASS");
  unsetenv("SPLASH_HEAD_Q6K");
  std::puts("host flag selection, grids and N=32768 K=2560 fp32 outputs PASS");
}
void identity(MetalBackend &backend) {
  std::mt19937 rng(2606); uint32_t checks = 0;
  for (uint32_t k : {256u, 768u, 2304u, 2560u})
    for (uint32_t n : {64u, 128u, 192u, 256u, 576u, 4096u}) {
      const auto w = weights(backend, n, k, rng);
      const uint32_t stride = n + 128, offset = 64;
      const uint64_t bytes = uint64_t(8) * stride * 4;
      auto input = buffer(backend, uint64_t(8) * k * 2);
      auto reference = buffer(backend, bytes + 512), candidate = buffer(backend, bytes + 512);
      auto ref = backend.view(reference, 256, bytes), out = backend.view(candidate, 256, bytes);
      for (uint32_t rows = 1; rows <= 8; ++rows) for (bool spread : {false, true}) {
        inputRows(input, k, rows, rng);
        std::memset(reference.contents(), 0xa5, bytes + 512);
        CommandGraph base; add(base, w, input, ref, n, k, stride, offset, spread, nullptr);
        static_cast<void>(backend.submitCommandAsync(base.dispatches()).wait());
        for (const auto &v : variants()) {
          if (n % v.columns) continue; // whole output tiles, as ops::Linear requires
          std::memset(candidate.contents(), 0xa5, bytes + 512);
          CommandGraph graph; add(graph, w, input, out, n, k, stride, offset, spread, &v);
          static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
          // Compare active fp32 rows, untouched column padding and guards.
          // Padded NaNs are compared too: both tiles execute eight rows.
          if (std::memcmp(reference.contents(), candidate.contents(), bytes + 512))
            throw std::runtime_error("memcmp " + v.flag() + " N=" + std::to_string(n) + " K=" +
              std::to_string(k) + " rows=" + std::to_string(rows) + " spread=" + std::to_string(spread));
          ++checks;
        }
      }
      std::printf("identity N=%u K=%u rows=1..8 spread=0,1 ok\n", n, k); std::fflush(stdout);
    }
  std::printf("PASS %u full-buffer memcmp checks (guards, padding, wrap and staging tails)\n", checks);
}
void benchmark(MetalBackend &backend, uint32_t n, uint32_t rounds, const std::string &choice) {
  constexpr uint32_t k = 2560;
  if (!n || n % 256 || !rounds) throw std::invalid_argument("N must fill 256 columns; rounds > 0");
  std::mt19937 rng(26);
  // More than the SLC, at least two distinct full head images, as the
  // projection harness uses a cold ring. Sequential reuse inside one command.
  const uint64_t bytes = uint64_t(n) * (k / 32 * 24 + k / 256 * 20);
  const uint32_t copies = std::max<uint32_t>(2, uint32_t(((768ull << 20) + bytes - 1) / bytes));
  std::vector<Weights> ring;
  for (uint32_t c = 0; c < copies; ++c) ring.push_back(weights(backend, n, k, rng));
  auto input = buffer(backend, 8 * k * 2), output = buffer(backend, uint64_t(8) * n * 4);
  std::vector<Variant> selected;
  for (const auto &v : variants()) if (choice == "all" || v.flag() == choice) selected.push_back(v);
  if (selected.empty()) throw std::invalid_argument("unknown candidate");
  std::printf("%s N=%u K=2560, %u cold copies, %llu bytes/pass; median of %u AB/BA rounds\n",
              backend.capabilities().deviceName.c_str(), n, copies, (unsigned long long)bytes, rounds);
  for (uint32_t rows = 1; rows <= 8; ++rows) {
    inputRows(input, k, rows, rng);
    auto reference = buffer(backend, uint64_t(8) * n * 4);
    CommandGraph base; add(base, ring[0], input, reference, n, k, n, 0, true, nullptr);
    static_cast<void>(backend.submitCommandAsync(base.dispatches()).wait());
    for (const auto &v : selected) {
      CommandGraph graph; add(graph, ring[0], input, output, n, k, n, 0, true, &v);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      if (std::memcmp(reference.contents(), output.contents(), 8ull * n * 4))
        throw std::runtime_error("exact-shape memcmp " + v.flag() + " rows=" + std::to_string(rows));
    }
  }
  std::puts("exact-shape rows=1..8 full-buffer memcmp PASS");
  for (uint32_t rows : {1u, 8u}) {
    inputRows(input, k, rows, rng);
    std::vector<std::vector<double>> samples(selected.size() + 1);
    const auto time = [&](uint32_t index) {
      CommandGraph graph;
      for (const auto &w : ring) add(graph, w, input, output, n, k, n, 0, true, index ? &selected[index - 1] : nullptr);
      return backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds * 1e3 / copies;
    };
    for (uint32_t i = 0; i < samples.size(); ++i) for (double spent = 0; spent < 100;) spent += time(i) * copies;
    for (uint32_t r = 0; r < rounds; ++r) for (uint32_t j = 0; j < samples.size(); ++j) {
      const uint32_t i = r % 2 ? uint32_t(samples.size() - 1 - j) : j;
      samples[i].push_back(time(i));
    }
    double baseline = 0;
    for (uint32_t i = 0; i < samples.size(); ++i) {
      auto &s = samples[i]; std::sort(s.begin(), s.end()); const double ms = s[s.size() / 2];
      if (!i) baseline = ms;
      std::printf("rows=%u %-10s %.6f ms %.2f GB/s %.3fx [min %.6f max %.6f] bits=PASS\n", rows,
                  i ? selected[i - 1].flag().c_str() : "baseline", ms, bytes / (ms * 1e6), baseline / ms, s.front(), s.back());
    }
    std::fflush(stdout);
  }
}
}
int main(int argc, char **argv) {
  @autoreleasepool { try {
    if (argc < 3) throw std::invalid_argument("head-q6k METALLIB --test | --bench N [rounds] [variant|all]");
    MetalBackend backend(argv[1]);
    if (std::string(argv[2]) == "--test" && argc == 3) { hostSelection(backend); identity(backend); }
    else if (std::string(argv[2]) == "--bench" && argc >= 4 && argc <= 6)
      benchmark(backend, uint32_t(std::stoul(argv[3])), argc > 4 ? uint32_t(std::stoul(argv[4])) : 20,
                argc > 5 ? argv[5] : "all");
    else throw std::invalid_argument("invalid arguments");
    return 0;
  } catch (const std::exception &e) { std::fprintf(stderr, "head-q6k FAIL: %s\n", e.what()); return 1; } }
}

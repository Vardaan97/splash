#pragma once
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "ops/Linear.hpp"
#include "ops/Qwen4.hpp"
#include <cstring>
#include <random>
#include <stdexcept>

namespace splash::test {
// Force staged plans on every GPU; compare the actual standalone kernels,
// including the bf16 rounding boundary, rather than a host approximation.
inline void hyperFusionIdentity(metal::MetalBackend &backend) {
  using namespace ops;
  using namespace gguf_reference;
  std::mt19937 rng(814);
  const auto buffer = [&](uint64_t bytes) { return sharedBuffer(backend, bytes); };
  const auto upload = [&](const auto &values) {
    auto b = buffer(values.size() * sizeof(values[0]));
    std::memcpy(b.contents(), values.data(), b.sizeBytes());
    return b;
  };
  const auto projection = [&](uint32_t n, uint32_t k) {
    const auto native = makeNative(Q80, n, k, rng);
    const auto packed = repack(Q80, native, n, k, nullptr);
    return Projection(n, k, BlockWeights{{QuantizedSegment::planes(Q80, n, k,
        upload(packed.w0), {}, upload(packed.meta))}});
  };
  const auto same = [&](const auto &a, const auto &b, uint64_t bytes, const char *what) {
    if (std::memcmp(a.contents(), b.contents(), bytes)) throw std::runtime_error(what);
  };
  const Linear linear(backend.capabilities());
  const auto down = projection(320, 10240);
  for (uint32_t rows : {8u, 16u, 24u, 32u})
    for (uint32_t splits : {1u, 2u, 4u, 8u})
      for (bool spread : {false, true})
        for (uint32_t inject : {0u, 1u}) {
        const LinearConfig config{.tile = LinearTile::GgufStaged, .splits = splits, .spread = spread};
        auto plain = Linear::plan({{320, 10240}, rows, LinearPhase::Decode}, config, FloatOutput::BFloat16);
        auto fused = Linear::plan({{320, 10240}, rows, LinearPhase::Decode, LinearEpilogue::HyperLow},
                                   config, FloatOutput::BFloat16);
        const uint32_t storage = plain.storageRows();
        std::vector<uint16_t> x(storage * 10240);
        for (auto &v : x) v = floatToBf16(float(int(rng() % 2001) - 1000) / 1000);
        auto input = upload(x), ref = buffer(storage * 320 * 2), got = buffer(storage * 320 * 2);
        const auto size = fused.scratchSize();
        LinearScratch scratch{.partials = buffer(std::max<uint64_t>(size.partials, 16)),
                              .counters = buffer(std::max<uint64_t>(size.counters, 16))};
        std::memset(scratch.counters.contents(), 0, scratch.counters.sizeBytes());
        std::vector<float> stream(storage * 10240), norms(10240, 1.0f), injection(4 * 10240, 0.01f);
        for (auto &v : stream) v = float(int(rng() % 2001) - 1000) / 500;
        auto streams = upload(stream), normWeights = upload(norms), injectWeights = upload(injection);
        auto weights = buffer(storage * 4 * sizeof(float));
        metal::CommandGraph graph;
        Qwen4::addNorm(graph, streams, {normWeights, true}, inject ? injectWeights : metal::MetalBuffer{},
                      input, weights, {2560, 4, 320}, storage);
        linear.add(graph, {.input = input, .output = ref, .scratch = scratch}, down, plain);
        Qwen4::addLow(graph, ref, {2560, 4, 320}, rows);
        linear.add(graph, {.input = input, .output = got, .scratch = scratch}, down, fused);
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        same(ref, got, uint64_t(rows) * 320 * 2, "hyper low differs bitwise");
        const auto *counts = static_cast<const uint32_t *>(scratch.counters.contents());
        for (uint64_t i = 0; i < size.counters / 4; ++i)
          if (counts[i]) throw std::runtime_error("hyper low leaves counters nonzero");
      }
  const auto up = projection(10240, 320);
  for (uint32_t rows : {8u, 16u, 24u, 32u})
    for (bool spread : {false, true})
      for (uint32_t inject : {0u, 1u}) {
        const LinearConfig config{.tile = LinearTile::GgufStaged, .spread = spread};
        auto plain = Linear::plan({{10240, 320}, rows, LinearPhase::Decode}, config, FloatOutput::BFloat16);
        auto fused = Linear::plan({{10240, 320}, rows, LinearPhase::Decode, LinearEpilogue::HyperMix},
                                  config, FloatOutput::BFloat16);
        const uint32_t storage = plain.storageRows();
        std::vector<uint16_t> x(storage * 320), norm(storage * 10240);
        for (auto &v : x) v = floatToBf16(float(int(rng() % 2001) - 1000) / 1000);
        for (auto &v : norm) v = floatToBf16(float(int(rng() % 2001) - 1000) / 500);
        auto input = upload(x), normalized = upload(norm);
        auto gateRef = buffer(storage * 10240 * 2), gateGot = buffer(storage * 10240 * 2);
        auto mixedRef = buffer(storage * 2560 * 2), mixedGot = buffer(storage * 2560 * 2);
        std::memset(gateGot.contents(), 0xA5, gateGot.sizeBytes());
        std::memset(mixedGot.contents(), 0xA5, mixedGot.sizeBytes());
        auto counters = buffer(fused.scratchSize().counters);
        std::memset(counters.contents(), 0, counters.sizeBytes());
        std::vector<float> stream(storage * 10240), norms(10240, 1.0f), injection(4 * 10240, 0.01f);
        for (auto &v : stream) v = float(int(rng() % 2001) - 1000) / 500;
        auto streams = upload(stream), normWeights = upload(norms), injectWeights = upload(injection);
        auto weights = buffer(storage * 4 * sizeof(float));
        metal::CommandGraph graph;
        Qwen4::addNorm(graph, streams, {normWeights, true}, inject ? injectWeights : metal::MetalBuffer{},
                      normalized, weights, {2560, 4, 320}, storage);
        linear.add(graph, {.input = input, .output = gateRef}, up, plain);
        Qwen4::addMix(graph, normalized, gateRef, mixedRef, {2560, 4, 320}, rows);
        linear.add(graph, {.input = input, .output = gateGot, .scratch = {.counters = counters},
                          .hyperNormalized = normalized, .hyperMixed = mixedGot}, up, fused);
        // Reuse the arrival counters in the same command, testing reset.
        linear.add(graph, {.input = input, .output = gateGot, .scratch = {.counters = counters},
                          .hyperNormalized = normalized, .hyperMixed = mixedGot}, up, fused);
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        same(gateRef, gateGot, uint64_t(rows) * 10240 * 2, "hyper gate differs bitwise");
        same(mixedRef, mixedGot, uint64_t(rows) * 2560 * 2, "hyper mixed differs bitwise");
        for (const auto &[output, columns] : {std::pair{gateGot, 10240u}, std::pair{mixedGot, 2560u}}) {
          const auto *bytes = static_cast<const uint8_t *>(output.contents());
          for (uint64_t i = uint64_t(rows) * columns * 2; i < output.sizeBytes(); ++i)
            if (bytes[i] != 0xA5) throw std::runtime_error("hyper mix writes a padding row");
        }
        const auto *counts = static_cast<const uint32_t *>(counters.contents());
        for (uint64_t i = 0; i < counters.sizeBytes() / 4; ++i)
          if (counts[i]) throw std::runtime_error("hyper mix leaves counters nonzero");
      }

  const auto out = projection(2560, 6144);
  for (uint32_t rows : {8u, 16u, 24u, 32u})
    for (uint32_t splits : {1u, 2u, 4u, 8u})
      for (bool spread : {false, true})
        for (uint32_t inject : {0u, 1u}) {
          const LinearConfig config{.tile = LinearTile::GgufStaged, .splits = splits, .spread = spread};
          auto plain = Linear::plan({{2560, 6144}, rows, LinearPhase::Decode}, config, FloatOutput::BFloat16);
          auto fused = Linear::plan({{2560, 6144}, rows, LinearPhase::Decode, LinearEpilogue::HyperCombine},
                                    config, FloatOutput::BFloat16);
          const uint32_t storage = plain.storageRows();
          std::vector<uint16_t> x(storage * 6144);
          std::vector<float> stream(storage * 10240), weight(storage * 4, float(inject));
          for (auto &v : x) v = floatToBf16(float(int(rng() % 2001) - 1000) / 1000);
          for (auto &v : stream) v = float(int(rng() % 2001) - 1000) / 500;
          if (inject) for (auto &v : weight) v = 0.1f + float(rng() % 1000001) / 555557.0f;
          auto input = upload(x), ref = upload(stream), got = upload(stream), weights = upload(weight);
          auto branch = buffer(storage * 2560 * 2), unused = buffer(storage * 2560 * 2);
          std::memset(unused.contents(), 0xA5, unused.sizeBytes());
          const auto size = fused.scratchSize();
          LinearScratch scratch{.partials = buffer(std::max<uint64_t>(size.partials, 16)),
                                .counters = buffer(std::max<uint64_t>(size.counters, 16))};
          std::memset(scratch.counters.contents(), 0, scratch.counters.sizeBytes());
          metal::CommandGraph graph;
          linear.add(graph, {.input = input, .output = branch, .scratch = scratch}, out, plain);
          Qwen4::addCombine(graph, ref, branch, weights, {2560, 4, 320}, rows);
          linear.add(graph, {.input = input, .output = unused, .scratch = scratch,
                            .hyperCombine = {got, weights}}, out, fused);
          static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
          same(ref, got, ref.sizeBytes(), "hyper projection streams differ bitwise");
          const auto *bytes = static_cast<const uint8_t *>(unused.contents());
          for (uint64_t i = 0; i < unused.sizeBytes(); ++i)
            if (bytes[i] != 0xA5) throw std::runtime_error("hyper combine writes branch");
        }

}
} // namespace splash::test

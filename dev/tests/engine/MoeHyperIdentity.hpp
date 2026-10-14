#pragma once
#include "TestBuffers.hpp"
#include "LinearNumerics.hpp"
#include "ops/Qwen4.hpp"
#include "metal/abi/MoE.h"
#include <cstring>
#include <random>
#include <stdexcept>
namespace splash::test {
// Exhaust every row count and every legal first/last bound. The route
// indirection models both live expert rows and dropped routes' shared rows.
inline void moeHyperIdentity(metal::MetalBackend &backend) {
  std::mt19937 rng(419);
  const auto upload = [&](const auto &v) {
    auto b = sharedBuffer(backend, v.size() * sizeof(v[0]));
    std::memcpy(b.contents(), v.data(), b.sizeBytes());
    return b;
  };
  constexpr uint32_t hidden = 256, routes = 11;
  for (uint32_t rows = 1; rows <= 32; ++rows) {
    std::vector<uint16_t> expert(rows * routes * hidden), zero(rows * hidden, 0);
    std::vector<float> routing(rows * routes), streams(rows * 4 * hidden), weights(rows * 4);
    std::vector<uint32_t> indices(rows * routes);
    for (auto &v : expert) v = floatToBf16(float(int(rng() % 2001) - 1000) / 300);
    for (auto &v : routing) v = float(rng() % 1001) / 1000;
    for (auto &v : streams) v = float(int(rng() % 2001) - 1000) / 500;
    auto experts = upload(expert), residual = upload(zero), routingBuffer = upload(routing);
    auto branch = sharedBuffer(backend, rows * hidden * 2);
    for (uint32_t first = 0; first < 8; ++first)
      for (uint32_t last = first; last <= 8; ++last)
        for (uint32_t inject : {0u, 1u}) {
          for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t s = 0; s < routes; ++s)
              indices[r * routes + s] = (r % 8 < first || r % 8 >= last) ? r * routes + 10 :
                                        (r * routes + s + 3) % (rows * routes);
          for (auto &v : weights) v = inject ? 0.1f + float(rng() % 1000001) / 555557.0f : 0.0f;
          auto routeRows = upload(indices), ref = upload(streams), got = upload(streams), w = upload(weights);
          metal::CommandGraph graph;
          const MoeCombineParams params{rows, hidden, routes};
          graph.add("moe_combine", {experts, routeRows, routingBuffer, residual, branch}, params,
                    {rows, 1, 1}, {256, 1, 1});
          ops::Qwen4::addCombine(graph, ref, branch, w, {hidden, 4, 64}, rows);
          graph.add("moe_combine_hyper", {experts, routeRows, routingBuffer, residual, got, w}, params,
                    {rows, 1, 1}, {256, 1, 1});
          static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
          if (std::memcmp(ref.contents(), got.contents(), ref.sizeBytes()))
            throw std::runtime_error("MoE hyper streams differ bitwise");
        }
  }
}
} // namespace splash::test

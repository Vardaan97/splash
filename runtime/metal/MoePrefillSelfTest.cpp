#include "metal/MoePrefillPolicy.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace splash::metal {
// Small, deterministic nonzero planes exercise every staged dequantizer.
// No counter, split-K, spread, or inject mode is used by this prefill tile.
bool moePrefillSelfTest(MetalBackend &backend) {
  constexpr uint32_t K = 256, N = GGUF_TILE_COLUMNS, Rows = GGUF_MOE_PREFILL_ROWS;
  const auto buffer = [&](uint64_t bytes) {
    auto b = backend.allocateBuffer(std::max<uint64_t>(bytes, 1), BufferStorage::Shared, "m64-selftest");
    std::memset(b.contents(), 0, b.sizeBytes());
    return b;
  };
  try {
    auto x = buffer(Rows * K * 2), ref = buffer(Rows * N * 2), got = buffer(Rows * N * 2);
    auto gate = buffer(Rows * N * 2), aux = buffer(Rows * N * 2);
    auto t32 = buffer(2 * sizeof(MoeTileDescriptor)), t64 = buffer(sizeof(MoeTileDescriptor));
    auto c32 = buffer(sizeof(uint32_t)), c64 = buffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(c32.contents()) = 2;
    *static_cast<uint32_t *>(c64.contents()) = 1;
    for (uint32_t i = 0; i < Rows * K; ++i)
      static_cast<__bf16 *>(x.contents())[i] = __bf16(float(int(i * 17 % 101) - 50) / 128);
    for (uint32_t fmt = 0; fmt < GGUF_FMT_COUNT; ++fmt) {
      const auto &f = kQuantFormats[fmt];
      auto w0 = buffer(N * (K / 32) * f.plane0_bytes);
      auto w1 = buffer(N * (K / 32) * f.plane1_bytes);
      auto meta = buffer(N * (K / 32 / f.meta_groups) * f.meta_bytes);
      // Bytes below 0x40 also keep direct half weights/scales finite.
      for (auto b : {w0, w1, meta})
        for (uint64_t i = 0; i < b.sizeBytes(); ++i)
          static_cast<uint8_t *>(b.contents())[i] = uint8_t(1 + (i * 13 + fmt * 7) % 47);
      for (uint32_t live : {1u, 8u, 16u, 32u, 33u, 64u}) {
        *static_cast<MoeTileDescriptor *>(t64.contents()) = {0, live};
        auto *d = static_cast<MoeTileDescriptor *>(t32.contents());
        d[0] = {0, std::min(live, 32u)}; d[1] = {0, live > 32 ? live - 32 : 0};
        *static_cast<uint32_t *>(c32.contents()) = live > 32 ? 2 : 1;
        const MoeGgufExpertParams p{K, N, 1, fmt, fmt, 0};
        CommandGraph graph;
        const auto pass = [&](const char *name, MetalBuffer tiles, MetalBuffer count, MetalBuffer out,
                              MetalBuffer a, uint32_t columns, uint32_t groups) {
          graph.add(name, {x, tiles, count, w0, w1, meta, w0, w1, meta, out, a}, p,
                    {columns, groups, 1}, {GGUF_MOE_PREFILL_THREADS, 1, 1});
        };
        pass("moe_expert_gguf_m32_a", t32, c32, ref, aux, 1, 2);
        pass("moe_expert_gguf_m64_a", t64, c64, got, aux, 1, 1);
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        if (std::memcmp(ref.contents(), got.contents(), live * N * 2))
          throw MetalBackendError("m64_a differs, format " + std::to_string(fmt));
        CommandGraph pair;
        pair.add("moe_expert_gguf_m32_gate_up",
                 {x, t32, c32, w0, w1, meta, w0, w1, meta, w0, w1, meta, w0, w1, meta, ref},
                 p, {2, 2, 1}, {GGUF_MOE_PREFILL_THREADS, 1, 1});
        pair.add("moe_expert_gguf_m64_gate_up",
                 {x, t64, c64, w0, w1, meta, w0, w1, meta, w0, w1, meta, w0, w1, meta, got},
                 p, {1, 1, 1}, {GGUF_MOE_PREFILL_THREADS, 1, 1});
        pair.add("moe_expert_gguf_m64_a", {x, t64, c64, w0, w1, meta, w0, w1, meta, gate, aux},
                 p, {1, 1, 1}, {GGUF_MOE_PREFILL_THREADS, 1, 1});
        pair.add("moe_expert_gguf_m64_g", {x, t64, c64, w0, w1, meta, w0, w1, meta, aux, gate},
                 p, {1, 1, 1}, {GGUF_MOE_PREFILL_THREADS, 1, 1});
        static_cast<void>(backend.submitCommandAsync(pair.dispatches()).wait());
        if (std::memcmp(ref.contents(), got.contents(), live * N * 2) ||
            std::memcmp(ref.contents(), aux.contents(), live * N * 2))
          throw MetalBackendError("m64_gate_up/g differs, format " + std::to_string(fmt));
      }
    }
    std::fprintf(stderr, "kernel-selftest: moe_m64 PASS on Apple%u\n", backend.capabilities().appleGpuFamily);
    return true;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "kernel-selftest: moe_m64 differs on Apple%u: %s; fallback\n",
                 backend.capabilities().appleGpuFamily, e.what());
    return false;
  }
}
}

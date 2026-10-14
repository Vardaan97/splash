#include "ops/AttentionPolicy.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/PagedAttention.hpp"
#include "metal/abi/Qwen4.h"
#include "metal/abi/PrefillPipe.h"
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
namespace splash::ops {
namespace {
thread_local bool baselineOnly = false;
std::atomic<bool> passed{false};
const std::string &requested() {
  static const std::string mode = [] {
    const char *p = std::getenv("SPLASH_ATTN_PIPE");
    return p && (std::string_view(p) == "prob2" || std::string_view(p) == "qk2" || std::string_view(p) == "rows4") ? std::string(p) : std::string();
  }();
  return mode;
}
}
void attentionPipeBaseline(bool baseline) { baselineOnly = baseline; }
uint32_t attentionPipeRows() {
  return !baselineOnly && passed.load() && requested() == "rows4" ? SPLASH_PREFILL_PIPE_TILE_ROWS : kv::kPrefillAttentionTileRows;
}
std::string_view attentionPipeReduce(std::string_view baseline) {
  if (attentionPipeRows() == kv::kPrefillAttentionTileRows) return baseline;
  if (baseline == "prefill_attention_reduce") return "prefill_attention_reduce_rows4";
  if (baseline == "prefill_attention_reduce_kv2_g8") return "prefill_attention_reduce_kv2_g8_rows4";
  return "prefill_attention_reduce_kv2_g12_rows4";
}
bool attentionStatsEnabled() {
  static const bool enabled = [] {
    const char *p = std::getenv("SPLASH_ATTN_STATS");
    return p && std::string_view(p) == "1";
  }();
  return enabled;
}
void logAttentionPages(metal::MetalBackend &backend, uint64_t request, const void *mask,
    uint32_t words, uint32_t committed, uint32_t rows, bool sparse) {
  if (!attentionStatsEnabled()) return;
  (void)backend;
  const auto *bits = static_cast<const uint32_t *>(mask);
  for (uint32_t first = 0; first < rows; first += kv::kPrefillAttentionTileRows) {
    const uint32_t active = std::min(kv::kPrefillAttentionTileRows, rows-first);
    const uint32_t pages = (committed+first+active+kv::kPageTokens-1)/kv::kPageTokens;
    uint32_t attended = 0;
    for (uint32_t page = 0; page < pages; ++page) {
      uint32_t unionBits = 0;
      if (sparse) {
        const uint32_t block = page * kv::kPageTokens / QWEN4_QSA_BLOCK;
        for (uint32_t row = 0; row < active; ++row)
          unionBits |= bits[uint64_t(first+row)*words+block/32] >> (block%32);
      }
      attended += !sparse || (unionBits & ((1u << (kv::kPageTokens/QWEN4_QSA_BLOCK))-1));
    }
    std::cerr << "attention-stats: last-attention-layer request=" << request << " tile=" << first/kv::kPrefillAttentionTileRows
              << " committed=" << committed << " attended_pages=" << attended << " total_pages=" << pages << '\n';
  }
}
std::string attentionPipePipeline(std::string_view baseline) {
  const bool enabled = !baselineOnly && passed.load() && !requested().empty() &&
      (requested() != "rows4" || baseline.starts_with("prefill_"));
  return std::string(baseline) + (enabled ? "_" + requested() : "");
}
namespace {
bool probe(metal::MetalBackend &backend, const std::string &mode) {
  using namespace metal;
  for (const auto format : {kv::Format::Int8, kv::Format::BFloat16}) {
    for (const uint32_t heads : {16u, 24u}) {
      for (const uint32_t kvHeads : {2u, 4u}) {
        if (kvHeads == 4 && heads != 24) continue;
        constexpr uint32_t pages = 16, stride = 32, rows = 8, history = 257;
        const uint32_t data = kvHeads * SPLASH_TARGET_KV_BLOCK_TOKENS * SPLASH_KV_HEAD_DIMENSION *
                              (format == kv::Format::Int8 ? 1 : 2);
        const uint32_t scale = format == kv::Format::Int8 ? kvHeads * SPLASH_TARGET_KV_BLOCK_TOKENS * 4 : 0;
        auto allocate = [&](uint64_t bytes) { return backend.allocateBuffer(bytes, BufferStorage::Shared, "attention-selftest"); };
        auto extent = allocate(uint64_t(pages) * 2 * (data + scale));
        auto *bytes = static_cast<std::byte *>(extent.contents());
        std::memset(bytes, 0, extent.sizeBytes());
        for (uint32_t page = 0; page < pages; ++page) {
          for (uint32_t tensor : {SPLASH_KV_KEYS, SPLASH_KV_VALUES}) {
            auto *slab = bytes + splash_kv_offset(pages, data, scale, 0, tensor, page);
            if (format == kv::Format::Int8) {
              for (uint32_t i = 0; i < data; ++i)
                reinterpret_cast<int8_t *>(slab)[i] = int((i * 13 + page * 7 + tensor * 3) % 127) - 63;
            } else {
              for (uint32_t i = 0; i < data / 2; ++i) {
                const float value = float(int((i * 13 + page * 7 + tensor * 3) % 127) - 63) / 64;
                reinterpret_cast<uint16_t *>(slab)[i] = std::bit_cast<uint32_t>(value) >> 16;
              }
            }
          }
          if (scale)
            for (uint32_t tensor : {SPLASH_KV_KEY_SCALES, SPLASH_KV_VALUE_SCALES}) {
              auto *scales = reinterpret_cast<float *>(bytes + splash_kv_offset(pages, data, scale, 0, tensor, page));
              for (uint32_t i = 0; i < scale / 4; ++i) scales[i] = 0.015625f + (i % 3) * 0.001f;
            }
        }
        auto table = allocate(pages * sizeof(SplashKvPage));
        for (uint32_t p = 0; p < pages; ++p)
          static_cast<SplashKvPage *>(table.contents())[p] = splash_kv_page_entry(extent.gpuAddress(), p);
        auto queries = allocate(uint64_t(stride) * heads * SPLASH_KV_HEAD_DIMENSION * 2);
        for (uint64_t i = 0; i < queries.sizeBytes() / 2; ++i) {
          const float value = float(int(i * 17 % 127) - 63) / 128;
          static_cast<uint16_t *>(queries.contents())[i] = std::bit_cast<uint32_t>(value) >> 16;
        }
        const uint32_t group = heads / kvHeads, fused = rows * group;
        auto mask = allocate(rows * 4 * sizeof(uint32_t));
        for (uint32_t r = 0; r < rows; ++r)
          for (uint32_t w = 0; w < 4; ++w)
            static_cast<uint32_t *>(mask.contents())[r * 4 + w] = w == 2 ? 0u : (0x55555555u >> (r & 1));
        const std::string suffix = kvHeads == 4 ? "" : heads == 24 ? "_kv2_g12" : "_kv2_g8";
        const std::string baseline = "prefill_attention_" + std::string(scale ? "q8" : "bf16") + "_split" + suffix;
        for (uint32_t splits : {1u, 2u, 4u, 8u, 16u, 128u}) {
          const uint64_t partialBytes = uint64_t(kvHeads) * splits * fused * SPLASH_KV_HEAD_DIMENSION * 4;
          const uint64_t statBytes = uint64_t(kvHeads) * splits * fused * 2 * 4;
          auto partial = allocate(partialBytes), stats = allocate(statBytes);
          const SplashPrefillAttentionParams p{history, rows, stride, pages,
              splash_kv_layer(pages, data, scale, 0), splits};
          auto output = allocate(queries.sizeBytes());
          std::vector<std::byte> referencePartial(partialBytes), referenceStats(statBytes), referenceOutput(output.sizeBytes());
          for (uint32_t arm = 0; arm < 2; ++arm) {
            std::memset(partial.contents(), 0xa5, partialBytes);
            std::memset(stats.contents(), 0xa5, statBytes);
            std::memset(output.contents(), 0, output.sizeBytes());
            CommandGraph graph;
            const std::string name = baseline + (arm ? "_" + mode : "");
            const bool rows4 = arm && mode == "rows4";
            const uint32_t tiles = rows4 ? 2 : 1;
            if (kvHeads == 2 && heads == 24)
              graph.add(name, {queries, partial, stats, table}, p, {mask}, Qwen4QsaMaskParams{4, 0}, {kvHeads, tiles, splits});
            else graph.add(name, {queries, partial, stats, table}, p, {kvHeads, tiles, splits});
            graph.add("prefill_attention_reduce" + suffix + (rows4 ? "_rows4" : ""),
                      {partial, stats, output}, p, {kvHeads, rows4 ? fused/2 : fused, tiles});
            (void)backend.submitCommandAsync(graph.command()).wait();
            if (!arm) {
              std::memcpy(referencePartial.data(), partial.contents(), partialBytes);
              std::memcpy(referenceStats.data(), stats.contents(), statBytes);
              std::memcpy(referenceOutput.data(), output.contents(), output.sizeBytes());
            } else if (mode == "rows4") {
              if (std::memcmp(referenceOutput.data(), output.contents(), output.sizeBytes())) return false;
            } else if (std::memcmp(referencePartial.data(), partial.contents(), partialBytes) ||
                       std::memcmp(referenceStats.data(), stats.contents(), statBytes)) return false;
          }
        }
      }
    }
  }
  return true;
}
}
void attentionKernelSelfTest(metal::MetalBackend &backend) {
  if (baselineOnly) return;
  const char *test = std::getenv("SPLASH_KERNEL_SELFTEST");
  const bool explicitTest = test && std::string_view(test) == "1";
  if (requested().empty() && !explicitTest) return;
  // Apple9 compiles these MPP variants but cannot prove Apple10 NA identity.
  // Remain unavailable on Apple9; never infer Apple10 promotion from it.
  if (backend.capabilities().appleGpuFamily < 10) {
    std::cerr << "kernel-selftest: attention pipe unavailable on Apple" << backend.capabilities().appleGpuFamily << '\n';
    return;
  }
  bool ok = true;
  for (const std::string mode : {"qk2", "prob2", "rows4"}) {
    if (!explicitTest && requested() != mode) continue;
    try {
      const bool same = probe(backend, mode);
      std::cerr << "kernel-selftest: " << mode << (same ? " identical" : " differs") << " on Apple10\n";
      ok &= same;
    } catch (const std::exception &e) {
      std::cerr << "kernel-selftest: " << mode << " unavailable: " << e.what() << '\n';
      ok = false;
    }
  }
  passed.store(ok);
  if (explicitTest && !ok) throw std::runtime_error("attention kernel self-test failed; fallback forced");
}
} // namespace splash::ops

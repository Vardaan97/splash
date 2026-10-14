#include "metal/KernelSelfTest.hpp"
#include "ops/PrefillCompactPolicy.hpp"
#include "metal/abi/PrefillCompact.h"
#include "ops/BufferExtent.hpp"
#include <atomic>
#include <bit>
#include <cmath>
#include <vector>
#include <fstream>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
namespace splash::ops {
namespace {
thread_local bool baselineOnly = false;
std::atomic<bool> ready{false};
constexpr uint32_t splits = SPLASH_PREFILL_COMPACT_SPLITS;
constexpr uint32_t heads = 2, group = 12;
constexpr uint32_t fused = SPLASH_PREFILL_ATTENTION_TILE_ROWS * group;
uint64_t statWords(uint32_t rows) {
  return uint64_t(kv::prefillAttentionTiles(rows)) * heads * splits * fused * 2;
}
// Full physical-context capacity handles dense/adversarial bitmaps without
// trusting model top-k constants. The padding supports every gathered cell.
constexpr uint32_t listStride = 1 +
    (SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS + QWEN4_QSA_BLOCK - 1) / QWEN4_QSA_BLOCK;
}
void prefillCompactBaseline(bool baseline) { baselineOnly = baseline; }
bool prefillCompactRequested() {
  static const bool requested = [] {
    const char *flag = std::getenv("SPLASH_PREFILL_QSA_COMPACT");
    if (flag && std::string_view(flag) == "1")
      std::cerr << "PF7: production enable refused: long-context quality gate pending; use validate only in the gate queue\n";
    return flag && std::string_view(flag) == "validate";
  }();
  return requested;
}
void capturePrefillDistribution(const float *row, uint32_t vocabulary) {
  static const std::string path = [] {
    const char *p = std::getenv("SPLASH_PREFILL_QSA_CAPTURE");
    return p ? std::string(p) : std::string();
  }();
  if (path.empty()) return;
  static std::mutex mutex;
  std::lock_guard lock(mutex);
  static std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(&vocabulary), sizeof(vocabulary));
  output.write(reinterpret_cast<const char *>(row), uint64_t(vocabulary) * sizeof(float));
  output.flush();
  if (!output) throw std::runtime_error("PF7 full-vocabulary capture write failed");
}
bool prefillCompactReady() { return !baselineOnly && ready.load(); }
AttentionWorkspace prefillCompactWorkspace(uint32_t rows) {
  const uint64_t tiles = kv::prefillAttentionTiles(rows);
  return {tiles * heads * splits * fused * SPLASH_KV_HEAD_DIMENSION * sizeof(float),
          (statWords(rows) + tiles * listStride) * sizeof(uint32_t)};
}
void addPrefillCompact(metal::CommandGraph &graph, SplashKvLayer layer,
    metal::MetalBuffer queries, metal::MetalBuffer output, metal::MetalBuffer partials,
    metal::MetalBuffer statistics, metal::MetalBuffer table,
    const kv::ChunkedPrefillParams &chunk, QsaMask mask, kv::Format format) {
  // U2 uses a short history. This marker proves long-context collectors
  // exercised production graph selection rather than merely passing U2.
  if (chunk.committed_tokens >= SPLASH_PREFILL_TOKEN_BUDGET) {
    static thread_local bool reported = false;
    if (!reported) {
      std::cerr << "PF7 runtime compact dispatched context=" << chunk.committed_tokens << " rows=" << chunk.chunk_tokens << '\n';
      reported = true;
    }
  }
  const auto workspace = prefillCompactWorkspace(chunk.chunk_tokens);
  requireBytes(partials, workspace.partialsBytes, "compact attention partials");
  requireBytes(statistics, workspace.statisticsBytes, "compact attention statistics/list");
  const uint32_t requiredWords = (uint64_t(chunk.committed_tokens) + chunk.chunk_tokens +
      QWEN4_QSA_BLOCK * 32 - 1) / (QWEN4_QSA_BLOCK * 32);
  if (!mask.words || mask.words < requiredWords) throw std::invalid_argument("short compact QSA bitmap stride");
  requireBytes(mask.bits, (uint64_t(mask.row0) + chunk.chunk_tokens) * mask.words * sizeof(uint32_t), "compact QSA bitmap");
  const SplashPrefillCompactParams p{{chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
      chunk.page_table_entries, layer, splits}, {mask.words, mask.row0},
      static_cast<uint32_t>(statWords(chunk.chunk_tokens)), listStride};
  const uint32_t tiles = kv::prefillAttentionTiles(chunk.chunk_tokens);
  graph.add("qwen4_qsa_prefill_compact", {mask.bits, statistics}, p, {tiles, 1, 1});
  graph.add(format == kv::Format::Int8 ? "prefill_attention_q8_compact_kv2_g12" : "prefill_attention_bf16_compact_kv2_g12",
      {queries, partials, statistics, table, mask.bits}, p, {heads, tiles, splits});
  graph.add("prefill_attention_reduce_kv2_g12", {partials, statistics, output}, p.attention,
      {heads, fused, tiles});
}

namespace {
bool compactProbe(metal::MetalBackend &backend, const std::string &) {
  using namespace metal;
  for (const auto format : {kv::Format::Int8, kv::Format::BFloat16}) {
    for (const uint32_t heads : {24u}) {
      for (const uint32_t kvHeads : {2u}) {
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
        const uint32_t group = heads / kvHeads;
        auto mask = allocate(rows * 4 * sizeof(uint32_t));
        for (uint32_t r = 0; r < rows; ++r)
          for (uint32_t w = 0; w < 4; ++w)
            static_cast<uint32_t *>(mask.contents())[r * 4 + w] = w == 2 ? 0u : (0x55555555u >> (r & 1));
        const std::string suffix = kvHeads == 4 ? "" : heads == 24 ? "_kv2_g12" : "_kv2_g8";
        const std::string baseline = "prefill_attention_" + std::string(scale ? "q8" : "bf16") + "_split" + suffix;
        const auto workspace = prefillCompactWorkspace(rows);
        const auto baselinePlan = PagedAttention::prefillPlan(rows, heads, {1, kvHeads, SPLASH_KV_HEAD_DIMENSION, format});
        auto partial = allocate(std::max(workspace.partialsBytes, baselinePlan.workspace.partialsBytes));
        auto stats = allocate(std::max(workspace.statisticsBytes, baselinePlan.workspace.statisticsBytes));
        auto output = allocate(queries.sizeBytes());
        const kv::ChunkedPrefillParams chunk{history, rows, stride, pages,
              splash_kv_layer(pages, data, scale, 0)};
        const auto plan = PagedAttention::prefillPlan(rows, heads, {1, kvHeads, SPLASH_KV_HEAD_DIMENSION, format});
        std::vector<uint16_t> reference(output.sizeBytes() / 2), repeated(reference.size());
        for (uint32_t pattern = 0; pattern < 4; ++pattern) {
          for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t w = 0; w < 4; ++w)
              static_cast<uint32_t *>(mask.contents())[r * 4 + w] =
                  pattern == 0 ? ~0u : pattern == 1 ? (0x55555555u >> (r & 1)) :
                  pattern == 2 ? (w == 0 ? 0u : ~0u) : 0u;
          // A poisoned row0 prefix detects accidental indexing through row 0.
          for (uint32_t arm = 0; arm < 3; ++arm) {
            std::memset(output.contents(), 0, output.sizeBytes());
            CommandGraph graph;
            if (!arm) {
              const SplashPrefillAttentionParams p{history, rows, stride, pages, chunk.kv, plan.splits};
              graph.add(baseline, {queries, partial, stats, table}, p,
                        {mask}, Qwen4QsaMaskParams{4, 0}, plan.splitGroups);
              graph.add("prefill_attention_reduce_kv2_g12", {partial, stats, output}, p, plan.reduceGroups);
            } else addPrefillCompact(graph, chunk.kv, queries, output, partial, stats, table,
                                     chunk, {mask, 4, 0}, format);
            (void)backend.submitCommandAsync(graph.command()).wait();
            if (!arm) std::memcpy(reference.data(), output.contents(), output.sizeBytes());
            else if (arm == 1) {
              std::memcpy(repeated.data(), output.contents(), output.sizeBytes());
              double dot = 0, aa = 0, bb = 0;
              float maximum = 0;
              for (uint32_t h = 0; h < kvHeads; ++h)
                for (uint32_t i = 0; i < rows * group * SPLASH_KV_HEAD_DIMENSION; ++i) {
                  const uint64_t index = uint64_t(h) * stride * group * SPLASH_KV_HEAD_DIMENSION + i;
                  const float a = std::bit_cast<float>(uint32_t(reference[index]) << 16);
                  const float b = std::bit_cast<float>(uint32_t(repeated[index]) << 16);
                  if (!std::isfinite(a) || !std::isfinite(b)) return false;
                  dot += a * b; aa += a * a; bb += b * b;
                  maximum = std::max(maximum, std::fabs(a - b));
                }
              if (maximum > 0.02f || ((aa || bb) && (!(aa && bb) || dot / std::sqrt(aa * bb) < 0.999))) return false;
            } else if (std::memcmp(repeated.data(), output.contents(), output.sizeBytes())) return false;
          }
        }
      }
    }
  }
  return true;
}
}
void prefillCompactSelfTest(metal::MetalBackend &backend) {
  if (baselineOnly) return;
  const char *test = std::getenv("SPLASH_KERNEL_SELFTEST");
  const bool explicitTest = test && std::string_view(test) == "1";
  if (!prefillCompactRequested()) return;
  if (backend.capabilities().appleGpuFamily != 10) {
    std::cerr << "kernel-selftest: PF7 unavailable on Apple" << backend.capabilities().appleGpuFamily << '\n';
    return;
  }
  bool ok = false;
  try { ok = compactProbe(backend, ""); }
  catch (const std::exception &e) { std::cerr << "kernel-selftest: PF7 " << e.what() << '\n'; }
  ready.store(ok);
  std::cerr << "kernel-selftest: PF7 " << (ok ? "tolerance pass; repeat bitwise" : "failed; fallback forced") << '\n';
  if (explicitTest && !ok) throw std::runtime_error("PF7 device self-test failed");
}
namespace { [[maybe_unused]] const bool registered = metal::registerKernelSelfTest(prefillCompactSelfTest); }
} // namespace splash::ops

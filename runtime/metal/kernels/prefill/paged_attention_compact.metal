#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/abi/PrefillCompact.h"
// Port of the compact decode design, 6c81c78: ascending bitmap union,
// gather into Page32-shaped threadgroup operands, causal per-row mask,
// online softmax and fixed-order LSE merge. This changes numerics (class N).

[[max_total_threads_per_threadgroup(SPLASH_PREFILL_COMPACT_THREADS)]]
kernel void qwen4_qsa_prefill_compact(
    device const uint *mask [[buffer(0)]], device uint *scratch [[buffer(1)]],
    constant SplashPrefillCompactParams &p [[buffer(2)]],
    uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  // Bounded serial scan per query tile, shared by both KV heads. A full-mask
  // capacity (not a top-k assumption) handles dense and adversarial bitmaps.
  if (tid) return;
  const uint first = tile * SPLASH_PREFILL_ATTENTION_TILE_ROWS;
  if (first >= p.attention.rows) return;
  const uint rows = min(uint(SPLASH_PREFILL_ATTENTION_TILE_ROWS), p.attention.rows - first);
  const uint visible = p.attention.committed_tokens + first + rows;
  const uint blocks = (visible + QWEN4_QSA_BLOCK - 1) / QWEN4_QSA_BLOCK;
  device uint *list = scratch + p.list_offset + ulong(tile) * p.list_stride;
  uint count = 0;
  for (uint word = 0; word < p.mask.mask_words && word * 32 < blocks; ++word) {
    uint bits = 0;
    for (uint row = 0; row < rows; ++row)
      bits |= mask[ulong(p.mask.row0 + first + row) * p.mask.mask_words + word];
    while (bits) {
      const uint bit = ctz(bits), block = word * 32 + bit;
      if (block < blocks) list[1 + count++] = block;
      bits &= bits - 1;
    }
  }
  const uint padded = (count + QWEN4_QSA_COMPACT_BLOCKS - 1) / QWEN4_QSA_COMPACT_BLOCKS * QWEN4_QSA_COMPACT_BLOCKS;
  for (uint i = count; i < padded; ++i) list[1 + i] = QWEN4_QSA_NO_BLOCK;
  list[0] = count;
}

template <uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_compact_softmax_lane(threadgroup const float *scores, threadgroup bfloat *probabilities,
                                        threadgroup float *row_max, threadgroup float *row_sum,
                                        threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
                                        threadgroup const float *key_scales, threadgroup const float *value_scales,
                                        threadgroup const uint *blocks, uint committed_tokens, uint active_rows,
                                        uint thread_index, device const uint *block_mask, uint mask_words) {
  constexpr uint N = SplashKvPageTokens;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = N / TokensPerLane;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  const uint query_row = fused_row / QueryHeadsPerKVHead;
  const uint limit = committed_tokens + min(query_row, active_rows - 1) + 1;
  device const uint *row_bits = block_mask + ulong(min(query_row, active_rows - 1)) * mask_words;
  const uint b0 = blocks[column / 4], b1 = blocks[column / 4 + 1];
  uint allowed = 0;
  if (b0 != QWEN4_QSA_NO_BLOCK && ((row_bits[b0 / 32] >> (b0 % 32)) & 1u)) allowed |= 0x0Fu;
  if (b1 != QWEN4_QSA_NO_BLOCK && ((row_bits[b1 / 32] >> (b1 % 32)) & 1u)) allowed |= 0xF0u;
  // Tokens past the row's causal limit: positions b * 4 + t.
  uint visible = 0;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    const uint b = j < 4 ? b0 : b1;
    if (b != QWEN4_QSA_NO_BLOCK && b * 4 + (j & 3) < limit) visible |= 1u << j;
  }
  allowed &= visible;
  threadgroup const float4 *scores4 = reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N + column);
  threadgroup const float4 *ks4 = reinterpret_cast<threadgroup const float4 *>(key_scales + column);
  threadgroup const float4 *vs4 = reinterpret_cast<threadgroup const float4 *>(value_scales + column);
  float score[TokensPerLane];
  {
    float4 low = scores4[0], high = scores4[1];
    low *= ks4[0];
    high *= ks4[1];
    low *= 0.0625f;
    high *= 0.0625f;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = ((allowed >> j) & 1u) ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = ((allowed >> j) & 1u) ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max ? 1.0f : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f) atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  const float4 low_scales = vs4[0], high_scales = vs4[1];
  const float4 low(((allowed >> 0) & 1u) ? probability[0] * low_scales.x : 0.0f,
                   ((allowed >> 1) & 1u) ? probability[1] * low_scales.y : 0.0f,
                   ((allowed >> 2) & 1u) ? probability[2] * low_scales.z : 0.0f,
                   ((allowed >> 3) & 1u) ? probability[3] * low_scales.w : 0.0f);
  const float4 high(((allowed >> 4) & 1u) ? probability[4] * high_scales.x : 0.0f,
                    ((allowed >> 5) & 1u) ? probability[5] * high_scales.y : 0.0f,
                    ((allowed >> 6) & 1u) ? probability[6] * high_scales.z : 0.0f,
                    ((allowed >> 7) & 1u) ? probability[7] * high_scales.w : 0.0f);
  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(probabilities + fused_row * N + column);
  probabilities4[0] = bfloat4(low);
  probabilities4[1] = bfloat4(high);
}


template <uint KVHeads, uint QueryHeadsPerKVHead, typename CacheElement>
[[max_total_threads_per_threadgroup(SPLASH_PREFILL_COMPACT_THREADS)]]
kernel void prefill_attention_compact(
    device bfloat *queries [[buffer(0)]], device float *partials [[buffer(1)]],
    device float *statistics [[buffer(2)]], device const SplashKvPage *page_table [[buffer(3)]],
    device const uint *block_mask [[buffer(4)]], constant SplashPrefillCompactParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint RowsPerTile = SPLASH_PREFILL_ATTENTION_TILE_ROWS;
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = SplashKvHeadDimension;
  constexpr uint Blocks = QWEN4_QSA_COMPACT_BLOCKS;
  static_assert(Blocks * 4 == N, "a tile gathers one page's worth of blocks");
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr bool Alias = !Quantized;
  constexpr uint TileBytes = N * D * sizeof(CacheElement);
  constexpr uint RegionBytes = Alias ? (TileBytes > M * N * 4 ? TileBytes : M * N * 4) : TileBytes;
  alignas(16) threadgroup uchar region[RegionBytes];   // keys [token][dimension], then values [dimension][token]
  alignas(16) threadgroup float own_scores[Alias ? 4 : M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup CacheElement *tile = reinterpret_cast<threadgroup CacheElement *>(region);
  threadgroup float *scores = Alias ? reinterpret_cast<threadgroup float *>(region) : own_scores;
  alignas(16) threadgroup float key_scales[N];
  alignas(16) threadgroup float value_scales[N];
  threadgroup uint blocks[Blocks];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float previous_scale[M];
  threadgroup atomic_uint rescale;
  const uint kv_head = group.x, tile_index = group.y, split = group.z;
  constant SplashPrefillAttentionParams &params = p.attention;
  const uint tile_start = tile_index * RowsPerTile;
  if (!splash_prefill_attention_contract_valid(params) || kv_head >= KVHeads ||
      tile_start >= params.rows || split >= params.split_count || !p.mask.mask_words) return;
  const uint committed_tokens = params.committed_tokens + tile_start;
  const uint active_rows = min(RowsPerTile, params.rows - tile_start);
  const ulong slot = (ulong(tile_index) * KVHeads + kv_head) * params.split_count + split;
  device const uint *list = reinterpret_cast<device const uint *>(statistics) + p.list_offset +
                            ulong(tile_index) * p.list_stride;
  device const uint *ids = list + 1;
  device const uint *lane_mask = block_mask + ulong(p.mask.row0 + tile_start) * p.mask.mask_words;
  // Fixed ascending cell order within each split; the reduce recomputes the
  // same page-derived number of written slots, including empty compact slots.
  const uint pages = splash_attention_pages(committed_tokens + active_rows);
  const uint per_split = splash_attention_pages_per_split(pages, params.split_count);
  const uint written = (pages + per_split - 1) / per_split;
  if (split >= written) return;
  const uint tiles = (list[0] + Blocks - 1) / Blocks;
  const uint per_tile_split = (tiles + written - 1) / written;
  const uint tile_begin = min(tiles, split * per_tile_split), tile_end = min(tiles, tile_begin + per_tile_split);
  device bfloat *tile_queries = queries + (ulong(kv_head) * params.chunk_stride + tile_start) * QueryHeadsPerKVHead * D;
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.template slice<N, M>(0, 0);
  tensor<threadgroup CacheElement, dextents<int, 2>, tensor_inline> kt(tile, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  tensor<threadgroup CacheElement, dextents<int, 2>, tensor_inline> vtt(tile, dextents<int, 2>{N, D}, array<int, 2>{1, N});
  const SplashKvAddressing<KVHeads, CacheElement> addressing(params.kv, kv_head);
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  auto v0 = vtt.template slice<N, D>(0, 0);
  constexpr auto qk_descriptor = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor = matmul2d_descriptor(M, D, N, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
  const bool running_full = uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index)
    if (running_full || running.is_valid_element(index)) running[index] = 0.0f;

  for (uint t = tile_begin; t < tile_end; ++t) {
    if (thread_index < Blocks) blocks[thread_index] = ids[t * Blocks + thread_index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Keys: thread (token row r, chunk c of 32 elements) copies its block's token row; the scales of token r with it.
    {
      constexpr uint Chunk = 32 * sizeof(CacheElement) / 16;   // uint4 per 32 elements
      const uint r = thread_index / 8, c = thread_index % 8;
      const uint b = blocks[r / 4];
      threadgroup uint4 *dst = reinterpret_cast<threadgroup uint4 *>(tile + r * D + c * 32);
      if (b == QWEN4_QSA_NO_BLOCK) {
#pragma unroll
        for (uint i = 0; i < Chunk; ++i) dst[i] = uint4(0);
        if (c == 0) { key_scales[r] = 1.0f; value_scales[r] = 1.0f; }
      } else {
        const SplashKvPageTensors<CacheElement> page = addressing.page(page_table[b / Blocks]);
        const uint token = (b % Blocks) * 4 + r % 4;
        device const uint4 *src = reinterpret_cast<device const uint4 *>(page.keys + token * D + c * 32);
#pragma unroll
        for (uint i = 0; i < Chunk; ++i) dst[i] = src[i];
        if (c == 0) {
          if constexpr (Quantized) { key_scales[r] = page.key_scales[token]; value_scales[r] = page.value_scales[token]; }
          else { key_scales[r] = 1.0f; value_scales[r] = 1.0f; }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto page_scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    qk.run(q0, k0, page_scores);
    if constexpr (Alias) threadgroup_barrier(mem_flags::mem_threadgroup);   // every simdgroup is done with the keys
    page_scores.store(st.template slice<N, M>(0, 0));
    if (thread_index == 0) atomic_store_explicit(&rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      constexpr uint Lanes = N / 8 * RowsPerTile * QueryHeadsPerKVHead;
      for (uint l = thread_index; l < Lanes; l += 256)
        splash_compact_softmax_lane<QueryHeadsPerKVHead, RowsPerTile>(
            scores, probabilities, row_max, row_sum, previous_scale, &rescale, key_scales, value_scales, blocks,
            committed_tokens, active_rows, l, lane_mask, p.mask.mask_words);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index)) continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    // Values: thread d copies dimension d's four tokens of each block (4 or 8 bytes) over the keys (or scores).
    {
      typedef typename conditional<Quantized, uint, uint2>::type Word;
      const uint d = thread_index;
      threadgroup Word *dst = reinterpret_cast<threadgroup Word *>(tile + d * N);
#pragma unroll
      for (uint j = 0; j < Blocks; ++j) {
        const uint b = blocks[j];
        if (b == QWEN4_QSA_NO_BLOCK) { dst[j] = Word(0); continue; }
        const SplashKvPageTensors<CacheElement> page = addressing.page(page_table[b / Blocks]);
        dst[j] = *reinterpret_cast<device const Word *>(page.values + d * N + (b % Blocks) * 4);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pv.run(p0, v0, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(target.template slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

using PrefillCompactKernel = void(device bfloat *, device float *, device float *, device const SplashKvPage *,
    device const uint *, constant SplashPrefillCompactParams &, uint3, uint);
template [[host_name("prefill_attention_q8_compact_kv2_g12")]] kernel PrefillCompactKernel prefill_attention_compact<2, 12, int8_t>;
template [[host_name("prefill_attention_bf16_compact_kv2_g12")]] kernel PrefillCompactKernel prefill_attention_compact<2, 12, bfloat>;

#pragma once
#include "metal/kernels/common/paged_attention_tile.h"

template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile,
          typename CacheElement, bool ProbabilityDoubleBuffer, bool QueryLookahead>
inline void splash_paged_attention_pipe_tile(
    device bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale_storage,
    uint thread_index, device const uint *block_mask = nullptr, uint mask_words = 0,
    uint partition_visible = 0) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = SplashKvHeadDimension;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(partition_visible ? partition_visible : visible_tokens);
  uint per_split = splash_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto key_type = tensor(static_cast<device CacheElement *>(nullptr),
                         dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type = tensor(static_cast<device CacheElement *>(nullptr),
                           dextents<int, 2>{N, D}, array<int, 2>{1, N});
  const SplashKvAddressing<KVHeads, CacheElement> addressing(kv, kv_head);
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = key_type.template slice<D, N>(0, 0);
  auto v0 = value_type.template slice<N, D>(0, 0);
  // QK writes a complete page score tile, the softmax applies an INT8 page's
  // key scales and PV accumulates the running output.
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  // Uniform capacity partitions the logical tile across eight 32-lane groups.
  // Equality proves no padding; otherwise only valid entries may be accessed.
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  uint iteration = 0;
  // Separate score storage lets QK(p+1) precede softmax(p). The token order,
  // partition and running rescale/PV chain remain exactly the baseline's.
  uint page = page_begin;
  while (page < page_end && block_mask &&
         !splash_qsa_page_attended(block_mask, mask_words, active_rows, page)) ++page;
  for (; page < page_end; ++iteration) {
    const uint parity = iteration & 1u;
    threadgroup atomic_uint *rescale = rescale_storage + (ProbabilityDoubleBuffer ? parity : 0);
    threadgroup float *current_scores = scores + (QueryLookahead ? parity * M * N : 0);
    threadgroup bfloat *current_probabilities = probabilities + (ProbabilityDoubleBuffer ? parity * M * N : 0);
    auto current_st = tensor(current_scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto current_pt = tensor(current_probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto current_p0 = current_pt.template slice<N, M>(0, 0);
    uint next = page + 1;
    while (next < page_end && block_mask &&
           !splash_qsa_page_attended(block_mask, mask_words, active_rows, next)) ++next;
    const SplashKvPageTensors<CacheElement> tensors =
        addressing.page(page_table[page]);
    uint token_start = page * N;
    auto kt = tensor(tensors.keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt = tensor(tensors.values, dextents<int, 2>{N, D}, array<int, 2>{1, N});
    device const float *key_scales = tensors.key_scales;
    device const float *value_scales = tensors.value_scales;

    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    // One full-dimension product initializes the score CT through MPP.
    auto ks = kt.template slice<D, N>(0, 0);
    if (!QueryLookahead || iteration == 0) {
      qk.run(q0, ks, page_scores);
      page_scores.store(current_st.template slice<N, M>(0, 0));
    }
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (QueryLookahead) {
      if (next < page_end) {
        const auto nt = addressing.page(page_table[next]);
        auto nkt = tensor(nt.keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
        qk.run(q0, nkt.template slice<D, N>(0, 0), page_scores);
        auto nst = tensor(scores + (1u - parity) * M * N,
                          dextents<int, 2>{N, M}, array<int, 2>{1, N});
        page_scores.store(nst.template slice<N, M>(0, 0));
      }
    }
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, Quantized>(
        current_scores, current_probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales),
        reinterpret_cast<device const float4 *>(value_scales), token_start,
        visible_tokens, committed_tokens, active_rows, thread_index, block_mask, mask_words);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    auto vs = vt.template slice<N, D>(0, 0);
    pv.run(current_p0, vs, running);
    // The following QK score barrier completes all readers before the next
    // softmax overwrites row statistics. Probability and rescale ping-pong
    // prevent a fast simdgroup from resetting data still read by PV(p).
    if constexpr (!ProbabilityDoubleBuffer)
      threadgroup_barrier(mem_flags::mem_threadgroup);
    page = next;
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// Four-row subtiles retain the original eight-row tile's page partition.
template <uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_attention_pipe_reduce_row(
    device const float *partials, device const float *statistics,
    device bfloat *tile_output, uint committed_tokens, uint active_rows,
    uint splits, ulong head_slot, uint fused_row, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values, uint partition_visible) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = SplashKvHeadDimension;
  if (fused_row / QueryHeadsPerKVHead >= active_rows) {
    tile_output[fused_row * D + thread_index] = bfloat(0.0f);
    return;
  }
  const uint pages = splash_attention_pages(partition_visible ? partition_visible : committed_tokens + active_rows);
  const uint per_split = splash_attention_pages_per_split(pages, splits);
  const uint written = (pages + per_split - 1) / per_split;
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const ulong stat = ((head_slot + thread_index) * M + fused_row) * 2;
  const float maximum = thread_index < written ? statistics[stat] : -INFINITY;
  const float group_maximum = simd_max(maximum);
  if (lane == 0) group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float row_maximum = simd_max(lane < 8 ? group_values[lane] : -INFINITY);
  if (thread_index < written)
    weights[thread_index] = fast::exp(maximum - row_maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  // Keep both accumulations in the original split order. A parallel sum of
  // the denominator changes speculative acceptance on real model prompts.
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    numerator += weight *
        partials[((head_slot + split) * M + fused_row) * D + thread_index];
    denominator += weight * statistics[stat + 1];
  }
  tile_output[fused_row * D + thread_index] =
      bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}

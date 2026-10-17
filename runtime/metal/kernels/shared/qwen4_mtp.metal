#include "metal/abi/KernelABI.h"
#include "metal/abi/Qwen4.h"
#include "metal/abi/HyperConnection.h"
#include "metal/kernels/common/rms_inverse.h"
#include <metal_stdlib>
using namespace metal;
constant constexpr uint kRowThreads = HYPER_CONNECTION_ROW_THREADS;
constant constexpr uint kSimdgroups = kRowThreads / 32;
constant constexpr uint kStreams = HYPER_CONNECTION_MAXIMUM_STREAMS;
inline void reduce_streams(thread float (&values)[kStreams], uint streams, threadgroup float *scratch,
                           uint lane, uint sg) {
  for (uint s = 0; s < streams; ++s) {
    const float sum = simd_sum(values[s]);
    if (lane == 0) scratch[sg * kStreams + s] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 0; s < streams; ++s) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += scratch[g * kStreams + s];
    values[s] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void qwen4_mtp_tokens(device const uint *anchors [[buffer(0)]],
                             device const uint *proposals [[buffer(1)]],
                             device uint *tokens [[buffer(2)]],
                             constant Qwen4MtpParams &p [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
  if (index >= p.rows) return;
  const uint lane = index / p.lane_rows, r = min(index % p.lane_rows, p.limit - 1);
  tokens[index] = r == 0 ? anchors[lane * p.lane_rows] : proposals[lane * p.proposals + r - 1];
}

// One lane's rows: per stream [enorm(embedding) ; hnorm_s(streams_s)] (bf16
// rows of 2 * hidden), the input of the head's eh projection, stream-major:
// stream s of row r of the dispatch at row s * lane_rows + p.row + r.
kernel void qwen4_mtp_input(device const bfloat *embedding [[buffer(0)]],
                            device const float *carried [[buffer(1)]],
                            device const float *history [[buffer(2)]],
                            device const float *embed_norm [[buffer(3)]],
                            device const float *hidden_norm [[buffer(4)]],
                            device bfloat *output [[buffer(5)]],
                            constant Qwen4MtpParams &p [[buffer(6)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * kStreams];
  if (row >= p.rows) return;
  const uint source = min(row, p.limit - 1);
  const uint width = p.streams * p.hidden;
  device const bfloat *e = embedding + ulong(source) * p.hidden;
  device const float *h = source == 0 ? carried : history + ulong(source - 1) * width;
  float sums[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < p.streams; ++s)
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const float v = h[s * p.hidden + d];
      sums[s] += v * v;
    }
  reduce_streams(sums, p.streams, scratch, lane, sg);
  float embed_sum[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint d = tid; d < p.hidden; d += kRowThreads) {
    const float v = float(e[d]);
    embed_sum[0] += v * v;
  }
  reduce_streams(embed_sum, 1, scratch, lane, sg);
  const float embed_scale = rsqrt(embed_sum[0] / p.hidden + kRmsEpsilon);
  for (uint s = 0; s < p.streams; ++s) {
    const float scale = rsqrt(sums[s] / p.hidden + kRmsEpsilon);
    device bfloat *o = output + (ulong(s) * p.lane_rows + p.row + row) * 2 * p.hidden;
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      o[d] = bfloat(float(e[d]) * embed_scale * embed_norm[d]);
      o[p.hidden + d] = bfloat(h[s * p.hidden + d] * scale * hidden_norm[s * p.hidden + d]);
    }
  }
}

// fp32 streams [rows][streams][hidden] from the eh projection's
// stream-major bf16 rows [streams][rows][hidden].
kernel void qwen4_mtp_widen(device const bfloat *projected [[buffer(0)]],
                            device float *streams [[buffer(1)]],
                            constant Qwen4MtpParams &p [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  const uint width = p.streams * p.hidden;
  if (index >= p.rows * width) return;
  const uint row = index / width, c = index % width, s = c / p.hidden, d = c % p.hidden;
  streams[index] = float(projected[(ulong(s) * p.rows + row) * p.hidden + d]);
}

// One threadgroup per lane: the argmax of the lane's logits row p.row, ties
// toward the lower id, as its proposal p.row (and, p.last, every later one)
// and its first sparse candidate. A row without a finite logit drafts 0.
kernel void qwen4_mtp_argmax(device const float *logits [[buffer(0)]],
                             device uint *proposals [[buffer(1)]],
                             device uint *candidates [[buffer(2)]],
                             constant Qwen4MtpParams &p [[buffer(3)]],
                             uint lane_index [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[32];
  threadgroup uint best_ids[32];
  device const float *x = logits + (ulong(lane_index) * p.lane_rows + p.row) * p.vocabulary;
  float best = -INFINITY;
  uint id = 0xffffffffu;
  for (uint v = tid; v < p.vocabulary; v += threads) {
    const float value = x[v];
    if (value > best) {
      best = value;
      id = v;
    }
  }
  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(best, offset);
    const uint other_id = simd_shuffle_down(id, offset);
    if (other > best || (other == best && other_id < id)) {
      best = other;
      id = other_id;
    }
  }
  if (lane == 0) {
    best_values[sg] = best;
    best_ids[sg] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  const uint groups = (threads + 31) / 32;
  for (uint g = 1; g < groups; ++g)
    if (best_values[g] > best || (best_values[g] == best && best_ids[g] < id)) {
      best = best_values[g];
      id = best_ids[g];
    }
  if (id >= p.vocabulary) id = 0;
  const uint end = p.last ? p.proposals : p.row + 1;
  for (uint slot = p.row; slot < end; ++slot) {
    proposals[lane_index * p.proposals + slot] = id;
    candidates[(lane_index * p.proposals + slot) * p.candidates] = id;
  }
}

// A lane's (or sequence's) carried streams: one row of its streams.
kernel void qwen4_mtp_carry(device const float *streams [[buffer(0)]],
                            device float *carried [[buffer(1)]],
                            device const uint *retained [[buffer(2)]],
                            constant Qwen4MtpParams &p [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
  const uint width = p.streams * p.hidden;
  if (index >= width) return;
  const uint row = (p.counted ? retained[0] : p.limit) - 1;
  carried[index] = streams[ulong(row) * width + index];
}

// The PLE rows of each lane's verify rows (model/Qwen4Steps.cpp qwen4PleRows
// on the CPU): heads of the 2- to ngram-gram hashes of the row's token and
// those before it, cut at EOS or the sequence's start.
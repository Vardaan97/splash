#include "metal/abi/KernelABI.h"
#include "metal/abi/Qwen4.h"
#include "metal/abi/QuantFormat.h"
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
  streams[index] = float(projected[(ulong(s) * p.lane_rows + row) * p.hidden + d]);
}

// A stable softmax mass, kept separate from the existing argmax reduction.
inline void qwen4_mtp_mass(thread float &maximum, thread float &mass, float value) {
  if (!isfinite(value)) {
    // NaNs and positive infinity make trimming unsafe. Negative infinity
    // contributes no mass (and an entirely empty row remains invalid).
    if (isnan(value) || value == INFINITY) mass = NAN;
    return;
  }
  const float next = max(maximum, value);
  mass = mass * exp(maximum - next) + exp(value - next);
  maximum = next;
}

inline void qwen4_mtp_mass_merge(thread float &maximum, thread float &mass, float other, float other_mass) {
  if (isnan(mass) || isnan(other_mass)) { mass = NAN; return; }
  if (other_mass == 0) return;
  if (mass == 0) { maximum = other; mass = other_mass; return; }
  const float next = max(maximum, other);
  mass = mass * exp(maximum - next) + other_mass * exp(other - next);
  maximum = next;
}

inline float qwen4_mtp_probability(float maximum, float mass, uint tid, uint lane, uint sg, uint threads,
                                    threadgroup float *maxima, threadgroup float *masses) {
  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(maximum, offset);
    const float other_mass = simd_shuffle_down(mass, offset);
    if (lane + offset < 32) qwen4_mtp_mass_merge(maximum, mass, other, other_mass);
  }
  if (lane == 0) { maxima[sg] = maximum; masses[sg] = mass; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return 0;
  for (uint g = 1; g < (threads + 31) / 32; ++g)
    qwen4_mtp_mass_merge(maximum, mass, maxima[g], masses[g]);
  return mass > 0 && isfinite(mass) ? 1.0f / mass : NAN;
}

// One threadgroup per lane: the argmax of the lane's logits row p.row, over
// p.vocabulary logits: the tokens themselves or, Mapped, the logits of a
// draft vocabulary's tokens, ascending, of which `tokens` names each. Ties
// go toward the lower index, so toward the lower token either way, as its
// proposal p.row (and, p.last, every later one) and its first sparse
// candidate. A row without a finite logit drafts 0.
template <bool Mapped>
inline void qwen4_mtp_argmax_row(device const float *logits, device uint *proposals, device uint *candidates,
                                 device const uint *tokens, device float *confidence, constant Qwen4MtpParams &p, uint lane_index, uint tid,
                                 uint threads, uint lane, uint sg, threadgroup float *best_values,
                                 threadgroup uint *best_ids, threadgroup float *maxima, threadgroup float *masses) {
  device const float *x = logits + (ulong(lane_index) * p.lane_rows + p.row) * p.vocabulary;
  float maximum = -INFINITY, mass = 0;
  float best = -INFINITY;
  uint id = 0xffffffffu;
  for (uint v = tid; v < p.vocabulary; v += threads) {
    const float value = x[v];
    qwen4_mtp_mass(maximum, mass, value);
    if (value > best) {
      best = value;
      id = v;
    }
  }
  const float probability = qwen4_mtp_probability(maximum, mass, tid, lane, sg, threads, maxima, masses);
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
  else if (Mapped) id = tokens[id];
  const uint end = p.last ? p.proposals : p.row + 1;
  for (uint slot = p.row; slot < end; ++slot) {
    confidence[lane_index * p.proposals + slot] = probability;
    proposals[lane_index * p.proposals + slot] = id;
    candidates[(lane_index * p.proposals + slot) * p.candidates] = id;
  }
}

kernel void qwen4_mtp_argmax(device const float *logits [[buffer(0)]],
                             device uint *proposals [[buffer(1)]],
                             device uint *candidates [[buffer(2)]],
                             device float *confidence [[buffer(3)]],
                             constant Qwen4MtpParams &p [[buffer(4)]],
                             uint lane_index [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[32];
  threadgroup uint best_ids[32];
  threadgroup float maxima[32], masses[32];
  qwen4_mtp_argmax_row<false>(logits, proposals, candidates, nullptr, confidence, p, lane_index, tid, threads, lane, sg,
                              best_values, best_ids, maxima, masses);
}

kernel void qwen4_mtp_argmax_mapped(device const float *logits [[buffer(0)]],
                                    device uint *proposals [[buffer(1)]],
                                    device uint *candidates [[buffer(2)]],
                                    device const uint *tokens [[buffer(3)]],
                                    device float *confidence [[buffer(4)]],
                                    constant Qwen4MtpParams &p [[buffer(5)]],
                                    uint lane_index [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint threads [[threads_per_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[32];
  threadgroup uint best_ids[32];
  threadgroup float maxima[32], masses[32];
  qwen4_mtp_argmax_row<true>(logits, proposals, candidates, tokens, confidence, p, lane_index, tid, threads, lane, sg,
                             best_values, best_ids, maxima, masses);
}

// Takes (value, t) as the best so far when it is larger, or as large with a
// lower token.
inline void qwen4_consider(thread float &best, thread uint &token, float value, uint t) {
  if (value > best || (value == best && t < token)) {
    best = value;
    token = t;
  }
}

// The byte offset of unit `unit` of `row` in planes [rows / T][units][T][bytes].
inline ulong qwen4_plane_offset(uint row, uint units, uint unit, uint bytes) {
  return ((ulong(row / QUANT_TILE_ROWS) * units + unit) * QUANT_TILE_ROWS + row % QUANT_TILE_ROWS) * bytes;
}

// One threadgroup per lane: the argmax of the lane's row p.row over the
// draft vocabulary's logits (`tokens` naming each) and the lane's context
// rows' logits (Qwen4MtpParams context_*), ties toward the lower token, as its
// proposal p.row (and, p.last, every later one) and its first candidate.
kernel void qwen4_mtp_argmax_context(device const float *logits [[buffer(0)]],
                                     device uint *proposals [[buffer(1)]],
                                     device uint *candidates [[buffer(2)]],
                                     device const uint *tokens [[buffer(3)]],
                                     device const float *context_logits [[buffer(4)]],
                                     device const uint *context_tokens [[buffer(5)]],
                                     device const Qwen4MtpContextLane *lanes [[buffer(6)]],
                                     device float *confidence [[buffer(7)]],
                                     constant Qwen4MtpParams &p [[buffer(8)]],
                                     uint lane_index [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint threads [[threads_per_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[32];
  threadgroup uint best_tokens[32];
  threadgroup float maxima[32], masses[32];
  float maximum = -INFINITY, mass = 0;
  float best = -INFINITY;
  uint token = 0xffffffffu;
  device const float *x = logits + (ulong(lane_index) * p.lane_rows + p.row) * p.vocabulary;
  for (uint v = tid; v < p.vocabulary; v += threads) {
    qwen4_mtp_mass(maximum, mass, x[v]);
    qwen4_consider(best, token, x[v], tokens[v]);
  }
  const Qwen4MtpContextLane own = lanes[lane_index];
  const uint count = min(own.count, p.context_capacity);
  device const float *y = context_logits + ulong(lane_index) * p.context_slot + ulong(p.row) * p.context_capacity;
  device const uint *named = context_tokens + ulong(own.state_lane) * p.context_capacity;
  for (uint v = tid; v < count; v += threads) {
    qwen4_mtp_mass(maximum, mass, y[v]);
    qwen4_consider(best, token, y[v], named[v]);
  }
  const float probability = qwen4_mtp_probability(maximum, mass, tid, lane, sg, threads, maxima, masses);
  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(best, offset);
    const uint other_token = simd_shuffle_down(token, offset);
    qwen4_consider(best, token, other, other_token);
  }
  if (lane == 0) {
    best_values[sg] = best;
    best_tokens[sg] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  const uint groups = (threads + 31) / 32;
  for (uint g = 1; g < groups; ++g) qwen4_consider(best, token, best_values[g], best_tokens[g]);
  // No finite logit drafts token 0, as without a draft vocabulary.
  if (token == 0xffffffffu) token = 0;
  const uint end = p.last ? p.proposals : p.row + 1;
  for (uint slot = p.row; slot < end; ++slot) {
    confidence[lane_index * p.proposals + slot] = probability;
    proposals[lane_index * p.proposals + slot] = token;
    candidates[(lane_index * p.proposals + slot) * p.candidates] = token;
  }
}

// One thread per (pair, group): copies the group's plane bytes of each
// pair's source row to its destination row, and its meta unit when the group
// starts one. Planes are [rows / T][units][T][unit bytes].
kernel void qwen4_copy_head_rows(device const uchar *source0 [[buffer(0)]],
                                 device const uchar *source1 [[buffer(1)]],
                                 device const uchar *source_meta [[buffer(2)]],
                                 device uchar *target0 [[buffer(3)]],
                                 device uchar *target1 [[buffer(4)]],
                                 device uchar *target_meta [[buffer(5)]],
                                 device const uint2 *pairs [[buffer(6)]],
                                 constant Qwen4HeadRowsParams &p [[buffer(7)]],
                                 uint2 index [[thread_position_in_grid]]) {
  const uint pair = index.x, group = index.y;
  if (pair >= p.count || group >= p.groups) return;
  const uint2 rows = pairs[pair];
  const ulong from0 = qwen4_plane_offset(rows.x, p.groups, group, p.plane0_bytes);
  const ulong to0 = qwen4_plane_offset(rows.y, p.groups, group, p.plane0_bytes);
  for (uint b = 0; b < p.plane0_bytes; ++b) target0[to0 + b] = source0[from0 + b];
  if (p.plane1_bytes) {
    const ulong from1 = qwen4_plane_offset(rows.x, p.groups, group, p.plane1_bytes);
    const ulong to1 = qwen4_plane_offset(rows.y, p.groups, group, p.plane1_bytes);
    for (uint b = 0; b < p.plane1_bytes; ++b) target1[to1 + b] = source1[from1 + b];
  }
  if (group % p.meta_groups == 0) {
    const uint units = p.groups / p.meta_groups, unit = group / p.meta_groups;
    const ulong from = qwen4_plane_offset(rows.x, units, unit, p.meta_bytes);
    const ulong to = qwen4_plane_offset(rows.y, units, unit, p.meta_bytes);
    for (uint b = 0; b < p.meta_bytes; ++b) target_meta[to + b] = source_meta[from + b];
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

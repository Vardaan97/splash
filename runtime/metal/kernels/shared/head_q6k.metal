// Opt-in Q6_K logits tile. Batch only dequantization and staging: every
// accumulator still consumes the baseline's K32 matrix operations in exactly
// the same order, including the original 64-column tile's rotated walk.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"

template <ushort Columns, ushort Depth>
[[max_total_threads_per_threadgroup(Columns)]]
kernel void head_q6k(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                     device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]],
                     device float *output [[buffer(4)]], constant GgufDecodeParams &p [[buffer(8)]],
                     uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
  using F = FmtQ6K;
  threadgroup half stage[Columns * 32 * Depth];
  threadgroup half2 tl[1];
  threadgroup half *my = stage + sg * 32 * 32 * Depth;
  const uint origin = group * Columns + sg * 32;
  const uint groups = p.input_size / 32, units = groups / F::MetaGroups;
  const uint tile = origin / QUANT_TILE_ROWS, row = origin % QUANT_TILE_ROWS + lane;
  device uchar *p0 = w0 + (ulong(tile) * groups * QUANT_TILE_ROWS + row) * F::P0;
  device uchar *p1 = w1 + (ulong(tile) * groups * QUANT_TILE_ROWS + row) * F::P1;
  device uchar *pm = meta + (ulong(tile) * units * QUANT_TILE_ROWS + row) * F::MetaBytes;
  auto a = tensor(input, dextents<int, 2>{int(p.input_size), 8}, array<int, 2>{1, int(p.input_size)});
  constexpr auto descriptor = matmul2d_descriptor(8, 32, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto acc = staged_accumulator<8, 32, 32>(input, p.input_size, my);
  gguf_zero(acc);
  const uint sf = staged_first_step(origin / 64, 0, groups, p.spread);
  for (uint run = 0; run < 2; ++run) {
    const uint begin = run ? 0 : sf, end = run ? sf : groups;
    for (uint first = begin; first < end; first += Depth) {
      typename F::Payload packed[Depth];
      typename F::Meta headers[Depth];
#pragma unroll
      for (ushort j = 0; j < Depth; ++j) {
        const uint g = first + j;
        if (g < end) {
          packed[j] = F::load(p0 + ulong(g) * QUANT_TILE_ROWS * F::P0,
                              p1 + ulong(g) * QUANT_TILE_ROWS * F::P1);
          headers[j] = F::loadMeta(pm + ulong(g / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
        }
      }
#pragma unroll
      for (ushort j = 0; j < Depth; ++j)
        if (first + j < end)
          dequant32<F>(packed[j], headers[j], (first + j) % F::MetaGroups, tl, my + j * 1024 + lane * 32);
      simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (ushort j = 0; j < Depth; ++j) {
        if (first + j < end) {
          tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt(my + j * 1024,
              dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
          auto b = bt.slice<32, 32>(0, 0);
          auto x = a.slice<32, 8>((first + j) * 32, 0);
          operation.run(x, b, acc);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  gguf_elements(acc, [&](uint r, uint c, float value) {
    output[ulong(r) * p.out_stride + p.out_offset + origin + c] = value;
  });
}
// Pure dispatch-width variant: keep the baseline's double-buffered stage,
// payload prefetch and meta reuse verbatim, with more/fewer independent SGs.
template <ushort Columns>
[[max_total_threads_per_threadgroup(Columns)]]
kernel void head_q6k_grouped(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                     device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]],
                     device float *output [[buffer(4)]], constant GgufDecodeParams &p [[buffer(8)]],
                     uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[Columns * 64];
  threadgroup half2 tl[1];
  const uint origin = group * Columns + sg * 32, groups = p.input_size / 32;
  threadgroup half *my = stage + sg * kStagedSimdgroupStage;
  auto acc = staged_accumulator<8, 32, 32>(input, p.input_size, my);
  gguf_zero(acc);
  staged_accumulate<FmtQ6K, 8, 32, 32>(input, w0, w1, meta, p.input_size, origin, my, tl,
                                     lane, 0, groups, staged_first_step(origin / 64, 0, groups, p.spread), acc);
  gguf_elements(acc, [&](uint r, uint c, float value) {
    output[ulong(r) * p.out_stride + p.out_offset + origin + c] = value;
  });
}
#define GROUPED(C) \
  template [[host_name("gguf_head_q6k_c" #C "_d0")]] kernel void head_q6k_grouped<C>( \
      device bfloat *, device uchar *, device uchar *, device uchar *, device float *, \
      constant GgufDecodeParams &, uint, uint, uint);
GROUPED(32) GROUPED(64) GROUPED(128) GROUPED(256)
#undef GROUPED
#define HEAD(C, D) \
  template [[host_name("gguf_head_q6k_c" #C "_d" #D)]] kernel void head_q6k<C, D>( \
      device bfloat *, device uchar *, device uchar *, device uchar *, device float *, \
      constant GgufDecodeParams &, uint, uint, uint);
HEAD(32, 1) HEAD(32, 2) HEAD(32, 4) HEAD(32, 8)
HEAD(64, 1) HEAD(64, 2) HEAD(64, 4) HEAD(64, 8)
HEAD(128, 1) HEAD(128, 2) HEAD(128, 4)
#undef HEAD

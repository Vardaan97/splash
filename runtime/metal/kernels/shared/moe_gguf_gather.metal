#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

// Opt-in grouped M8 gather fold. The K32 accumulation sequence is unchanged;
// device identity tests must qualify the threadgroup-input operand on each family.
kernel void moe_expert_gguf_m8_gather_gate_up(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                    device const uint *tile_count [[buffer(2)]], device uchar *gw0 [[buffer(3)]],
                                    device uchar *gw1 [[buffer(4)]], device uchar *gmeta [[buffer(5)]],
                                    device uchar *gsw0 [[buffer(6)]], device uchar *gsw1 [[buffer(7)]],
                                    device uchar *gsmeta [[buffer(8)]], device uchar *uw0 [[buffer(9)]],
                                    device uchar *uw1 [[buffer(10)]], device uchar *umeta [[buffer(11)]],
                                    device uchar *usw0 [[buffer(12)]], device uchar *usw1 [[buffer(13)]],
                                    device uchar *usmeta [[buffer(14)]], device bfloat *output [[buffer(15)]],
                                    device const uint *routes [[buffer(16)]],
                                    constant MoeGatherExpertParams &params [[buffer(17)]],
                                    uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constant auto &p = params.expert;
  const uint routes_per_row = params.routes_per_row;
  constexpr ushort Rows = 8;
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const bool up = simd_group == 1;
  const MoeGgufSegment s = up ? moe_gguf_segment(tile.expert, p, uw0, uw1, umeta, usw0, usw1, usmeta)
                              : moe_gguf_segment(tile.expert, p, gw0, gw1, gmeta, gsw0, gsw1, gsmeta);
  device bfloat *x = input;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    static_assert(R * GGUF_STAGED_COLUMNS * sizeof(bfloat) <= kStagedSimdgroupStage * sizeof(half), "the gate fits a stage");
    // One weights stage plus the gathered input occupy less than the two
    // existing weights stages. Both SIMD groups own disjoint storage.
    threadgroup bfloat *gathered = reinterpret_cast<threadgroup bfloat *>(my + 32 * 32);
    auto a = tensor(gathered, dextents<int, 2>{32, R}, array<int, 2>{1, 32});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt(my, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto as = a.template slice<32, R>(0, 0);
    auto bs = bt.slice<32, 32>(0, 0);
    constexpr auto descriptor = matmul2d_descriptor(R, 32, 32, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<1>> operation;
    auto acc = operation.template get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
    gguf_zero(acc);
    quant_format_switch(s.format, [&](auto format) {
      using F = decltype(format);
      quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
      StagedColumn<F> weights;
      weights.begin(s.w0, s.w1, s.meta, p.input_size, origin, simd_lane, 0, true);
      bfloat next_input[R] = {};
      const auto read_input = [&](uint row, uint step) {
        const uint route = p.in_place ? row * routes_per_row : routes[group.y * Rows + row];
        return route == ~0u ? bfloat(0) :
            x[ulong(route / routes_per_row) * p.input_size + step * 32 + simd_lane];
      };
      if (params.input_ahead)
        for (uint row = 0; row < R; ++row) next_input[row] = read_input(row, 0);
      for (uint step = 0; step < p.input_size / 32; ++step) {
        weights.stage(step, tl, my, simd_lane);
#pragma unroll
        for (uint row = 0; row < R; ++row) {
          gathered[row * 32 + simd_lane] = params.input_ahead ? next_input[row] : read_input(row, step);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (step + 1 < p.input_size / 32) {
          weights.load(step, step + 1);
          if (params.input_ahead)
            for (uint row = 0; row < R; ++row) next_input[row] = read_input(row, step + 1);
        }
        operation.run(as, bs, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    });
    if (!up) gguf_elements(acc, [&](uint row, uint column, float v) { gate[row * GGUF_STAGED_COLUMNS + column] = bfloat(v); });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up)
      gguf_elements(acc, [&](uint row, uint column, float v) {
        output[out + ulong(row) * p.output_size + origin + column] =
            bfloat(float(bfloat(v)) * splash_silu(float(gate[row * GGUF_STAGED_COLUMNS + column])));
      });
  };
  moe_live_rows<Rows>(tile.rows, run);
}

#pragma once

// Parameters of the Flash-Next MTP head (kernels/shared/qwen4_mtp.metal).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The MTP head (model/Qwen4Exp.hpp Qwen4MtpWeights) over a lane's verify
// rows. Row r of a lane pairs the residual streams of position r - 1 (row 0:
// the lane's carried streams, else history row r - 1) with the embedding of
// its token; rows from `limit` on repeat row limit - 1. qwen4_mtp_tokens
// builds a draft step's tokens: row r the anchor (r = 0) or proposal r - 1.
// qwen4_mtp_argmax writes the argmax of each lane's row `row` as its
// proposal `row` (and, on the last step, every later one) and its first
// sparse candidate. qwen4_mtp_carry keeps one row of streams: row
// retained - 1 (a GPU count) or, without one, row limit - 1.
struct Qwen4MtpParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t lane_rows;
  uint32_t limit;
  uint32_t vocabulary;
  uint32_t proposals;  // proposals per lane
  uint32_t candidates; // sparse candidates per proposal
  uint32_t row;        // argmax: the row; input: the first output row
  uint32_t last;       // argmax: whether to fill the later proposals
  uint32_t counted;    // carry: whether `retained` is bound
  // argmax with context rows (qwen4_mtp_argmax_context): each lane's logits
  // of its context rows, at context_slot floats per lane and
  // context_capacity per row, and each state lane's tokens of them,
  // context_capacity per lane; a lane's Qwen4MtpContextLane gives its count
  // and state lane.
  uint32_t context_capacity;
  uint32_t context_slot;
  uint32_t pad[3];
};

static_assert(sizeof(Qwen4MtpParams) == 64, "Qwen4 MTP parameters are 64 bytes on both sides");

struct Qwen4MtpContextLane {
  uint32_t count;
  uint32_t state_lane;
  uint32_t pad[2];
};

static_assert(sizeof(Qwen4MtpContextLane) == 16, "Qwen4 MTP context lanes are 16 bytes on both sides");

// Rows of a quantized tensor's planes (metal/abi/QuantFormat.h) copied to
// rows of another of the same format and width (qwen4_copy_head_rows):
// `count` (source row, destination row) pairs; `groups` groups of 32 inputs
// per row, `meta_groups` of them per meta unit.
struct Qwen4HeadRowsParams {
  uint32_t count;
  uint32_t groups;
  uint32_t meta_groups;
  uint32_t plane0_bytes;
  uint32_t plane1_bytes;
  uint32_t meta_bytes;
  uint32_t pad[2];
};

static_assert(sizeof(Qwen4HeadRowsParams) == 32, "Qwen4 head row parameters are 32 bytes on both sides");

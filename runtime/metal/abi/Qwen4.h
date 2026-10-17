#pragma once

// Parameters of Qwen3.8-Flash-Next's QSA indexer (kernels/shared/qwen4_qsa.metal,
// ops/Qwen4.cpp), whose block bitmaps are the block masks of its
// full-attention layers' sparse attention.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/PagedAttention.h"

// QSA (Qwen Sparse Attention) of a full-attention layer: the indexer's
// QWEN4_QSA_HEADS query heads and one key head of QWEN4_QSA_DIMENSION values;
// blocks of QWEN4_QSA_BLOCK tokens, those of the sparse attention's block
// masks (metal/abi/PagedAttention.h); `top_blocks` blocks per query row past
// which the row attends sparsely. The layer's index holds the pooled key of
// every complete block (bf16 [dimension]): a lane's index table holds an
// entry per logical page (SplashKvPage), the page's blocks' keys
// `index_offset` bytes into the entry's extent, after those of the pages
// before it in the extent (ops::QsaIndex). A block pools the raw keys of its
// tokens: a step's rows' (bf16 [row][dimension], which the project writes)
// and, before a lane's first row, the lane's pending keys, those of its open
// block (bf16 [slot][dimension] of QWEN4_QSA_PENDING slots, the token at
// position b * QWEN4_QSA_BLOCK + slot of the block b that is not complete),
// which its state keeps.
#define QWEN4_QSA_DIMENSION 128u
#define QWEN4_QSA_HEADS 4u
#define QWEN4_QSA_BLOCK SPLASH_ATTENTION_BLOCK_TOKENS
#define QWEN4_QSA_PENDING (QWEN4_QSA_BLOCK - 1u)
// The score kernel's tiles: QWEN4_QSA_SCORE_ROWS rows by
// QWEN4_QSA_SCORE_BLOCKS blocks, a threadgroup of QWEN4_QSA_SCORE_THREADS
// threads each.
#define QWEN4_QSA_SCORE_ROWS 8u
#define QWEN4_QSA_SCORE_BLOCKS 32u
#define QWEN4_QSA_SCORE_THREADS 256u
// The select kernel's threads, a threadgroup per row.
#define QWEN4_QSA_SELECT_THREADS 1024u
// A QSA dispatch serves `lanes` lanes, each through its own index table: a
// verify step's lanes, or a prefill sequence as one. Lane l's `rows` rows
// follow the rows of the lanes before it in the queries, the keys, the scores
// and the bitmaps, the first at logical position first_position[l] +
// row_offset. A dispatch of one lane selects a long sequence's rows in
// batches of the score scratch's rows: the batch from row `row_offset` of its
// queries and bitmaps, scored into the scratch's first rows. The pool pools
// lane l's counts[l] blocks, its records lane_records records apart.
struct Qwen4QsaParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t first_position[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t counts[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lane_records;
  uint32_t row_offset;
  uint32_t top_blocks;
  uint32_t mask_words;   // 32-bit words of a row's block bitmap
  uint32_t score_stride; // scores per row of the score scratch
  uint32_t index_offset; // the layer's index in every index entry's extent, in bytes
};

static_assert(sizeof(Qwen4QsaParams) == 32 + 8 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Qwen4 QSA parameters are the same size on both sides");

// The pending keys of each of `lanes` lanes after the first retained[l] of
// its `rows` rows of a step from first_position[l], in `layers` layers: the
// keys of the rows' last block that they do not complete, from the step's
// keys (`layer_rows` rows between two layers') and, before the lane's first
// row, its pending keys before the step. qwen4_qsa_pending takes the
// retained rows from these parameters (a prefill sequence's, or those before
// a state it captures), qwen4_qsa_commit from the counts a verify wrote.
struct Qwen4QsaPendingParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t layer_rows;
  uint32_t first_position[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t retained[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(Qwen4QsaPendingParams) == 12 + 8 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Qwen4 QSA pending parameters are the same size on both sides");

// One block to pool: its first token's logical position and RoPE position
// (t, h, w; text repeats one), to which its pooled key is rotated.
struct Qwen4QsaBlock {
  uint32_t position;
  uint32_t rope[3];
};

static_assert(sizeof(Qwen4QsaBlock) == 16,
              "Qwen4 QSA blocks are 16 bytes on both sides");

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

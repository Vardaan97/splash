# Experimental expert tile

`SPLASH_EXPERT_TILE=1` opts eligible eight-row expert plans into a per-lane,
two-way K-split tile. The default is off. It takes 2560 hidden / 640
intermediate, routed Q5_K gate/up, shared Q5_K or Q8_0 gate/up, and Q5_1
or Q8_0 down. Other shapes/formats and 32-row tiles keep the usual kernels.
Eligible short prefill plans also use it for row-width testing.

Gate/up stays fused. Each dequantized, half-rounded weight is reused across
the tile's live rows. Two SIMD groups own contiguous halves of K; one final
threadgroup barrier combines partial sums. Gate and up round to bf16 before
SiLU and multiplication. Down keeps the existing route-order combine pass.

The accumulation tree changes, so results are checked against the existing
fp64 projection intervals in `GgufFormatReference.hpp`, including half
weight rounding and the nonlinear gate/up interval. Bitwise equality with
the staged tile is not promised. `make check` includes `expert-tile-test`
with shader validation: widths 1–8, B1–B4, live rows, and both down formats.

`build/engine-tests/expert-tile-benchmark build/splash.metallib 20 q51`
(or `q80`) reports mirrored current/candidate expert-pass timings at the
model shape. Its synthetic bank holds 128 experts with top-10 routing.
Selected experts rotate through the bank between samples, with routing
preparation outside the timed command. Cache reuse still exists; the printed
logical GB/s is not DRAM bandwidth. M5 timing and real-model output/acceptance checks are
required before enabling this path by default. No performance gain is
claimed from build or correctness tests.

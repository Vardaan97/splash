# Experimental Q6_K logits head

`SPLASH_HEAD_Q6K` is off by default. `1` selects `c128d0`; explicit variants
are `c{32,64,128}d{0,1,2,4,8}` except `c128d8`, plus `c256d0`.
Unknown values retain the baseline. Selection is restricted to a complete
single-segment fp32 Q6_K decode head at K=2560, N=32768/49152/65536/248320,
one K partition and at most eight rows of storage. Other projections retain
their existing path and scratch requirements.

Depth 0 changes only dispatch width and reuses the original staged accumulation
helper. Other depths batch loads/dequantization/staging but issue the same K32
matrix operations. All preserve the original 64-column tile's spread seed and
K walk, including wrap tails. They do not split/reassociate the reduction.
Rows remain padded to eight, as in the original logits kernel.

Build `build/engine-tests/head-q6k` and `build/engine-tests/decode-profile`.

```
head-q6k splash.metallib --test
head-q6k splash.metallib --bench 248320 40 all
head-q6k splash.metallib --bench 32768 40 c128d0
```

The identity matrix compares the whole output allocation, including guards,
column padding and NaN row padding, for rows 1–8, both K walks, multiple K/N
sizes, and every eligible candidate. It also checks flag selection/fallbacks,
production dispatch dimensions, and shape-tagged profiling. Exact-shape
benchmarks check all eight active-row counts before timing rows 1 and 8.
Benchmark weights are synthetic valid repacked Q6_K planes. A ring of distinct
allocations exceeds 768 MiB and uses at least two copies. Reported bytes are
logical plane bytes, not a DRAM counter.

`SPLASH_PROFILE_HEAD_SHAPES=1` adds N/K/split/spread/offset/plane-byte tags to
Q6_K dispatch names in the instrumented backend only. `decode-profile` also
prints exact-count weighted logical bytes/cycle for those entries. This
separates full, restricted draft and context heads that otherwise have the
same pipeline name. The ordinary table and production backend remain unchanged
when this profiling flag is absent.

Apple9 identity is not proof of Apple10 lowering identity. Run the identity
matrix on the target GPU before timing. No variant has an established isolated
speedup yet; leave the flag off pending that measurement.

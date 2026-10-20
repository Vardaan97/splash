# DFlash2 projection and PLE experiments

These began as design notes against e92c00c8 plus the opt-in serving controls.
This branch implements `SPLASH_DFLASH_PROJECTION_FORMAT=af4g64|af8g64|bf16`;
see `DEVELOPMENT.md` under Drafts for the current recipe, memory accounting,
identity and tiled-BF16 execution qualification. The default remains af4g64.
The speculative PLE prefetch section below remains design only.

## Original af8g64 design rationale

Changing only the projection format in `DraftCheckpoint.cpp` is insufficient:
`GgufPreparation.cpp::quantizeRows` accepts only af4g64, and its group writer
packs two 4-bit codes per byte. The format ABI and generic block loader
already support af8g64. The remaining work spans preparation, sizing,
identity, and independent byte-level fixtures. This exceeds a small format
switch; budget about 2–3 engineering days including validation on an idle
Mac. This is a scope estimate, not a performance estimate.

The proposed control is `SPLASH_DFLASH_PROJECTION_FORMAT=af8g64`, with unset
or `af4g64` preserving the current path exactly and all other values refused
before memory planning. Resolve it once at native startup. Do not read the
environment in individual image writers: idle release reconstructs weights
later, and its preparation recipe must remain immutable.

The implementation would make these changes:

1. In `model/DraftCheckpoint.hpp`, define a restricted projection-format
   option with af4g64 as its default. Carry it in `ModelDescriptor.hpp`,
   separately from the trained draft dimensions, so `ModelSizing.cpp` and
   `ModelFactory.cpp` consume the same selection. Parse the control with a
   pure helper in `engine/NativeArguments.cpp` and set the copied descriptor
   in `main.mm::bootstrapConfig` before resource planning. Every target
   family uses this same option. No target or vision format changes.
2. Pass the format through `planDraftImages`, `draftImageBytes`,
   `DraftCheckpointLoader`, `layerImage`, `modelImage`, and `Builder` in
   `DraftCheckpoint.cpp`. Use it for the source row type, `ggufRowBytes`,
   and `ImageBuilder::planes`. Capture the selection in `Planned` for later
   image restoration. All context, selector-hidden, convolution-kernel,
   attention and MLP projections use it; norms, convolution base kernels
   and selector codebooks remain copied as BF16. The vocabulary projection
   remains the target's projection.
3. In `GgufPreparation.cpp::quantizeGroup`, keep the existing 4-bit branch's
   arithmetic and packing unchanged. Add an 8-bit branch with 255 as the
   maximum code, the same finite checks, range/bias choice, rounding and
   BF16 scale/bias storage, and one byte per code. In `quantizeRows`, accept
   only af4g64 or af8g64 for this BF16 reader, select their format metadata,
   require whole groups, and compute group offsets with that format's block
   bytes. af4g64 has 36 native bytes per 64 weights; af8g64 has 68. Repacking
   must use the selected format's planes and metadata, including padding.
4. `WeightStore.cpp::readQuantizedSegment` and `readBlockProjection` already
   decode the descriptor's format and check its plane sizes; retain those
   checks. Extend `draft_preparation_test.mm` to assert af8g64 for every
   loaded draft projection and correct dimensions, then exercise the
   existing linear planning/dispatch paths at all supported lane widths.
   `RuntimeArenas.hpp::DecodeArena::linearScratchSize` must be audited
   against the plans of the af8g64 projections. Add workspace changes only
   if those plans require them; do not infer scratch from weight-byte ratios.
5. `ModelSizing.cpp::streamableWeights` must call `draftImageBytes` with the
   descriptor's format. It then feeds the existing fixed-weight memory
   budget. Compare planned image bytes with the sum actually loaded and
   `DFlashDraftWeights::actualAllocatedBytes`. Count aligned image planes and
   descriptors, not a blanket doubling: only projection payloads grow.
6. Key derived images explicitly by source identity, projection format,
   quantizer version and layout. Add an optional per-image content identity
   to `WeightImages.hpp::ImagePlan`, passed through `WeightImages::load` to
   `WeightFile`, defaulting to the owner's identity. For af8g64 draft images,
   derive it with `weightDigest` from the existing source identity plus a
   canonical `draft-projection af8g64 quantizer-v1` suffix. Leave af4g64's
   identity and relative paths unchanged. The per-image override keeps the
   target-only manifest unchanged while the combined manifest differs in
   `weightManifestFingerprint`; `RuntimeResources.mm::makeRuntimeCacheIdentity`
   and `persistentCacheNamespace` then isolate composite target/draft states.
   Never reuse a cached draft state across these two projection formats.
7. Report the resolved `draft_projection_format` in `/status` from immutable
   model metadata. Update `DEVELOPMENT.md` with the control, byte accounting
   and cache behavior once it exists.

There is no persistent prepared-weight image cache on this base.
`WeightImages::load` allocates and writes each image in memory; `restore`
replays its captured writer. `LegacyWeightCache.hpp` only reports an older
cache. Do not reintroduce it for this experiment. Any future prepared-image
cache must use the complete derived-image key above. The active persistent
prefix cache is a different cache, whose composite-state identity must
change with the draft's numerics even on the same binary and source root.

Validation to add with that implementation:

- CPU parser cases: unset/default, both formats, malformed and unknown values.
  CPU planning cases: every projection descriptor, exact aligned bytes,
  unchanged copied tensors, and identical default af4g64 plans.
- Independent 8-bit affine fixture rows in `run_draft_preparation.py`,
  including zeros, positive and negative constants, signed ranges, halfway
  rounding, and non-finite refusal. Keep every existing af4g64 golden intact.
  Extend `draft_preparation_test.mm` for image bytes, loader formats and
  release/restore identity in both modes.
- CPU identity regressions in `model_loading_test.cpp` and
  `runtime_resources_test.mm`: same source/default preserves identities;
  af8g64 changes the combined manifest and persistent namespace, with the
  target-only manifest unchanged. Reject a cross-format composite restore.
- Metal projection/draft oracles at lane widths 1/2/4; exact agreement with
  the independent quantized reference for each mode, not identical text
  between different weight formats. Recheck actual memory against its plan.

After those tests pass, compare default versus af8g64 on one binary/root,
first with full-row verification to isolate projection quality, then with
adaptive verification. Measure accepted proposals, selected verify rows,
cycle/GPU time, tokens/ms, TTFT and memory/suspensions. More accepted drafts
alone do not establish a throughput improvement. No af8g64 benchmark arm is
included in the current executable arms because the control does not exist.

## DFlash2 speculative PLE prefetch

Design only. This is not a trivial transplant. DFlash2 produces its proposals
in one parallel block; its selector and thinking-token replacements finish
at the end of `Runtime.cpp::Impl::encodeDraftBatchGraph`. The target's
`Qwen4Steps::addVerifyInput` hashes those proposals and immediately stages
streamed PLE rows in an awake host step. A prefetch issued at that same host
step has almost no compute left to hide the read behind. An extra graph
split or GPU-to-host synchronization can cost more than it saves.

A candidate implementation would:

1. Add an opt-in, default-off `SPLASH_DFLASH_PLE_PREFETCH=1` control, active
   only for a target with a streamed PLE table. Do not add a stage for other
   target families or for a fully resident table.
2. Signal a host hint after `DFlashDraft::addSelection` and
   `sampling.addThinkingDrafts`, when the actual proposals are ready, before
   the target hash/stage consumes them. Identify independent GPU work that
   can continue after that signal before adding synchronization. If there
   is none, stop here and retain the existing path.
3. Snapshot each lane's committed n-gram history and pending anchor while
   planning the command, append the final proposed tokens only after the
   signal, and compute hints with `Qwen4Steps.cpp::qwen4PleRows`. Keep the
   history's position relative to that copied suffix, with its start/EOS
   rules intact. Proposals are hints only; the target still hashes and
   stages its actual verify inputs. Reuse the store's bounded, low-priority
   `SparseRowStore::prefetch` API and existing counters.
4. Give every submitted command its own immutable lane/history/position
   storage. A completion or cancellation wrapper must release all captured
   buffers on graph construction, preparation or submission failure, normal
   completion, request cancellation, suspension and shutdown. Never leave a
   callback waiting for an event that a failed submission cannot signal.
   Do not read arena proposals after their command's lifetime or after the
   next batch can overwrite them.
5. Keep the current verify `stage` correctness path. Decode currently calls
   `store.stage` without prefetch adoption, while prefill adopts hints. If
   adoption for decode is needed to avoid duplicate in-flight reads, gate it
   on this control and validate its reader priorities and failure behavior.
   It must never wait on a hint that cannot finish.

Tests would cover CPU hint hashes against the target for sequence start,
EOS, suffix-relative positions, different n-grams and lane widths; bounded
queue/drop behavior; and all cancellation/submission-failure lifetimes.
Metal/runtime tests would check identical tokens, states and cache identity
with hints off/on, and an empty hint path when no table streams.

Estimate 1–2 engineering days plus an isolated timing window. The recorded
0.64–0.72 ms of staging in a roughly 31–34 ms cycle is only an upper bound
on direct savings; at most about 2% is a hypothesis, not a measured gain.
Measure added host/event overhead, stage hit/read counts and whole-cycle
cost before retaining the implementation. There is no prefetch arm or
implemented control in this change.

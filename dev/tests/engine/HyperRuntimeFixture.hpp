#pragma once
#include "GgufFormatReference.hpp"
#include "TestChecks.hpp"
#include "engine/Types.hpp"
#include "model/Qwen4DraftVocabulary.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include <unordered_map>
namespace splash::test::mtp {
using namespace splash;
using namespace splash::model;
using splash::test::require;
constexpr uint32_t hidden = 256, vocabulary = 512;

std::vector<uint32_t> contextTokens() {
  std::vector<uint32_t> tokens;
  for (uint32_t token = 2; token < vocabulary && tokens.size() < 3; ++token)
    if (!inQwen4DraftVocabulary(token)) tokens.push_back(token);
  require(tokens.size() == 3, "fixture has three outside-vocabulary context tokens");
  return tokens;
}

struct Fixture {
  metal::MetalBackend backend;
  bool contextWeights;
  bool stagedHyper = false;
  LoadedModel model;
  ops::ExecutionPlans operators;
  std::unique_ptr<kv::PageStorage> pages;
  std::unique_ptr<QwenStateStorage> states;
  std::unique_ptr<Runtime> runtime;

  metal::MetalBuffer bytes(const void *data, uint64_t size) {
    auto buffer = backend.allocateBuffer(size, metal::BufferStorage::Shared, "mtp-runtime-fixture");
    std::memcpy(buffer.contents(), data, size);
    return buffer;
  }
  template<class T> metal::MetalBuffer upload(const std::vector<T> &values) {
    return bytes(values.data(), values.size() * sizeof(T));
  }
  ops::NormWeights norm(uint32_t width) { return {upload(std::vector<float>(width, 1.0F)), true}; }
  ops::Projection floats(uint32_t n, uint32_t k) {
    return {n, k, {{ops::QuantizedSegment::floats(n, k, upload(std::vector<float>(uint64_t{n} * k)))}}};
  }
  ops::Projection denseSignal(uint32_t n, uint32_t k) {
    std::mt19937 rng(n * 31u + k);
    std::vector<float> values(uint64_t{n} * k);
    for (auto &v : values) v = float(int(rng() % 2001) - 1000) / 200000.0f;
    return {n, k, {{ops::QuantizedSegment::floats(n, k, upload(values))}}};
  }
  ops::QuantizedSegment quantized(uint32_t n, uint32_t k, bool head = false) {
    // Q8_0's exact +/-1 rows, all other rows zero, repacked by the shared reference.
    std::vector<uint8_t> native(uint64_t{n} * k / 32 * 34);
    if (head) {
      std::vector<std::pair<uint32_t, int>> rows{{0, 1}, {1, -1}};
      if (contextWeights && n == vocabulary) {
        const auto tokens = contextTokens();
        rows.insert(rows.end(), {{tokens[0], 2}, {tokens[1], 3}, {tokens[2], -4}});
      }
      for (const auto &[row, code] : rows)
        for (uint32_t group = 0; group < k / 32; ++group) {
          uint8_t *block = native.data() + (uint64_t{row} * k / 32 + group) * 34;
          const uint16_t scale = gguf_reference::f2h(1.0F);
          std::memcpy(block, &scale, 2);
          std::memset(block + 2, static_cast<uint8_t>(code), 32);
        }
    }
    const auto planes = gguf_reference::repack(gguf_reference::Q80, native, n, k, nullptr);
    return ops::QuantizedSegment::planes(GGUF_FMT_Q80, n, k, upload(planes.w0), {}, upload(planes.meta));
  }
  Qwen4HyperWeights hyper(bool inject = true) {
    std::vector<float> injection(4 * 4 * hidden);
    if (stagedHyper)
      for (uint32_t i = 0; i < injection.size(); ++i) injection[i] = float(int(i % 23) - 11) / 200000.0f;
    return {norm(4 * hidden),
            stagedHyper ? ops::Projection(256, 4 * hidden, {{quantized(256, 4 * hidden, true)}}) : floats(256, 4 * hidden),
            stagedHyper ? ops::Projection(4 * hidden, 256, {{quantized(4 * hidden, 256, true)}}) : floats(4 * hidden, 256),
            inject ? upload(injection) : metal::MetalBuffer{}};
  }
  ops::BlockMoeWeights moe() {
    ops::BlockExpertProjection forward{quantized(16 * 256, hidden), quantized(256, hidden)};
    ops::BlockExpertProjection down{quantized(16 * hidden, 256), quantized(hidden, 256)};
    return {floats(16, hidden).blocks().segments.front(), floats(1, hidden).blocks().segments.front(),
            forward, forward, down};
  }
  explicit Fixture(const char *library, bool adaptive = true, bool withContext = false, bool zeroDepth = false, bool lookup = false, bool forceStagedHyper = false)
      : backend(library), contextWeights(withContext), stagedHyper(forceStagedHyper),
        operators([&] { auto caps = backend.capabilities(); if (forceStagedHyper) caps.appleGpuFamily = 10; return caps; }()) {
    setenv("SPLASH_MTP_ADAPTIVE", adaptive ? "1" : "0", 1);
    setenv("SPLASH_MTP_DRAFTS", "7", 1);
    setenv("SPLASH_MTP_DEPTH", "cost", 1);
    Qwen4ExpWeights weights;
    auto &layout = weights.layout;
    layout.maximumContextTokens = 4096;
    layout.layers = 1;
    layout.hiddenSize = hidden;
    layout.vocabularySize = vocabulary;
    // Retain Flash-Next's compiled attention/GDN head geometry; shrink only
    // the hidden width, layer count and experts.
    layout.fullAttentionPeriod = 4;
    layout.experts = 16;
    layout.expertsPerToken = 4;
    layout.expertIntermediateSize = 256;
    layout.hyperRank = 256;
    layout.hiddenCaptureLayers = {0};
    layout.maskToken = 511;
    layout.stopTokens = {509, 510};
    // One zero PLE head keeps hashing and history commits on their real path.
    layout.pleNgram = 2;
    layout.pleHeadsPerNgram = 1;
    layout.pleHeadDimension = 256;
    layout.pleConvolutionTaps = 2;
    layout.pleLayer = 0;
    const uint32_t pleRows = zeroDepth ? 32 : 1;
    weights.pleHash = {{1, 1, 1, 1}, {0}, {pleRows}, pleRows};
    layout.mtpLayers = 1;
    layout.mtpDraftTokens = 256;
    std::vector<uint16_t> embeddings(uint64_t{vocabulary} * hidden, 0x3f80);
    // The negative lane emits token 1 forever; the positive lane token 0.
    std::vector<uint32_t> negative{1, 11};
    if (contextWeights) negative.push_back(contextTokens()[2]);
    for (uint32_t token : negative)
      std::fill_n(embeddings.begin() + token * hidden, hidden, 0xbf80);
    weights.tokenEmbedding = {vocabulary, hidden, {upload(embeddings), GGUF_FMT_BF16}};
    weights.logitsProjection = {vocabulary, hidden, {{quantized(vocabulary, hidden, true)}}};
    weights.logitsProjection.destination = ops::FloatOutput::Float32;
    Qwen4ExpLayerWeights trunk;
    trunk.mixerHyper = hyper();
    trunk.ffnHyper = hyper();
    std::vector<uint16_t> convolution(4 * layout.convolutionDimension);
    if (stagedHyper)
      for (uint32_t c = 0; c < layout.convolutionDimension; ++c) convolution[c * 4 + 3] = 0x3f80;
    trunk.mixer = QwenGdnWeights{stagedHyper ? denseSignal(layout.packedGdnWidth, hidden) : floats(layout.packedGdnWidth, hidden),
                               upload(convolution),
                               upload(std::vector<float>(layout.gdnValueHeads)),
                               upload(std::vector<float>(layout.gdnValueHeads)),
                               norm(layout.gdnHeadDimension), stagedHyper ? ops::Projection(hidden, layout.attentionWidth, {{quantized(hidden, layout.attentionWidth, true)}}) : floats(hidden, layout.attentionWidth)};
    trunk.ffn = moe();
    std::vector<uint16_t> pleTable(uint64_t{pleRows} * 256);
    if (zeroDepth)
      for (uint32_t row = 0; row < pleRows; ++row)
        std::fill_n(pleTable.begin() + row * 256, 256, row % 2 ? 0xbf80 : 0x3f80);
    weights.pleTable = {pleRows, 256, {upload(pleTable), GGUF_FMT_BF16}};
    trunk.ple = Qwen4PleWeights{floats(4 * hidden, 256), floats(hidden, 256),
                               norm(4 * hidden), norm(4 * hidden), norm(4 * hidden),
                               upload(std::vector<float>(2 * 4 * hidden))};
    if (zeroDepth) {
      // Nonconstant PLE rows leave their sign in the committed history;
      // an all-zero table would hide a hash of the old lookup padding.
      std::vector<float> identity(uint64_t{hidden} * 256);
      for (uint32_t i = 0; i < hidden; ++i) identity[i * 256 + i] = 1.0F;
      trunk.ple->value = {hidden, 256, {{ops::QuantizedSegment::floats(hidden, 256, upload(identity))}}};
    }
    weights.layers.push_back(trunk);
    weights.finalHyper = hyper(false);
    Qwen4MtpWeights mtp;
    mtp.block.mixerHyper = hyper();
    mtp.block.ffnHyper = hyper();
    mtp.block.mixer = QwenAttentionWeights{floats(layout.packedFullWidth, hidden), norm(256), norm(256),
                                           stagedHyper ? ops::Projection(hidden, layout.attentionWidth, {{quantized(hidden, layout.attentionWidth, true)}}) : floats(hidden, layout.attentionWidth)};
    mtp.block.indexer = Qwen4IndexerWeights{floats(512, hidden), floats(128, hidden), norm(128), norm(128)};
    mtp.block.ffn = moe();
    mtp.embedHidden = floats(hidden, 2 * hidden); // all-zero MTP streams propose token 0
    if (contextWeights) {
      // Preserve the token embedding's sign; only copied context head rows
      // can propose the target's outside-vocabulary winner.
      std::vector<float> identity(uint64_t{hidden} * 2 * hidden);
      for (uint32_t i = 0; i < hidden; ++i) identity[i * 2 * hidden + i] = 1.0F;
      mtp.embedHidden = {hidden, 2 * hidden, {{ops::QuantizedSegment::floats(hidden, 2 * hidden, upload(identity))}}};
    }
    mtp.embedNorm = norm(hidden);
    mtp.hiddenNorm = norm(4 * hidden);
    mtp.head = hyper(false);
    std::vector<uint32_t> tokens;
    for (uint32_t token = 0; token < vocabulary && tokens.size() < 256; ++token)
      if (inQwen4DraftVocabulary(token)) tokens.push_back(token);
    require(tokens.size() == 256 && tokens[0] == 0 && tokens[1] == 1,
            "fixture draft head maps static vocabulary rows");
    mtp.draftHead = Qwen4DraftHead{{256, hidden, {{quantized(256, hidden, true)}}}, upload(tokens)};
    mtp.draftHead->logits.destination = ops::FloatOutput::Float32;
    weights.mtp = mtp;
    if (lookup) {
      // Keep one real trunk attention layer when the MTP head is absent.
      weights.layers.push_back(mtp.block);
      layout.layers = layout.fullAttentionPeriod = 2;
      weights.mtp.reset(); layout.mtpLayers = 0;
    }
    model.target = weights;
    model.draft.layout = nullDraftLayout(hidden, vocabulary, hidden);
    model.descriptor.target = layout;
    model.descriptor.draftModel = false;
    model.descriptor.visionSource = VisionSource::None;
    model.descriptor.targetKvLayout = layout.kvLayout();
    model.descriptor.stateLayout = {layout.gdnStateLayout(), model.draft.layout.stateLayout()};
    model.descriptor.capabilities.maximumContextTokens = layout.maximumContextTokens;
    const metal::AllocationAdmission admit = [](uint64_t, const std::function<void()> &allocate) {
      allocate(); return metal::AllocationResult{};
    };
    const auto kvLayout = model.targetKvLayout(kv::Format::Int8);
    const uint32_t extent = kvLayout.minimumExtentPages();
    const uint32_t pageCount = ((64 + extent - 1) / extent) * extent;
    pages = std::make_unique<kv::PageStorage>(backend, admit, kvLayout, pageCount, extent);
    for (uint32_t i = 0; i < pageCount / extent; ++i)
      require(bool(pages->allocateExtent(i)), "fixture extent allocation");
    states = std::make_unique<QwenStateStorage>(backend, admit, model.stateLayout(ops::GdnStateFormat::Float32), nullptr);
    RuntimeContext context{backend, model, *pages, *states, operators};
    context.mtpDepth.probeCycles = context.mtpDepth.exploreCycles = 0;
    context.mtpDepth.measuredCosts = false; // deterministic plans, independent of GPU timings
    if (zeroDepth) context.mtpDepth.priorRatios = {1.0, 100.0, 100.0, 100.0, 100.0};
    runtime = std::make_unique<Runtime>(context);
  }
  engine::EngineRequest request(uint64_t id, std::vector<uint32_t> prompt, bool constrained = false) {
    engine::EngineRequest result;
    result.id = id; result.prompt = std::move(prompt); result.maxNewTokens = 2048;
    result.constraint = constrained ? ConstraintMode::TokenMask : ConstraintMode::None;
    return result;
  }
  ModelStepResult prefill(const engine::EngineRequest &request, uint32_t lane) {
    runtime->beginColdRequest(request.modelView(), lane);
    return replay(request, lane);
  }
  ModelStepResult replay(const engine::EngineRequest &request, uint32_t lane) {
    const std::array<uint32_t, 1> table{lane};
    const uint32_t count = static_cast<uint32_t>(request.prompt.size());
    BatchPlan plan{.kind = WorkKind::Prefill, .items = {{request.id, count}}};
    ModelBatchItem item{request.id, 0, count, table}; item.inputTokens = request.prompt;
    item.pageTableRevision = ++revisions[request.id];
    return runtime->submit(plan, std::span(&item, 1), {})->wait().front();
  }
  std::unordered_map<uint64_t, uint64_t> revisions;
  std::array<std::array<uint32_t, 16>, 4> tables{};
  std::unique_ptr<ModelBatchTicket> submit(std::span<const uint64_t> ids, std::span<const uint32_t> lanes,
                                          bool constrained = false) {
    BatchPlan plan{.kind = WorkKind::Decode, .constrained = constrained};
    std::vector<ModelBatchItem> items;
    for (uint32_t i = 0; i < ids.size(); ++i) {
      plan.items.push_back({ids[i], 0});
      // Disjoint page ranges, enough for the acceptance-training history.
      for (uint32_t page = 0; page < 16; ++page) tables[i][page] = lanes[i] + page * 4;
      ModelBatchItem item{ids[i], states->metadata(lanes[i]).lengths.targetTokens, 0, tables[i]};
      item.pageTableRevision = ++revisions[ids[i]];
      items.push_back(item);
    }
    return runtime->submit(plan, items, [] {});
  }
  ModelStepResult decode(uint64_t id, uint32_t lane) {
    return submit(std::span(&id, 1), std::span(&lane, 1))->wait().front();
  }
};

} // namespace splash::test::mtp

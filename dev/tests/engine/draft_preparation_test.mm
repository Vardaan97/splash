// Prepared draft images against independent native-row fixtures, retained
// default goldens, poisoned padding, loader formats and release/restore.
// Usage: draft-preparation METALLIB FIXTURE GOLDENS
#include "GgufFixtures.hpp"
#include "TestAdmission.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <cstring>
#include <cstdlib>
#include <optional>
#include <vector>

using namespace gguf_fixtures;
using namespace gguf_reference;

namespace {

namespace ops = splash::ops;
using splash::metal::MetalBackend;

// The draft fixture: two layers of width 256, two KV heads.
model::DFlashDraftLayout tinyDraftLayout() {
  model::DFlashDraftLayout layout;
  layout.layers = 2;
  layout.hiddenSize = 256;
  layout.vocabularySize = 256;
  layout.dynamicSize = 256;
  layout.qkvSize = 512;
  layout.attentionSize = 256;
  layout.intermediateSize = 256;
  layout.attentionHeadDimension = 64;
  layout.rotaryTheta = 10'000'000.0F;
  layout.targetLayerCount = 1;
  layout.selectorRank = 256;
  layout.kvHeads = 2;
  return layout;
}

std::vector<uint8_t> tensorBytes(const model::SourceTensor &tensor) {
  std::vector<uint8_t> bytes(tensor.bytes);
  tensor.read(0, bytes);
  return bytes;
}

std::vector<uint8_t> bytesOf(std::span<const uint8_t> bytes) { return {bytes.begin(), bytes.end()}; }

// Whether `image` holds `bytes` at `offset`.
bool holds(const std::vector<uint8_t> &image, uint64_t offset, const std::vector<uint8_t> &bytes) {
  return offset + bytes.size() <= image.size() && std::equal(bytes.begin(), bytes.end(), image.begin() + offset);
}

// Each copy of the planned image as stored in the checkpoint, and the planes
// of each projection the CPU reference's of its parts' expected rows.
void checkImage(const std::vector<uint8_t> &image, const model::gguf::Image &plan,
                const model::SafetensorsCheckpoint &checkpoint, const model::SafetensorsCheckpoint &expected) {
  for (const model::gguf::Copy &copy : plan.copies)
    check(holds(image, copy.destination, tensorBytes(checkpoint.require(copy.source.name))),
          plan.name + " " + copy.source.name + ": as stored");
  for (const model::gguf::Repack &step : plan.repacks) {
    std::vector<uint8_t> native;
    for (const model::gguf::TensorRows &source : step.sources) {
      const std::vector<uint8_t> rows = tensorBytes(expected.require(source.name));
      native.insert(native.end(), rows.begin(), rows.end());
    }
    const Fmt f = Fmt(step.format);
    native.resize(size_t(step.rows) * rowBytes(f, uint32_t(step.columns)), 0);
    const Packed planes = repack(f, native, uint32_t(step.rows), uint32_t(step.columns), nullptr);
    check(holdsPackedPlanes(image, step, planes),
          plan.name + " " + step.sources.front().name + ": exact packed planes of independent native rows");
    std::vector<uint8_t> actual(native.size(), 0xFF);
    model::readRepackRows(step, 0, step.rows, actual.data());
    check(actual == native, plan.name + ": CPU preparation equals the independent native rows");
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 4) {
      std::fprintf(stderr, "usage: draft-preparation METALLIB FIXTURE GOLDENS\n");
      return 2;
    }
    const std::filesystem::path root(argv[2]);
    const Goldens hashes = goldens(argv[3], @"draft_images");
    MetalBackend backend(argv[1]);
    const model::DFlashDraftLayout layout = tinyDraftLayout();
    const auto checkFormat = [&](model::DraftProjectionFormat format, const std::filesystem::path &fixture,
                                 const char *expectedDirectory) {
    guarded(std::string("draft preparation ") + std::string(model::draftProjectionFormatName(format)), [&] {
      model::WeightImages images(backend, "fixture");
      model::DraftCheckpointLoader loader(backend, images, fixture, layout, format);
      const model::DFlashDraftWeights draft = model::loadDFlashDraftWeights(backend, loader, layout);
      const model::SafetensorsCheckpoint checkpoint(fixture), expected(fixture / expectedDirectory);
      const std::vector<model::gguf::Image> plans = model::planDraftImages(checkpoint, layout, format);
      const auto contents = images.contents();
      check(contents.size() == plans.size(), "the draft loads every planned image");
      uint64_t plannedBytes = 0;
      for (const auto &plan : plans) plannedBytes += plan.bytes;
      uint64_t declaredBytes = 0;
      for (const auto &file : draft.files) {
        declaredBytes += file.declaredBytes;
        check(file.contentIdentity == model::draftProjectionIdentity("fixture", format), "per-image recipe identity");
      }
      check(draft.actualAllocatedBytes == plannedBytes && declaredBytes == plannedBytes &&
                plannedBytes == model::draftImageBytes(fixture, layout, format), "planner/loader byte totals agree");
      std::vector<std::vector<uint8_t>> loaded;
      for (size_t index = 0; index < plans.size(); ++index) {
        const auto &image = contents[index];
        loaded.push_back(bytesOf(image.bytes));
        checkImage(loaded.back(), plans[index], checkpoint, expected);
        const auto buffer = backend.allocateBuffer(plans[index].bytes, splash::metal::BufferStorage::Shared, "image");
        std::memset(buffer.contents(), 0xFF, plans[index].bytes);
        model::writeGgufImage(backend, buffer, plans[index]);
        check(bytesOf(model::contentsOf(buffer)) == loaded.back(), plans[index].name + ": every byte written");
        for (const auto &fill : plans[index].fills)
          check(holds(loaded.back(), fill.offset, fill.bytes), "every descriptor has its planned bytes");
        if (format == model::DraftProjectionFormat::Af4g64)
          checkGolden(hashes, std::string(image.component), image.bytes, std::string(image.component) + ": golden");
      }
      const char *environment = std::getenv("SPLASH_DFLASH_PROJECTION_FORMAT");
      const std::optional<std::string> original = environment ? std::optional<std::string>(environment) : std::nullopt;
      // Restoration must use the captured recipe even after the environment
      // changes to a different valid selection.
      setenv("SPLASH_DFLASH_PROJECTION_FORMAT",
             format == model::DraftProjectionFormat::Af4g64 ? "bf16" : "af4g64", 1);
      images.release();
      while (!images.restore(splash::test::admitAll)) {
      }
      for (size_t index = 0; index < loaded.size(); ++index)
        check(bytesOf(images.contents()[index].bytes) == loaded[index],
              std::string(images.contents()[index].component) + ": restored");
      if (original) setenv("SPLASH_DFLASH_PROJECTION_FORMAT", original->c_str(), 1);
      else unsetenv("SPLASH_DFLASH_PROJECTION_FORMAT");
      // Every loaded projection keeps the captured format and trained sizes.
      const auto projection = [&](const ops::Projection &p, uint32_t n, uint32_t k) {
        return p.outputSize == n && p.inputSize == k && p.blocks().segments.size() == 1 &&
               p.blocks().segments.front().formatId == model::draftProjectionFormatId(format);
      };
      bool read = draft.layers.size() == layout.layers && draft.files.size() == layout.layers + 1 &&
                  projection(draft.contextProjection, layout.hiddenSize, layout.targetHiddenSize()) &&
                  projection(draft.selectorProjection, layout.selectorRank, layout.hiddenSize);
      for (const auto &layer : draft.layers)
        read = read && projection(layer.attentionDynamic, layout.dynamicSize, layout.hiddenSize) &&
               projection(layer.qkvProjection, layout.qkvSize, layout.hiddenSize) &&
               projection(layer.outputProjection, layout.hiddenSize, layout.attentionSize) &&
               projection(layer.mlpDynamic, layout.dynamicSize, layout.hiddenSize) &&
               projection(layer.gateProjection, layout.intermediateSize, layout.hiddenSize) &&
               projection(layer.upProjection, layout.intermediateSize, layout.hiddenSize) &&
               projection(layer.downProjection, layout.hiddenSize, layout.intermediateSize);
      check(read, "the draft loader reads the draft images");
    });
    };
    checkFormat(model::DraftProjectionFormat::Af4g64, root, "expected");
    checkFormat(model::DraftProjectionFormat::Bfloat16, root, "expected-bf16");
    checkFormat(model::DraftProjectionFormat::Af8g64, root, "expected-af8g64");
    checkFormat(model::DraftProjectionFormat::Bfloat16, root / "bf16-extremes", "expected-bf16");
    std::printf("%s (%d failures)\n", failures ? "draft preparation tests FAILED" : "draft preparation tests passed",
                failures);
    return failures ? 1 : 0;
  }
}

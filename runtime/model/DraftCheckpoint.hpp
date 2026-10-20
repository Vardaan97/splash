#pragma once

#include "model/GgufImage.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace splash::model {

struct DFlashDraftLayout;

// A preparation recipe, separate from the draft's trained geometry. The
// default retains the original image bytes and content identity.
enum class DraftProjectionFormat : uint8_t { Af4g64, Af8g64, Bfloat16 };
[[nodiscard]] inline std::string_view draftProjectionFormatName(DraftProjectionFormat format) {
  switch (format) {
  case DraftProjectionFormat::Af4g64: return "af4g64";
  case DraftProjectionFormat::Af8g64: return "af8g64";
  case DraftProjectionFormat::Bfloat16: return "bf16";
  }
  throw WeightStoreError("unknown draft projection format");
}
[[nodiscard]] uint32_t draftProjectionFormatId(DraftProjectionFormat format);
// Checks every projection before any image allocation (BF16 needs K % 256).
void requireDraftProjectionGeometry(const DFlashDraftLayout &layout, DraftProjectionFormat format);
// The execution policy is tiled planes with automatic linear dispatch. A
// version change isolates numerics even when image sizes stay the same.
[[nodiscard]] std::string draftProjectionIdentity(std::string_view source, DraftProjectionFormat format,
                                                  uint32_t recipeVersion = 1);

// A DFlash2 checkpoint as its repository releases it, config.json and BF16
// safetensors -> the draft's block images (DFlashDraft.cpp), laid out as a
// target's are (model/GgufImage.hpp): projections prepared into the selected
// tiled planes, every other tensor copied as stored. The checkpoint is planned
// once; each image is written into memory when it is opened.
class DraftCheckpointLoader final {
public:
  DraftCheckpointLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                        const DFlashDraftLayout &layout, DraftProjectionFormat format = DraftProjectionFormat::Af4g64);
  DraftCheckpointLoader(const DraftCheckpointLoader &) = delete;
  DraftCheckpointLoader &operator=(const DraftCheckpointLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    Planned(const std::filesystem::path &directory, DraftProjectionFormat selection)
        : checkpoint(directory), format(selection) {}
    SafetensorsCheckpoint checkpoint;
    const DraftProjectionFormat format;
    std::string contentIdentity;
    std::vector<gguf::Image> images; // layers, then model.bin
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
};

// The images of a draft checkpoint, in the order DFlashDraft.cpp reads them:
// the layers, then model.bin. The checkpoint outlives them: their rows read
// its tensors.
[[nodiscard]] std::vector<gguf::Image> planDraftImages(const SafetensorsCheckpoint &checkpoint,
                                                     const DFlashDraftLayout &layout,
                                                     DraftProjectionFormat format = DraftProjectionFormat::Af4g64);

// The bytes of every image of the draft checkpoint in directory.
[[nodiscard]] uint64_t draftImageBytes(const std::filesystem::path &directory, const DFlashDraftLayout &layout,
                                      DraftProjectionFormat format = DraftProjectionFormat::Af4g64);

} // namespace splash::model

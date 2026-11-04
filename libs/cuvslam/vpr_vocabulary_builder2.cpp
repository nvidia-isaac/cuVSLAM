/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include "cuvslam/vpr_vocabulary_builder2.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "slam/vpr/vpr_vocabulary.h"

#include "cuvslam/internal.h"

namespace cuvslam {

class VprVocabularyBuilder::Impl {
public:
  std::unique_ptr<slam::vpr::VocabularyBuilder> builder;
};

VprVocabularyBuilder::VprVocabularyBuilder(const Slam::VprSettings& settings) : impl(std::make_unique<Impl>()) {
  CheckVocabularyFit(settings);
  slam::vpr::VprOptions options;
  options.enabled = true;
  options.model_path = settings.model_path;
  options.min_training_keyframes = settings.vocabulary_min_keyframes;
  options.max_training_keyframes = settings.vocabulary_max_keyframes;
  impl->builder = std::make_unique<slam::vpr::VocabularyBuilder>(options);
}

VprVocabularyBuilder::VprVocabularyBuilder(VprVocabularyBuilder&& other) noexcept = default;

VprVocabularyBuilder::~VprVocabularyBuilder() = default;

uint32_t VprVocabularyBuilder::GetCapacity() const { return static_cast<uint32_t>(impl->builder->Capacity()); }

uint32_t VprVocabularyBuilder::GetImageCount() const { return static_cast<uint32_t>(impl->builder->Size()); }

void VprVocabularyBuilder::AddImage(const ImageData& image) {
  const slam::vpr::VprImage vpr_image = MakeVprImage(image, true);
  THROW_INVALID_ARG_IF(vpr_image.Empty(), "VprVocabularyBuilder::AddImage needs an image with pixels.");
  impl->builder->AddFrame(vpr_image);
}

void VprVocabularyBuilder::Save(std::string_view path) { impl->builder->Save(std::string(path)); }

}  // namespace cuvslam

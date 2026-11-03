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

#include "slam/vpr/vpr_vocabulary.h"

#include <cstdint>
#include <filesystem>
#include <fstream>

namespace cuvslam::slam::vpr {

namespace {

/// Far above any vocabulary, so a wrong path, like a map file, is refused unread.
constexpr uintmax_t kMaxVocabularyBytes = uintmax_t{64} << 20;

}  // namespace

bool ReadVocabularyFile(const std::string& path, Blob& bytes) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return false;
  }
  const uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size == 0 || size > kMaxVocabularyBytes) {
    return false;
  }
  bytes.resize(static_cast<size_t>(size));
  std::ifstream file(path, std::ios::binary);
  return static_cast<bool>(file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)));
}

std::unique_ptr<VocabularyFitter> VocabularyFitter::Create(const VprOptions& options, std::string& error) {
  if (!options.enabled || !options.vocabulary_path.empty()) {
    error = "VPR: a vocabulary is fitted with place recognition on and no vocabulary_path.";
    return nullptr;
  }
  std::unique_ptr<IVpr> backend = CreateVpr(options, error);
  if (!backend) {
    return nullptr;
  }
  if (backend->TrainingCapacity() == 0) {
    error = std::string("VPR: the ") + backend->Name() + " backend fits no vocabulary.";
    return nullptr;
  }
  return std::unique_ptr<VocabularyFitter>(new VocabularyFitter(std::move(backend), options.min_training_keyframes));
}

VocabularyFitter::VocabularyFitter(std::unique_ptr<IVpr> backend, size_t min_frames)
    : backend_(std::move(backend)), min_frames_(min_frames), capacity_(backend_->TrainingCapacity()) {}

VocabularyFitter::~VocabularyFitter() = default;

bool VocabularyFitter::AddFrame(const VprImage& image, std::string& error) {
  if (fitted_) {
    error = "VPR: the vocabulary is already fitted; add frames before saving it.";
    return false;
  }
  if (size_ >= capacity_) {
    error = "VPR: a vocabulary is fitted to at most " + std::to_string(capacity_) +
            " frames; spread them over the recordings.";
    return false;
  }
  const KeyFrameId node_id = size_;
  backend_->AddFrame(node_id, image);
  if (!backend_->HasNode(node_id)) {
    error = std::string("VPR: ") + backend_->Name() + " could not describe the frame, see the log.";
    return false;
  }
  ++size_;
  return true;
}

bool VocabularyFitter::Save(const std::string& path, std::string& error) {
  if (!fitted_) {
    if (size_ < min_frames_) {
      error = "VPR: a vocabulary is fitted to at least " + std::to_string(min_frames_) + " frames, not " +
              std::to_string(size_) + ".";
      return false;
    }
    backend_->Finalize();
    for (size_t node_id = 0; node_id < size_; ++node_id) {
      backend_->RemoveNode(node_id);
    }
    fitted_ = true;
  }
  Blob blob;
  BlobWriter writer(blob);
  backend_->Serialize(writer);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
  file.close();  // flushes, so fail() also sees write errors
  if (file.fail()) {
    error = "VPR: could not write the vocabulary \"" + path + "\".";
    return false;
  }
  return true;
}

}  // namespace cuvslam::slam::vpr

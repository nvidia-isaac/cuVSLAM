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
#include <stdexcept>
#include <string>

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

VocabularyBuilder::VocabularyBuilder(const VprOptions& options)
    : backend_(CreateVpr(options)),
      min_frames_(options.min_training_keyframes),
      capacity_(options.max_training_keyframes) {}

void VocabularyBuilder::AddFrame(const VprImage& image) {
  if (fitted_) {
    throw std::invalid_argument("VPR: the vocabulary is already fitted; add frames before saving it.");
  }
  if (size_ >= capacity_) {
    throw std::invalid_argument("VPR: a vocabulary is fitted to at most " + std::to_string(capacity_) +
                                " frames; spread them over the recordings.");
  }
  const KeyFrameId node_id = size_;
  backend_->AddFrame(node_id, image);
  if (!backend_->HasNode(node_id)) {
    throw std::runtime_error(std::string("VPR: ") + backend_->Name() + " could not describe the frame, see the log.");
  }
  ++size_;
}

void VocabularyBuilder::Save(const std::string& path) {
  if (!fitted_) {
    if (size_ < min_frames_) {
      throw std::invalid_argument("VPR: a vocabulary is fitted to at least " + std::to_string(min_frames_) +
                                  " frames, not " + std::to_string(size_) + ".");
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
    throw std::runtime_error("VPR: could not write the vocabulary \"" + path + "\".");
  }
}

}  // namespace cuvslam::slam::vpr

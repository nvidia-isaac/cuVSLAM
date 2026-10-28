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
#include <iterator>
#include <stdexcept>

namespace cuvslam::slam::vpr {

namespace {

/// Far above any vocabulary, so a wrong path, like a map file, is refused unread.
constexpr uintmax_t kMaxVocabularyBytes = uintmax_t{64} << 20;

// An interrupted save must not leave a truncated vocabulary.
void WriteAtomically(const Blob& blob, const std::string& path) {
  const std::string temporary = path + ".tmp";
  std::error_code error;
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    file.close();  // flushes, so fail() also sees write errors
    if (file.fail()) {
      std::filesystem::remove(temporary, error);
      throw std::runtime_error("VPR: could not write the vocabulary \"" + path + "\".");
    }
  }
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw std::runtime_error("VPR: could not write the vocabulary \"" + path + "\": " + error.message() + ".");
  }
}

}  // namespace

Blob ReadVocabularyFile(const std::string& path) {
  Blob bytes;
  std::error_code error;
  if (std::filesystem::is_regular_file(path, error) && std::filesystem::file_size(path, error) <= kMaxVocabularyBytes) {
    try {
      std::ifstream file(path, std::ios::binary);
      bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
      if (!file.good() && !file.eof()) {
        bytes.clear();
      }
    } catch (const std::ios_base::failure&) {
      bytes.clear();  // libstdc++ throws on a read error
    }
  }
  if (bytes.empty()) {
    throw std::runtime_error("VPR: could not read the vocabulary \"" + path + "\".");
  }
  return bytes;
}

VocabularyFitter::VocabularyFitter(const VprOptions& options) {
  if (!options.enabled || !options.vocabulary_path.empty()) {
    throw std::invalid_argument("VPR: a vocabulary is fitted with place recognition on and no vocabulary_path.");
  }
  backend_ = CreateVpr(options);
  capacity_ = backend_->TrainingCapacity();
  if (capacity_ == 0) {
    throw std::invalid_argument(std::string("VPR: the ") + backend_->Name() + " backend fits no vocabulary.");
  }
  min_frames_ = options.min_training_keyframes;
}

VocabularyFitter::~VocabularyFitter() = default;

void VocabularyFitter::AddFrame(const VprImage& image) {
  if (fitted_) {
    throw std::invalid_argument("VPR: the vocabulary is already fitted; add frames before saving it.");
  }
  if (size_ >= capacity_) {
    throw std::invalid_argument("VPR: a vocabulary is fitted to at most " + std::to_string(capacity_) +
                                " frames of this engine; spread them over the recordings.");
  }
  const KeyFrameId node_id = size_;
  backend_->AddFrame(node_id, image);
  if (!backend_->HasNode(node_id)) {
    throw std::runtime_error(std::string("VPR: ") + backend_->Name() + " could not describe the frame, see the log.");
  }
  ++size_;
}

void VocabularyFitter::Save(const std::string& path) {
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
  WriteAtomically(blob, path);
}

}  // namespace cuvslam::slam::vpr

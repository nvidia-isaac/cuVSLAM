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

#pragma once

#include <memory>
#include <string>

#include "slam/common/blob.h"
#include "slam/vpr/ivpr.h"

namespace cuvslam::slam::vpr {

/// The whole file at `path`.
/// @throws std::runtime_error naming `path` if it is missing, unreadable or empty.
Blob ReadVocabularyFile(const std::string& path);

/// Fits a vocabulary once, offline, for VprOptions::vocabulary_path: maps frames into a backend, fits it, removes the
/// frames again and writes what the backend's Serialize() then holds. Not thread safe.
class VocabularyFitter {
public:
  /// @throws what CreateVpr() throws; std::invalid_argument if `options` are disabled, name a vocabulary_path, or
  /// fit no vocabulary.
  explicit VocabularyFitter(const VprOptions& options);
  ~VocabularyFitter();

  /// Most frames fitted to, every patch descriptor of each.
  size_t Capacity() const { return capacity_; }

  /// Frames added so far.
  size_t Size() const { return size_; }

  /// @throws std::invalid_argument at Capacity() or after Save(); std::runtime_error if the backend cannot describe it.
  void AddFrame(const VprImage& image);

  /// Fit to the frames added (first call only), then replace `path` atomically.
  /// @throws std::invalid_argument with fewer than VprOptions::min_training_keyframes frames;
  /// std::runtime_error naming `path` if it cannot be written.
  void Save(const std::string& path);

private:
  std::unique_ptr<IVpr> backend_;
  size_t min_frames_ = 0;
  size_t capacity_ = 0;
  size_t size_ = 0;
  bool fitted_ = false;
};

}  // namespace cuvslam::slam::vpr

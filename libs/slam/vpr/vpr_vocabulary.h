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

/// The whole file at `path` into `bytes`; false unless it is a readable, non-empty regular file of at most 64 MiB.
bool ReadVocabularyFile(const std::string& path, Blob& bytes);

/// Fits a vocabulary once, offline, for VprOptions::vocabulary_path: maps frames into a backend, fits it, removes the
/// frames again and writes what the backend's Serialize() then holds. Not thread safe.
class VocabularyBuilder {
public:
  /// nullptr with `error` set if `options` are disabled, name a vocabulary_path, or make no backend that fits one.
  static std::unique_ptr<VocabularyBuilder> Create(const VprOptions& options, std::string& error);
  ~VocabularyBuilder();

  /// Most frames fitted to.
  size_t Capacity() const { return capacity_; }

  /// Frames added so far.
  size_t Size() const { return size_; }

  /// The vocabulary is fitted, by the first Save().
  bool Fitted() const { return fitted_; }

  /// False with `error` set at Capacity(), once Fitted(), or if the backend cannot describe `image`.
  bool AddFrame(const VprImage& image, std::string& error);

  /// Fit to the frames added on the first call, then write `path`; false with `error` set if either step fails.
  bool Save(const std::string& path, std::string& error);

private:
  VocabularyBuilder(std::unique_ptr<IVpr> backend, size_t min_frames);

  std::unique_ptr<IVpr> backend_;
  size_t min_frames_ = 0;
  size_t capacity_ = 0;
  size_t size_ = 0;
  bool fitted_ = false;
};

}  // namespace cuvslam::slam::vpr

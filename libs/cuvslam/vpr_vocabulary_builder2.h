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

#include <cstdint>
#include <memory>
#include <string_view>

#include "cuvslam/cuvslam2.h"

namespace cuvslam {

/**
 * @brief Fits the place recognition vocabulary of one operating domain, for `Slam::VprSettings::vocabulary_path`
 *
 * Fits it as a session fits its own, to images spread over recordings of the domain. Needs what
 * `Slam::Config::use_vpr` needs. Not thread safe.
 */
class CUVSLAM_API VprVocabularyBuilder {
public:
  /**
   * Load the engine; sessions need an engine of the same network.
   * @param[in] settings `model_path`, and `vocabulary_min_keyframes` and `vocabulary_max_keyframes` counted in
   * images; the rest is ignored
   * @throws std::invalid_argument if `vocabulary_min_keyframes` is 0 or above `vocabulary_max_keyframes`
   * @throws std::runtime_error if the engine cannot be loaded, or cuVSLAM was built without TensorRT
   */
  explicit VprVocabularyBuilder(const Slam::VprSettings& settings);

  /**
   * Destructor
   */
  ~VprVocabularyBuilder();

  /// @return most images fitted to, `Slam::VprSettings::vocabulary_max_keyframes`; spread them out
  uint32_t GetCapacity() const;

  /**
   * Describe an image of the camera SLAM maps, its first primary camera.
   * @param[in] image 8 bit mono or RGB image in CPU or GPU memory
   * @throws std::invalid_argument if `image` is unusable, `GetCapacity()` images were added, or after Save()
   * @throws std::runtime_error if the engine fails on it
   */
  void AddImage(const ImageData& image);

  /**
   * Fit the vocabulary to the images added (first call only), then write it to `path`.
   * @param[in] path vocabulary file to write
   * @throws std::invalid_argument with fewer than `Slam::VprSettings::vocabulary_min_keyframes` images
   * @throws std::runtime_error if `path` cannot be written
   */
  void Save(std::string_view path);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace cuvslam

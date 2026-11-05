
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

#include "sof/selection.h"

#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "sof/image_pyramid_float.h"
#include "utils/image_loader.h"

namespace test::sof {
namespace {

// Existing tracks leave some bins short of their share, which must stay short when no features are desired
TEST(GoodFeaturesToTrackDetector, AddsNoFeaturesWhenNoneDesired) {
  cuvslam::utils::ImageLoaderT loader;
  ASSERT_TRUE(loader.load(std::string{CUVSLAM_TEST_ASSETS} + "sof/left.png"));
  const cuvslam::ImageMatrixT image = loader.getImage().cast<float>();
  cuvslam::sof::ImagePyramidT pyramid(image);
  pyramid.buildPyramid();
  cuvslam::sof::GradientPyramidT gradient;
  gradient.set(pyramid);

  cuvslam::sof::GoodFeaturesToTrackDetector detector;
  std::vector<cuvslam::Vector2T> features;
  detector.computeGFTTAndSelectFeatures(gradient, 0, 0, 0, 0, nullptr, cuvslam::sof::TracksVector{}, 450, features);
  ASSERT_FALSE(features.empty());
  cuvslam::sof::TracksVector tracks;
  tracks.add(0, features);

  std::vector<cuvslam::Vector2T> selected;
  detector.computeGFTTAndSelectFeatures(gradient, 0, 0, 0, 0, nullptr, tracks, 0, selected);
  EXPECT_TRUE(selected.empty()) << selected.size() << " features added";
}

}  // namespace
}  // namespace test::sof

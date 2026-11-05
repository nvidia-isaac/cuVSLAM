
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
using namespace cuvslam;
using namespace cuvslam::sof;

// Tops up existing tracks, as a keyframe does. The bins apportion the existing and the desired features together,
// so bins short of their share still have room when no new features are desired.
TEST(GoodFeaturesToTrackDetector, SelectsNoMoreThanDesired) {
  utils::ImageLoaderT loader;
  ASSERT_TRUE(loader.load(std::string{CUVSLAM_TEST_ASSETS} + "sof/left.png"));
  const ImageMatrixT image = loader.getImage().cast<float>();
  ImagePyramidT pyramid(image);
  pyramid.buildPyramid();
  GradientPyramidT gradient;
  gradient.set(pyramid);

  GoodFeaturesToTrackDetector detector;
  std::vector<Vector2T> features;
  detector.computeGFTTAndSelectFeatures(gradient, 0, 0, 0, 0, nullptr, TracksVector{}, 450, features);
  ASSERT_FALSE(features.empty());
  TracksVector tracks;
  tracks.add(0, features);

  for (const size_t desired : {0, 1, 5}) {
    std::vector<Vector2T> selected;
    detector.computeGFTTAndSelectFeatures(gradient, 0, 0, 0, 0, nullptr, tracks, desired, selected);
    EXPECT_LE(selected.size(), desired);
  }
}

}  // namespace
}  // namespace test::sof

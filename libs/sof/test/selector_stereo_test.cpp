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

#include "sof/selector_stereo.h"

#include "common/include_gtest.h"
#include "sof/sof_config.h"

#include <cstddef>
#include <vector>

namespace test::sof {
namespace {

using namespace cuvslam;

cuvslam::sof::TracksVector MakeTracks(size_t count) {
  cuvslam::sof::TracksVector tracks;
  std::vector<Vector2T> points;
  points.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    points.emplace_back(static_cast<float>(i), static_cast<float>(i));
  }
  tracks.add(0, points);
  tracks.sort();
  return tracks;
}

cuvslam::sof::Settings SettingsWithThreshold(float survivor_from_last) {
  cuvslam::sof::Settings settings;
  settings.feature_selection_settings.survivor_from_last = survivor_from_last;
  return settings;
}

// The threshold comes from the frame's settings, so one selector follows per-frame overrides.
TEST(SelectorStereoTest, UsesPerFrameSurvivorThreshold) {
  const cuvslam::sof::Settings defaults;
  cuvslam::sof::SelectorStereo selector;
  const cuvslam::sof::TracksVector initial = MakeTracks(100);

  EXPECT_TRUE(selector.select(initial, defaults));  // the first frame always detects features
  selector.set_tracks(initial);

  cuvslam::sof::TracksVector current = initial;
  for (size_t i = 0; i < 50; ++i) {
    current.kill(i);
  }

  // 50% of the tracks survived since the last detection.
  EXPECT_FALSE(selector.select(current, defaults));                     // 50 >= 41 (default)
  EXPECT_FALSE(selector.select(current, SettingsWithThreshold(50.f)));  // 50 >= 50
  EXPECT_TRUE(selector.select(current, SettingsWithThreshold(60.f)));   // 50 < 60
}

}  // namespace
}  // namespace test::sof

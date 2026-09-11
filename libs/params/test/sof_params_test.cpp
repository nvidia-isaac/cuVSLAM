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

#include "common/include_gtest.h"
#include "params/registry.h"
#include "sof/sof_config.h"

namespace cuvslam::params {
namespace {

class SofParamsTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.Add("sof", settings);
    registry.Add("sof.feature_selection", settings.feature_selection_settings);
  }

  sof::Settings settings;
  Registry registry;
};

TEST_F(SofParamsTest, ManualMulticamModeIsNotReachableByName) {
  // Manual mode needs an accompanying camera setup, so no scalar value may select it.
  EXPECT_THROW(registry.Set("sof.multicam_mode", "manual", Source::File), std::runtime_error);
  EXPECT_EQ(settings.multicam_mode, sof::Settings{}.multicam_mode);
}

TEST_F(SofParamsTest, NestedSelectorSettingsGetTheirOwnPrefix) {
  registry.Set("sof.feature_selection.survivor_from_last", "55", Source::Api);
  EXPECT_EQ(settings.feature_selection_settings.survivor_from_last, 55.f);
}

TEST_F(SofParamsTest, StructuralFieldsStayOffTheTuningSurface) {
  // multicam_setup is camera topology, not a value a sweep can vary.
  EXPECT_THROW(registry.Resolve("sof.multicam_setup"), std::invalid_argument);
  EXPECT_THROW(registry.Resolve("multicam_setup"), std::invalid_argument);
}

TEST_F(SofParamsTest, AbbreviationMatchesWholeSegmentsOnly) {
  // `lr_tracker` ends in the characters "tracker" but not in the segment "tracker", so the
  // abbreviation stays unambiguous. Without the segment rule every underscore in a name would
  // create a spurious collision.
  EXPECT_EQ(registry.Resolve("tracker"), "sof.tracker");
  EXPECT_EQ(registry.Resolve("lr_tracker"), "sof.lr_tracker");

  registry.Set("lr_tracker", "klt", Source::CommandLine);
  EXPECT_EQ(settings.lr_tracker, sof::TrackerType::KLT);
  EXPECT_EQ(settings.tracker, sof::Settings{}.tracker) << "the sibling parameter must be untouched";
}

TEST_F(SofParamsTest, SurvivorFromLastIsReachableByItsBareName) {
  // Lives in a nested group, so this also covers abbreviation across a multi-segment prefix.
  EXPECT_EQ(registry.Resolve("survivor_from_last"), "sof.feature_selection.survivor_from_last");
  EXPECT_EQ(registry.Resolve("feature_selection.survivor_from_last"), "sof.feature_selection.survivor_from_last");
}

}  // namespace
}  // namespace cuvslam::params

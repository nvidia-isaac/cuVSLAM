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

#include <algorithm>
#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "odometry/svo_config.h"
#include "params/registry.h"

namespace cuvslam::params {
namespace {

/// Registers the solver settings the way Odometry does, so the key layout under test is the real one.
class SolverParamsTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.Add("sof", settings.sof);
    registry.Add("sof.feature_selection", settings.sof.feature_selection_settings);
    registry.Add("kf", settings.kf);
    registry.Add("sba", settings.sba);
    registry.Add("sm", settings.sm);
    registry.Add("vo_pnp", settings.vo_pnp);
    registry.Add("inertial_stereo_pnp", settings.inertial_stereo_pnp);
    registry.Add("imu_pnp", settings.imu_pnp);
    registry.Add("icp", settings.icp);
  }

  odom::TrackPerFrameSettings settings;
  Registry registry;
};

TEST_F(SolverParamsTest, TheTwoPnpInstancesAreIndependent) {
  // Both share one descriptor list, so this guards against the list accidentally binding to a
  // single instance.
  registry.Set("vo_pnp.huber", "0.5", Source::Api);
  EXPECT_EQ(settings.vo_pnp.huber, 0.5f);
  EXPECT_EQ(settings.inertial_stereo_pnp.huber, pnp::PNPSettings::InertialSettings().huber);

  registry.Set("inertial_stereo_pnp.huber", "0.25", Source::Api);
  EXPECT_EQ(settings.inertial_stereo_pnp.huber, 0.25f);
  EXPECT_EQ(settings.vo_pnp.huber, 0.5f);
}

TEST_F(SolverParamsTest, InertialStereoPnpKeepsItsOwnDefaults) {
  // TrackPerFrameSettings seeds this instance from PNPSettings::InertialSettings(), and List()
  // reports defaults from a default-constructed PNPSettings, so the two legitimately differ.
  const std::vector<ParamInfo> infos = registry.List();
  const auto huber =
      std::find_if(infos.begin(), infos.end(), [](const ParamInfo& i) { return i.key == "inertial_stereo_pnp.huber"; });
  ASSERT_NE(huber, infos.end());
  EXPECT_EQ(huber->value, "0.100000001") << "current value comes from InertialSettings()";
}

TEST_F(SolverParamsTest, OverrideFrameSelectionIsNotAParameter) {
  // It is the one value that varies per frame, so it arrives as a per-frame hint. Exposing it as a
  // parameter would silently apply one keyframe decision to every frame.
  EXPECT_THROW(registry.Resolve("kf.override_frame_selection"), std::invalid_argument);
  EXPECT_THROW(registry.Resolve("override_frame_selection"), std::invalid_argument);
}

TEST_F(SolverParamsTest, EveryKeyIsUniqueAcrossGroups) {
  std::vector<std::string> keys;
  for (const ParamInfo& info : registry.List()) {
    keys.push_back(info.key);
  }
  std::vector<std::string> sorted = keys;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end()) << "duplicate parameter key";
  EXPECT_EQ(keys.size(), sorted.size());
}

TEST_F(SolverParamsTest, WritesLandInTheStructThatOdometryActuallyPassesDown) {
  registry.Set("sof.num_desired_tracks", "321", Source::File);
  registry.Set("kf.survivor_from_last", "60", Source::File);
  registry.Set("sba.num_sba_iterations", "9", Source::File);
  registry.Set("sm.min_num_kf_for_gravity", "30", Source::File);
  registry.Set("icp.blending_alpha", "0.4", Source::File);

  EXPECT_EQ(settings.sof.num_desired_tracks, 321);
  EXPECT_EQ(settings.kf.survivor_from_last, 60.f);
  EXPECT_EQ(settings.sba.num_sba_iterations, 9);
  EXPECT_EQ(settings.sm.min_num_kf_for_gravity, 30u);
  EXPECT_EQ(settings.icp.blending_alpha, 0.4f);
}

TEST_F(SolverParamsTest, UnsignedFieldRejectsNegativeValues) {
  EXPECT_THROW(registry.Set("sm.min_num_kf_for_gravity", "-1", Source::File), std::runtime_error);
  EXPECT_EQ(settings.sm.min_num_kf_for_gravity, pipelines::StateMachineSettings{}.min_num_kf_for_gravity);
}

TEST_F(SolverParamsTest, SurvivorPercentageIsRangeChecked) {
  EXPECT_THROW(registry.Set("kf.survivor_from_last", "101", Source::File), std::runtime_error);
  EXPECT_THROW(registry.Set("icp.blending_alpha", "1.2", Source::File), std::runtime_error);
}

TEST_F(SolverParamsTest, DumpCoversTheWholeSolverConfiguration) {
  const std::vector<ParamInfo> infos = registry.List();
  // 12 sof + 1 feature_selection + 2 kf + 8 sba + 5 sm + 9 vo_pnp + 9 inertial_stereo_pnp
  // + 3 imu_pnp + 9 icp
  EXPECT_EQ(infos.size(), 58u);
  for (const ParamInfo& info : infos) {
    EXPECT_FALSE(info.doc.empty()) << info.key << " has no documentation";
  }
}

}  // namespace
}  // namespace cuvslam::params

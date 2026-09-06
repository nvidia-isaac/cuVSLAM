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

#include "odometry/svo_config.h"

#include "common/include_gtest.h"

namespace cuvslam::odom {
namespace {

/// Construction-time settings with every field that a frame re-reads set away from its default.
Settings MakeConfiguredSettings() {
  Settings settings;
  settings.sof_settings.box3_prefilter = true;
  settings.sof_settings.border_top = 20;
  settings.sof_settings.border_bottom = 21;
  settings.sof_settings.border_left = 10;
  settings.sof_settings.border_right = 11;
  settings.sof_settings.num_desired_tracks = 321;
  settings.kf_settings.survivor_from_last = 55.f;
  settings.kf_settings.max_timedelta_between_kfs_s = 7;
  settings.sba_settings.num_sba_iterations = 9;
  settings.sm_settings.min_num_kf_for_gravity = 30;
  return settings;
}

// box3_prefilter and the border fields are read from the per-frame struct on every frame
// (MonoSOF builds the pyramid and masks features from them), so a per-frame struct built from type
// defaults discards Odometry::Config::use_denoising and Rig::Camera::border_*.
TEST(MakeTrackPerFrameSettings, CarriesTheDenoisingAndBorderSettingsAFrameRereads) {
  const TrackPerFrameSettings per_frame = MakeTrackPerFrameSettings(MakeConfiguredSettings());

  EXPECT_TRUE(per_frame.sof.box3_prefilter);
  EXPECT_EQ(per_frame.sof.border_top, 20);
  EXPECT_EQ(per_frame.sof.border_bottom, 21);
  EXPECT_EQ(per_frame.sof.border_left, 10);
  EXPECT_EQ(per_frame.sof.border_right, 11);
}

TEST(MakeTrackPerFrameSettings, CarriesEverySubStructThatExistsInBothPlaces) {
  const TrackPerFrameSettings per_frame = MakeTrackPerFrameSettings(MakeConfiguredSettings());

  EXPECT_EQ(per_frame.sof.num_desired_tracks, 321);
  EXPECT_EQ(per_frame.kf.survivor_from_last, 55.f);
  EXPECT_EQ(per_frame.kf.max_timedelta_between_kfs_s, 7);
  EXPECT_EQ(per_frame.sba.num_sba_iterations, 9);
  EXPECT_EQ(per_frame.sm.min_num_kf_for_gravity, 30u);
}

TEST(MakeTrackPerFrameSettings, LeavesSolverSettingsWithNoCounterpartAtTheirDefaults) {
  const TrackPerFrameSettings per_frame = MakeTrackPerFrameSettings(MakeConfiguredSettings());

  EXPECT_EQ(per_frame.vo_pnp.huber, pnp::PNPSettings{}.huber);
  EXPECT_EQ(per_frame.imu_pnp.max_iteration, pipelines::InertialPnPSettings{}.max_iteration);
  // Seeded from InertialSettings() by TrackPerFrameSettings itself, not from Settings.
  EXPECT_EQ(per_frame.inertial_stereo_pnp.huber, pnp::PNPSettings::InertialSettings().huber);
}

TEST(MakeTrackPerFrameSettings, DefaultSettingsYieldDefaultPerFrameSettings) {
  const TrackPerFrameSettings per_frame = MakeTrackPerFrameSettings(Settings{});

  EXPECT_EQ(per_frame.sof.box3_prefilter, sof::Settings{}.box3_prefilter);
  EXPECT_EQ(per_frame.sof.border_top, sof::Settings{}.border_top);
  EXPECT_EQ(per_frame.sof.num_desired_tracks, sof::Settings{}.num_desired_tracks);
  EXPECT_EQ(per_frame.kf.survivor_from_last, KeyFrameSettings{}.survivor_from_last);
}

TEST(MakeTrackPerFrameSettings, KeyframeOverrideStaysUnsetBecauseItIsAPerFrameDecision) {
  Settings settings = MakeConfiguredSettings();
  settings.kf_settings.override_frame_selection = true;

  // The field exists on KeyFrameSettings, so it does travel; what matters is that nothing in
  // construction sets it, leaving automatic keyframe selection in place by default.
  EXPECT_FALSE(MakeTrackPerFrameSettings(Settings{}).kf.override_frame_selection.has_value());
  EXPECT_TRUE(MakeTrackPerFrameSettings(settings).kf.override_frame_selection.value_or(false));
}

}  // namespace
}  // namespace cuvslam::odom

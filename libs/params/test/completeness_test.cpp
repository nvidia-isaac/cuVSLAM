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
#include "cuvslam/cuvslam_params.h"
#include "odometry/svo_config.h"
#include "params/registry.h"

/**
 * @file completeness_test.cpp
 *
 * Guards against a field being added to a described struct and not described.
 *
 * A descriptor list is written by hand, so nothing in the compiler notices when a struct grows a
 * field that no parameter reaches. The failure is silent and unpleasant: the new setting simply
 * cannot be set from a file, the command line or Python, and it is missing from every report.
 *
 * Adding a field almost always changes the struct's size, so each described struct is pinned to
 * the size and field count it was last reviewed at. When one of these fails, look at what changed
 * and then either describe the new field or record it below as a deliberate omission, and update
 * the numbers.
 *
 * These are review prompts, not correctness assertions -- a field that lands entirely inside
 * existing padding will slip through, and the sizes assume an LP64 target.
 */

namespace cuvslam::params {
namespace {

/// Number of parameters a struct describes, including any nested groups listed separately.
template <typename T>
constexpr size_t DescribedCount() {
  return Fields<T>::kList.size();
}

TEST(ParameterCompleteness, SofSettings) {
  // Deliberately not parameters: multicam_setup (manual camera topology needs more than a scalar)
  // and feature_selection_settings (described separately, registered under its own prefix).
  EXPECT_EQ(sizeof(sof::Settings), 72u) << "sof::Settings changed shape; revisit its parameters";
  EXPECT_EQ(DescribedCount<sof::Settings>(), 12u);
  EXPECT_EQ(sizeof(sof::SelectorStereoSettings), 4u);
  EXPECT_EQ(DescribedCount<sof::SelectorStereoSettings>(), 1u);
}

TEST(ParameterCompleteness, KeyFrameSettings) {
  // Deliberately not a parameter: override_frame_selection is a per-frame decision, not config.
  EXPECT_EQ(sizeof(odom::KeyFrameSettings), 24u) << "KeyFrameSettings changed shape";
  EXPECT_EQ(DescribedCount<odom::KeyFrameSettings>(), 2u);
}

TEST(ParameterCompleteness, SbaSettings) {
  // Deliberately not a parameter: mode follows from the odometry mode and backend at construction.
  EXPECT_EQ(sizeof(sba::Settings), 36u) << "sba::Settings changed shape";
  EXPECT_EQ(DescribedCount<sba::Settings>(), 8u);
}

TEST(ParameterCompleteness, StateMachineSettings) {
  EXPECT_EQ(sizeof(pipelines::StateMachineSettings), 40u) << "StateMachineSettings changed shape";
  EXPECT_EQ(DescribedCount<pipelines::StateMachineSettings>(), 5u);
}

TEST(ParameterCompleteness, InertialPnPSettings) {
  EXPECT_EQ(sizeof(pipelines::InertialPnPSettings), 12u) << "InertialPnPSettings changed shape";
  EXPECT_EQ(DescribedCount<pipelines::InertialPnPSettings>(), 3u);
}

TEST(ParameterCompleteness, PnpSettings) {
  // Deliberately not a parameter: verbose is a debugging aid.
  EXPECT_EQ(sizeof(pnp::PNPSettings), 40u) << "PNPSettings changed shape";
  EXPECT_EQ(DescribedCount<pnp::PNPSettings>(), 9u);
}

TEST(ParameterCompleteness, IcpSettings) {
  // Deliberately not a parameter: verbose is a debugging aid.
  EXPECT_EQ(sizeof(pnp::ICPSettings), 40u) << "ICPSettings changed shape";
  EXPECT_EQ(DescribedCount<pnp::ICPSettings>(), 9u);
}

TEST(ParameterCompleteness, PublicConfigs) {
  // Deliberately not parameters: rgbd_settings and multisensor_settings are described separately
  // and registered under their own prefixes.
  EXPECT_EQ(DescribedCount<Odometry::Config>(), 15u) << "Odometry::Config changed; revisit its parameters";
  EXPECT_EQ(DescribedCount<Odometry::RGBDSettings>(), 3u);
  EXPECT_EQ(DescribedCount<Odometry::MultisensorSettings>(), 3u);
  EXPECT_EQ(DescribedCount<Slam::Config>(), 12u) << "Slam::Config changed; revisit its parameters";
}

/// Every parameter of every described struct must round-trip through its own string form.
///
/// Catches a descriptor whose parse and format disagree, which would otherwise show up as a
/// dumped configuration that cannot be reloaded.
template <typename T>
void ExpectRoundTrip(std::string_view prefix) {
  T settings;
  Registry registry;
  registry.Add(prefix, settings);
  for (const ParamInfo& info : registry.List()) {
    EXPECT_NO_THROW(registry.Set(info.key, info.value, Source::Api)) << info.key << " = " << info.value;
    EXPECT_EQ(registry.Get(info.key), info.value) << info.key << " does not survive a round trip";
  }
}

TEST(ParameterCompleteness, EveryDefaultRoundTrips) {
  ExpectRoundTrip<sof::Settings>("sof");
  ExpectRoundTrip<sof::SelectorStereoSettings>("fs");
  ExpectRoundTrip<odom::KeyFrameSettings>("kf");
  ExpectRoundTrip<sba::Settings>("sba");
  ExpectRoundTrip<pipelines::StateMachineSettings>("sm");
  ExpectRoundTrip<pipelines::InertialPnPSettings>("imu_pnp");
  ExpectRoundTrip<pnp::PNPSettings>("vo_pnp");
  ExpectRoundTrip<pnp::ICPSettings>("icp");
  ExpectRoundTrip<Odometry::Config>("odometry");
  ExpectRoundTrip<Odometry::RGBDSettings>("rgbd");
  ExpectRoundTrip<Odometry::MultisensorSettings>("multisensor");
  ExpectRoundTrip<Slam::Config>("slam");
}

}  // namespace
}  // namespace cuvslam::params

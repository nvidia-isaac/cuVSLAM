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

#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "params/registry.h"

/**
 * @file svo_config_test.cpp
 *
 * MakeTrackPerFrameSettings() must not drop anything.
 *
 * Asserting field by field would only restate the assignments, so instead every parameter of the
 * construction-time settings is changed away from its default, and the per-frame settings are then
 * required to report the same value for every one of them. A sub-struct that stops being copied,
 * or a newly described field that is never copied, shows up as a mismatch without this test
 * needing to know the field names.
 *
 * A sub-struct added to both types but never copied is still only caught once it is registered
 * below, which is the one case that needs a hand edit.
 */

namespace cuvslam::odom {
namespace {

using params::Registry;
using params::Source;

/// Registers the sub-structs that exist in both Settings and TrackPerFrameSettings, under the same
/// names on both sides so their listings can be compared key by key.
Registry RegisterStored(Settings& settings) {
  Registry registry;
  registry.Add("sof", settings.sof_settings);
  registry.Add("sof.feature_selection", settings.sof_settings.feature_selection_settings);
  registry.Add("kf", settings.kf_settings);
  registry.Add("sba", settings.sba_settings);
  registry.Add("sm", settings.sm_settings);
  return registry;
}

Registry RegisterPerFrame(TrackPerFrameSettings& settings) {
  Registry registry;
  registry.Add("sof", settings.sof);
  registry.Add("sof.feature_selection", settings.sof.feature_selection_settings);
  registry.Add("kf", settings.kf);
  registry.Add("sba", settings.sba);
  registry.Add("sm", settings.sm);
  return registry;
}

/// A valid value for @p info that differs from its current one, derived from the reported type so
/// a newly described field needs no change here.
std::string DifferentValue(const params::ParamInfo& info) {
  if (info.type == "bool") {
    return info.value == "true" ? "false" : "true";
  }
  if (info.type == "bool|null") {
    return info.value == "null" ? "true" : "null";
  }
  if (info.type == "string") {
    return info.value + "-changed";
  }
  if (info.type == "int32[]") {
    return info.value == "7" ? "8" : "7";
  }
  if (info.type.rfind("enum{", 0) == 0) {
    // "enum{a|b|c}" -> the first spelling that is not the current one.
    const std::string names = info.type.substr(5, info.type.size() - 6);
    size_t start = 0;
    while (start <= names.size()) {
      const size_t bar = names.find('|', start);
      const std::string name = names.substr(start, bar == std::string::npos ? std::string::npos : bar - start);
      if (name != info.value) {
        return name;
      }
      if (bar == std::string::npos) {
        break;
      }
      start = bar + 1;
    }
    return info.value;
  }
  // Numeric: nudge upward, staying inside the ranges these settings declare.
  const double current = std::stod(info.value);
  const double changed = current + 1.0;
  if (info.type == "float") {
    return std::to_string(changed);
  }
  return std::to_string(static_cast<int64_t>(changed));
}

TEST(MakeTrackPerFrameSettings, CarriesEveryParameterOfEverySharedSubStruct) {
  Settings stored;
  Registry stored_registry = RegisterStored(stored);

  // Move every parameter off its default, so a value that fails to travel cannot coincidentally
  // match the default on the other side.
  std::vector<params::ParamInfo> expected;
  for (const params::ParamInfo& info : stored_registry.List()) {
    const std::string changed = DifferentValue(info);
    ASSERT_NO_THROW(stored_registry.Set(info.key, changed, Source::Api)) << info.key << " = " << changed;
  }
  expected = stored_registry.List();
  ASSERT_FALSE(expected.empty());

  TrackPerFrameSettings per_frame = MakeTrackPerFrameSettings(stored);
  Registry per_frame_registry = RegisterPerFrame(per_frame);

  for (const params::ParamInfo& info : expected) {
    EXPECT_EQ(per_frame_registry.Get(info.key), info.value) << info.key << " did not reach the per-frame settings";
  }
}

TEST(MakeTrackPerFrameSettings, PerturbationActuallyChangedEveryParameter) {
  // Guards the test above: if DifferentValue() ever returned the current value for some type, the
  // comparison would pass without proving anything.
  Settings settings;
  Registry registry = RegisterStored(settings);
  for (const params::ParamInfo& info : registry.List()) {
    EXPECT_NE(DifferentValue(info), info.value) << info.key << " (" << info.type << ") was not perturbed";
  }
}

TEST(MakeTrackPerFrameSettings, LeavesSolverSettingsWithNoCounterpartAtTheirDefaults) {
  Settings stored;
  stored.sof_settings.num_desired_tracks = 321;

  const TrackPerFrameSettings per_frame = MakeTrackPerFrameSettings(stored);

  // Settings holds no PnP or ICP configuration, so these keep the defaults TrackPerFrameSettings
  // gives them -- including the inertial fallback's own InertialSettings() seed.
  EXPECT_EQ(per_frame.vo_pnp.huber, pnp::PNPSettings{}.huber);
#ifdef USE_CUDA
  EXPECT_EQ(per_frame.icp.blending_alpha, pnp::ICPSettings{}.blending_alpha);
#endif
  EXPECT_EQ(per_frame.inertial_stereo_pnp.huber, pnp::PNPSettings::InertialSettings().huber);
}

}  // namespace
}  // namespace cuvslam::odom

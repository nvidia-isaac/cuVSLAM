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
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/include_gtest.h"
#include "cuvslam/cuvslam2.h"

namespace {

using cuvslam::Odometry;
using cuvslam::Rig;

constexpr int32_t kWidth = 640;
constexpr int32_t kHeight = 480;
constexpr float kFocal = 500.f;
constexpr float kBaseline = 0.1f;

Rig MakeStereoRig() {
  cuvslam::Camera camera;
  camera.size = {kWidth, kHeight};
  camera.focal = {kFocal, kFocal};
  camera.principal = {kWidth / 2.f, kHeight / 2.f};

  Rig rig;
  rig.cameras.push_back(camera);
  rig.cameras.push_back(camera);
  rig.cameras[1].rig_from_camera.translation = {kBaseline, 0.f, 0.f};
  return rig;
}

Odometry::Config MakeConfig() {
  Odometry::Config config;
  config.async_sba = false;
  return config;
}

/// Exercises the parameter API through the public interface only, the way a tool would.
class ParametersTest : public testing::Test {
protected:
  ParametersTest() : odometry_(MakeStereoRig(), MakeConfig()) {}

  /// Current value of a parameter, read back through GetParameters().
  std::string ValueOf(std::string_view key) const {
    const std::vector<Odometry::ParameterInfo> params = odometry_.GetParameters();
    const auto it =
        std::find_if(params.begin(), params.end(), [key](const Odometry::ParameterInfo& p) { return p.key == key; });
    EXPECT_NE(it, params.end()) << "no parameter named " << key;
    return it != params.end() ? std::string(it->value) : std::string{};
  }

  Odometry::ParameterInfo InfoOf(std::string_view key) const {
    const std::vector<Odometry::ParameterInfo> params = odometry_.GetParameters();
    const auto it =
        std::find_if(params.begin(), params.end(), [key](const Odometry::ParameterInfo& p) { return p.key == key; });
    EXPECT_NE(it, params.end()) << "no parameter named " << key;
    return it != params.end() ? *it : Odometry::ParameterInfo{};
  }

  Odometry odometry_;
};

TEST_F(ParametersTest, ReportsEveryParameterWithDocTypeAndDefault) {
  const std::vector<Odometry::ParameterInfo> params = odometry_.GetParameters();
  ASSERT_FALSE(params.empty());
  for (const Odometry::ParameterInfo& p : params) {
    EXPECT_FALSE(p.key.empty());
    EXPECT_FALSE(p.doc.empty()) << p.key << " has no documentation";
    EXPECT_FALSE(p.type.empty()) << p.key << " has no type";
    EXPECT_FALSE(p.value.empty()) << p.key << " has no value";
    EXPECT_EQ(p.source, "default") << p.key << " was not assigned, so it must report as a default";
  }
}

TEST_F(ParametersTest, SetsAParameterAndReportsItsNewValueAndSource) {
  odometry_.SetParameter("sof.num_desired_tracks", "300");

  const Odometry::ParameterInfo info = InfoOf("sof.num_desired_tracks");
  EXPECT_EQ(info.value, "300");
  EXPECT_EQ(info.source, "api");
  EXPECT_NE(info.default_value, "300") << "the default must stay distinct from the current value";
}

TEST_F(ParametersTest, AcceptsAnUnambiguousSuffixSoShortNamesWork) {
  odometry_.SetParameter("num_desired_tracks", "250");
  EXPECT_EQ(ValueOf("sof.num_desired_tracks"), "250");
}

TEST_F(ParametersTest, RejectsAnAmbiguousSuffix) {
  // Both vo_pnp and icp have lambda and max_iteration, and both run in this mode.
  EXPECT_THROW(odometry_.SetParameter("lambda", "0.1"), std::invalid_argument);
  EXPECT_THROW(odometry_.SetParameter("max_iteration", "5"), std::invalid_argument);
}

TEST_F(ParametersTest, WhatCountsAsAmbiguousDependsOnTheMode) {
  // `huber` names both vo_pnp.huber and inertial_stereo_pnp.huber, but the inertial solvers are
  // not registered without an IMU, so here it resolves to the only one this tracker reads. This is
  // a consequence of not exposing settings the mode ignores, and is worth pinning: the same
  // abbreviation is ambiguous on an inertial tracker.
  // 0.25 is exactly representable, so it survives the round trip through the reported form
  // unchanged; values like 0.03 come back as 0.0299999993 because the float formatting is exact.
  EXPECT_NO_THROW(odometry_.SetParameter("huber", "0.25"));
  EXPECT_EQ(ValueOf("vo_pnp.huber"), "0.25");
}

TEST_F(ParametersTest, RejectsAnUnknownName) {
  EXPECT_THROW(odometry_.SetParameter("sof.no_such_knob", "1"), std::invalid_argument);
  EXPECT_THROW(odometry_.SetParameter("no_such_group.knob", "1"), std::invalid_argument);
}

TEST_F(ParametersTest, RejectsAValueOfTheWrongTypeAndKeepsThePreviousOne) {
  const std::string before = ValueOf("sof.num_desired_tracks");
  EXPECT_THROW(odometry_.SetParameter("sof.num_desired_tracks", "many"), std::runtime_error);
  EXPECT_EQ(ValueOf("sof.num_desired_tracks"), before);
}

TEST_F(ParametersTest, RejectsAnOutOfRangeValueAndKeepsThePreviousOne) {
  const std::string before = ValueOf("icp.blending_alpha");
  EXPECT_THROW(odometry_.SetParameter("icp.blending_alpha", "2.5"), std::runtime_error);
  EXPECT_EQ(ValueOf("icp.blending_alpha"), before);
  EXPECT_EQ(InfoOf("icp.blending_alpha").source, "default") << "a rejected value must not record a source";
}

TEST_F(ParametersTest, DoesNotExposeParametersTheCurrentModeNeverReads) {
  // This tracker has no IMU, so the state machine and inertial solvers never run. Their settings
  // are not registered at all rather than being settable and then ignored, so they are reported as
  // unknown and never appear in the listing.
  EXPECT_THROW(odometry_.SetParameter("sm.min_num_kf_for_gravity", "30"), std::invalid_argument);
  EXPECT_THROW(odometry_.SetParameter("imu_pnp.max_iteration", "10"), std::invalid_argument);

  for (const Odometry::ParameterInfo& p : odometry_.GetParameters()) {
    EXPECT_NE(p.key.substr(0, 3), "sm.") << p.key << " should not be listed without an IMU";
    EXPECT_NE(p.key.substr(0, 8), "imu_pnp.") << p.key << " should not be listed without an IMU";
  }
}

TEST_F(ParametersTest, SetsEnumParametersByName) {
  odometry_.SetParameter("sof.tracker", "klt");
  EXPECT_EQ(ValueOf("sof.tracker"), "klt");
  EXPECT_THROW(odometry_.SetParameter("sof.tracker", "sideways"), std::runtime_error);
}

TEST_F(ParametersTest, ADefaultConfigLeavesEveryParameterAtItsDefault) {
  // A pristine Config must not make any parameter differ from its own default. Otherwise every
  // report lists that parameter as tuning on every run, and the entry says more about two structs
  // disagreeing on a default than about the run. Guards Odometry::Config and the settings structs
  // it feeds from drifting apart -- multicam_mode did exactly that.
  Odometry defaults(MakeStereoRig(), Odometry::Config{});
  for (const Odometry::ParameterInfo& p : defaults.GetParameters()) {
    EXPECT_EQ(p.value, p.default_value) << p.key << " differs from its default under a default Config";
  }
}

TEST_F(ParametersTest, ReportsTheSettingsTheTrackerWasConstructedWith) {
  Odometry::Config config = MakeConfig();
  config.use_denoising = true;
  Rig rig = MakeStereoRig();
  rig.cameras[0].border_top = 20;
  Odometry configured(rig, config);

  const std::vector<Odometry::ParameterInfo> params = configured.GetParameters();
  const auto find = [&params](std::string_view key) {
    const auto it =
        std::find_if(params.begin(), params.end(), [key](const Odometry::ParameterInfo& p) { return p.key == key; });
    return it != params.end() ? std::string(it->value) : std::string{};
  };
  // Construction-time configuration reaches the parameters a frame actually uses, so a dump of the
  // parameters describes the run rather than describing the type defaults.
  EXPECT_EQ(find("sof.box3_prefilter"), "true");
  EXPECT_EQ(find("sof.border_top"), "20");
}

TEST_F(ParametersTest, ReportedValuesRoundTripThroughSetParameter) {
  const std::vector<Odometry::ParameterInfo> params = odometry_.GetParameters();
  std::vector<std::pair<std::string, std::string>> snapshot;
  for (const Odometry::ParameterInfo& p : params) {
    snapshot.emplace_back(std::string(p.key), std::string(p.value));
  }
  // Every reported value must be accepted back verbatim; otherwise a dumped configuration cannot
  // be replayed. Nothing needs skipping here: a parameter is only listed if this tracker reads it.
  for (const auto& [key, value] : snapshot) {
    EXPECT_NO_THROW(odometry_.SetParameter(key, value)) << key << " = " << value;
  }
}

class ParametersFileTest : public ParametersTest {
protected:
  void TearDown() override {
    if (!path_.empty()) {
      std::remove(path_.c_str());
    }
  }

  void WriteFile(const std::string& contents, const std::string& name) {
    path_ = ::testing::TempDir() + "/" + name;
    std::ofstream file(path_);
    file << contents;
  }

  std::string path_;
};

TEST_F(ParametersFileTest, LoadsParametersAndRecordsTheFileAsTheirSource) {
  WriteFile(
      "# tuning\n"
      "sof.num_desired_tracks: 275\n"
      "vo_pnp.huber = 0.05\n",
      "public_params.txt");

  EXPECT_EQ(odometry_.LoadParameters(path_), 2u);
  EXPECT_EQ(ValueOf("sof.num_desired_tracks"), "275");
  EXPECT_EQ(InfoOf("vo_pnp.huber").source, "file");
}

TEST_F(ParametersFileTest, ReportsTheLineNumberOfABadEntry) {
  WriteFile("sof.num_desired_tracks: 275\nsof.nonsense: 3\n", "public_params_bad.txt");

  try {
    odometry_.LoadParameters(path_);
    FAIL() << "expected a failure";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(":2:"), std::string::npos) << e.what();
  }
}

TEST_F(ParametersFileTest, MissingFileThrows) {
  EXPECT_THROW(odometry_.LoadParameters("/nonexistent/params.txt"), std::runtime_error);
}

}  // namespace

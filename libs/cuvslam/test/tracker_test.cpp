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

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/include_gtest.h"
#include "cuvslam/cuvslam2.h"

namespace {

using cuvslam::Odometry;
using cuvslam::Rig;
using cuvslam::Slam;
using cuvslam::Tracker;

static_assert(std::is_same_v<decltype(std::declval<Tracker&>().GetOdometry()), const Odometry&>);
static_assert(std::is_same_v<decltype(std::declval<Tracker&>().GetSlam()), Slam&>);

constexpr int32_t kWidth = 640;
constexpr int32_t kHeight = 480;
constexpr float kFocal = 500.f;
constexpr float kBaseline = 0.1f;
constexpr int64_t kFramePeriodNs = 33'333'333;

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

/// Checks that the rig is rejected as invalid, with a message naming the field that is at fault.
testing::AssertionResult RejectedWith(const Rig& rig, std::string_view field) {
  try {
    Tracker{rig};
  } catch (const std::invalid_argument& e) {
    if (std::string_view{e.what()}.find(field) != std::string_view::npos) {
      return testing::AssertionSuccess();
    }
    return testing::AssertionFailure() << "rejected with \"" << e.what() << "\", which does not name " << field;
  }
  return testing::AssertionFailure() << "rig was accepted";
}

/// Run the bundler and SLAM in the calling thread so the tests do not depend on background workers.
struct SyncConfig {
  Odometry::Config odometry;
  Slam::Config slam;
};

SyncConfig MakeSyncConfig() {
  SyncConfig cfg;
  cfg.odometry.async_sba = false;
  cfg.slam.sync_mode = true;
  cfg.slam.enable_reading_internals = true;
  return cfg;
}

class TrackerTest : public testing::Test {
protected:
  /// A blank stereo frame with a strictly increasing timestamp.
  ///
  /// These tests cover the Tracker interface - what it constructs, what it delegates, and how it
  /// behaves with SLAM disabled - so the image content is irrelevant. Tracking behaviour itself is
  /// covered by the Python test suite, which drives real datasets.
  Odometry::ImageSet NextFrame() {
    timestamp_ns_ += kFramePeriodNs;
    return {MakeImage(left_, timestamp_ns_, 0), MakeImage(right_, timestamp_ns_, 1)};
  }

  Rig rig{MakeStereoRig()};

private:
  static cuvslam::Image MakeImage(const std::vector<uint8_t>& pixels, int64_t timestamp_ns, uint32_t camera_index) {
    return cuvslam::Image{{pixels.data(), kWidth, kHeight, kWidth, cuvslam::ImageData::Encoding::MONO,
                           cuvslam::ImageData::DataType::UINT8, false /* is_gpu_mem */},
                          timestamp_ns,
                          camera_index};
  }

  std::vector<uint8_t> left_ = std::vector<uint8_t>(static_cast<size_t>(kWidth) * kHeight, 0);
  std::vector<uint8_t> right_ = std::vector<uint8_t>(static_cast<size_t>(kWidth) * kHeight, 0);
  int64_t timestamp_ns_{0};
};

TEST_F(TrackerTest, SlamIsDisabledByDefault) {
  Tracker tracker{rig};

  EXPECT_FALSE(tracker.IsSlamEnabled());
  EXPECT_THROW(tracker.GetSlam(), std::logic_error);

  const auto result = tracker.Track(NextFrame());
  EXPECT_TRUE(result.odometry.world_from_rig.has_value());
  EXPECT_FALSE(result.slam.has_value());
}

TEST_F(TrackerTest, GtAlignModeRequiresManualDispatch) {
  SyncConfig cfg = MakeSyncConfig();
  cfg.slam.gt_align_mode = true;

  EXPECT_THROW(Tracker(rig, cfg.odometry, &cfg.slam), std::invalid_argument);
}

TEST_F(TrackerTest, RejectsNonFiniteCalibration) {
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

  // a non-finite calibration makes every projection NaN, and the range tests on the way in are all
  // `<` or `<=`, which a NaN passes. Each rig must be turned away by the check for its own field,
  // not by some unrelated one further down.
  Rig bad_focal{rig};
  bad_focal.cameras[0].focal = {kNan, kFocal};
  EXPECT_TRUE(RejectedWith(bad_focal, "Focal length"));

  Rig bad_principal{rig};
  bad_principal.cameras[0].principal = {kWidth / 2.f, kNan};
  EXPECT_TRUE(RejectedWith(bad_principal, "Principal point"));

  Rig bad_distortion{rig};
  bad_distortion.cameras[0].distortion.model = cuvslam::Distortion::Model::Fisheye;
  bad_distortion.cameras[0].distortion.parameters = {0.f, kNan, 0.f, 0.f};
  EXPECT_TRUE(RejectedWith(bad_distortion, "Distortion parameters"));

  Rig bad_pose{rig};
  bad_pose.cameras[1].rig_from_camera.translation = {kNan, 0.f, 0.f};
  EXPECT_TRUE(RejectedWith(bad_pose, "rig_from_camera"));

  // Slam is constructible on its own, so it has to reject the rig without Odometry's help
  EXPECT_THROW(Slam(bad_pose, {0}), std::invalid_argument);
}

// The per-camera checks run from SetTrackerRigAndIntrinsics, which Slam calls, but the whole-rig
// ones live in CheckCameras, which only Odometry used to reach.
TEST_F(TrackerTest, StandaloneSlamChecksTheWholeRig) {
  Rig mismatched_resolutions{rig};
  mismatched_resolutions.cameras[1].size = {kWidth / 2, kHeight};

  EXPECT_THROW(Slam(mismatched_resolutions, {0}), std::invalid_argument);
}

TEST_F(TrackerTest, TrackReturnsSlamPoseWhenEnabled) {
  SyncConfig cfg = MakeSyncConfig();
  Tracker tracker{rig, cfg.odometry, &cfg.slam};

  EXPECT_TRUE(tracker.IsSlamEnabled());
  EXPECT_NO_THROW(tracker.GetSlam());

  const auto result = tracker.Track(NextFrame());
  ASSERT_TRUE(result.odometry.world_from_rig.has_value());
  EXPECT_TRUE(result.slam.has_value());

  Slam::Metrics metrics;
  EXPECT_NO_THROW(tracker.GetSlam().GetSlamMetrics(metrics));
}

// A SLAM configuration must turn on the exports SLAM depends on, whatever the odometry config said.
TEST_F(TrackerTest, SlamConfigEnablesRequiredExports) {
  SyncConfig cfg = MakeSyncConfig();
  cfg.odometry.enable_observations_export = false;
  cfg.odometry.enable_landmarks_export = false;

  Tracker tracker{rig, cfg.odometry, &cfg.slam};
  tracker.Track(NextFrame());

  EXPECT_NO_THROW(tracker.GetOdometry().GetLastObservations(0));
  EXPECT_NO_THROW(tracker.GetOdometry().GetLastLandmarks());

  // The caller's config is an input, not scratch space.
  EXPECT_FALSE(cfg.odometry.enable_observations_export);
  EXPECT_FALSE(cfg.odometry.enable_landmarks_export);
}

// Without SLAM the exports stay as configured, which is what makes the test above meaningful.
TEST_F(TrackerTest, ExportsStayDisabledWithoutSlam) {
  SyncConfig cfg = MakeSyncConfig();
  cfg.odometry.enable_observations_export = false;
  cfg.odometry.enable_landmarks_export = false;

  Tracker tracker{rig, cfg.odometry};
  tracker.Track(NextFrame());

  EXPECT_THROW(tracker.GetOdometry().GetLastObservations(0), std::invalid_argument);
  EXPECT_THROW(tracker.GetOdometry().GetLastLandmarks(), std::invalid_argument);
}

TEST_F(TrackerTest, ExposesUnderlyingComponents) {
  SyncConfig cfg = MakeSyncConfig();
  Tracker tracker{rig, cfg.odometry, &cfg.slam};

  EXPECT_FALSE(tracker.GetOdometry().GetPrimaryCameras().empty());

  tracker.Track(NextFrame());

  // Data layer reading is reached through the SLAM accessor rather than mirrored on Tracker.
  Slam& slam = tracker.GetSlam();
  slam.EnableReadingData(Slam::DataLayer::Landmarks, 1000);
  EXPECT_NO_THROW(slam.ReadLandmarks(Slam::DataLayer::Landmarks));
}

// Moving must carry the odometry and SLAM instances over intact. Note that a blank scene has nothing
// to match against, so the tracker legitimately reports no pose from the second frame onwards; this
// test is about the moved-to tracker still owning working components, not about tracking success.
TEST_F(TrackerTest, MoveKeepsComponentsUsable) {
  SyncConfig cfg = MakeSyncConfig();
  Tracker tracker{rig, cfg.odometry, &cfg.slam};
  tracker.Track(NextFrame());

  Tracker moved{std::move(tracker)};

  EXPECT_TRUE(moved.IsSlamEnabled());
  EXPECT_NO_THROW(moved.GetSlam());
  EXPECT_FALSE(moved.GetOdometry().GetPrimaryCameras().empty());
  EXPECT_NO_THROW(moved.Track(NextFrame()));
}

}  // namespace

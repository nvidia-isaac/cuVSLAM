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

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>

#include "camera/camera.h"
#include "common/include_gtest.h"
#include "slam/async_slam/async_slam.h"
#include "sof/image_context.h"

namespace test::slam {
namespace {

using namespace std::chrono_literals;

cuvslam::camera::Rig MakeSingleCameraRig(std::unique_ptr<cuvslam::camera::ICameraModel>& camera) {
  const cuvslam::Vector2T resolution = {640, 480};
  const cuvslam::Vector2T focal = {320, 240};
  const cuvslam::Vector2T principal = {320, 240};
  camera =
      cuvslam::camera::CreateCameraModel(resolution, focal, principal, cuvslam::Distortion::Model::Pinhole, nullptr, 0);

  cuvslam::camera::Rig rig;
  rig.intrinsics[0] = camera.get();
  rig.num_cameras = 1;
  rig.camera_from_rig[0].setIdentity();
  return rig;
}

cuvslam::slam::Images MakeImages(cuvslam::FrameId frame_id, int64_t timestamp_ns) {
  const cuvslam::ImageShape shape{640, 480};
  auto image = std::make_shared<cuvslam::sof::ImageContext>(shape, false, false);

  cuvslam::ImageMeta meta{};
  meta.shape = shape;
  meta.frame_id = frame_id;
  meta.timestamp = timestamp_ns;
  meta.frame_number = static_cast<int>(frame_id);
  meta.camera_index = 0;
  image->set_image_meta(meta);

  cuvslam::slam::Images images(1, nullptr);
  images[0] = image;
  return images;
}

cuvslam::Slam::LocalizationSettings MakeLocalizationSettings() {
  cuvslam::Slam::LocalizationSettings settings{};
  settings.horizontal_search_radius = 1.5f;
  settings.vertical_search_radius = 0.5f;
  settings.horizontal_step = 0.5f;
  settings.vertical_step = 0.25f;
  settings.angular_step_rads = 0.1f;
  return settings;
}

}  // namespace

// Note: propagation of PoseGraphOptimizerOptions, spatial-index settings, and active cameras
// (async_slam_localize.cpp L143-156) requires a real map database and cannot be covered here.

TEST(AsyncSlam, LocalizeInMap_ReproduceMode_BothCallbacksCalledSynchronously) {
  // In reproduce_mode LocalizeInMap calls ProcessInputSynchronously() before returning,
  // so both callbacks must have fired by the time the call returns.
  std::unique_ptr<cuvslam::camera::ICameraModel> camera;
  const cuvslam::camera::Rig rig = MakeSingleCameraRig(camera);
  cuvslam::slam::AsyncSlamOptions options;
  options.use_gpu = false;
  options.reproduce_mode = true;
  options.loop_closure_solver_type = cuvslam::slam::LoopClosureSolverType::kDummy;

  cuvslam::slam::AsyncSlam slam(rig, {0}, options);

  bool start_called = false;
  bool finish_called = false;
  bool finish_succeeded = true;

  const cuvslam::Slam::LocalizeStartCB start_cb = [&] { start_called = true; };
  const cuvslam::Slam::LocalizeFinishCB finish_cb = [&](const cuvslam::Result<cuvslam::Pose>& result) {
    finish_called = true;
    finish_succeeded = result.data.has_value();
  };

  const std::string missing_map_path = "/tmp/cuvslam_missing_localization_map_repro_" +
                                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

  slam.LocalizeInMap(missing_map_path, 1'000, cuvslam::Isometry3T::Identity(), {}, MakeLocalizationSettings(), start_cb,
                     finish_cb);

  EXPECT_TRUE(start_called);
  EXPECT_TRUE(finish_called);
  EXPECT_FALSE(finish_succeeded);  // missing database → finish_cb receives an error
}

TEST(AsyncSlam, LocalizeInMap_AsyncMode_GetSlamPoseRemainsFiniteWhileLocalizationIsInFlight) {
  // Verifies that GetSlamPose() (caller thread) is safe while the worker thread is executing
  // LocalizeInMapCmd::Execute(). With a missing map the worker returns at OpenDatabase() and
  // never reaches the slam_mutex_-protected section (async_slam_localize.cpp L143-150);
  // testing that path requires a real map database which is not available here.
  std::unique_ptr<cuvslam::camera::ICameraModel> camera;
  const cuvslam::camera::Rig rig = MakeSingleCameraRig(camera);
  cuvslam::slam::AsyncSlamOptions options;
  options.use_gpu = false;
  options.reproduce_mode = false;
  options.loop_closure_solver_type = cuvslam::slam::LoopClosureSolverType::kDummy;

  std::mutex mutex;
  std::condition_variable cv;
  bool localization_started = false;
  bool release_localization = false;
  bool finish_called = false;

  const cuvslam::Slam::LocalizeStartCB start_cb = [&] {
    std::unique_lock<std::mutex> lock(mutex);
    localization_started = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release_localization; });
  };
  const cuvslam::Slam::LocalizeFinishCB finish_cb = [&](const cuvslam::Result<cuvslam::Pose>&) {
    std::lock_guard<std::mutex> lock(mutex);
    finish_called = true;
    cv.notify_all();
  };

  const std::string missing_map_path = "/tmp/cuvslam_missing_localization_map_async_" +
                                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

  cuvslam::slam::AsyncSlam slam(rig, {0}, options);
  slam.LocalizeInMap(missing_map_path, 1'000, cuvslam::Isometry3T::Identity(), {}, MakeLocalizationSettings(), start_cb,
                     finish_cb);

  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return localization_started; }));
  }

  // Caller thread reads pose while worker thread is inside LocalizeInMapCmd::Execute().
  EXPECT_TRUE(slam.GetSlamPose().matrix().allFinite());

  {
    std::lock_guard<std::mutex> lock(mutex);
    release_localization = true;
    cv.notify_all();
  }
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 10s, [&] { return finish_called; }));
  }
}

TEST(AsyncSlam, TrackResultCanRunWhileLocalizeInMapIsInFlight) {
  std::unique_ptr<cuvslam::camera::ICameraModel> camera;
  const cuvslam::camera::Rig rig = MakeSingleCameraRig(camera);

  cuvslam::slam::AsyncSlamOptions options;
  options.use_gpu = false;
  options.reproduce_mode = false;
  options.pose_for_frame_required = true;
  options.loop_closure_solver_type = cuvslam::slam::LoopClosureSolverType::kDummy;

  // Synchronisation objects must be declared before slam so they outlive the background
  // SLAM thread: C++ destroys locals in reverse declaration order, so slam (declared last)
  // is destroyed first. Its destructor calls thread_.join(), which blocks until the thread
  // finishes. The thread may still call finish_cb_ / start_cb_ at that point, so mutex and
  // cv must still be alive.
  std::mutex mutex;
  std::condition_variable cv;
  bool localization_started = false;
  bool release_localization = false;
  bool localization_finished = false;
  bool localization_succeeded = true;

  const cuvslam::Slam::LocalizeStartCB start_cb = [&] {
    std::unique_lock<std::mutex> lock(mutex);
    localization_started = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release_localization; });
  };
  const cuvslam::Slam::LocalizeFinishCB finish_cb = [&](const cuvslam::Result<cuvslam::Pose>& result) {
    std::lock_guard<std::mutex> lock(mutex);
    localization_succeeded = result.data.has_value();
    localization_finished = true;
    cv.notify_all();
  };

  const std::string missing_map_path = "/tmp/cuvslam_missing_localization_map_" +
                                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

  cuvslam::slam::AsyncSlam slam(rig, {0}, options);

  slam.LocalizeInMap(missing_map_path, 1'000, cuvslam::Isometry3T::Identity(), {}, MakeLocalizationSettings(), start_cb,
                     finish_cb);

  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return localization_started; }));
  }

  cuvslam::odom::IVisualOdometry::VOFrameStat stat{};
  stat.keyframe = true;
  cuvslam::Isometry3T delta = cuvslam::Isometry3T::Identity();
  delta.translation().x() = 1.f;

  slam.TrackResult(1, 1'000, stat, MakeImages(1, 1'000), delta);
  const cuvslam::Isometry3T slam_pose = slam.GetSlamPose();
  EXPECT_TRUE(slam_pose.matrix().allFinite());

  {
    std::lock_guard<std::mutex> lock(mutex);
    release_localization = true;
    cv.notify_all();
  }
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, 30s, [&] { return localization_finished; }));
    EXPECT_FALSE(localization_succeeded);
  }
}

}  // namespace test::slam

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
#include <cmath>
#include <random>
#include <vector>

#include "camera/camera.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "common/include_gtest.h"
#include "map/map.h"
#include "pipelines/track_online_multi.h"

namespace test {
using namespace cuvslam;

namespace {

constexpr float kFocal = 718.856f;
constexpr float kStep = 2.65f;  // highway speed, m per frame at 10 Hz

camera::PinholeCameraModel MakeKittiCamera() {
  return camera::PinholeCameraModel(Vector2T(1241.f, 376.f), Vector2T(kFocal, kFocal), Vector2T(607.19f, 185.22f));
}

camera::Rig MakeStereoRig(const camera::ICameraModel& camera) {
  camera::Rig rig;
  rig.num_cameras = 2;
  rig.camera_from_rig[0].setIdentity();
  rig.camera_from_rig[1].setIdentity();
  rig.camera_from_rig[1].translate(Vector3T(-0.537f, 0.f, 0.f));
  rig.intrinsics[0] = &camera;
  rig.intrinsics[1] = &camera;
  return rig;
}

struct Frame {
  std::vector<pipelines::Landmark> landmarks;
  std::vector<camera::Observation> keyframe;  // seen from the previous pose, the identity
  sof::MulticamObservations current;          // seen after one step forward
};

// Far points, road edges and optionally a car feature stereo-mismatched 2.57 m ahead, which the step passes
Frame MakeHighwayFrame(const camera::Rig& rig, bool car) {
  Frame frame;
  frame.current.resize(rig.num_cameras);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> unit(0.f, 1.f);
  std::normal_distribution<float> noise(0.f, 0.5f / kFocal);
  const Matrix2T info = Matrix2T::Identity() * (kFocal * kFocal / 9.f);
  Isometry3T rig_from_world = Isometry3T::Identity();
  rig_from_world.translate(Vector3T(0.f, 0.f, -kStep));
  auto observe = [&](TrackId id, const Vector3T& point, const Isometry3T& from_world, bool keyframe) {
    for (int cam = 0; cam < rig.num_cameras; cam++) {
      const Vector3T p = rig.camera_from_rig[cam] * from_world * point;
      if (p.z() < 0.1f || std::abs(p.x() / p.z()) > 0.86f || std::abs(p.y() / p.z()) > 0.26f) {
        continue;
      }
      const camera::Observation obs(static_cast<CameraId>(cam), id,
                                    Vector2T(p.x() / p.z() + noise(rng), p.y() / p.z() + noise(rng)), info);
      if (keyframe) {
        frame.keyframe.push_back(obs);
      } else {
        frame.current[cam].push_back(obs);
      }
    }
  };
  auto add = [&](const Vector3T& point, const Isometry3T& seen_now_from) {
    const TrackId id = frame.landmarks.size() + 1;
    frame.landmarks.push_back({id, point});
    observe(id, point, Isometry3T::Identity(), true);
    observe(id, point, seen_now_from, false);
  };
  for (int i = 0; i < 130; i++) {
    const float z = 100.f + 600.f * unit(rng);
    add(Vector3T(z * (-0.7f + 1.4f * unit(rng)), z * (-0.2f + 0.2f * unit(rng)), z), rig_from_world);
  }
  for (int i = 0; i < 35; i++) {
    const float side = unit(rng) < 0.5f ? -1.f : 1.f;
    add(Vector3T(side * (2.f + 10.f * unit(rng)), 1.6f, 6.f + 24.f * unit(rng)), rig_from_world);
  }
  if (car) {
    add(Vector3T(1.688f, 0.015f, 2.566f), Isometry3T::Identity());
  }
  for (auto& cam_obs : frame.current) {
    std::sort(cam_obs.begin(), cam_obs.end(),
              [](const camera::Observation& a, const camera::Observation& b) { return a.id < b.id; });
  }
  return frame;
}

// IN: world_from_rig - prediction; OUT: solved pose
bool Solve(const Frame& frame, const camera::Rig& rig, Isometry3T& world_from_rig) {
  map::UnifiedMap map(10);
  map.add_keyframe(0, map::State{}, sba_imu::IMUPreintegration{}, frame.keyframe, frame.landmarks);
  sba::Settings sba_settings;
  sba_settings.mode = sba::Disabled;
  pipelines::SolverSfMMulti solver(map, rig, sba_settings);
  Matrix6T static_info_exp;
  return solver.solveNextFrame(1, sof::FrameState::None, frame.current, world_from_rig, static_info_exp, {});
}

}  // namespace

// From the previous pose the solve cannot get past the car feature's depth; from the prediction it is already past
TEST(SolverSfMMulti, PredictionGetsPastWrongDepthLandmark) {
  const camera::PinholeCameraModel camera = MakeKittiCamera();
  const camera::Rig rig = MakeStereoRig(camera);
  Isometry3T world_from_rig = Isometry3T::Identity();
  world_from_rig.translate(Vector3T(0.05f, 0.f, kStep + 0.2f));  // constant-velocity prediction, a little off
  ASSERT_TRUE(Solve(MakeHighwayFrame(rig, true), rig, world_from_rig));
  EXPECT_NEAR(world_from_rig.translation().z(), kStep, 0.05f);
}

// Without a spurious minimum the solve from the previous pose is kept, bit for bit
TEST(SolverSfMMulti, PredictionKeepsUsualSolveOtherwise) {
  const camera::PinholeCameraModel camera = MakeKittiCamera();
  const camera::Rig rig = MakeStereoRig(camera);
  const Frame frame = MakeHighwayFrame(rig, false);
  Isometry3T with_prediction = Isometry3T::Identity();
  with_prediction.translate(Vector3T(0.05f, 0.f, kStep + 0.2f));
  Isometry3T without_prediction = Isometry3T::Identity();  // the previous pose
  ASSERT_TRUE(Solve(frame, rig, with_prediction));
  ASSERT_TRUE(Solve(frame, rig, without_prediction));
  EXPECT_NEAR(without_prediction.translation().z(), kStep, 0.05f);
  EXPECT_EQ(with_prediction.matrix(), without_prediction.matrix());
}

}  // namespace test

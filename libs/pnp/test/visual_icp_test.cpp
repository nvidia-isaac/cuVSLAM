
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

#include <unordered_map>
#include <vector>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

#include "camera/camera.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "math/twist.h"

#include "pnp/visual_icp.h"

namespace test::visual_icp {

using namespace cuvslam;

namespace {

// Synthetic pinhole camera: 640x480, focal=320, principal=(320,240).
const Vector2T kResolution{640.f, 480.f};
const Vector2T kFocal{320.f, 320.f};
const Vector2T kPrincipal{320.f, 240.f};

camera::PinholeCameraModel MakePinhole() { return camera::PinholeCameraModel(kResolution, kFocal, kPrincipal); }

camera::Rig MakeRig(const camera::ICameraModel& cam) {
  camera::Rig rig;
  rig.num_cameras = 1;
  rig.camera_from_rig[0] = Isometry3T::Identity();
  rig.intrinsics[0] = &cam;
  return rig;
}

// A 5x5 grid of landmarks a fixed distance along the camera's z axis. cuVSLAM is z-forward, so a
// negative z puts the whole grid behind the camera.
std::unordered_map<TrackId, Vector3T> MakeGrid(float z) {
  std::unordered_map<TrackId, Vector3T> landmarks;
  TrackId id = 0;
  for (int x = -2; x <= 2; x++) {
    for (int y = -2; y <= 2; y++) {
      landmarks[id++] = Vector3T(0.3f * static_cast<float>(x), 0.3f * static_cast<float>(y), z);
    }
  }
  return landmarks;
}

// Observes every landmark in normalized image coordinates, including the ones behind the camera -
// deciding which of those it can use is the solver's job.
std::vector<camera::Observation> ProjectLandmarks(const std::unordered_map<TrackId, Vector3T>& landmarks,
                                                  const Isometry3T& cam_from_world) {
  std::vector<camera::Observation> observations;
  observations.reserve(landmarks.size());
  for (const auto& [id, lm_world] : landmarks) {
    const Vector3T p_cam = cam_from_world * lm_world;
    const float inv_z = 1.f / p_cam.z();
    observations.emplace_back(CameraId{0}, id, Vector2T(p_cam.x() * inv_z, p_cam.y() * inv_z), Matrix2T::Identity());
  }
  return observations;
}

float PoseError(const Isometry3T& lhs, const Isometry3T& rhs) {
  Vector6T twist;
  math::Log(twist, lhs.inverse() * rhs);
  return twist.norm();
}

}  // namespace

// Every landmark sits behind the camera and there is no depth term to fall back on, so the solve
// has no residual anywhere. It has to report that instead of handing back the untouched pose as a
// success backed by an all-zero information matrix.
TEST(VisualICP, SolveFailsWhenNoTermHasResiduals) {
  auto cam = MakePinhole();
  const auto rig = MakeRig(cam);

  const auto landmarks = MakeGrid(-5.f);
  const auto observations = ProjectLandmarks(landmarks, Isometry3T::Identity());
  ASSERT_EQ(observations.size(), landmarks.size());

  const cuvslam::pnp::VisualICP solver(rig);
  const cuvslam::pnp::ICPSettings settings;

  Isometry3T rig_from_world = Isometry3T::Identity();
  Matrix6T info = Matrix6T::Ones();

  EXPECT_FALSE(solver.solve(rig_from_world, info, observations, landmarks, settings));
  EXPECT_TRUE(rig_from_world.isApprox(Isometry3T::Identity()));
  EXPECT_TRUE(info.isZero()) << "an empty sum of J^T J is zero, not stale:\n" << info;
}

// The pose already fits the observations exactly, so the cost is zero while the residuals are
// perfectly real. A cost of zero must not be read as an absent term: the solve still succeeds, and
// it still reports the information those residuals carry. The level also has to reach that answer
// without scoring a gain ratio against the zero cost, which is why nothing here comes back NaN.
TEST(VisualICP, SolveAcceptsAPerfectFit) {
  auto cam = MakePinhole();
  const auto rig = MakeRig(cam);

  const auto landmarks = MakeGrid(3.f);
  const auto observations = ProjectLandmarks(landmarks, Isometry3T::Identity());

  const cuvslam::pnp::VisualICP solver(rig);
  const cuvslam::pnp::ICPSettings settings;

  Isometry3T rig_from_world = Isometry3T::Identity();
  Matrix6T info = Matrix6T::Zero();

  EXPECT_TRUE(solver.solve(rig_from_world, info, observations, landmarks, settings));
  EXPECT_TRUE(rig_from_world.isApprox(Isometry3T::Identity()));
  EXPECT_FALSE(info.isZero());
  EXPECT_TRUE(info.allFinite()) << info;
}

// A healthy reprojection-only problem still converges - the empty-term shortcut must not fire on
// data the solver can actually use.
TEST(VisualICP, SolveRecoversAPerturbedPose) {
  auto cam = MakePinhole();
  const auto rig = MakeRig(cam);

  const Isometry3T gt_rig_from_world = Isometry3T::Identity();
  const auto landmarks = MakeGrid(3.f);
  const auto observations = ProjectLandmarks(landmarks, gt_rig_from_world);

  Vector6T perturbation;
  perturbation << 0.02f, -0.01f, 0.015f, 0.05f, -0.03f, 0.04f;
  Isometry3T delta;
  math::Exp(delta, perturbation);

  const cuvslam::pnp::VisualICP solver(rig);
  const cuvslam::pnp::ICPSettings settings;

  Isometry3T rig_from_world = gt_rig_from_world * delta;
  const float initial_error = PoseError(gt_rig_from_world, rig_from_world);
  Matrix6T info = Matrix6T::Zero();

  EXPECT_TRUE(solver.solve(rig_from_world, info, observations, landmarks, settings));
  EXPECT_LT(PoseError(gt_rig_from_world, rig_from_world), 0.1f * initial_error);
  EXPECT_FALSE(info.isZero());
}

}  // namespace test::visual_icp

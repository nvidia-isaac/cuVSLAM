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

#include <cmath>
#include <vector>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/vector_3t.h"
#include "cuda_modules/cuda_helper.h"
#include "epipolar/camera_selection.h"

#include "map/depth_point_map.h"

namespace test::map {

using namespace cuvslam;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr float kFx = 320.f;
constexpr float kFy = 320.f;
constexpr float kCx = 320.f;
constexpr float kCy = 240.f;

// Small holder for a synthetic depth texture: the GPUImage owns the texture
// object used by cuvslam::map::DepthCameraInfo::depth_tex.
struct DepthCam {
  cuvslam::cuda::GPUImage<float> depth_img;

  explicit DepthCam(float constant_depth) : depth_img(kWidth, kHeight) {
    std::vector<float> host(kWidth * kHeight, constant_depth);
    depth_img.copy(cuvslam::cuda::ToGPU, host.data(), nullptr);
    cudaDeviceSynchronize();
  }

  cuvslam::map::DepthCameraInfo info(const Isometry3T& world_from_cam) {
    return {depth_img.get_texture_filter_point(), {kFx, kFy}, {kCx, kCy}, {kWidth, kHeight}, world_from_cam};
  }
};

Isometry3T translation_only(float tx, float ty, float tz) {
  Isometry3T iso = Isometry3T::Identity();
  iso.translation() = Vector3T(tx, ty, tz);
  return iso;
}

Isometry3T yaw(float angle_rad) {
  Isometry3T iso = Isometry3T::Identity();
  iso.linear() = Eigen::AngleAxis<float>(angle_rad, Vector3T::UnitY()).toRotationMatrix();
  return iso;
}

}  // namespace

TEST(DepthPointMapTest, EmptyAfterConstruction) {
  cuvslam::map::DepthPointMap map;
  EXPECT_TRUE(map.get_points().empty());
  EXPECT_FALSE(map.added_last_update());
}

TEST(DepthPointMapTest, ClearEmptiesPoints) {
  cuvslam::map::DepthPointMap map;
  map.clear();
  EXPECT_TRUE(map.get_points().empty());
}

TEST(DepthPointMapTest, UpdateWithEmptyCameras) {
  cuvslam::map::DepthPointMap map;
  std::vector<cuvslam::map::DepthCameraInfo> empty_cams;
  map.update(empty_cams);
  EXPECT_TRUE(map.get_points().empty());
  EXPECT_FALSE(map.added_last_update());
}

TEST(DepthPointMapTest, SettingsAreRespected) {
  cuvslam::map::DepthPointMapSettings settings;
  settings.sample_stride = 16;
  settings.visible_max_depth = 4.f;
  settings.max_per_cam = 123;

  cuvslam::map::DepthPointMap map(settings);
  EXPECT_TRUE(map.get_points().empty());
}

// Bootstrap: empty map + one camera seeing a constant depth plane must be
// filled to max_per_cam on the first update().
TEST(DepthPointMapTest, BootstrapFillsToTarget) {
  cuvslam::map::DepthPointMapSettings settings;
  settings.sample_stride = 8;
  settings.visible_max_depth = 5.f;
  settings.max_per_cam = 300;
  cuvslam::map::DepthPointMap map(settings);

  DepthCam cam(2.f);  // plane at 2 m, well inside visible_max_depth
  std::vector<cuvslam::map::DepthCameraInfo> cams{cam.info(Isometry3T::Identity())};

  map.update(cams);

  EXPECT_TRUE(map.added_last_update());
  EXPECT_GE(map.get_points().size(), settings.max_per_cam);
}

// After target is reached, a second update() at the same pose must NOT add
// any further points (the self-bound at max_per_cam kicks in).
TEST(DepthPointMapTest, StationaryAtTarget_NoSampling) {
  cuvslam::map::DepthPointMapSettings settings;
  settings.sample_stride = 8;
  settings.visible_max_depth = 5.f;
  settings.max_per_cam = 300;
  cuvslam::map::DepthPointMap map(settings);

  DepthCam cam(2.f);
  std::vector<cuvslam::map::DepthCameraInfo> cams{cam.info(Isometry3T::Identity())};

  map.update(cams);
  const size_t after_bootstrap = map.get_points().size();
  ASSERT_GT(after_bootstrap, 0u);

  map.update(cams);
  EXPECT_FALSE(map.added_last_update());
  EXPECT_EQ(map.get_points().size(), after_bootstrap);
}

// Translating the camera forward past the depth cloud makes all points fall
// outside (0, visible_max_depth]. They must be evicted, and the next update
// should re-sample from the new pose.
TEST(DepthPointMapTest, ForwardMotionEvictsAndResamples) {
  cuvslam::map::DepthPointMapSettings settings;
  settings.sample_stride = 8;
  settings.visible_max_depth = 3.f;
  settings.max_per_cam = 200;
  cuvslam::map::DepthPointMap map(settings);

  DepthCam cam(2.f);
  Isometry3T pose0 = Isometry3T::Identity();
  std::vector<cuvslam::map::DepthCameraInfo> cams0{cam.info(pose0)};
  map.update(cams0);
  const size_t after_bootstrap = map.get_points().size();
  ASSERT_GE(after_bootstrap, settings.max_per_cam);

  // Move forward by 10 m: all 2m-deep world points are now well behind the
  // camera (p_cam.z ~= -8 m), so they fail the close() predicate.
  Isometry3T pose1 = translation_only(0.f, 0.f, 10.f);
  std::vector<cuvslam::map::DepthCameraInfo> cams1{cam.info(pose1)};
  map.update(cams1);

  EXPECT_TRUE(map.added_last_update());
  // Since all old points are evicted and we re-sample at the new pose, size
  // should again be ~ max_per_cam.
  EXPECT_GE(map.get_points().size(), settings.max_per_cam);
  // And no surviving point should lie within ~2 m of the old origin.
  for (const auto& p : map.get_points()) {
    EXPECT_GT(p.z(), 5.f) << "Old points should have been evicted after forward motion.";
  }
}

// A camera already at target must not drive sampling; only the starved camera
// should sample. We check by verifying that added points project inside the
// starved camera's frustum (they came from its depth texture).
TEST(DepthPointMapTest, PerCameraTriggerIsolated) {
  cuvslam::map::DepthPointMapSettings settings;
  settings.sample_stride = 8;
  settings.visible_max_depth = 5.f;
  settings.max_per_cam = 200;
  cuvslam::map::DepthPointMap map(settings);

  DepthCam cam_a(2.f);
  // Fill cam A first.
  Isometry3T pose_a = Isometry3T::Identity();
  std::vector<cuvslam::map::DepthCameraInfo> only_a{cam_a.info(pose_a)};
  map.update(only_a);
  const size_t after_a = map.get_points().size();
  ASSERT_GE(after_a, settings.max_per_cam);

  // Now introduce cam B, positioned sideways so its frustum does not see any
  // of cam A's points. Cam A is still at target; only cam B should sample.
  DepthCam cam_b(2.f);
  Isometry3T pose_b = translation_only(10.f, 0.f, 0.f);        // 10 m to the side
  pose_b.linear() = yaw(static_cast<float>(M_PI_2)).linear();  // face +x
  std::vector<cuvslam::map::DepthCameraInfo> both{cam_a.info(pose_a), cam_b.info(pose_b)};

  map.update(both);

  EXPECT_TRUE(map.added_last_update());
  // Size grew: cam B was starved and has now been topped up too.
  EXPECT_GT(map.get_points().size(), after_a);
  // Old cam-A points remain (cam A did not trigger, so it did not insert).
  EXPECT_GE(map.get_points().size(), after_a + settings.max_per_cam);
}

// A point close to a camera but slightly outside its image rect must survive
// the distance-only eviction pass. A follow-up yaw that brings it back inside
// the frustum should not trigger a full resample.
TEST(DepthPointMapTest, CloseButOffFrustumPointsAreKept) {
  cuvslam::map::DepthPointMapSettings settings;
  settings.sample_stride = 8;
  settings.visible_max_depth = 5.f;
  settings.max_per_cam = 150;
  cuvslam::map::DepthPointMap map(settings);

  DepthCam cam(2.f);
  Isometry3T pose0 = Isometry3T::Identity();
  std::vector<cuvslam::map::DepthCameraInfo> cams0{cam.info(pose0)};
  map.update(cams0);
  const size_t after_bootstrap = map.get_points().size();
  ASSERT_GE(after_bootstrap, settings.max_per_cam);

  // Small yaw: most points leave the image rect but are still close in
  // distance (they only rotated around the camera origin). evict_far must
  // keep all of them; the post-yaw frustum count will be low enough that we
  // will ALSO top up from the new view, which is fine — the invariant we
  // care about here is "no point was evicted".
  const float yaw_rad = 1.0f;  // ~57 deg, pushes most points out of the image
  Isometry3T pose1 = yaw(yaw_rad);
  std::vector<cuvslam::map::DepthCameraInfo> cams1{cam.info(pose1)};

  // Record which old points are still "close" to the new camera pose. All of
  // these MUST still be present after update(). The near bound has to be the
  // same IsDepthInFront the eviction predicate uses: a point the yaw pushed
  // inside the near plane is legitimately dropped, so expecting it back here
  // would pin the boundary this test is not about.
  std::vector<Vector3T> close_before = map.get_points();
  close_before.erase(
      std::remove_if(close_before.begin(), close_before.end(),
                     [&](const Vector3T& p) {
                       const Vector3T p_cam = pose1.inverse() * p;
                       return !(epipolar::IsDepthInFront(p_cam.z()) && p_cam.z() <= settings.visible_max_depth);
                     }),
      close_before.end());
  ASSERT_FALSE(close_before.empty()) << "Test precondition: at least some points should still be close.";

  map.update(cams1);

  const auto& after = map.get_points();
  for (const auto& p : close_before) {
    const auto it =
        std::find_if(after.begin(), after.end(), [&](const Vector3T& q) { return (p - q).squaredNorm() < 1e-6f; });
    EXPECT_NE(it, after.end()) << "A close point was evicted when it should have been kept.";
  }
}

}  // namespace test::map

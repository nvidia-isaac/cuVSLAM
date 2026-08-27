
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

#include "epipolar/camera_selection.h"
#include "common/include_gtest.h"
#include "epipolar/camera_projection.h"
#include "epipolar/test/generate_points_in_cube_test.h"

namespace test::epipolar {
using namespace cuvslam;

namespace {

///-------------------------------------------------------------------------------------------------
/// @brief Generates points in front of both cameras. A pre-requisite to invoking this
/// method is that camera1 is aligned and centered on the world coordinate system, i.e.
/// camera1=Identity in the frame of reference.
///
/// @param camera2   The second camera.
/// @param numPoints Number of points to generate.
///
/// @return The points in front of both cameras.
///-------------------------------------------------------------------------------------------------
Vector3TVector GeneratePointsInFrontOfBothCameras(const Isometry3T& camera2, size_t numPoints) {
  // Tranform from reference coordinate space to camera2 local coordinate space
  Isometry3T toLocalCamera2Space = camera2.inverse();

  // Generate some points that we know will be in front of both cameras (+Z forward, OpenCV).
  Vector3TVector points3D = GeneratePointsInCube(numPoints, Vector3T(3.0, 3.0, 4.0), Vector3T(8.0, 8.0, 14.0));

  std::for_each(points3D.cbegin(), points3D.cend(), [&toLocalCamera2Space](const Vector3T& point3D) {
    Vector3T point3DInCamera2Space = toLocalCamera2Space * point3D;
    ASSERT_GT(point3DInCamera2Space.z(), cuvslam::epipolar::FrustumProperties::MINIMUM_HITHER);
  });
  return points3D;
}

///-------------------------------------------------------------------------------------------------
/// @brief Generates points behind both cameras. A pre-requisite to invoking this method is
/// that camera1 is aligned and centered on the world coordinate system, i.e. camera1=Identity in
/// the frame of reference.
///
/// @param camera2   The second camera.
/// @param numPoints Number of points to generate.
///
/// @return The points behind both cameras.
///-------------------------------------------------------------------------------------------------
Vector3TVector GeneratePointsBehindBothCameras(const Isometry3T& camera2, size_t numPoints) {
  // Tranform from reference coordinate space to camera2 local coordinate space
  Isometry3T toLocalCamera2Space = camera2.inverse();

  // Points with negative Z in the reference frame are behind camera1 (identity, +Z forward).
  Vector3TVector points3D = GeneratePointsInCube(numPoints, Vector3T(-6.0, -6.0, -15.0), Vector3T(-3.0, -3.0, -10.0));

  // Remove points that are still in front of camera2 (keep only behind w.r.t. cam2).
  auto pred = [&](const Vector3T& t) {
    return (toLocalCamera2Space * t).z() > cuvslam::epipolar::FrustumProperties::MINIMUM_HITHER;
  };
  points3D.erase(std::remove_if(points3D.begin(), points3D.end(), pred), points3D.end());

  return points3D;
}
}  // namespace

///-------------------------------------------------------------------------------------------------
/// @brief The goal of these tests is to ensure correct functionality of the camera selection
/// methods. The basic framework of the tests is to setup 2 cameras, camera1 being the reference
/// and camera2 is related to camera1 by a fixed transform. Tests are run to ensure that
/// IntersectRaysInCameraSpace will provide the proper reconstruction of the 3D point from 2 rays
/// expressed in the reference space. It also ensures that IntersectRaysInCameraSpace returns the
/// correct value for points, in front of both cameras, 1 camera or 0 camera. The
/// CountPointsInFrontOfCameras method is also tested to ensure it returns the proper number of
/// points.
///-------------------------------------------------------------------------------------------------
class CameraSelectionTest : public testing::Test {
protected:
  void PrepareData(const Vector3T& point3D) {
    base_ = Vector3T(1.0f, 2.0f, 3.0f);
    rotation_ =
        Rotation3T(Vector3T(static_cast<float>(-PI / 7.0f), static_cast<float>(PI / 8.0f), 0.0f), AngleUnits::Radian);

    camera2_ = Translation3T(base_) * rotation_;
    expected3DPoint_ = point3D;

    dir1_ = point3D;

    if (dir1_.norm() > epsilon()) {
      dir1_.normalize();
    }

    dir2_ = camera2_.inverse() * expected3DPoint_;

    if (dir1_.norm() > epsilon()) {
      dir2_.normalize();
    }
  }

protected:
  Isometry3T camera2_;
  Rotation3T rotation_;
  Vector3T base_, dir1_, dir2_, expected3DPoint_;
  Vector3TVector pointsInFrontOfCameras_, pointsBehindCameras_;
};

TEST_F(CameraSelectionTest, IntersectRaysInCameraSpace_HappyPath) {
  PrepareData(Vector3T(-5.0f, -5.0f, -5.0f));

  Vector3T actual3DPoint;
  cuvslam::epipolar::IntersectRaysInReferenceSpace(camera2_, dir1_, dir2_, actual3DPoint);

  EXPECT_LE((expected3DPoint_ - actual3DPoint).norm(), 0.0001f);
}

TEST_F(CameraSelectionTest, IntersectRaysInCameraSpace_InFrontOfBothCameras) {
  PrepareData(Vector3T(4.0f, 3.0f, 5.0f));

  Vector3T actual3DPoint;
  EXPECT_TRUE(cuvslam::epipolar::IntersectRaysInReferenceSpace(camera2_, dir1_, dir2_, actual3DPoint));
}

TEST_F(CameraSelectionTest, IntersectRaysInCameraSpace_BehindOneCamera) {
  PrepareData(Vector3T(2.0f, 2.0f, 1.0f));

  Vector3T actual3DPoint;
  EXPECT_FALSE(cuvslam::epipolar::IntersectRaysInReferenceSpace(camera2_, dir1_, dir2_, actual3DPoint));
}

TEST_F(CameraSelectionTest, IntersectRaysInCameraSpace_BehindBothCameras) {
  PrepareData(Vector3T(2.0f, 2.0f, -10.0f));

  Vector3T actual3DPoint;
  EXPECT_FALSE(cuvslam::epipolar::IntersectRaysInReferenceSpace(camera2_, dir1_, dir2_, actual3DPoint));
}

TEST_F(CameraSelectionTest, CountPointsInFrontOfCameras_AllPointInFrontOfCameras) {
  PrepareData(Vector3T(0.0f, 0.0f, 0.0f));

  const size_t expectedCount = 67;
  pointsInFrontOfCameras_ = GeneratePointsInFrontOfBothCameras(camera2_, expectedCount);

  Vector2TPairVector testData;
  cuvslam::epipolar::GeneratePairOf2DPointsFrom3DPoints(pointsInFrontOfCameras_, Isometry3T::Identity(), camera2_,
                                                        &testData);

  size_t actualCount = cuvslam::epipolar::CountPointsInFrontOfCameras(testData.cbegin(), testData.cend(), camera2_);

  EXPECT_EQ(expectedCount, actualCount);
}

TEST_F(CameraSelectionTest, CountPointsInFrontOfCameras_MixedPointsInFrontAndBehindCameras) {
  PrepareData(Vector3T(0.0f, 0.0f, 0.0f));

  // Generate the points in front of the cameras
  const size_t expectedCount = 56;
  pointsInFrontOfCameras_ = GeneratePointsInFrontOfBothCameras(camera2_, expectedCount);

  // Generate the points behind the cameras
  pointsBehindCameras_ = GeneratePointsBehindBothCameras(camera2_, 37);

  // Concatenate the two vectors.
  Vector3TVector points3D = pointsInFrontOfCameras_;
  points3D.insert(points3D.end(), pointsBehindCameras_.begin(), pointsBehindCameras_.end());

  Vector2TPairVector testData;
  cuvslam::epipolar::GeneratePairOf2DPointsFrom3DPoints(points3D, Isometry3T::Identity(), camera2_, &testData);

  size_t actualCount = cuvslam::epipolar::CountPointsInFrontOfCameras(testData.cbegin(), testData.cend(), camera2_);

  // Test that CountPointsInFrontOfCameras counted only the points in front of both cameras.
  EXPECT_EQ(expectedCount, actualCount);
}

TEST(NearPlaneGuard, IsDepthInFront_TreatsTheBoundaryAsTooClose) {
  const float hither = cuvslam::epipolar::FrustumProperties::MINIMUM_HITHER;

  EXPECT_TRUE(cuvslam::epipolar::IsDepthInFront(hither * 2.0f));
  EXPECT_TRUE(cuvslam::epipolar::IsDepthInFront(hither * 1.01f));

  // A point sitting exactly on the near plane counts as too close, not as visible.
  EXPECT_FALSE(cuvslam::epipolar::IsDepthInFront(hither));
  EXPECT_FALSE(cuvslam::epipolar::IsDepthInFront(hither * 0.99f));
  EXPECT_FALSE(cuvslam::epipolar::IsDepthInFront(0.0f));
  EXPECT_FALSE(cuvslam::epipolar::IsDepthInFront(-hither));
}

TEST(NearPlaneGuard, IsPointInFrontInLocalSpace_SharesTheSameBoundary) {
  const float hither = cuvslam::epipolar::FrustumProperties::MINIMUM_HITHER;
  const Isometry3T identity = Isometry3T::Identity();

  EXPECT_TRUE(cuvslam::epipolar::IsPointInFrontInLocalSpace(Vector3T(0.0f, 0.0f, hither * 2.0f), identity));
  EXPECT_FALSE(cuvslam::epipolar::IsPointInFrontInLocalSpace(Vector3T(0.0f, 0.0f, hither), identity));
}

}  // namespace test::epipolar

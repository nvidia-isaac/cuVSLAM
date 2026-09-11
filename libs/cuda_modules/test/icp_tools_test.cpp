
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

#include <limits>
#include <vector>

#include "common/include_gtest.h"
#include "cuda_modules/icp_tools.h"

namespace test {
using namespace cuvslam;

namespace {

constexpr size_t kImageSize = 64;
constexpr float kFocal = 100.f;
constexpr float kPrincipal = 32.f;
constexpr float kHuber = 5e-2f;

// A four-plane level filled with the same constant depth. The photometric and point-to-point
// kernels only read curr_depth, but all four textures have to be live.
struct ConstantDepthLevel {
  explicit ConstantDepthLevel(float depth)
      : depth_image(ImageMatrixT::Constant(kImageSize, kImageSize, depth)),
        zeros(ImageMatrixT::Zero(kImageSize, kImageSize)),
        depth(depth_image),
        image(zeros),
        grad_x(zeros),
        grad_y(zeros) {}

  cuda::Level level() const { return {depth, image, grad_x, grad_y}; }

  ImageMatrixT depth_image;
  ImageMatrixT zeros;
  cuda::GPUImageT depth;
  cuda::GPUImageT image;
  cuda::GPUImageT grad_x;
  cuda::GPUImageT grad_y;
};

// Tracks that all project onto the principal point, with the landmark one metre in front of the
// camera. Whether a track survives is then decided by the level's depth alone.
std::vector<cuda::GPUICPTools::ObsLmPair> TracksAtPrincipalPoint(size_t count) {
  cuda::GPUICPTools::ObsLmPair track;
  track.obs_xy = {0.f, 0.f};
  track.lm_xyz = {0.f, 0.f, 1.f};
  return std::vector<cuda::GPUICPTools::ObsLmPair>(count, track);
}

}  // namespace

// A depth of zero is below the kernels' D_MIN, so no track survives either kernel. The outputs are
// unspecified on this path by contract - the caller must not read them - so only the report matters.
TEST(Cuda, IcpToolsReportsNoResidualsWhenEveryTrackIsRejected) {
  const float nan = std::numeric_limits<float>::quiet_NaN();

  ConstantDepthLevel level(0.f);
  const auto tracks = TracksAtPrincipalPoint(16);

  cuda::GPUICPTools icp_tools;
  float cost = nan;
  Vector6T rhs = Vector6T::Constant(nan);
  Matrix6T hessian = Matrix6T::Constant(nan);

  const bool have_residuals = icp_tools.match_and_reduce(cost, rhs, hessian, {kFocal, kFocal}, {kPrincipal, kPrincipal},
                                                         level.level(), Isometry3T::Identity(), kHuber, tracks);

  EXPECT_FALSE(have_residuals);
}

// The same requirement for the mixed case: a depth of two metres against a landmark one metre out is
// too far off for the photometric kernel to accept, but well inside the point-to-point kernel's
// range. The point-to-point block accumulates with +=, so it must not read back the caller's
// storage.
TEST(Cuda, IcpToolsDoesNotAccumulateOntoUninitialisedOutputs) {
  const float nan = std::numeric_limits<float>::quiet_NaN();

  ConstantDepthLevel level(2.f);
  const auto tracks = TracksAtPrincipalPoint(16);

  cuda::GPUICPTools icp_tools;
  float cost = nan;
  Vector6T rhs = Vector6T::Constant(nan);
  Matrix6T hessian = Matrix6T::Constant(nan);

  const bool have_residuals = icp_tools.match_and_reduce(cost, rhs, hessian, {kFocal, kFocal}, {kPrincipal, kPrincipal},
                                                         level.level(), Isometry3T::Identity(), kHuber, tracks);

  EXPECT_TRUE(have_residuals);
  EXPECT_TRUE(std::isfinite(cost)) << cost;
  EXPECT_TRUE(rhs.allFinite()) << rhs.transpose();
  EXPECT_TRUE(hessian.allFinite()) << hessian;

  // With the photometric term empty, point-to-point carries the whole ICP weight rather than its
  // 1 - alpha_v share, so the cost is the mean Huber loss of the residual itself. Each track sees a
  // one-metre gap between landmark and measured depth, giving a squared residual of 1, which is
  // above the Huber knee: delta * sqrt(1) - 0.5 * delta^2 with delta = kHuber.
  const float expected = kHuber - 0.5f * kHuber * kHuber;
  EXPECT_NEAR(cost, expected, 1e-6f);
}

// An empty track list takes the early return, which reports the same way.
TEST(Cuda, IcpToolsReportsNoResidualsWithoutTracks) {
  const float nan = std::numeric_limits<float>::quiet_NaN();

  ConstantDepthLevel level(1.f);

  cuda::GPUICPTools icp_tools;
  float cost = nan;
  Vector6T rhs = Vector6T::Constant(nan);
  Matrix6T hessian = Matrix6T::Constant(nan);

  const bool have_residuals = icp_tools.match_and_reduce(cost, rhs, hessian, {kFocal, kFocal}, {kPrincipal, kPrincipal},
                                                         level.level(), Isometry3T::Identity(), kHuber, {});

  EXPECT_FALSE(have_residuals);
}

}  // namespace test

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

#include <cunls/common/cuda_stream.h>
#include <cunls/common/device_vector.h>
#include <cunls/factor/prior/se3_prior_factor_batch.h>
#include <cunls/minimizer/levenberg_marquardt_minimizer.h>
#include <cunls/minimizer/problem.h>
#include <cunls/robustifier/scaled_loss_function_batch.h>
#include <cunls/robustifier/trivial_loss_function_batch.h>
#include <cunls/state/se3_state_batch.h>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "math/point_to_plane_factor.h"
#include "math/twist.h"

namespace test::pnp {

using namespace cuvslam;
// cuNLS headers pull in libcu++'s global ::cuda namespace.
namespace cuda = cuvslam::cuda;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kStride = 64;
constexpr float kFx = 320.f;
constexpr float kFy = 320.f;
constexpr float kCx = 320.f;
constexpr float kCy = 240.f;
constexpr float kDepth = 2.f;

void ToRowMajor(const Isometry3T& iso, float* out) {
  const auto& m = iso.matrix();
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      out[r * 4 + c] = m(r, c);
    }
  }
}

// Runs the point-to-plane kernel over a constant-depth image at the state world_from_rig.
class PointToPlaneEvaluator {
public:
  PointToPlaneEvaluator(const Isometry3T& cam_from_rig, const cuda::GPUPlane& plane) : depth_img_(kWidth, kHeight) {
    std::vector<float> depth_host(kWidth * kHeight, kDepth);
    depth_img_.copy(cuda::ToGPU, depth_host.data(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    float cam_from_rig_host[16];
    ToRowMajor(cam_from_rig, cam_from_rig_host);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cam_from_rig_), 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cam_from_rig_, cam_from_rig_host, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_plane_), sizeof(cuda::GPUPlane)));
    CUDA_CHECK(cudaMemcpy(d_plane_, &plane, sizeof(cuda::GPUPlane), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_state_), 16 * sizeof(float)));
    std::vector<float*> state_ptrs(kNumFactors, d_state_);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_state_ptrs_), kNumFactors * sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_state_ptrs_, state_ptrs.data(), kNumFactors * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_res_), kNumFactors * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_jac_), kNumFactors * 6 * sizeof(float)));
  }

  ~PointToPlaneEvaluator() {
    cudaFree(d_cam_from_rig_);
    cudaFree(d_plane_);
    cudaFree(d_state_);
    cudaFree(d_state_ptrs_);
    cudaFree(d_res_);
    cudaFree(d_jac_);
  }

  void evaluate(const Isometry3T& world_from_rig, std::vector<float>& res, std::vector<float>* jac) {
    float state_host[16];
    ToRowMajor(world_from_rig, state_host);
    CUDA_CHECK(cudaMemcpy(d_state_, state_host, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cuda::point_to_plane_evaluate(d_plane_, 1, kStride, depth_img_.get_texture_filter_point(),
                                             float2{kFx, kFy}, float2{kCx, kCy}, int2{kWidth, kHeight}, d_state_ptrs_,
                                             d_cam_from_rig_, d_res_, jac ? d_jac_ : nullptr, kNumFactors, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    res.resize(kNumFactors);
    CUDA_CHECK(cudaMemcpy(res.data(), d_res_, kNumFactors * sizeof(float), cudaMemcpyDeviceToHost));
    if (jac) {
      jac->resize(kNumFactors * 6);
      CUDA_CHECK(cudaMemcpy(jac->data(), d_jac_, kNumFactors * 6 * sizeof(float), cudaMemcpyDeviceToHost));
    }
  }

  cudaTextureObject_t depth_texture() const { return depth_img_.get_texture_filter_point(); }
  const cuda::GPUPlane* plane() const { return d_plane_; }
  const float* cam_from_rig() const { return d_cam_from_rig_; }

  static constexpr size_t kNumFactors =
      static_cast<size_t>((kWidth + kStride - 1) / kStride) * ((kHeight + kStride - 1) / kStride);

private:
  cuda::GPUImage<float> depth_img_;
  float* d_cam_from_rig_ = nullptr;
  cuda::GPUPlane* d_plane_ = nullptr;
  float* d_state_ = nullptr;
  float** d_state_ptrs_ = nullptr;
  float* d_res_ = nullptr;
  float* d_jac_ = nullptr;
};

// Builds the camera-frame depth plane z = kDepth in the world frame, offset by `offset` meters.
cuda::GPUPlane MakeDepthPlane(const Isometry3T& world_from_cam, float offset) {
  const Vector3T normal_world = world_from_cam.linear() * Vector3T::UnitZ();
  const Vector3T point_world = world_from_cam * Vector3T(0.f, 0.f, kDepth);
  cuda::GPUPlane plane;
  plane.normal = {normal_world.x(), normal_world.y(), normal_world.z()};
  plane.d = -normal_world.dot(point_world) + offset;
  return plane;
}

}  // namespace

// Verify the 1x6 Jacobian against central differences along the cuNLS SE(3) update
// W' = W * Exp(xi) of the state W = world_from_rig, with non-identity poses and extrinsics.
// The pixel grid and the sampled depth do not depend on the state, so the residual is smooth in xi.
TEST(PointToPlaneTest, JacobianMatchesFiniteDifferences) {
  Vector6T rig_from_world_twist, cam_from_rig_twist;
  rig_from_world_twist << 0.2f, -0.3f, 0.1f, 0.5f, -0.4f, 1.f;
  cam_from_rig_twist << -0.04f, 0.06f, 0.03f, 0.05f, 0.02f, -0.03f;
  Isometry3T rig_from_world, cam_from_rig;
  math::Exp(rig_from_world, rig_from_world_twist);
  math::Exp(cam_from_rig, cam_from_rig_twist);
  const Isometry3T world_from_rig = rig_from_world.inverse();
  const Isometry3T world_from_cam = world_from_rig * cam_from_rig.inverse();

  // A 1 cm offset (inside the kernel's 3 cm association gate) makes the residuals nonzero.
  PointToPlaneEvaluator evaluator(cam_from_rig, MakeDepthPlane(world_from_cam, 0.01f));
  std::vector<float> residuals, jacobians;
  evaluator.evaluate(world_from_rig, residuals, &jacobians);

  constexpr float kEps = 1e-3f;
  std::vector<std::vector<float>> numeric(6);
  for (int k = 0; k < 6; k++) {
    Vector6T xi = Vector6T::Zero();
    Isometry3T step;
    std::vector<float> res_plus, res_minus;

    xi[k] = kEps;
    math::Exp(step, xi);
    evaluator.evaluate(world_from_rig * step, res_plus, nullptr);

    xi[k] = -kEps;
    math::Exp(step, xi);
    evaluator.evaluate(world_from_rig * step, res_minus, nullptr);

    numeric[k].resize(PointToPlaneEvaluator::kNumFactors);
    for (size_t f = 0; f < PointToPlaneEvaluator::kNumFactors; f++) {
      numeric[k][f] = (res_plus[f] - res_minus[f]) / (2.f * kEps);
    }
  }

  // Grid cells whose center falls outside the image (the last row here) are zeroed by the kernel.
  size_t checked = 0;
  for (size_t f = 0; f < PointToPlaneEvaluator::kNumFactors; f++) {
    if (residuals[f] == 0.f) {
      continue;
    }
    EXPECT_NEAR(residuals[f], 0.01f, 1e-4f) << "Factor " << f << " should be associated with the plane";
    for (int j = 0; j < 6; j++) {
      EXPECT_NEAR(jacobians[f * 6 + j], numeric[j][f], 1e-3f) << "Jacobian mismatch at factor=" << f << " col=" << j;
    }
    ++checked;
  }
  EXPECT_EQ(checked, static_cast<size_t>((kWidth / kStride) * (kHeight / kStride)));
}

// Run PointToPlaneCostFunctionBatch inside the cuNLS Levenberg-Marquardt minimizer, as the
// multisensor pose estimator does. A single plane constrains only 3 of the 6 pose DOF, so a weak
// SE(3) prior at the initial pose keeps the normal equations nonsingular.
TEST(PointToPlaneTest, MinimizerMovesPoseOntoPlane) {
  Vector6T rig_from_world_twist, cam_from_rig_twist;
  rig_from_world_twist << 0.2f, -0.3f, 0.1f, 0.5f, -0.4f, 1.f;
  cam_from_rig_twist << -0.04f, 0.06f, 0.03f, 0.05f, 0.02f, -0.03f;
  Isometry3T rig_from_world, cam_from_rig;
  math::Exp(rig_from_world, rig_from_world_twist);
  math::Exp(cam_from_rig, cam_from_rig_twist);
  const Isometry3T world_from_rig = rig_from_world.inverse();

  PointToPlaneEvaluator evaluator(cam_from_rig, MakeDepthPlane(world_from_rig * cam_from_rig.inverse(), 0.01f));

  std::vector<float> world_from_rig_host(16);
  ToRowMajor(world_from_rig, world_from_rig_host.data());
  cunls::DeviceVector<float> d_state(world_from_rig_host);
  cunls::DeviceVector<float> d_prior(world_from_rig_host);

  math::PointToPlaneCostFunctionBatch plane_factor(evaluator.plane(), 1, kStride, evaluator.depth_texture(),
                                                   float2{kFx, kFy}, float2{kCx, kCy}, int2{kWidth, kHeight},
                                                   evaluator.cam_from_rig());
  cunls::SE3PriorFactorBatch prior_factor(reinterpret_cast<const cunls::SE3Transform*>(d_prior.data()), 1);
  prior_factor.SetNumActiveFactors(1);
  cunls::ScaledLossFunctionBatch<cunls::TrivialLossFunctionBatch> prior_loss(1e-4f);

  cunls::SE3StateBatch states(d_state.data(), 1);
  states.SetNumActiveStates(1);

  cunls::Problem problem;
  problem.AddFactorBatch(&plane_factor, std::vector<float*>(plane_factor.NumActiveFactors(), states.StateDevicePtr(0)));
  problem.AddFactorBatch(&prior_factor, &prior_loss, {states.StateDevicePtr(0)});
  problem.AddStateBatch(&states);

  cunls::LevenbergMarquardtMinimizerOptions options;
  options.base_options.max_num_iterations = 20;
  options.base_options.sparse_linear_solver_type = cunls::SparseLinearSolverType::DenseQR;
  cunls::LevenbergMarquardtMinimizer minimizer(options);
  cunls::CudaStream stream;
  minimizer.Minimize(stream.GetStream(), problem);
  ASSERT_EQ(cudaStreamSynchronize(stream.GetStream()), cudaSuccess);

  d_state.CopyToHost(world_from_rig_host.data(), 16);
  Isometry3T solved = Isometry3T::Identity();
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 4; c++) {
      solved.matrix()(r, c) = world_from_rig_host[r * 4 + c];
    }
  }

  std::vector<float> residuals;
  evaluator.evaluate(solved, residuals, nullptr);
  size_t checked = 0;
  for (size_t f = 0; f < PointToPlaneEvaluator::kNumFactors; f++) {
    // Grid cells outside the image report 0; every other one started at the 1 cm offset.
    if (residuals[f] == 0.f) {
      continue;
    }
    EXPECT_LT(std::fabs(residuals[f]), 1e-3f) << "Factor " << f << " not moved onto the plane";
    ++checked;
  }
  EXPECT_EQ(checked, static_cast<size_t>((kWidth / kStride) * (kHeight / kStride)));
}

}  // namespace test::pnp

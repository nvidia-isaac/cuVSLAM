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
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace test::pnp {

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr float kFx = 320.f;
constexpr float kFy = 320.f;
constexpr float kCx = 320.f;
constexpr float kCy = 240.f;

void MakeIdentity(float* M) {
  for (int i = 0; i < 16; i++) M[i] = 0.f;
  M[0] = M[5] = M[10] = M[15] = 1.f;
}

void MakeSE3(float rx, float ry, float rz, float tx, float ty, float tz, float* M) {
  float angle = std::sqrt(rx * rx + ry * ry + rz * rz);
  float c, s, one_minus_c;
  float ux = 0, uy = 0, uz = 0;
  if (angle > 1e-8f) {
    ux = rx / angle;
    uy = ry / angle;
    uz = rz / angle;
    c = std::cos(angle);
    s = std::sin(angle);
    one_minus_c = 1.f - c;
  } else {
    c = 1.f;
    s = 0.f;
    one_minus_c = 0.f;
  }
  M[0] = c + ux * ux * one_minus_c;
  M[1] = ux * uy * one_minus_c - uz * s;
  M[2] = ux * uz * one_minus_c + uy * s;
  M[3] = tx;
  M[4] = uy * ux * one_minus_c + uz * s;
  M[5] = c + uy * uy * one_minus_c;
  M[6] = uy * uz * one_minus_c - ux * s;
  M[7] = ty;
  M[8] = uz * ux * one_minus_c - uy * s;
  M[9] = uz * uy * one_minus_c + ux * s;
  M[10] = c + uz * uz * one_minus_c;
  M[11] = tz;
  M[12] = 0.f;
  M[13] = 0.f;
  M[14] = 0.f;
  M[15] = 1.f;
}

float3 TransformPointCPU(const float* M, float3 p) {
  return {M[0] * p.x + M[1] * p.y + M[2] * p.z + M[3], M[4] * p.x + M[5] * p.y + M[6] * p.z + M[7],
          M[8] * p.x + M[9] * p.y + M[10] * p.z + M[11]};
}

void InvertSE3CPU(const float* A, float* C) {
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      C[r * 4 + c] = A[c * 4 + r];
    }
    C[r * 4 + 3] = -(A[0 * 4 + r] * A[3] + A[1 * 4 + r] * A[7] + A[2 * 4 + r] * A[11]);
  }
  C[12] = C[13] = C[14] = 0.f;
  C[15] = 1.f;
}

void ComposeSE3CPU(const float* A, const float* B, float* C) {
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      C[r * 4 + c] = A[r * 4 + 0] * B[0 * 4 + c] + A[r * 4 + 1] * B[1 * 4 + c] + A[r * 4 + 2] * B[2 * 4 + c] +
                     A[r * 4 + 3] * B[3 * 4 + c];
    }
  }
}

std::vector<float> RenderDepthFromPoints(const float* cam_from_world, const std::vector<float3>& world_points) {
  std::vector<float> depth(kWidth * kHeight, 0.f);
  for (const auto& pw : world_points) {
    float3 pc = TransformPointCPU(cam_from_world, pw);
    if (pc.z < 0.01f) continue;
    float u = kFx * pc.x / pc.z + kCx;
    float v = kFy * pc.y / pc.z + kCy;
    int ui = static_cast<int>(std::round(u));
    int vi = static_cast<int>(std::round(v));
    for (int dy = -3; dy <= 3; dy++) {
      for (int dx = -3; dx <= 3; dx++) {
        int px = ui + dx, py = vi + dy;
        if (px >= 0 && px < kWidth && py >= 0 && py < kHeight) {
          depth[py * kWidth + px] = pc.z;
        }
      }
    }
  }
  return depth;
}

struct TestResources {
  cuvslam::cuda::GPUImage<float> depth_img;
  float3* d_landmarks = nullptr;
  float* d_cam_from_rig = nullptr;
  size_t N = 0;

  TestResources(const std::vector<float>& depth_host, const std::vector<float3>& world_points,
                const float* cam_from_rig_host)
      : depth_img(kWidth, kHeight), N(world_points.size()) {
    depth_img.copy(cuvslam::cuda::ToGPU, depth_host.data(), nullptr);
    cudaDeviceSynchronize();
    cudaMalloc(reinterpret_cast<void**>(&d_landmarks), N * sizeof(float3));
    cudaMemcpy(d_landmarks, world_points.data(), N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&d_cam_from_rig), 16 * sizeof(float));
    cudaMemcpy(d_cam_from_rig, cam_from_rig_host, 16 * sizeof(float), cudaMemcpyHostToDevice);
  }

  ~TestResources() {
    if (d_landmarks) cudaFree(d_landmarks);
    if (d_cam_from_rig) cudaFree(d_cam_from_rig);
  }

  // Run the projective ICP kernel at the state world_from_rig. residuals is 1 float per factor,
  // jacobians (if requested) is 6 floats per factor. The relative-depth rejection threshold is
  // hard-coded in the kernel translation unit (kMaxRelDepth in point_to_point_icp_kernel.cu).
  void evaluate(const float* state_host, std::vector<float>& res, std::vector<float>* jac) {
    float* d_state;
    cudaMalloc(reinterpret_cast<void**>(&d_state), 16 * sizeof(float));
    cudaMemcpy(d_state, state_host, 16 * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float*> h_ptrs(N, d_state);
    float** d_ptrs;
    cudaMalloc(reinterpret_cast<void**>(&d_ptrs), N * sizeof(float*));
    cudaMemcpy(d_ptrs, h_ptrs.data(), N * sizeof(float*), cudaMemcpyHostToDevice);

    float* d_res;
    cudaMalloc(reinterpret_cast<void**>(&d_res), N * sizeof(float));
    float* d_jac = nullptr;
    if (jac) cudaMalloc(reinterpret_cast<void**>(&d_jac), N * 6 * sizeof(float));

    cuvslam::cuda::point_to_point_icp_evaluate(d_landmarks,
                                               const_cast<float const* const*>(reinterpret_cast<float**>(d_ptrs)),
                                               d_cam_from_rig, depth_img.get_texture_filter_point(), float2{kFx, kFy},
                                               float2{kCx, kCy}, int2{kWidth, kHeight}, d_res, d_jac, N, nullptr);
    cudaDeviceSynchronize();

    res.resize(N);
    cudaMemcpy(res.data(), d_res, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (jac) {
      jac->resize(N * 6);
      cudaMemcpy(jac->data(), d_jac, N * 6 * sizeof(float), cudaMemcpyDeviceToHost);
      cudaFree(d_jac);
    }
    cudaFree(d_state);
    cudaFree(d_ptrs);
    cudaFree(d_res);
  }
};

}  // namespace

// At the ground-truth pose, residuals should be near zero.
TEST(PointToPointICPTest, ResidualZeroAtGT) {
  std::vector<float3> world_points;
  for (float x = -0.5f; x <= 0.5f; x += 0.25f) {
    for (float y = -0.3f; y <= 0.3f; y += 0.2f) {
      world_points.push_back({x, y, 2.f});
    }
  }

  float cam_from_world[16], cam_from_rig[16];
  MakeIdentity(cam_from_world);
  MakeIdentity(cam_from_rig);

  auto depth_host = RenderDepthFromPoints(cam_from_world, world_points);
  TestResources tr(depth_host, world_points, cam_from_rig);

  // Probe the rejection gate by also requesting Jacobians: a non-zero Jacobian indicates
  // the factor was kept (the gate writes both residual and Jacobian to zero on rejection).
  // At GT the residual is exactly 0 for every kept factor, so a residual-magnitude check
  // alone cannot distinguish "perfect match" from "rejected" — we need the Jacobian.
  float world_from_rig[16];
  InvertSE3CPU(cam_from_world, world_from_rig);
  std::vector<float> residuals, jacobians;
  tr.evaluate(world_from_rig, residuals, &jacobians);

  int num_active = 0;
  for (size_t i = 0; i < tr.N; i++) {
    bool jac_nonzero = false;
    for (int j = 0; j < 6; j++) {
      if (std::fabs(jacobians[i * 6 + j]) > 1e-12f) {
        jac_nonzero = true;
        break;
      }
    }
    if (jac_nonzero) {
      num_active++;
      EXPECT_LT(std::fabs(residuals[i]), 0.01f) << "Residual too large at GT for point " << i;
    }
  }
  EXPECT_GT(num_active, 0) << "No active factors at GT pose";
}

// Verify the 1x6 Jacobian against central differences along the cuNLS SE(3) update
// W' = W * Exp(xi) of the state W = world_from_rig, with a non-identity cam_from_rig.
// The depth texture is piecewise constant around each projected landmark, so the sampled
// depth does not change under the small probe steps and the differences isolate d(p_cam.z)/dxi.
TEST(PointToPointICPTest, JacobianMatchesFiniteDifferences) {
  std::vector<float3> world_points = {{0.1f, -0.2f, 2.f}, {0.5f, 0.3f, 3.f}, {-0.3f, 0.1f, 1.5f}};

  float rig_from_world[16], cam_from_rig[16], cam_from_world[16], world_from_rig[16];
  MakeSE3(0.05f, -0.03f, 0.02f, 0.1f, -0.05f, 0.02f, rig_from_world);
  MakeSE3(-0.04f, 0.06f, 0.03f, 0.05f, 0.02f, -0.03f, cam_from_rig);
  ComposeSE3CPU(cam_from_rig, rig_from_world, cam_from_world);
  InvertSE3CPU(rig_from_world, world_from_rig);

  auto depth_host = RenderDepthFromPoints(cam_from_world, world_points);
  TestResources tr(depth_host, world_points, cam_from_rig);

  std::vector<float> residuals, jacobians;
  tr.evaluate(world_from_rig, residuals, &jacobians);

  // Tangent ordering is (omega, t); a single nonzero component makes MakeSE3 an exact Exp.
  constexpr float kEps = 1e-3f;
  std::vector<std::vector<float>> numeric(6);
  for (int k = 0; k < 6; k++) {
    float xi[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    std::vector<float> res_plus, res_minus;
    float step[16], perturbed[16];

    xi[k] = kEps;
    MakeSE3(xi[0], xi[1], xi[2], xi[3], xi[4], xi[5], step);
    ComposeSE3CPU(world_from_rig, step, perturbed);
    tr.evaluate(perturbed, res_plus, nullptr);

    xi[k] = -kEps;
    MakeSE3(xi[0], xi[1], xi[2], xi[3], xi[4], xi[5], step);
    ComposeSE3CPU(world_from_rig, step, perturbed);
    tr.evaluate(perturbed, res_minus, nullptr);

    numeric[k].resize(tr.N);
    for (size_t pt = 0; pt < tr.N; pt++) {
      numeric[k][pt] = (res_plus[pt] - res_minus[pt]) / (2.f * kEps);
    }
  }

  int checked = 0;
  for (size_t pt = 0; pt < tr.N; pt++) {
    // Skip rejected correspondences (zero residual + zero Jacobian).
    bool jac_all_zero = true;
    for (int j = 0; j < 6; j++) {
      if (std::fabs(jacobians[pt * 6 + j]) > 1e-12f) {
        jac_all_zero = false;
        break;
      }
    }
    if (jac_all_zero) {
      continue;
    }

    for (int j = 0; j < 6; j++) {
      EXPECT_NEAR(jacobians[pt * 6 + j], numeric[j][pt], 2e-3f) << "Jacobian mismatch at point=" << pt << " col=" << j;
    }
    ++checked;
  }
  EXPECT_EQ(checked, static_cast<int>(tr.N)) << "Every landmark should produce an active factor";
}

// A depth-axis perturbation of the pose should produce a non-zero scalar residual equal
// to the signed depth difference (up to rejection gating).
TEST(PointToPointICPTest, ResidualMatchesDepthDifferenceForZPerturbation) {
  constexpr float kPlaneZ = 2.f;
  std::vector<float3> world_points = {{0.f, 0.f, kPlaneZ}, {0.5f, 0.f, kPlaneZ}, {0.f, 0.3f, kPlaneZ}};

  float gt_pose[16], cam_from_rig[16];
  MakeIdentity(gt_pose);
  MakeIdentity(cam_from_rig);

  std::vector<float> depth_host(kWidth * kHeight, kPlaneZ);
  TestResources tr(depth_host, world_points, cam_from_rig);

  // Translate the cam_from_world origin by +kDz along camera-z: every transformed
  // p_cam.z grows by kDz while the depth texture still reads 2.0, so r = p_cam.z - d = +kDz.
  constexpr float kDz = 0.02f;
  float perturbed[16], world_from_rig[16];
  MakeSE3(0, 0, 0, 0, 0, kDz, perturbed);
  InvertSE3CPU(perturbed, world_from_rig);

  std::vector<float> residuals;
  tr.evaluate(world_from_rig, residuals, nullptr);

  for (size_t i = 0; i < tr.N; i++) {
    EXPECT_NEAR(residuals[i], kDz, 1e-4f) << "Expected r = p_cam.z - d = +kDz at point " << i;
  }
}

// A lateral translation on a fronto-parallel plane is invisible to the projective
// depth-only residual (this is the documented trade-off vs. full 3-D ICP).
TEST(PointToPointICPTest, ResidualInvariantToLateralTranslationOnPlane) {
  constexpr float kPlaneZ = 2.f;
  std::vector<float3> world_points = {{0.f, 0.f, kPlaneZ}, {0.5f, 0.f, kPlaneZ}, {0.f, 0.3f, kPlaneZ}};

  float gt_pose[16], cam_from_rig[16];
  MakeIdentity(gt_pose);
  MakeIdentity(cam_from_rig);

  std::vector<float> depth_host(kWidth * kHeight, kPlaneZ);
  TestResources tr(depth_host, world_points, cam_from_rig);

  // 5 cm lateral translation: projection pixel moves, depth value unchanged.
  float perturbed[16], world_from_rig[16];
  MakeSE3(0, 0, 0, 0.05f, 0, 0, perturbed);
  InvertSE3CPU(perturbed, world_from_rig);

  std::vector<float> residuals;
  tr.evaluate(world_from_rig, residuals, nullptr);

  for (size_t i = 0; i < tr.N; i++) {
    EXPECT_NEAR(residuals[i], 0.f, 1e-4f) << "Expected zero residual at point " << i;
  }
}

// Relative-depth rejection gate: a large depth mismatch must be rejected (residual = 0),
// while a mismatch below the kernel's hard-coded threshold (kMaxRelDepth) must be kept.
TEST(PointToPointICPTest, RelativeDepthRejection) {
  std::vector<float3> world_points = {{0.f, 0.f, 2.f}};

  float cam_from_world[16], cam_from_rig[16];
  MakeIdentity(cam_from_world);
  MakeIdentity(cam_from_rig);

  // Case A (rejected): depth = 3.0 m while p_cam.z = 2.0 m -> |r|/z = 0.5 >> 2 %.
  {
    std::vector<float> depth_host(kWidth * kHeight, 3.f);
    TestResources tr(depth_host, world_points, cam_from_rig);
    std::vector<float> res;
    tr.evaluate(cam_from_world, res, nullptr);
    EXPECT_NEAR(res[0], 0.f, 1e-8f) << "Large depth mismatch must be rejected";
  }

  // Case B (accepted): depth = 2.01 m, p_cam.z = 2.0 m -> |r|/z = 0.005 < 2 %.
  {
    std::vector<float> depth_host(kWidth * kHeight, 2.01f);
    TestResources tr(depth_host, world_points, cam_from_rig);
    std::vector<float> res;
    tr.evaluate(cam_from_world, res, nullptr);
    EXPECT_NEAR(res[0], 2.f - 2.01f, 1e-4f) << "Small depth mismatch must be kept";
  }
}

// A non-identity cam_from_rig must be composed with the state; equivalent
// effective poses should produce equivalent residuals.
TEST(PointToPointICPTest, CamFromRigComposition) {
  std::vector<float3> world_points = {{0.f, 0.f, 3.f}};

  float world_from_rig[16], cam_from_rig[16];
  MakeIdentity(world_from_rig);
  MakeSE3(0, 0, 0, 0.2f, 0, 0, cam_from_rig);

  float cam_from_world[16], world_from_cam[16];
  MakeSE3(0, 0, 0, 0.2f, 0, 0, cam_from_world);
  InvertSE3CPU(cam_from_world, world_from_cam);

  auto depth_host = RenderDepthFromPoints(cam_from_world, world_points);

  TestResources tr1(depth_host, world_points, cam_from_rig);
  std::vector<float> res1;
  tr1.evaluate(world_from_rig, res1, nullptr);

  float identity[16];
  MakeIdentity(identity);
  TestResources tr2(depth_host, world_points, identity);
  std::vector<float> res2;
  tr2.evaluate(world_from_cam, res2, nullptr);

  EXPECT_NEAR(res1[0], res2[0], 1e-4f) << "cam_from_rig composition mismatch";
}

}  // namespace test::pnp

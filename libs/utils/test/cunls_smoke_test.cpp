
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

#include <iostream>

#include "common/include_gtest.h"

#include <cunls/common/cuda_stream.h>
#include <cunls/common/types.h>
#include <cunls/factor/prior/prior_vector_factor_batch.h>
#include <cunls/minimizer/gauss_newton_minimizer.h>
#include <cunls/minimizer/levenberg_marquardt_minimizer.h>
#include <cunls/minimizer/problem.h>
#include <cunls/state/vector_state_batch.h>

namespace test::utils {

static constexpr int kDim = 3;
static constexpr size_t kNumVectors = 100;

using Vec = cunls::Vector<kDim>;

TEST(CuNLSSmokeTest, ProblemConsistency) {
  std::cout << "[CuNLS] Running ProblemConsistency smoke test..." << std::endl;

  std::vector<Vec> host_values(kNumVectors);
  std::vector<Vec> host_observations(kNumVectors);
  for (size_t i = 0; i < kNumVectors; i++) {
    float x = static_cast<float>(i);
    host_values[i].fill(x);
    host_observations[i].fill(x - 1.f);
  }

  cunls::DeviceVector<Vec> d_values(host_values);
  cunls::DeviceVector<Vec> d_observations(host_observations);

  cunls::VectorStateBatch<kDim> states(reinterpret_cast<const float*>(d_values.data()), kNumVectors);
  cunls::PriorVectorFactorBatch<kDim> factors(d_observations.data(), kNumVectors);

  std::vector<float*> state_ptrs;
  for (size_t i = 0; i < kNumVectors; i++) {
    state_ptrs.push_back(states.StateBlockDevicePtr(i));
  }

  cunls::Problem problem;
  problem.AddFactorBatch(&factors, state_ptrs);
  problem.AddStateBatch(&states);

  ASSERT_TRUE(problem.CheckConsistency());

  std::cout << "[CuNLS] ProblemConsistency PASSED" << std::endl;
}

TEST(CuNLSSmokeTest, GaussNewtonMinimize) {
  std::cout << "[CuNLS] Running GaussNewtonMinimize smoke test..." << std::endl;

  std::vector<Vec> host_values(kNumVectors);
  std::vector<Vec> host_observations(kNumVectors);
  for (size_t i = 0; i < kNumVectors; i++) {
    float x = static_cast<float>(i);
    host_values[i].fill(x);
    host_observations[i].fill(x - 1.f);
  }

  cunls::DeviceVector<Vec> d_values(host_values);
  cunls::DeviceVector<Vec> d_observations(host_observations);

  cunls::VectorStateBatch<kDim> states(reinterpret_cast<const float*>(d_values.data()), kNumVectors);
  cunls::PriorVectorFactorBatch<kDim> factors(d_observations.data(), kNumVectors);

  std::vector<float*> state_ptrs;
  for (size_t i = 0; i < kNumVectors; i++) {
    state_ptrs.push_back(states.StateBlockDevicePtr(i));
  }

  cunls::Problem problem;
  problem.AddFactorBatch(&factors, state_ptrs);
  problem.AddStateBatch(&states);

  cunls::CudaStream stream;
  cunls::GaussNewtonMinimizer minimizer;
  auto summary = minimizer.Minimize(stream.GetStream(), problem);
  ASSERT_EQ(cudaStreamSynchronize(stream.GetStream()), cudaSuccess);

  std::cout << "[CuNLS] GN iterations: " << summary.num_iterations << ", initial_cost: " << summary.initial_cost
            << ", final_cost: " << summary.final_cost << std::endl;

  EXPECT_GT(summary.initial_cost, 0.f);
  EXPECT_NEAR(summary.final_cost, 0.f, 1e-4f);

  std::vector<Vec> result(kNumVectors);
  d_values.CopyToHost(result.data(), kNumVectors);

  for (size_t i = 0; i < kNumVectors; i++) {
    for (int d = 0; d < kDim; d++) {
      ASSERT_NEAR(result[i][d], host_observations[i][d], 1e-3f);
    }
  }

  std::cout << "[CuNLS] GaussNewtonMinimize PASSED" << std::endl;
}

TEST(CuNLSSmokeTest, LevenbergMarquardtMinimize) {
  std::cout << "[CuNLS] Running LevenbergMarquardtMinimize smoke test..." << std::endl;

  std::vector<Vec> host_values(kNumVectors);
  std::vector<Vec> host_observations(kNumVectors);
  for (size_t i = 0; i < kNumVectors; i++) {
    float x = static_cast<float>(i);
    host_values[i].fill(x);
    host_observations[i].fill(x - 1.f);
  }

  cunls::DeviceVector<Vec> d_values(host_values);
  cunls::DeviceVector<Vec> d_observations(host_observations);

  cunls::VectorStateBatch<kDim> states(reinterpret_cast<const float*>(d_values.data()), kNumVectors);
  cunls::PriorVectorFactorBatch<kDim> factors(d_observations.data(), kNumVectors);

  std::vector<float*> state_ptrs;
  for (size_t i = 0; i < kNumVectors; i++) {
    state_ptrs.push_back(states.StateBlockDevicePtr(i));
  }

  cunls::Problem problem;
  problem.AddFactorBatch(&factors, state_ptrs);
  problem.AddStateBatch(&states);

  cunls::CudaStream stream;
  cunls::LevenbergMarquardtMinimizer minimizer;
  auto summary = minimizer.Minimize(stream.GetStream(), problem);
  ASSERT_EQ(cudaStreamSynchronize(stream.GetStream()), cudaSuccess);

  std::cout << "[CuNLS] LM iterations: " << summary.num_iterations << ", initial_cost: " << summary.initial_cost
            << ", final_cost: " << summary.final_cost << std::endl;

  EXPECT_GT(summary.num_iterations, 0u);
  EXPECT_GT(summary.initial_cost, 0.f);
  EXPECT_NEAR(summary.final_cost, 0.f, 1e-4f);

  std::cout << "[CuNLS] LevenbergMarquardtMinimize PASSED" << std::endl;
}

}  // namespace test::utils

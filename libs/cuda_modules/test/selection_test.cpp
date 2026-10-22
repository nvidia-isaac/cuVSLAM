
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
#include <cstdint>
#include <random>
#include <vector>

#include "common/image_matrix.h"
#include "common/include_gtest.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace test {
using namespace cuvslam;
using namespace cuvslam::cuda;

// select_features appends keypoints in whatever order its blocks run, so tied strengths must not keep that order.
TEST(Cuda, SortKeypointsOrdersTiesByPositionWhenAsked) {
  std::vector<Keypoint> input = {{7.f, 0.f, 2.f}, {1.f, 9.f, 3.f}, {3.f, 1.f, 2.f}, {5.f, 1.f, 2.f}, {2.f, 2.f, 1.f}};
  auto by_x = [](const Keypoint& a, const Keypoint& b) { return a.x < b.x; };
  std::sort(input.begin(), input.end(), by_x);

  const size_t size = input.size();
  GPUArrayPinned<Keypoint> keypoints(size);
  GPUOnlyArray<uint8_t> temp(sort_keypoints_get_temp_buffer_size(size));
  Stream s;
  const auto sort = [&](bool ties_by_position) {
    for (size_t i = 0; i < size; i++) {
      keypoints[i] = input[i];
    }
    keypoints.copy(GPUCopyDirection::ToGPU, s.get_stream());
    sort_keypoints(keypoints.ptr(), size, temp.ptr(), temp.size(), ties_by_position, s.get_stream());
    keypoints.copy(GPUCopyDirection::ToCPU, s.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(s.get_stream()));
    std::vector<Keypoint> sorted(size);
    for (size_t i = 0; i < size; i++) {
      sorted[i] = keypoints[i];
    }
    return sorted;
  };
  auto same_order = [](const std::vector<Keypoint>& a, const std::vector<Keypoint>& b) {
    return std::equal(a.begin(), a.end(), b.begin(), [](const Keypoint& l, const Keypoint& r) {
      return l.x == r.x && l.y == r.y && l.strength == r.strength;
    });
  };
  auto by_strength = [](const Keypoint& a, const Keypoint& b) { return a.strength > b.strength; };

  const std::vector<Keypoint> first = sort(true);
  std::vector<std::vector<Keypoint>> orders_without;
  do {
    const std::vector<Keypoint> sorted = sort(true);
    ASSERT_TRUE(std::is_sorted(sorted.begin(), sorted.end(), by_strength));
    ASSERT_TRUE(same_order(sorted, first));

    const std::vector<Keypoint> unordered_ties = sort(false);
    ASSERT_TRUE(std::is_sorted(unordered_ties.begin(), unordered_ties.end(), by_strength));
    if (std::none_of(orders_without.begin(), orders_without.end(),
                     [&](const std::vector<Keypoint>& order) { return same_order(order, unordered_ties); })) {
      orders_without.push_back(unordered_ties);
    }
  } while (std::next_permutation(input.begin(), input.end(), by_x));
  // without ties_by_position the order of the three keypoints of strength 2 follows the input
  EXPECT_GT(orders_without.size(), 1u);
}

// accumulateGFTT adds each 16x16 tile to the bin its center falls in; with tile_sums every launch gives equal bins.
TEST(Cuda, AccumulateGFTTWithTileSumsIsReproducible) {
  const uint2 size = {1241, 376};
  const uint2 num_bins = {8, 8};
  const uint2 bin_size = {size.x / num_bins.x, size.y / num_bins.y};
  const int tile = 16;

  ImageMatrix<float> image(size.y, size.x);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> value(0.f, 1.f);
  std::vector<double> expected(num_bins.x * num_bins.y, 0.0);
  for (unsigned y = 0; y < size.y; y++) {
    for (unsigned x = 0; x < size.x; x++) {
      image(y, x) = value(rng);
      const unsigned bin_x = (x / tile * tile + tile / 2) / bin_size.x;
      const unsigned bin_y = (y / tile * tile + tile / 2) / bin_size.y;
      if (bin_x < num_bins.x && bin_y < num_bins.y) {
        expected[bin_y * num_bins.x + bin_x] += image(y, x);
      }
    }
  }
  GPUImageT gpu_image(image);

  GPUArrayPinned<float> bins(expected.size());
  GPUOnlyArray<float> tile_sums(accumulateGFTT_get_tile_sums_size(size));
  Stream s;
  const auto accumulate = [&](float* tile_sums_ptr) {
    for (size_t i = 0; i < expected.size(); i++) {
      bins[i] = 0.f;
    }
    bins.copy(GPUCopyDirection::ToGPU, s.get_stream());
    CUDA_CHECK(accumulateGFTT(gpu_image.get_texture_filter_point(), size, bin_size, num_bins, bins.ptr(), tile_sums_ptr,
                              s.get_stream()));
    bins.copy(GPUCopyDirection::ToCPU, s.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(s.get_stream()));
    std::vector<float> result(expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
      result[i] = bins[i];
    }
    return result;
  };

  const std::vector<float> first = accumulate(tile_sums.ptr());
  const std::vector<float> atomic = accumulate(nullptr);
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(first[i], expected[i], 1e-5 * expected[i]) << "bin " << i;
    EXPECT_NEAR(atomic[i], expected[i], 1e-5 * expected[i]) << "bin " << i;
  }
  for (int launch = 0; launch < 100; launch++) {
    ASSERT_EQ(accumulate(tile_sums.ptr()), first) << "launch " << launch;
  }
}

}  // namespace test


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

#include "cuda_modules/selection_v2.h"

#include <array>
#include <cassert>

namespace cuvslam::cuda {

GPUSelection::GPUSelection(bool deterministic) : deterministic_(deterministic) {}

GPUSelection::~GPUSelection() { cudaFree(sort_temp_buffer_); }

void GPUSelection::computeGFTTAndSelectFeatures(const GPUGradientPyramid &gradientPyramid, int border_top,
                                                int border_bottom, int border_left, int border_right,
                                                const ImageMatrix<uint8_t> *input_mask,
                                                const std::vector<Vector2T> &AliveFeatures,
                                                size_t num_desired_new_features,
                                                std::vector<Vector2T> &newSelectedFeatures, cudaStream_t &stream) {
  int *kpCount_host = (int *)(&(bins_array_[0]));
  int *kpIndex_host = (int *)(&(bins_array_[1]));
  int *kpCount_device = (int *)(bins_array_.ptr());
  int *kpIndex_device = (int *)(bins_array_.ptr() + 1);

  const GPUImageT &gradX = gradientPyramid.gradX()[0];
  const GPUImageT &gradY = gradientPyramid.gradY()[0];
  if (!gftt_value_) {
    gftt_value_ = std::make_unique<GPUImageT>(gradX.cols(), gradX.rows());
    gftt_maximums_ = std::make_unique<GPUImageT>(gradX.cols(), gradX.rows());

    gftt_size_ = {(unsigned)gradX.cols(), (unsigned)gradX.rows()};

    bin_size_ = {
        (unsigned)(gftt_size_.x / NUM_BINS_X),
        (unsigned)(gftt_size_.y / NUM_BINS_Y),
    };
    if (deterministic_) {
      gftt_tile_sums_ = std::make_unique<GPUOnlyArray<float>>(accumulateGFTT_get_tile_sums_size(gftt_size_));
    }
  }

  if (!downsampled_maximums_) {
    downsampled_size_ = {(unsigned)gradX.cols() / 8, (unsigned)gradX.rows() / 8};

    max_tracks_ = downsampled_size_.x * downsampled_size_.y;
    keypoints_ = std::make_unique<GPUArrayPinned<Keypoint>>(max_tracks_);
    downsampled_maximums_ = std::make_unique<GPUImageT>(downsampled_size_.x, downsampled_size_.y);
    indices_ = std::make_unique<GPUImage<uint2>>(downsampled_size_.x, downsampled_size_.y);

    sort_temp_buffer_size_ = sort_keypoints_get_temp_buffer_size(max_tracks_);
    CUDA_CHECK(cudaMallocAsync(&sort_temp_buffer_, sort_temp_buffer_size_, stream));
  }

  size_t num_old_keypoints = std::min(max_tracks_, AliveFeatures.size());
  for (size_t i = 0; i < num_old_keypoints; i++) {
    const Vector2T &feature = AliveFeatures[i];
    keypoints_->operator[](i).x = feature.x();
    keypoints_->operator[](i).y = feature.y();
  }

  for (size_t i = 0; i < NUM_BINS_X * NUM_BINS_Y; i++) {
    bins_array_[i + 2] = 0.f;
  }
  *kpCount_host = 0;
  *kpIndex_host = 0;

  keypoints_->copy_top_n(GPUCopyDirection::ToGPU, num_old_keypoints, stream);
  bins_array_.copy(GPUCopyDirection::ToGPU, stream);

  {
    TRACE_EVENT ev_gftt = profiler_domain_.trace_event("gftt_.compute()", profiler_color_);
    gftt_.compute(gradX, gradY, *gftt_value_, stream);
  }

  CUDA_CHECK(accumulateGFTT(gftt_value_->get_texture_filter_point(), gftt_size_, bin_size_, num_bins_,
                            bins_array_.ptr() + 2, gftt_tile_sums_ ? gftt_tile_sums_->ptr() : nullptr, stream));

  CUDA_CHECK(non_max_suppression(gftt_value_->get_texture_filter_point(), gftt_size_, gftt_maximums_->ptr(),
                                 gftt_maximums_->pitch(), stream));

  if (!AliveFeatures.empty()) {
    CUDA_CHECK(filter_maximums(gftt_maximums_->ptr(), gftt_size_, gftt_maximums_->pitch(), keypoints_->ptr(),
                               AliveFeatures.size(), stream));
  }

  CUDA_CHECK(downsample_gftt_x8(gftt_maximums_->ptr(), gftt_maximums_->pitch(), downsampled_maximums_->ptr(),
                                downsampled_size_, downsampled_maximums_->pitch(), indices_->ptr(), indices_->pitch(),
                                stream));

  CUDA_CHECK(select_features(gftt_value_->ptr(), gftt_size_, gftt_value_->pitch(), downsampled_maximums_->ptr(),
                             downsampled_size_, downsampled_maximums_->pitch(), indices_->ptr(), indices_->pitch(),
                             keypoints_->ptr(), max_tracks_, kpCount_device, kpIndex_device, stream));

  bins_array_.copy(GPUCopyDirection::ToCPU, stream);
  cudaStreamSynchronize(stream);

  {
    TRACE_EVENT ev_gftt = profiler_domain_.trace_event("sort", profiler_color_);
    sort_keypoints(keypoints_->ptr(), kpCount_host[0], sort_temp_buffer_, sort_temp_buffer_size_, deterministic_,
                   stream);
  }

  float GFTTSum = 0.f;
  for (size_t i = 0; i < NUM_BINS_X * NUM_BINS_Y; i++) {
    GFTTSum += bins_array_[i + 2];
  }

  std::array<size_t, NUM_BINS_X * NUM_BINS_Y> nFeaturesFromPrevFrame{};
  std::array<size_t, NUM_BINS_X * NUM_BINS_Y> nFeatures{};
  std::array<size_t, NUM_BINS_X * NUM_BINS_Y> nAdded{};

  for (const Vector2T &feat : AliveFeatures) {
    uint2 bin_coords = {
        (unsigned)(feat.x() / (float)bin_size_.x),
        (unsigned)(feat.y() / (float)bin_size_.y),
    };
    if (bin_coords.x < NUM_BINS_X && bin_coords.y < NUM_BINS_Y) {
      nFeaturesFromPrevFrame[bin_coords.y * NUM_BINS_X + bin_coords.x]++;
    }
  }

  {
    TRACE_EVENT ev_gftt = profiler_domain_.trace_event("nFeatures", profiler_color_);
    const size_t nExpectedTracks = AliveFeatures.size() + num_desired_new_features;
    for (size_t i = 0; i < NUM_BINS_X * NUM_BINS_Y; i++) {
      const auto nExpectedFeatures = static_cast<size_t>(std::round(nExpectedTracks * bins_array_[i + 2] / GFTTSum));
      if (nExpectedFeatures > nFeaturesFromPrevFrame[i]) {
        nFeatures[i] = nExpectedFeatures - nFeaturesFromPrevFrame[i];
      }
    }
  }

  keypoints_->copy_top_n(GPUCopyDirection::ToCPU, kpCount_host[0], stream);
  cudaStreamSynchronize(stream);

  {
    TRACE_EVENT ev_gftt = profiler_domain_.trace_event("CopyKP", profiler_color_);
    for (int i = 0; i < kpCount_host[0]; i++) {
      float kp_x = keypoints_->operator[](i).x;
      float kp_y = keypoints_->operator[](i).y;

      assert(kp_x >= 0);
      assert(kp_x < gftt_size_.x);
      assert(kp_y >= 0);
      assert(kp_y < gftt_size_.y);

      if (kp_x < border_left || kp_x > gftt_size_.x - 1 - border_right || kp_y < border_top ||
          kp_y > gftt_size_.y - 1 - border_bottom) {
        continue;
      }

      if (input_mask && (*input_mask)(static_cast<size_t>(kp_y), static_cast<size_t>(kp_x)) != 0) {
        continue;
      }

      uint2 bin_coords = {
          (unsigned)(kp_x / (float)bin_size_.x),
          (unsigned)(kp_y / (float)bin_size_.y),
      };

      if (bin_coords.x < NUM_BINS_X && bin_coords.y < NUM_BINS_Y) {
        unsigned int bin_id = bin_coords.y * NUM_BINS_X + bin_coords.x;
        if (nAdded[bin_id] < nFeatures[bin_id]) {
          nAdded[bin_id]++;

          if (newSelectedFeatures.size() < num_desired_new_features) {
            newSelectedFeatures.emplace_back(kp_x, kp_y);
          } else {
            return;
          }
        }
      }
    }
  }
}

}  // namespace cuvslam::cuda

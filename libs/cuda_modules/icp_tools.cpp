
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

#include "cuda_modules/icp_tools.h"

const float alpha_v = 0.8f;

namespace cuvslam::cuda {
GPUICPTools::GPUICPTools() : pinned_photometric_(1 + 1 + 6 + 6 * 6), pinned_point_to_point_(1 + 1 + 6 + 6 * 6) {}

bool GPUICPTools::match_and_reduce(float& cost, Vector6T& rhs, Matrix6T& hessian, const Vector2T& focal,
                                   const Vector2T& principal, const Level& level, const Isometry3T& cam_from_world,
                                   const float& huber, const std::vector<ObsLmPair>& tracks) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("match_and_reduce");

  Inctinsics intr{focal.x(),
                  focal.y(),
                  principal.x(),
                  principal.y(),
                  static_cast<int>(level.curr_depth.cols()),
                  static_cast<int>(level.curr_depth.rows())};

  ImgTextures tex{level.curr_depth.get_texture_filter_linear(), level.curr_image.get_texture_filter_linear(),
                  level.curr_grad_x.get_texture_filter_linear(), level.curr_grad_y.get_texture_filter_linear()};

  Extrinsics exrt;
  {
    Isometry3T world_from_cam = cam_from_world.inverse();

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        exrt.cam_from_world.d_[i][j] = cam_from_world(i, j);
        exrt.world_from_cam.d_[i][j] = world_from_cam(i, j);
      }
    }
  }

  CUDA_CHECK(
      cudaMemsetAsync(pinned_photometric_.ptr(), 0, pinned_photometric_.size() * sizeof(float), s_.get_stream()));
  CUDA_CHECK(
      cudaMemsetAsync(pinned_point_to_point_.ptr(), 0, pinned_point_to_point_.size() * sizeof(float), s_.get_stream()));

  const size_t num_tracks = std::min(tracks.size(), pinned_tracks_.size());
  if (num_tracks == 0) {
    return false;
  }

  for (size_t i = 0; i < num_tracks; i++) {
    pinned_tracks_[i] = tracks.at(i);
  }

  pinned_tracks_.copy_top_n(ToGPU, num_tracks, s_.get_stream());

  {
    float* cost_ptr = pinned_photometric_.ptr();
    float* num_valid = cost_ptr + 1;
    float* rhs_ptr = num_valid + 1;
    float* H_ptr = rhs_ptr + 6;

    CUDA_CHECK(photometric(tex, intr, exrt, pinned_tracks_.ptr(), num_tracks, huber, cost_ptr, num_valid, rhs_ptr,
                           H_ptr, s_.get_stream()));
  }

  {
    float* cost_ptr = pinned_point_to_point_.ptr();
    float* num_valid = cost_ptr + 1;
    float* rhs_ptr = num_valid + 1;
    float* H_ptr = rhs_ptr + 6;

    CUDA_CHECK(point_to_point(tex, intr, exrt, pinned_tracks_.ptr(), num_tracks, huber, cost_ptr, num_valid, rhs_ptr,
                              H_ptr, s_.get_stream()));
  }

  pinned_photometric_.copy(ToCPU, s_.get_stream());
  pinned_point_to_point_.copy(ToCPU, s_.get_stream());
  CUDA_CHECK(cudaStreamSynchronize(s_.get_stream()));

  const float count_photometric = pinned_photometric_[1];
  const float count_p = pinned_point_to_point_[1];

  // Whichever term has residuals takes the whole weight; alpha_v only splits them when both do.
  // count_photometric flips between LM iterations - magic_depth gates on lm_cam - while count_p does
  // not, so leaving point-to-point on its 1 - alpha_v fraction would make a guess that lost every
  // photometric match come back ~5x cheaper than it is, and the accept test would take it.
  float alpha = alpha_v;
  if (count_photometric <= 0) {
    alpha = 0.f;
    // Nothing above wrote these, and the point-to-point block below accumulates with +=.
    cost = 0.0f;
    rhs = Vector6T::Zero();
    hessian = Matrix6T::Zero();
  } else if (count_p <= 0) {
    alpha = 1.f;
  }

  if (count_photometric > 0) {
    float w = alpha / count_photometric;
    cost = pinned_photometric_[0] * w;
    for (int i = 0; i < 6; i++) {
      rhs(i) = pinned_photometric_[2 + i] * w;
      for (int j = 0; j < 6; j++) {
        if (j > i) {
          hessian(i, j) = pinned_photometric_[8 + j * 6 + i] * w;
        } else {
          hessian(i, j) = pinned_photometric_[8 + i * 6 + j] * w;
        }
      }
    }
  }

  if (count_p > 0) {
    float w = (1.f - alpha) / count_p;

    cost += pinned_point_to_point_[0] * w;
    for (int i = 0; i < 6; i++) {
      rhs(i) += pinned_point_to_point_[2 + i] * w;
      for (int j = 0; j < 6; j++) {
        if (j > i) {
          hessian(i, j) += pinned_point_to_point_[8 + j * 6 + i] * w;
        } else {
          hessian(i, j) += pinned_point_to_point_[8 + i * 6 + j] * w;
        }
      }
    }
  }

  return count_photometric > 0 || count_p > 0;
}

void GPUICPTools::lift_points(const cuda::GPUImageT& dst_depth, const Vector2T& focal, const Vector2T& principal,
                              const std::vector<ObsLmPair>& tracks, std::vector<Vector3T>& landmarks) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("lift_points");

  landmarks.clear();

  size_t num_tracks = std::min(tracks.size(), pinned_tracks_.size());
  if (num_tracks > 0) {
    Inctinsics intr{focal.x(),
                    focal.y(),
                    principal.x(),
                    principal.y(),
                    static_cast<int>(dst_depth.cols()),
                    static_cast<int>(dst_depth.rows())};

    for (size_t i = 0; i < num_tracks; i++) {
      pinned_tracks_[i] = tracks.at(i);
    }

    pinned_tracks_.copy_top_n(ToGPU, num_tracks, s_.get_stream());

    CUDA_CHECK(lift(dst_depth.get_texture_filter_linear(), intr, pinned_tracks_.ptr(), num_tracks,
                    pinned_landmarks_.ptr(), s_.get_stream()));

    pinned_landmarks_.copy_top_n(ToCPU, num_tracks, s_.get_stream());

    landmarks.reserve(num_tracks);

    CUDA_CHECK(cudaStreamSynchronize(s_.get_stream()));

    for (size_t i = 0; i < num_tracks; i++) {
      const auto& lm = pinned_landmarks_[i];
      landmarks.push_back({lm.x, lm.y, lm.z});
    }
  }
}

}  // namespace cuvslam::cuda

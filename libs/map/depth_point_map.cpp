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

#include "map/depth_point_map.h"

#include <algorithm>

#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "epipolar/camera_selection.h"

namespace cuvslam::map {

namespace {

// Eviction predicate: the point is "close" to camera `cam` iff its camera-frame
// z is in (near plane, max_depth]. No frustum check — a point close but momentarily
// off the image rect is still worth keeping because small yaw / vibration will
// routinely bring it back into view.
inline bool close_to_cam(const Vector3T& p_world, const DepthCameraInfo& cam, float max_depth) {
  const Vector3T p_cam = cam.world_from_cam.inverse() * p_world;
  return epipolar::IsDepthInFront(p_cam.z()) && p_cam.z() <= max_depth;
}

// Counting predicate: close AND projects inside this camera's image rect. This
// is the quantity the ICP factor can actually exploit on the next solve.
inline bool close_and_visible(const Vector3T& p_world, const DepthCameraInfo& cam, float max_depth) {
  const Vector3T p_cam = cam.world_from_cam.inverse() * p_world;
  if (!epipolar::IsDepthInFront(p_cam.z()) || p_cam.z() > max_depth) {
    return false;
  }
  const float inv_z = 1.f / p_cam.z();
  const float u = cam.focal.x * p_cam.x() * inv_z + cam.principal.x;
  const float v = cam.focal.y * p_cam.y() * inv_z + cam.principal.y;
  return u >= 0.f && u < static_cast<float>(cam.image_size.x) && v >= 0.f && v < static_cast<float>(cam.image_size.y);
}

}  // namespace

DepthPointMap::DepthPointMap(const DepthPointMapSettings& settings) : settings_(settings) {}

void DepthPointMap::update(const std::vector<DepthCameraInfo>& depth_cameras) {
  TRACE_EVENT ev = profiler_domain_.trace_event("update");
  added_ = false;

  if (depth_cameras.empty()) {
    return;
  }

  {
    TRACE_EVENT evict_ev = profiler_domain_.trace_event("evict_far");
    evict_far(depth_cameras);
  }

  TRACE_EVENT sample_ev = profiler_domain_.trace_event("sample_points");
  for (const auto& cam : depth_cameras) {
    const size_t have = count_close_and_visible(cam);
    if (have >= settings_.max_per_cam) {
      continue;
    }
    add_samples(cam, settings_.max_per_cam - have);
    added_ = true;
  }
}

const std::vector<Vector3T>& DepthPointMap::get_points() const { return points_; }

bool DepthPointMap::added_last_update() const { return added_; }

void DepthPointMap::clear() { points_.clear(); }

void DepthPointMap::evict_far(const std::vector<DepthCameraInfo>& depth_cameras) {
  const float max_depth = settings_.visible_max_depth;
  points_.erase(std::remove_if(points_.begin(), points_.end(),
                               [&](const Vector3T& p) {
                                 for (const auto& cam : depth_cameras) {
                                   if (close_to_cam(p, cam, max_depth)) {
                                     return false;
                                   }
                                 }
                                 return true;
                               }),
                points_.end());
}

size_t DepthPointMap::count_close_and_visible(const DepthCameraInfo& cam) const {
  const float max_depth = settings_.visible_max_depth;
  size_t count = 0;
  for (const auto& p : points_) {
    if (close_and_visible(p, cam, max_depth)) {
      ++count;
    }
  }
  return count;
}

void DepthPointMap::add_samples(const DepthCameraInfo& cam, size_t n) {
  if (n == 0) {
    return;
  }

  cudaStream_t s = stream_.get_stream();

  d_count_[0] = 0;
  d_count_.copy(cuda::ToGPU, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  CUDA_CHECK(cuda::unproject_depth_points(cam.depth_tex, cam.focal, cam.principal, cam.image_size, settings_.depth_min,
                                          settings_.depth_max, d_sampled_points_.ptr(), d_count_.ptr(),
                                          kMaxSamplePoints, settings_.sample_stride, s));

  d_count_.copy(cuda::ToCPU, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  const int num_sampled = std::min(d_count_[0], kMaxSamplePoints);
  if (num_sampled == 0) {
    return;
  }

  // Copy the full sampled set to host so we can stride-pick a spatially
  // uniform subset of size `n`. Copying only `n` up front and then striding
  // over that smaller slice would yield fewer points than requested — that
  // was the bug in the previous implementation.
  d_sampled_points_.copy_top_n(cuda::ToCPU, num_sampled, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  const int want = static_cast<int>(std::min<size_t>(n, static_cast<size_t>(num_sampled)));
  const int step = std::max(1, num_sampled / want);
  const Isometry3T& w_from_c = cam.world_from_cam;

  int added = 0;
  for (int i = 0; i < num_sampled && added < want; i += step) {
    const float3& pt = d_sampled_points_[i];
    points_.push_back(w_from_c * Vector3T(pt.x, pt.y, pt.z));
    ++added;
  }
}

}  // namespace cuvslam::map

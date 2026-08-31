
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

#include "sof/epipolar_curves.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "epipolar/camera_selection.h"

namespace cuvslam::sof {
namespace {
constexpr int kNumDepthSamples = 50;

// Auto-detect anchors mapping stereo baseline (m) → sensible (min_depth, max_depth) range (m).
// Small stereo (7.5 cm avg baseline) is set for close-range indoor use; large stereo (KITTI-
// scale, ~0.5 m baseline) is set for outdoor / driving use. Baselines above kAutoBaselineLarge
// clamp to the large-baseline values.
constexpr float kAutoBaselineSmall = 0.075f;  // ~7.5 cm — small stereo (5–10 cm range midpoint)
constexpr float kAutoBaselineLarge = 0.5f;    // 0.5 m — KITTI-scale
constexpr float kAutoMinDepthSmall = 0.1f;    // 10 cm
constexpr float kAutoMinDepthLarge = 7.f;     // 7 m
constexpr float kAutoMaxDepthSmall = 20.f;    // 20 m
constexpr float kAutoMaxDepthLarge = 1000.f;  // 1 km

// Fills any finite negative `min_depth` / `max_depth` with baseline-interpolated defaults, then
// throws if the resulting range is not finite, strictly positive and monotonic. Only a finite
// negative value is the "auto" trigger; 0 stays an explicit input, and every non-finite input
// (NaN, ±inf) falls through to validation rather than being silently replaced. Left unchecked,
// +inf would reach `std::log` and turn every depth sample into NaN.
void AutoDetectDepthRange(const float baseline, float& min_depth, float& max_depth) {
  const float t = std::clamp((baseline - kAutoBaselineSmall) / (kAutoBaselineLarge - kAutoBaselineSmall), 0.f, 1.f);
  if (std::isfinite(min_depth) && min_depth < 0.f) {
    min_depth = kAutoMinDepthSmall + t * (kAutoMinDepthLarge - kAutoMinDepthSmall);
  }
  if (std::isfinite(max_depth) && max_depth < 0.f) {
    max_depth = kAutoMaxDepthSmall + t * (kAutoMaxDepthLarge - kAutoMaxDepthSmall);
  }
  if (!std::isfinite(min_depth) || !std::isfinite(max_depth) || !(min_depth > 0.f) || !(min_depth < max_depth)) {
    throw std::invalid_argument(
        "EpipolarCurves: min_depth / max_depth must be finite and satisfy 0 < min_depth < max_depth "
        "after auto-detection (pass a negative value to trigger auto-detect).");
  }
}
}  // namespace

void EpipolarCurves::Candidates(const Vector2T& uv_l_base, std::vector<Vector2T>& out) const {
  out.clear();
  const float v_f = uv_l_base.y() * inv_scale_;
  const float u_f = uv_l_base.x() * inv_scale_;

  const size_t top_image_w = curves_[0].size();
  const size_t top_image_h = curves_.size();
  if (v_f < 0.f || v_f >= static_cast<float>(top_image_h) || u_f < 0.f || u_f >= static_cast<float>(top_image_w)) {
    return;
  }
  const auto v0 = static_cast<size_t>(v_f);
  const auto u0 = static_cast<size_t>(u_f);
  const size_t v1 = v0 + 1;
  const size_t u1 = u0 + 1;
  // Need four corners around (u_f, v_f); bail if either right/bottom corner is out of range.
  if (v1 >= top_image_h || u1 >= top_image_w) {
    return;
  }
  const float dv = v_f - static_cast<float>(v0);
  const float du = u_f - static_cast<float>(u0);
  const float w00 = (1.0f - du) * (1.0f - dv);
  const float w01 = du * (1.0f - dv);
  const float w10 = (1.0f - du) * dv;
  const float w11 = du * dv;
  const auto& c00 = curves_[v0][u0];
  const auto& c01 = curves_[v0][u1];
  const auto& c10 = curves_[v1][u0];
  const auto& c11 = curves_[v1][u1];
  // Interpolate up to the shortest of the four corner curves. Adjacent corners have similar
  // geometry, so their lengths typically differ by at most a candidate or two.
  const size_t n = std::min({c00.size(), c01.size(), c10.size(), c11.size()});
  out.resize(n);
  for (size_t k = 0; k < n; ++k) {
    out[k] = w00 * c00[k] + w01 * c01[k] + w10 * c10[k] + w11 * c11[k];
  }
}

EpipolarCurves::EpipolarCurves(const camera::ICameraModel& cam_l, const camera::ICameraModel& cam_r,
                               const Isometry3T& right_from_left, const int top_level, const size_t top_width,
                               const size_t top_height, float min_depth, float max_depth)
    : inv_scale_(1.0f / static_cast<float>(1u << top_level)),
      curves_(top_height + 1, std::vector<std::vector<Vector2T>>(top_width + 1)) {
  const auto scale = static_cast<float>(1u << top_level);

  AutoDetectDepthRange(right_from_left.translation().norm(), min_depth, max_depth);

  const float log_min = std::log(min_depth);
  const float log_max = std::log(max_depth);
  const float log_step = (log_max - log_min) / static_cast<float>(kNumDepthSamples - 1);

  // Iterate CORNERS: (u, v) in [0, top_width] × [0, top_height]. Each corner sits at the base-level
  // position (u*scale, v*scale) — the top-left corner of top-pixel (u, v) mapped to base.
  for (size_t v = 0; v <= top_height; ++v) {
    auto& row = curves_[v];
    for (size_t u = 0; u <= top_width; ++u) {
      const Vector2T uv_l_base(static_cast<float>(u) * scale, static_cast<float>(v) * scale);
      Vector2T xy_l;
      if (!cam_l.normalizePoint(uv_l_base, xy_l)) {
        continue;
      }
      const Vector3T ray_l(xy_l.x(), xy_l.y(), 1.0f);
      auto& curve = row[u];
      // Typical dedup-keeps range from a handful (KITTI-scale, avg ~3) to a few dozen (short
      // baseline). 128 covers the practical worst case with negligible memory per corner.
      curve.reserve(128);

      // Distance-based dedup: enforce that consecutive stored entries are ≥ 1 top-level pixel
      // apart (Euclidean). Prevents storing (u_r, v_r) positions that are within 1 pixel of the
      // previously kept entry — the LK convergence basin trivially covers sub-pixel gaps.
      float last_u_top_f = -1e9f;
      float last_v_top_f = -1e9f;
      float log_d = log_max;
      for (int s = 0; s < kNumDepthSamples; ++s, log_d -= log_step) {
        const Vector3T p_r = right_from_left * (std::exp(log_d) * ray_l);
        if (!epipolar::IsDepthInFront(p_r.z())) {
          continue;
        }
        Vector2T uv_r_base;
        if (!cam_r.denormalizePoint({p_r.x() / p_r.z(), p_r.y() / p_r.z()}, uv_r_base)) {
          continue;
        }
        const float u_r_top_f = uv_r_base.x() * inv_scale_;
        const float v_r_top_f = uv_r_base.y() * inv_scale_;
        if (u_r_top_f < 0.0f || v_r_top_f < 0.0f) {
          continue;
        }
        if (u_r_top_f >= static_cast<float>(top_width) || v_r_top_f >= static_cast<float>(top_height)) {
          continue;
        }
        const float du = u_r_top_f - last_u_top_f;
        const float dv = v_r_top_f - last_v_top_f;
        if (du * du + dv * dv < 1.0f) {
          continue;
        }
        last_u_top_f = u_r_top_f;
        last_v_top_f = v_r_top_f;
        // Store the sub-pixel top-level position (converted to base-level) rather than snapping to
        // a pixel center — preserves the smooth curve for LK's initial guess.
        curve.emplace_back(u_r_top_f * scale, v_r_top_f * scale);
      }
    }
  }
}

}  // namespace cuvslam::sof

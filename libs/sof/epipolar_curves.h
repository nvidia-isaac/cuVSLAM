
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

#pragma once

#include <vector>

#include "camera/camera.h"
#include "common/isometry.h"
#include "common/vector_2t.h"

namespace cuvslam::sof {

// Precomputed L2R initial-guess table for the epipolar-scan LK tracker. Samples scene depth
// log-uniformly over the caller-supplied `[min_depth, max_depth]` range along the left viewing
// ray of each top-level pixel CORNER, projects through the right camera model (with distortion),
// and dedupes consecutive samples that fall within one top-level pixel of the last kept sample.
// Internally indexes the stored curves by LEFT top-level pixel corner; publicly,
// `Candidates(uv_l_base)` accepts a LEFT base-level pixel position and fills a caller-owned
// vector with RIGHT base-level candidate positions along the epipolar curve — bilinearly
// interpolated from the four surrounding corner curves, roughly ordered far-to-near (independent
// per-corner dedup makes the ordering approximate, not strict). Consumers feed those directly to
// the LK tracker as initial guesses and stop at the first success.
// Interpolation accuracy: on the lr_test pair the guesses sit ~0.01 px off the true epipolar curve
// on average (0.05 px at p99); the residual error is almost entirely ALONG the curve rather than
// across it, which is the per-corner dedup above and harmless to the scan. Worst near the borders.
class EpipolarCurves {
public:
  // `min_depth` / `max_depth` — the depth range (meters) sampled along the epipolar curve.
  // Any negative value (e.g. -1) auto-detects from the pair's baseline; see
  // `Odometry::Config::min_depth` in the public API for the auto-detection anchors.
  EpipolarCurves(const camera::ICameraModel& cam_l, const camera::ICameraModel& cam_r,
                 const Isometry3T& right_from_left, int top_level, size_t top_width, size_t top_height, float min_depth,
                 float max_depth);

  // For a LEFT base-level pixel, fills `out` with RIGHT base-level candidate positions along the
  // epipolar curve (roughly far-to-near — see class comment for why "roughly"), bilinearly
  // interpolated from the four surrounding top-level pixel corners. `out` is resized as needed;
  // caller should retain the vector between calls/frames to avoid reallocation. Left empty if
  // `uv_l_base` is out of range or NaN.
  void Candidates(const Vector2T& uv_l_base, std::vector<Vector2T>& out) const;

private:
  const float inv_scale_;
  // curves_[v][u] — for the LEFT top-level pixel CORNER at (u, v), the far-to-near list of RIGHT
  // base-level pixel positions along the epipolar curve. Corner grid: outer dim is top_height+1,
  // middle dim is top_width+1 (corners around each top-pixel). Candidates() bilinearly
  // interpolates between the four surrounding corners for arbitrary sub-corner queries.
  std::vector<std::vector<std::vector<Vector2T>>> curves_;
};

}  // namespace cuvslam::sof

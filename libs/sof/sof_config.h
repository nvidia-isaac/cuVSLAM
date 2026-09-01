
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

#include <optional>
#include <vector>

#include "camera/frustum_intersection_graph.h"

#include "sof/selector_stereo.h"
#include "sof/sof.h"

namespace cuvslam::sof {

struct Settings {
  // number of tracks for left selection
  int32_t num_desired_tracks = 450;

  // left camera border to ignore in pixels
  int32_t border_top = 0;
  int32_t border_bottom = 0;
  int32_t border_left = 0;
  int32_t border_right = 0;

  // image preprocessor
  bool box3_prefilter = false;

  bool ransac_filter = false;

  TrackerType tracker = TrackerType::LK;

  // left-to-right tracker (stereo only)
  TrackerType lr_tracker = TrackerType::LK;

  // Depth range (meters) sampled along the epipolar curve when generating LK initial guesses for
  // left-to-right (L2R) tracking. Used wherever MultiSOF tracks between overlapping camera pairs;
  // ignored by MonoSOF, which has no L2R stage.
  // Any negative value (e.g. -1) auto-detects from the pair baseline. See
  // `Odometry::Config::min_depth` in the public API for the auto-detection anchors.
  float min_depth = -1.f;
  float max_depth = -1.f;

  SelectorStereoSettings feature_selection_settings;

  camera::MulticameraMode multicam_mode = camera::MulticameraMode::Moderate;
  camera::MulticamManualSetup multicam_setup;
};

void OverrideMulticameraSettings(Settings& settings, const std::optional<camera::MulticameraMode>& multicam_mode,
                                 const camera::MulticamManualSetup& multicam_setup);

}  // namespace cuvslam::sof

// Included after Settings is fully defined to break the circular dependency
// with sof_mono_interface.h (which uses Settings in its function signatures).
#include "sof/sof_mono_interface.h"

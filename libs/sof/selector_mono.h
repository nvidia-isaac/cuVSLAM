
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

#include "sof/selector_interface.h"
#include "sof/sof.h"

namespace cuvslam::sof {

struct SelectorMonoSettings {
  // Initialize new feature tracks if the percent of survivor tracks
  // from the last feature initialization is less than this value.
  float survivor_from_last = 41.f;

  // Initialize new feature tracks if the percent of survivor tracks
  // from the penultimate feature initialization is less than this value.
  float survivor_from_penultimate = 10.f;

  // Initialize new feature tracks if the average motion
  // in normalized (/ width) values is higher than this value
  float av_motion_from_last = 0.194f;

  // New feature tracks are not initialized until motion > min_av_motion_to_initialization.
  float min_av_motion_to_initialization = 0.1f;
};

class SelectorMono : public ISelector {
public:
  SelectorMono(const SelectorMonoSettings& settings, bool mono);

  void set_image_width(uint32_t width) override;
  void reset_selector() override;
  // Uses SelectorMonoSettings; sof_settings has no mono selection fields.
  bool select(const TracksVector& cur_frame_tracks, const Settings& sof_settings) override;
  void set_tracks(const TracksVector& cur_frame_tracks) override;

private:
  SelectorMonoSettings settings_;
  bool mono_ = false;
  uint32_t width_ = 0;
  bool first_feature_initialization_selected_ = false;
  TracksVector penultimate_feature_initialization_tracks_;  // expected to be ordered by TrackId
  TracksVector last_feature_initialization_tracks_;         // expected to be ordered by TrackId
};

}  // namespace cuvslam::sof


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

#include "sof/selector_stereo.h"

#include "sof/sof_config.h"

namespace {
using namespace cuvslam::sof;

// p_sum_motion - average motion in pixels
uint32_t CalcNSurvivors(const TracksVector& frame1, const TracksVector& frame2) {
  const auto size1 = frame1.size();
  const auto size2 = frame2.size();

  uint32_t i1 = 0;
  uint32_t i2 = 0;
  uint32_t n_survivors = 0;  // number of alive tracks between frames

  while (i1 < size1) {
    const Track& t1 = frame1[i1];
    ++i1;

    if (t1.dead()) {
      continue;
    }

    while (i2 < size2) {
      const Track& t2 = frame2[i2];

      if (!t2.dead() && t1.id() == t2.id()) {
        ++n_survivors;
      }

      if ((i2 + 1 < size2) && (frame2[i2 + 1].id() <= t1.id())) {
        ++i2;
      } else {
        break;
      }
    }
  }

  return n_survivors;
}

float CalcSurvivorTracksPercentage(const TracksVector& frame1, uint32_t n_survivors) {
  const size_t n_frame1_tracks = frame1.get_num_alive();

  assert(n_survivors <= n_frame1_tracks);

  if (n_frame1_tracks == 0) {
    return 0;
  }

  const float percentage = 100.f * n_survivors / n_frame1_tracks;
  assert(0.f <= percentage && percentage <= 100.f);
  return percentage;
}

}  // namespace

namespace cuvslam::sof {
void SelectorStereo::set_image_width(uint32_t /* width */) {}

void SelectorStereo::reset_selector() {
  first_feature_initialization_selected_ = false;
  last_feature_initialization_tracks_.reset();
}

bool SelectorStereo::select(const TracksVector& cur_frame_tracks, const Settings& sof_settings) {
  if (!first_feature_initialization_selected_) {
    first_feature_initialization_selected_ = true;
    return true;
  }

  const uint32_t n_survivors_from_last = CalcNSurvivors(last_feature_initialization_tracks_, cur_frame_tracks);
  if (n_survivors_from_last == 0) {
    return true;
  }

  assert(CalcNSurvivors(last_feature_initialization_tracks_, cur_frame_tracks) == n_survivors_from_last);

  if (CalcSurvivorTracksPercentage(last_feature_initialization_tracks_, n_survivors_from_last) <
      sof_settings.feature_selection_settings.survivor_from_last) {
    return true;
  }

  return false;
}

void SelectorStereo::set_tracks(const TracksVector& cur_frame_tracks) {
  last_feature_initialization_tracks_ = cur_frame_tracks;
}

void KillTracksOnBorder(size_t w, size_t h, size_t border_top, size_t border_bottom, size_t border_left,
                        size_t border_right, sof::TracksVector& tracks) {
  for (size_t i = 0; i < tracks.size(); ++i) {
    const Track& track = tracks[i];
    if (track.dead()) {
      continue;
    }

    const Vector2T p = track.position();
    const float x = p.x();
    const float y = p.y();
    if (border_left <= x && x <= w - 1 - border_right && border_top <= y && y <= h - 1 - border_bottom) {
      continue;
    }
    tracks.kill(i);
  }
}

}  // namespace cuvslam::sof

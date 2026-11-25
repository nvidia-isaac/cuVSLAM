
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

#include "sof/selector_mono.h"

namespace cuvslam::sof {

namespace {
#ifndef NDEBUG
bool IsSorted(const TracksVector& v) {
#ifdef CHECKED_BUILD
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i - 1].id() >= v[i].id()) {
      return false;
    }
  }
#else
  (void)v;
#endif

  return true;
}
#endif

// p_sum_motion - average motion in pixels
uint32_t CalcNSurvivors(const TracksVector& frame1, const TracksVector& frame2, float* p_sum_motion = nullptr) {
  assert(IsSorted(frame1));
  assert(IsSorted(frame2));
  const auto size1 = frame1.size();
  const auto size2 = frame2.size();

  uint32_t i1 = 0;
  uint32_t i2 = 0;
  uint32_t n_survivors = 0;  // number of alive tracks between frames

  if (p_sum_motion != nullptr) {
    *p_sum_motion = 0.f;
  }

  while (i1 < size1) {
    const Track& t1 = frame1[i1];
    ++i1;

    if (t1.dead()) {
      continue;
    }

    while (i2 < size2) {
      const Track& t2 = frame2[i2];

      if (!t2.dead() && t1.id() == t2.id()) {
        if (p_sum_motion != nullptr) {
          *p_sum_motion += (t1.position() - t2.position()).norm();
        }

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

// return percent (0.f-100.f) of survivor tracks
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

SelectorMono::SelectorMono(const SelectorMonoSettings& settings, bool mono) : settings_(settings), mono_(mono) {}

void SelectorMono::set_image_width(uint32_t width) { width_ = width; }

void SelectorMono::reset_selector() { first_feature_initialization_selected_ = false; }

bool SelectorMono::select(const TracksVector& cur_frame_tracks, [[maybe_unused]] const Settings& sof_settings) {
  assert(IsSorted(cur_frame_tracks));

  if (!first_feature_initialization_selected_) {
    first_feature_initialization_selected_ = true;
    return true;
  }

  float sum_motion;
  const uint32_t n_survivors_from_last =
      CalcNSurvivors(last_feature_initialization_tracks_, cur_frame_tracks, &sum_motion);
  if (n_survivors_from_last == 0) {
    return true;
  }

  assert(CalcNSurvivors(last_feature_initialization_tracks_, cur_frame_tracks) == n_survivors_from_last);

  if (CalcSurvivorTracksPercentage(last_feature_initialization_tracks_, n_survivors_from_last) <
      settings_.survivor_from_last) {
    return true;
  }

  const uint32_t n_survivors_from_penultimate =
      CalcNSurvivors(penultimate_feature_initialization_tracks_, cur_frame_tracks);
  const float survivor_from_penultimate = settings_.survivor_from_penultimate;

  const float av_motion_in_pixels = sum_motion / n_survivors_from_last;
  const float av_motion_normalized = av_motion_in_pixels / width_;

  if (mono_ &&
      CalcSurvivorTracksPercentage(penultimate_feature_initialization_tracks_, n_survivors_from_penultimate) <=
          survivor_from_penultimate &&
      av_motion_normalized > settings_.min_av_motion_to_initialization) {
    return true;
  }

  return av_motion_normalized > settings_.av_motion_from_last;
}

void SelectorMono::set_tracks(const TracksVector& cur_frame_tracks) {
  assert(IsSorted(cur_frame_tracks));

  penultimate_feature_initialization_tracks_ = last_feature_initialization_tracks_;
  last_feature_initialization_tracks_ = cur_frame_tracks;
}

}  // namespace cuvslam::sof

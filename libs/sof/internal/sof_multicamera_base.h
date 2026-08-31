
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

#include "sof/sof_multicamera_interface.h"

#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

#include "common/camera_id.h"
#include "odometry/svo_config.h"
#include "profiler/profiler.h"

#include "sof/epipolar_curves.h"
#include "sof/kf_selector.h"
#include "sof/sof.h"

namespace cuvslam::sof {

// Multi-launch L2R scan stops advancing the candidate index once this fraction of observations
// is tracked. Later candidates are more likely to produce spurious matches for the still-
// untracked points, so dropping them cleanly beats capturing decoys.
inline constexpr float kL2REarlyExitFraction = 0.8f;

class MultiSOFBase : public IMultiSOF {
protected:
public:
  MultiSOFBase(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid, const Settings& sof_settings,
               const odom::KeyFrameSettings& keyframe_settings);

  bool trackNextFrame(const Sources& curr_sources, Images& curr_images, const Images& prev_images,
                      const Sources& masks_sources, const Isometry3T& predicted_world_from_rig,
                      MulticamObservations& observations, FrameState& state,
                      const odom::TrackPerFrameSettings& per_frame) final;

  void reset_keyframe_selector() override;

protected:
  virtual void LaunchTrackingPrimaryToSecondary(CameraId primary_id, CameraId secondary_id, const Sources& curr_sources,
                                                Images& curr_images,
                                                const std::vector<camera::Observation>& primary_obs,
                                                std::vector<camera::Observation>* secondary_obs = nullptr) = 0;
  virtual void GetTrackingResults(MulticamObservations& observations) = 0;
  virtual void StartKeyframe() = 0;

  struct TracksVectorAndCam {
    CameraId cam_id;
    std::reference_wrapper<const TracksVector> ref;
  };
  using MulticamTracksVector = std::vector<TracksVectorAndCam>;

  bool is_keyframe(const MulticamTracksVector& tracks, const int64_t current_timestamp_ns,
                   const odom::KeyFrameSettings& kf_settings);

  // Return the epipolar curve grid for the given (primary, secondary) pair, building it lazily on
  // first request. Derived classes call this after their per-implementation pyramids are built and
  // pass in the top-level dimensions from those pyramids. Left and right cameras are assumed to
  // have the same base resolution (hence the same top-level dimensions). Depth range is taken
  // from `sof::Settings::min_depth` / `max_depth` captured at construction (any negative
  // value, e.g. -1, auto-detects from baseline).
  const EpipolarCurves& GetOrBuildEpipolarCurves(CameraId primary_id, CameraId secondary_id, int top_level,
                                                 size_t top_width, size_t top_height);

  // LK search-radius cap for L2R initial guesses. With epipolar-guided starting points already
  // within a few pixels of the true match, LK only needs local refinement. Sized at an
  // empirically-tuned multiple of top-level pixel size (see .cpp for tuning notes) — large enough
  // to permit refinement, small enough to prevent lateral drift onto decoy features. Shared by
  // MultiSOFCPU and MultiSOFGPU so both paths use the same reach, and by LK and LKHorizontal alike
  // — rectified stereo gets the same radius as the general case.
  static float CrossCamSearchRadius(int top_level);

  camera::Rig rig_;
  camera::FrustumIntersectionGraph fid_;
  bool box_prefilter_ = false;
  float min_depth_ = -1.f;
  float max_depth_ = -1.f;
  std::list<std::unique_ptr<IMonoSOF>> mono_sof_;
  KFSelector kf_selector_;
  std::unordered_map<CameraId, TracksVector> last_kf_tracks_;
  int64_t last_kf_timestamp_ = 0;
  std::unordered_map<CameraId, std::unordered_map<CameraId, EpipolarCurves>> epipolar_curves_by_pair_;

  // keep allocated memory for is_keyframe
  TracksVector all_tracks_vec_;
  TracksVector last_kf_tracks_vec_;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("MultiSOF");
  const uint32_t profiler_color_ = 0x00FF00;
};

}  // namespace cuvslam::sof

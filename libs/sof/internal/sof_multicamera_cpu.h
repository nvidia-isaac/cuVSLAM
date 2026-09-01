
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

#include "sof/internal/sof_multicamera_base.h"

#include <memory>
#include <unordered_map>

#include "camera/camera.h"
#include "camera/frustum_intersection_graph.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "sof/feature_prediction_interface.h"
#include "sof/feature_tracker.h"
#include "sof/image_pyramid_float.h"
#include "sof/image_pyramid_u8.h"
#include "sof/kf_selector.h"
#include "sof/sof_config.h"
#include "sof/sof_mono_interface.h"

namespace cuvslam::sof {

class MultiSOFCPU : public MultiSOFBase {
public:
  MultiSOFCPU(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid,
              sof::FeaturePredictorPtr feature_predictor, const Settings& sof_settings,
              const odom::KeyFrameSettings& keyframe_settings);

  void reset() final;

private:
  // Per-observation winner slot — mirrors GPU's `winner_scratch[i]` (a `TrackData`) so both
  // implementations use one scratch vector per observation instead of parallel arrays.
  // `tracked` is `uint8_t` to avoid the bit-packed `std::vector<bool>` specialization.
  struct CPUWinner {
    Vector2T uvR;
    Matrix2T info;
    uint8_t tracked = 0;
  };

  struct PrimaryToSecondaryCPUTracker {
    std::unique_ptr<IFeatureTracker> tracker;
    // Per-frame / per-observation scratch. Kept as members (rather than locals) so allocations
    // and inner-vector capacities persist across Launch calls.
    std::vector<Vector2T> uvL;
    std::vector<std::vector<Vector2T>> cands;
    std::vector<CPUWinner> winners;
  };

  std::unordered_map<CameraId,                                        // primary cam id
                     std::unordered_map<CameraId,                     // secondary cam id
                                        PrimaryToSecondaryCPUTracker  // tracker primary -> secondary
                                        >>
      secondary_from_primary_sof_;

  void LaunchTrackingPrimaryToSecondary(CameraId primary_id, CameraId secondary_id, const Sources& curr_sources,
                                        Images& curr_images, const std::vector<camera::Observation>& primary_obs,
                                        std::vector<camera::Observation>* secondary_obs = nullptr) final;
  void GetTrackingResults(MulticamObservations& observations) final;
  void StartKeyframe() final;
};

}  // namespace cuvslam::sof

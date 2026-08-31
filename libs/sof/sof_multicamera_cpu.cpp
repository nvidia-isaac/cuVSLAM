
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

#include <memory>
#include <vector>

#include "sof/internal/sof_multicamera_cpu.h"
#include "sof/sof_create.h"

namespace cuvslam::sof {

MultiSOFCPU::MultiSOFCPU(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid,
                         sof::FeaturePredictorPtr feature_predictor, const Settings& sof_settings,
                         const odom::KeyFrameSettings& keyframe_settings)
    : MultiSOFBase(rig, fid, sof_settings, keyframe_settings) {
  const auto& primary_cams = fid_.primary_cameras();

  for (CameraId primary_cam_id : primary_cams) {
    const camera::ICameraModel& intrinsics = *rig_.intrinsics[primary_cam_id];
    auto selector = std::make_unique<SelectorStereo>(sof_settings.feature_selection_settings);
    mono_sof_.emplace_back(CreateMonoSOF(Implementation::kCPU, primary_cam_id, intrinsics, std::move(selector),
                                         feature_predictor, sof_settings));

    const auto& secondary_cams = fid_.secondary_cameras(primary_cam_id);

    auto& tracker_from_secondary_cam = secondary_from_primary_sof_[primary_cam_id];
    for (CameraId secondary_cam_id : secondary_cams) {
      tracker_from_secondary_cam[secondary_cam_id].tracker = CreateTracker(sof_settings.lr_tracker);
    }
  }
}

void MultiSOFCPU::LaunchTrackingPrimaryToSecondary(CameraId primary_id, CameraId secondary_id,
                                                   const Sources& curr_sources, sof::Images& curr_images,
                                                   const std::vector<camera::Observation>& primary_obs,
                                                   std::vector<camera::Observation>* secondary_obs) {
  ImageContextPtr primary_image = curr_images[primary_id];
  const ImageSource& secondary_source = curr_sources[secondary_id];
  ImageContextPtr secondary_image = curr_images[secondary_id];

  const camera::ICameraModel& intrinsicsP = *rig_.intrinsics[primary_id];
  const camera::ICameraModel& intrinsicsS = *rig_.intrinsics[secondary_id];

  PrimaryToSecondaryCPUTracker& pair = secondary_from_primary_sof_.at(primary_id).at(secondary_id);
  const std::unique_ptr<IFeatureTracker>& tracker = pair.tracker;
  assert(tracker != nullptr);

  secondary_image->build_cpu_image_pyramid(secondary_source, box_prefilter_);
  secondary_image->build_cpu_gradient_pyramid(tracker->isHorizontal());

  const ImagePyramidT& img_l = primary_image->cpu_image_pyramid();
  const ImagePyramidT& img_r = secondary_image->cpu_image_pyramid();

  const int top_l = img_l.getLevelsCount() - 1;
  const EpipolarCurves& epipolar_curves =
      GetOrBuildEpipolarCurves(primary_id, secondary_id, top_l, img_l[top_l].cols(), img_l[top_l].rows());

  const float cross_cam_search_radius = CrossCamSearchRadius(top_l);

  const size_t n = primary_obs.size();
  auto& uvL = pair.uvL;
  auto& cands = pair.cands;
  auto& winners = pair.winners;
  uvL.resize(n);
  cands.resize(n);  // inner vectors retain their allocation across frames
  winners.assign(n, CPUWinner{});

  size_t max_candidates = 0;
  for (size_t i = 0; i < n; ++i) {
    // A point that can't be projected gets no candidates: Candidates() would otherwise leave last
    // frame's list in place, and uvL[i] keeps whatever it held.
    if (!intrinsicsP.denormalizePoint(primary_obs[i].xy, uvL[i])) {
      cands[i].clear();
      continue;
    }
    epipolar_curves.Candidates(uvL[i], cands[i]);
    max_candidates = std::max(max_candidates, cands[i].size());
  }

  size_t tracked_count = 0;

  // Candidate-index outer loop, observation inner loop — matches GPU semantics: first successful
  // candidate wins per observation.
  for (size_t k = 0; k < max_candidates; ++k) {
    for (size_t i = 0; i < n; ++i) {
      if (winners[i].tracked || k >= cands[i].size()) {
        continue;
      }
      Vector2T uvR = cands[i][k];
      Matrix2T info;
      if (tracker->trackPoint(primary_image->cpu_gradient_pyramid(), secondary_image->cpu_gradient_pyramid(), img_l,
                              img_r, uvL[i], uvR, info, cross_cam_search_radius)) {
        winners[i].uvR = uvR;
        winners[i].info = info;
        winners[i].tracked = 1;
        ++tracked_count;
      }
    }

    // Stop advancing candidate index once we've tracked at least kL2REarlyExitFraction of
    // observations (shared with MultiSOFGPU).
    if (static_cast<float>(tracked_count) >= kL2REarlyExitFraction * static_cast<float>(n)) {
      break;
    }
  }

  // Publish successful matches in observation order.
  if (secondary_obs) {
    for (size_t i = 0; i < n; ++i) {
      if (!winners[i].tracked) {
        continue;
      }
      Vector2T xyR;
      Matrix2T info_xy;
      if (!intrinsicsS.normalizePoint(winners[i].uvR, xyR) ||
          !camera::ObservationInfoUVToXY(intrinsicsS, winners[i].uvR, xyR, winners[i].info, info_xy)) {
        continue;
      }
      secondary_obs->push_back({secondary_id, primary_obs[i].id, xyR, info_xy});
    }
  }
}

void MultiSOFCPU::GetTrackingResults(MulticamObservations&) { return; }

void MultiSOFCPU::StartKeyframe() {}

void MultiSOFCPU::reset() {
  for (auto& sof : mono_sof_) {
    sof->reset();
  }

  reset_keyframe_selector();
}

}  // namespace cuvslam::sof


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

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "common/vector_2t.h"
#include "sof/internal/sof_multicamera_gpu.h"
#include "sof/sof_create.h"

namespace cuvslam::sof {

std::unique_ptr<GPULKFeatureTracker> CreateGPUTracker(TrackerType type) {
  switch (type) {
    case TrackerType::LK:
      return std::make_unique<GPULKFeatureTracker>();
    case TrackerType::LKHorizontal:
      return std::make_unique<GPULKTrackerHorizontal>();
    default:
      TraceError("Unsupported GPU tracker type");
      return nullptr;
  }
}

MultiSOFGPU::MultiSOFGPU(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid,
                         FeaturePredictorPtr feature_predictor, const Settings& sof_settings,
                         const odom::KeyFrameSettings& keyframe_settings)
    : MultiSOFBase(rig, fid, sof_settings, keyframe_settings) {
  const auto& primary_cams = fid_.primary_cameras();

  for (CameraId primary_cam_id : primary_cams) {
    const camera::ICameraModel& intrinsics = *rig_.intrinsics[primary_cam_id];
    auto selector = std::make_unique<SelectorStereo>(sof_settings.feature_selection_settings);
    mono_sof_.emplace_back(CreateMonoSOF(Implementation::kGPU, primary_cam_id, intrinsics, std::move(selector),
                                         feature_predictor, sof_settings));

    const auto& secondary_cams = fid_.secondary_cameras(primary_cam_id);

    auto& tracker_from_secondary_cam = secondary_from_primary_sof_[primary_cam_id];
    for (CameraId secondary_cam_id : secondary_cams) {
      auto tracker_ptr = CreateGPUTracker(sof_settings.lr_tracker);
      // CreateGPUTracker returns nullptr for tracker types with no GPU implementation. Fail here
      // rather than on the first Launch: asserts are compiled out in release builds, so the null
      // would otherwise surface as a crash inside track_points.
      if (tracker_ptr == nullptr) {
        throw std::invalid_argument("MultiSOFGPU: lr_tracker type has no GPU implementation");
      }
      tracker_from_secondary_cam[secondary_cam_id].tracker = std::move(tracker_ptr);
    }
  }
}

void MultiSOFGPU::LaunchTrackingPrimaryToSecondary(CameraId primary_id, CameraId secondary_id,
                                                   const Sources& curr_sources, Images& curr_images,
                                                   const std::vector<camera::Observation>& primary_obs,
                                                   std::vector<camera::Observation>* /* secondary_obs */) {
  const ImageContextPtr primary_image = curr_images[primary_id];
  const ImageSource& secondary_source = curr_sources[secondary_id];
  const ImageContextPtr secondary_image = curr_images[secondary_id];

  const camera::ICameraModel& intrinsicsP = *rig_.intrinsics[primary_id];

  PrimaryToSecondaryGPUTracker& pair = secondary_from_primary_sof_.at(primary_id).at(secondary_id);
  assert(pair.tracker != nullptr);

  GPUArrayPinned<TrackData>& tracks_data = pair.tracks_data;
  Stream& stream = pair.stream;

  // Build secondary pyramids.
  secondary_image->build_gpu_image_pyramid(secondary_source, box_prefilter_, stream.get_stream());
  secondary_image->build_gpu_gradient_pyramid(true, stream.get_stream());

  const auto& pyr_l_gpu = primary_image->gpu_image_pyramid();
  const int top_l = static_cast<int>(pyr_l_gpu.getLevelsCount()) - 1;

  const EpipolarCurves& epipolar_curves =
      GetOrBuildEpipolarCurves(primary_id, secondary_id, top_l, pyr_l_gpu[top_l].cols(), pyr_l_gpu[top_l].rows());

  const float cross_cam_search_radius = CrossCamSearchRadius(top_l);

  const size_t n = primary_obs.size();
  auto& uvL = pair.uvL;
  auto& cands = pair.cands;
  auto& winners = pair.winners;
  uvL.resize(n);
  cands.resize(n);  // inner vectors retain their allocation across frames
  // `assign` rather than `resize`: guarantees every slot starts as a zero-init `TrackData`, so a
  // newly grown vector element cannot leak stale fields into a later `winners[i] = ...` copy.
  winners.assign(n, TrackData{});

  size_t max_candidates = 0;
  for (size_t i = 0; i < n; ++i) {
    // A point that can't be projected gets no candidates: Candidates() would otherwise leave last
    // frame's list in place, and uvL[i] still reaches the kernel as `data.track`.
    if (!intrinsicsP.denormalizePoint(primary_obs[i].xy, uvL[i])) {
      uvL[i].setZero();
      cands[i].clear();
      continue;
    }
    epipolar_curves.Candidates(uvL[i], cands[i]);
    max_candidates = std::max(max_candidates, cands[i].size());
  }

  size_t tracked_count = 0;

  // Multi-launch scan: iterate candidate index k = 0..max_candidates-1. Points that have already
  // succeeded, or that ran out of candidates, are given a benign (zero) offset — the kernel still
  // processes them (batched-launch limitation) but their results are discarded host-side. First
  // successful candidate wins, matching CPU semantics.
  for (size_t k = 0; k < max_candidates; ++k) {
    bool any_active = false;
    for (size_t i = 0; i < n; ++i) {
      TrackData& data = tracks_data[i];
      data.ncc_threshold = 0.8f;
      data.track = {uvL[i].x(), uvL[i].y()};
      data.track_status = false;
      data.search_radius_px = cross_cam_search_radius;
      // Zero info[] every iteration so a winner copy cannot inherit stale covariance from an
      // earlier k iteration for the same point.
      std::fill_n(data.info, 4, 0.f);
      const std::vector<Vector2T>& cands_i = cands[i];
      // Skip already-won points and points past the end of their candidate list. Both get the
      // sentinel `{0, 0}` offset; the winner-update loop ignores them.
      if (winners[i].track_status || k >= cands_i.size()) {
        data.offset = {0.f, 0.f};
      } else {
        const Vector2T offset = cands_i[k] - uvL[i];
        data.offset = {offset.x(), offset.y()};
        any_active = true;
      }
    }

    // Early exit: if no point still has an unclaimed real candidate, remaining k values are
    // pure sentinel work — skip them entirely.
    if (!any_active) {
      break;
    }

    tracks_data.copy_top_n(ToGPU, n, stream.get_stream());
    pair.tracker->track_points(primary_image->gpu_gradient_pyramid(), secondary_image->gpu_gradient_pyramid(),
                               primary_image->gpu_image_pyramid(), secondary_image->gpu_image_pyramid(), tracks_data, n,
                               stream.get_stream());
    tracks_data.copy_top_n(ToCPU, n, stream.get_stream());
    cudaStreamSynchronize(stream.get_stream());

    for (size_t i = 0; i < n; ++i) {
      // Skip already-won points and sentinel results (ran out of candidates).
      if (winners[i].track_status || k >= cands[i].size()) {
        continue;
      }
      if (tracks_data[i].track_status) {
        winners[i] = tracks_data[i];
        ++tracked_count;
      }
    }

    // Stop advancing candidate index once we've tracked at least kL2REarlyExitFraction of
    // observations (shared with MultiSOFCPU).
    if (static_cast<float>(tracked_count) >= kL2REarlyExitFraction * static_cast<float>(n)) {
      break;
    }
  }

  // Publish winners so GetTrackingResults sees per-point best track_status + track uv.
  for (size_t i = 0; i < n; ++i) {
    if (winners[i].track_status) {
      tracks_data[i] = winners[i];
    } else {
      tracks_data[i].track_status = false;
      tracks_data[i].track = {uvL[i].x(), uvL[i].y()};
    }
  }
  pair.was_launched = true;
}

void MultiSOFGPU::GetTrackingResults(MulticamObservations& observations) {
  std::vector<std::vector<camera::Observation>> secondary_observations(observations.size());
  const auto& primary_cams = fid_.primary_cameras();

  for (CameraId primary_id : primary_cams) {
    if (primary_id >= observations.size()) {
      continue;
    }
    const std::vector<camera::Observation>& primary_obs = observations[primary_id];
    const auto& secondary_cams = fid_.secondary_cameras(primary_id);

    for (CameraId secondary_id : secondary_cams) {
      PrimaryToSecondaryGPUTracker& pair = secondary_from_primary_sof_.at(primary_id).at(secondary_id);
      if (!pair.was_launched) {
        continue;
      }

      GPUArrayPinned<TrackData>& tracks_data = pair.tracks_data;
      Stream& stream = pair.stream;

      const camera::ICameraModel& intrinsicsS = *rig_.intrinsics[secondary_id];

      cudaStreamSynchronize(stream.get_stream());

      Matrix2T info;
      Vector2T xyR;
      Vector2T uvR;
      // secondary observations can be added, so only iterate over primary observations
      // FIXME: tracks_data.size() should return data size and not capacity, fix GPUArray[Pinned] and get rid of
      // obs_sizes
      for (size_t i = 0; i < primary_obs.size(); i++) {
        auto& data = tracks_data[i];
        const TrackId& trackId = primary_obs[i].id;
        if (data.track_status) {
          uvR << data.track.x, data.track.y;
          info << data.info[0], data.info[1], data.info[2], data.info[3];
          Matrix2T info_xy;
          if (!intrinsicsS.normalizePoint(uvR, xyR) ||
              !camera::ObservationInfoUVToXY(intrinsicsS, uvR, xyR, info, info_xy)) {
            continue;
          }

          secondary_observations[secondary_id].push_back({secondary_id, trackId, xyR, info_xy});
        }
      }
    }
  }

  for (CameraId cam_id = 0; cam_id < secondary_observations.size(); ++cam_id) {
    auto& obs_vector = secondary_observations[cam_id];
    auto& x = observations[cam_id];
    std::move(obs_vector.begin(), obs_vector.end(), std::back_inserter(x));
  }
}

void MultiSOFGPU::StartKeyframe() {
  for (auto& [_, x] : secondary_from_primary_sof_) {
    for (auto& [cam_id, pair] : x) {
      pair.was_launched = false;
    }
  }
}

void MultiSOFGPU::reset() {
  for (const auto& sof : mono_sof_) {
    sof->reset();
  }

  reset_keyframe_selector();
}

}  // namespace cuvslam::sof

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

#include "slam/slam/loop_closure_solver/lcs_simple.h"

#include <unordered_set>

#include "camera/observation.h"
#include "camera/rig.h"
#include "epipolar/camera_selection.h"
#include "epipolar/fundamental_ransac.h"

#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"
#include "slam/slam/loop_closure_solver/pnp_ransac_hypothesis.h"

namespace cuvslam::slam {

ILoopClosureSolverPtr CreateLoopClosureSolverSimple(const camera::Rig& rig, RansacType ransac_type, bool randomized,
                                                    LSIGrid::FetchStrategy fetch_strategy) {
  return new LoopClosureSolverSimple(rig, ransac_type, randomized, fetch_strategy);
}

LoopClosureSolverSimple::LoopClosureSolverSimple(const camera::Rig& rig, RansacType ransac_type, bool randomized,
                                                 LSIGrid::FetchStrategy fetch_strategy)
    : pnp_(rig), rig_(rig) {
  this->ransac_type_ = ransac_type;
  this->randomized_ = randomized;
  this->fetch_strategy_ = fetch_strategy;
}
LoopClosureSolverSimple::~LoopClosureSolverSimple() {}

bool LoopClosureSolverSimple::Solve(const LoopClosureTask& task, const LSIGrid& landmarks_spatial_index,
                                    const IFeatureDescriptorOps* feature_descriptor_ops, Isometry3T& world_from_rig,
                                    Matrix6T& pose_covariance, std::vector<LandmarkInSolver>* landmarks,
                                    DiscardLandmarkCB* discard_landmark_cb,
                                    KeyframeInSightCB* keyframe_in_sight_cb) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("lcs_simple.Solve()", profiler_color_);

  bool camera_res = false;
  // statistic
  this->ransac_per_tracked_ = 1;

  std::vector<LandmarkInfo> landmark_candidates;
  this->SelectLandmarksCandidates(task, landmarks_spatial_index, feature_descriptor_ops, landmark_candidates,
                                  discard_landmark_cb, keyframe_in_sight_cb);

  int points_count = static_cast<int>(landmark_candidates.size());
  if (points_count < 3) {
    return false;  // Surveyed tracking method operates on 3 points or more
  }

  auto& guess_world_from_rig = task.guess_world_from_rig;
  auto rig_from_world_guess = guess_world_from_rig.inverse();

  std::unordered_map<TrackId, Vector3T> pnp_landmarks;
  std::vector<camera::Observation> pnp_observations;

  if (this->ransac_type_ != RansacType::kNone) {
    std::vector<bool> landmark_candidates_ok;
    if (this->ransac_type_ == RansacType::kPnP) {
      points_count = PnpRansacFilter(rig_from_world_guess, landmark_candidates, landmark_candidates_ok);
    } else if (this->ransac_type_ == RansacType::kFundamental) {
      // TODO check if it is used!
      // remove if not
      points_count = FundamentalRansacFilter(landmark_candidates, landmark_candidates_ok);
    }
    // statistic
    this->ransac_per_tracked_ = points_count / static_cast<float>(landmark_candidates.size() + 1);

    if (points_count < 0) {
      return false;
    } else {
      // ransac is ok
      if (points_count < 3) {
        return false;  // Surveyed tracking method operates on 3 points or more
      }

      pnp_landmarks.reserve(points_count);
      pnp_observations.reserve(points_count);
      for (size_t i = 0; i < landmark_candidates.size(); i++) {
        auto& landmark_stat = landmark_candidates[i];
        if (!landmark_candidates_ok[i]) {
          if (discard_landmark_cb) {
            (*discard_landmark_cb)(landmark_stat.id, LP_RANSAC_FAILED);
          }
          continue;
        }

        pnp_landmarks.insert({landmark_stat.id, landmark_stat.xyz});
        pnp_observations.emplace_back(landmark_stat.cam_id, landmark_stat.id, landmark_stat.uv_norm,
                                      landmark_stat.info);
      }
    }
  } else {
    points_count = static_cast<int>(landmark_candidates.size());
    pnp_landmarks.reserve(points_count);
    pnp_observations.reserve(points_count);
    for (int i = 0; i < points_count; i++) {
      auto& landmark_stat = landmark_candidates[i];
      pnp_landmarks.insert({landmark_stat.id, landmark_stat.xyz});
      pnp_observations.emplace_back(landmark_stat.cam_id, landmark_stat.id, landmark_stat.uv_norm, landmark_stat.info);
    }
  }

  Matrix6T precision;
  TRACE_EVENT ev_refinepose = profiler_domain_.trace_event("RefinePose()", profiler_color_);
  camera_res =
      pnp_.solve(rig_from_world_guess, precision, pnp_observations, pnp_landmarks, pnp::PNPSettings::LCSettings());

  ev_refinepose.Pop();

  const int min_observations = 13;
  if (points_count < min_observations) {
    camera_res = false;
  }

  if (discard_landmark_cb) {
    LandmarkProbe landmark_probe = camera_res ? LP_SOLVER_OK : LP_PNP_FAILED;
    for (auto& landmark_stat : landmark_candidates) {
      (*discard_landmark_cb)(landmark_stat.id, landmark_probe);
    }
  }

  if (!camera_res) {
    return false;
  }

  // task result
  pose_covariance = precision.ldlt().solve(Matrix6T::Identity());
  world_from_rig = rig_from_world_guess.inverse();

  if (landmarks) {
    TRACE_EVENT ev_landmarks = profiler_domain_.trace_event("fetch landmarks data", profiler_color_);

    landmarks->clear();
    landmarks->reserve(landmark_candidates.size());
    // for (auto& landmark : landmark_candidates) {
    for (const auto& obs : pnp_observations) {
      LandmarkInSolver lins;
      lins.id = obs.id;
      lins.uv_norm = obs.xy;
      landmarks->push_back(lins);
    }
  }

  return camera_res;
}
// can be called after Solve()
float LoopClosureSolverSimple::GetRansacPerTracked() const { return ransac_per_tracked_; }

void LoopClosureSolverSimple::SelectLandmarksCandidates(const LoopClosureTask& task,
                                                        const LSIGrid& landmarks_spatial_index,
                                                        const IFeatureDescriptorOps* feature_descriptor_ops,
                                                        std::vector<LandmarkInfo>& landmark_candidates,
                                                        DiscardLandmarkCB* discard_landmark_cb,
                                                        KeyframeInSightCB* keyframe_in_sight_cb) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("lcs_simple.SelectLandmarksCandidates()", profiler_color_);
  // filled if keyframe_in_sight_cb is not null
  std::unordered_set<KeyFrameId> keyframes_in_sight;

  struct Track2dInputExtra {
    LandmarkId id;
    Vector3T xyz;
    Vector2T uv_guess_norm;
  };

  landmark_candidates.clear();

  LSIGrid::QueryOptions query_options;
  query_options.fetch_strategy = fetch_strategy_;

  std::vector<LandmarkId> lsi_candidate_ids;
  {
    std::vector<CameraId> cam_ids;
    for (CameraId cam_id = 0; cam_id < task.current_images.size(); ++cam_id) {
      if (task.current_images[cam_id] != nullptr) {
        cam_ids.push_back(cam_id);
      }
    }
    landmarks_spatial_index.QueryLandmarksByCameraPose(task.guess_world_from_rig, cam_ids, *task.pose_graph_hypothesis,
                                                       query_options,
                                                       [&](LandmarkId id) { lsi_candidate_ids.push_back(id); });
  }

  for (CameraId cam_id = 0; cam_id < task.current_images.size(); ++cam_id) {
    const auto& image = task.current_images[cam_id];
    if (image == nullptr) {
      continue;
    }
    auto camera_from_world_guess = rig_.camera_from_rig[cam_id] * task.guess_world_from_rig.inverse();
    auto& intrinsics = *rig_.intrinsics[cam_id];
    Matrix2T default_info = camera::ObservationInfoUVToNormUV(intrinsics, camera::GetDefaultObservationInfoUV());

    std::vector<IFeatureDescriptorOps::Track2dInput> maptoimage_input;
    std::vector<Track2dInputExtra> track2d_input_extra;
    std::vector<FeatureDescriptor> descriptors;

    for (LandmarkId id : lsi_candidate_ids) {
      const Vector3T xyz = landmarks_spatial_index.GetLandmarkOrStagedCoords(id, *task.pose_graph_hypothesis);
      const Vector3T point_guess = camera_from_world_guess * xyz;
      // same near plane the triangulator and the visibility test use — a landmark nearer than this
      // was never triangulated, and a failed lookup comes back as the origin, at zero depth
      if (!epipolar::IsDepthInFront(point_guess.z())) {
        continue;
      }
      const Vector2T uv_guess_norm = point_guess.topRows(2) / point_guess.z();
      Vector2T uv_guess;
      if (!intrinsics.denormalizePoint(uv_guess_norm, uv_guess)) {
        continue;
      }

      if (keyframe_in_sight_cb) {
        landmarks_spatial_index.QueryLandmarkRelations(id, [&](KeyFrameId kf_id, const Vector2T&) {
          keyframes_in_sight.insert(kf_id);
          return true;
        });
      }

      const FeatureDescriptor fd = landmarks_spatial_index.GetLandmarkFeatureDescriptor(id);
      if (!fd) {
        continue;
      }
      descriptors.push_back(fd);
      uint32_t fd_index = descriptors.size() - 1;

      IFeatureDescriptorOps::Track2dInput track2d_input(fd_index);
      track2d_input.has_predict_uv = true;
      track2d_input.predict_uv = uv_guess;
      maptoimage_input.push_back(track2d_input);
      track2d_input_extra.push_back({id, xyz, uv_guess_norm});
    }

    // batch tracking
    TRACE_EVENT ev_maptoimage = profiler_domain_.trace_event("MapToImage");
    std::vector<IFeatureDescriptorOps::Track2dOutput> maptoimage_output;
    feature_descriptor_ops->MapsToImage(image, descriptors, maptoimage_input, maptoimage_output);
    ev_maptoimage.Pop();

    // fill landmark_candidates
    for (size_t i = 0; i < maptoimage_input.size(); i++) {
      const auto& input_extra = track2d_input_extra[i];
      const auto& output = maptoimage_output[i];

      // IFeatureDescriptorOps::Track2dInfo info;
      bool res = output.successed;
      Vector2T uv = output.uv;

      const float ncc_threshold = 0.9f;
      if (output.ncc < ncc_threshold) {
        res = false;
      }

      if (!res) {
        if (discard_landmark_cb) {
          (*discard_landmark_cb)(input_extra.id, LP_TRACKING_FAILED);
        }
        continue;
      }

      Vector2T uv_norm(0, 0);
      if (!intrinsics.normalizePoint(uv, uv_norm)) {
        continue;
      }

      Matrix2T info_m;
      info_m << output.info[0], output.info[1], output.info[2], output.info[3];

      // convert info to normalized uv space
      Matrix2T info_m_norm = default_info;
      // info_m_norm = focal.transpose() * info_m * focal;

      LandmarkInfo landmark{input_extra.id, input_extra.xyz, cam_id, uv_norm, input_extra.uv_guess_norm,
                            output.ncc,     info_m_norm};
      landmark_candidates.push_back(landmark);
    }
  }

  // call keyframe_in_sight_cb
  if (keyframe_in_sight_cb) {
    for (auto& kf_id : keyframes_in_sight) {
      (*keyframe_in_sight_cb)(kf_id);
    }
  }
}

// return -1 if ransac unavailable
int LoopClosureSolverSimple::PnpRansacFilter(const Isometry3T& rig_from_world,
                                             const std::vector<LandmarkInfo>& landmark_candidates,
                                             std::vector<bool>& landmark_candidates_ok) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("lcs_simple.PnpRansacFilter()", profiler_color_);

  if (landmark_candidates.size() < 5) {
    return 0;
  }

  std::vector<PnpRansacTrackData> sampleSequence;
  sampleSequence.reserve(landmark_candidates.size());
  for (auto& lm : landmark_candidates) {
    sampleSequence.push_back(PnpRansacTrackData(lm.xyz, {lm.cam_id, lm.id, lm.uv_norm, lm.info}));
  }

  math::Ransac<PnpRansacHypothesis> pnpr(rig_);
  if (!randomized_) {
    pnpr.getRandomGenerator().seed(0);
  }
  pnpr.setConfidence(0.95f);
  pnpr.setInitialRigFromWorld(rig_from_world);
  pnpr.setThreshold(0.05f);  // Threshold
  Isometry3T pose;
  if (!pnpr(pose, sampleSequence.begin(), sampleSequence.end())) {
    return -1;
  }

  int inner_count = 0;
  landmark_candidates_ok.resize(landmark_candidates.size());
  for (size_t i = 0; i < landmark_candidates.size(); ++i) {
    bool inner = pnpr.isInlier(pose, sampleSequence[i]);
    landmark_candidates_ok[i] = inner;
    if (inner) {
      inner_count++;
    }
  }
  return inner_count;
}

// return -1 if ransac unavailable
int LoopClosureSolverSimple::FundamentalRansacFilter(const std::vector<LandmarkInfo>& landmark_candidates,
                                                     std::vector<bool>& landmark_candidates_ok) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("lcs_simple.FundamentalRansacFilter()", profiler_color_);

  if (landmark_candidates.size() < 10) {
    return 0;
  }

  Vector2TPairVector sampleSequence;
  for (auto& lm : landmark_candidates) {
    sampleSequence.emplace_back(lm.uv_guess_norm, lm.uv_norm);
  }

  const float RANSAC_ACCURACY_THRESHOLD = 0.002f;
  const float threshold = RANSAC_ACCURACY_THRESHOLD;

  landmark_candidates_ok.clear();
  landmark_candidates_ok.resize(landmark_candidates.size());

  Matrix3T essential;
  math::Ransac<epipolar::Fundamental> fr;
  fr.setOptions(math::Ransac<epipolar::Fundamental>::Epipolar, threshold, 0.002f);
  if (!fr(essential, sampleSequence.begin(), sampleSequence.end())) {
    return -1;
  }
  int inner_count = 0;
  for (size_t i = 0; i < landmark_candidates.size(); ++i) {
    bool inner = fr.isInlier(essential, sampleSequence[i]);
    landmark_candidates_ok[i] = inner;
    if (inner) {
      inner_count++;
    }
  }
  return inner_count;
}

}  // namespace cuvslam::slam

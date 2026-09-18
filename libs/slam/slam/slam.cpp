
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

#include "slam/slam/slam.h"

#include "camera/observation.h"
#include "common/log_types.h"
#include "common/rerun.h"

#include "slam/map/database/lmdb_slam_database.h"
#include "slam/map/spatial_index/lsi_grid.h"
#include "slam/slam/slam_check_hypothesis.h"

namespace cuvslam::slam {

LocalizerAndMapper::LocalizerAndMapper(const camera::Rig& rig, FeatureDescriptorType descriptor_type, bool use_gpu)
    : rig_(rig), map_(rig, descriptor_type, use_gpu), pnp_(rig), random_generator_(random_device_()) {}

LocalizerAndMapper::~LocalizerAndMapper() { SlamStdout("Destroyed LocalizerAndMapper instance. "); }

void LocalizerAndMapper::SetReproduceMode(bool reproduce_mode) {
  reproduce_mode_ = reproduce_mode;
  if (reproduce_mode) {
    random_generator_.seed(0);
  }
}

bool LocalizerAndMapper::GetReproduceMode() const { return reproduce_mode_; }

void LocalizerAndMapper::SetLandmarksSpatialIndex(const SpatialIndexOptions& options) {
  float size = options.cell_size;

  // default size
  if (size <= 0) {
    // default size if not set
    size = 0.25f;

    // default size based on "pupillary distance"
    // TODO: cover multi-cam setup
    if (rig_.num_cameras == 2) {
      auto left_right = rig_.camera_from_rig[0] * rig_.camera_from_rig[1].inverse();
      auto pd = left_right.translation().norm();
      // constant here
      size = pd * 4;
    }
  }

  map_.landmarks_spatial_index_ =
      std::make_shared<LSIGrid>(*map_.feature_descriptor_ops_, rig_, size, options.max_landmarks_in_cell);
}

SpatialIndexOptions LocalizerAndMapper::GetLandmarksSpatialIndexOptions() const {
  SpatialIndexOptions opt;
  opt.cell_size = map_.landmarks_spatial_index_->GetCellSize();
  opt.max_landmarks_in_cell = map_.landmarks_spatial_index_->GetMaxLandmarksInCell();

  return opt;
}

void LocalizerAndMapper::SetKeyframesLimit(int max_keyframes_count) { max_keyframes_count_ = max_keyframes_count; }

int LocalizerAndMapper::GetKeyframesLimit() const { return max_keyframes_count_; }

bool LocalizerAndMapper::SetPoseGraphOptimizerOptions(const PoseGraphOptimizerOptions& options) {
  if (options.type != Simple && options.type != Dummy) {
    return false;
  }
  pg_options_ = options;
  pose_graph_optimizer_ = options.type == Simple ? "simple" : "";
  return true;
}

PoseGraphOptimizerOptions LocalizerAndMapper::GetPoseGraphOptimizerOptions() const { return pg_options_; }

void LocalizerAndMapper::RegisterRemoveNodeCB() {
  // PoseGraph holds a single removal callback, so everything that has to follow a keyframe merge
  // is dispatched from this one lambda. Each branch checks whether its feature is switched on,
  // which lets the callback be installed before either feature is configured.
  const PoseGraph::RemoveNodeCB remove_node_cb =
      [this](const KeyFrameId keyframe_id, const KeyFrameId instead_keyframe_id, const Isometry3T& to_instead) -> void {
    if (keep_track_poses_) {
      ChangeKeyframeInfo change_keyframe_info;
      change_keyframe_info.new_node = instead_keyframe_id;
      change_keyframe_info.old_node_to_new = to_instead;
      keyframe_removed_[keyframe_id] = change_keyframe_info;
    }
    if (vpr_map_) {
      vpr_map_->OnKeyframeRemoved(keyframe_id, instead_keyframe_id);
    }
  };
  map_.pose_graph_.RegisterRemoveNodeCB(remove_node_cb);
}

void LocalizerAndMapper::SetKeepTrackPoses(bool keep_track_poses) {
  // if true - CalcFramePose() will working
  keep_track_poses_ = keep_track_poses;

  if (keep_track_poses_) {
    RegisterRemoveNodeCB();
  }
}

void LocalizerAndMapper::SetVprOptions(const vpr::VprOptions& options) {
  if (options.type == vpr::VprType::kNone) {
    vpr_map_.reset();
    return;
  }
  vpr_map_ = std::make_unique<vpr::VprMap>(options);
  RegisterRemoveNodeCB();
}

bool LocalizerAndMapper::IsVprEnabled() const { return vpr_map_ && vpr_map_->Enabled(); }

size_t LocalizerAndMapper::GetVprMapSize() const { return vpr_map_ ? vpr_map_->Size() : 0; }

bool LocalizerAndMapper::VprHeadNeedsImage() const {
  if (!IsVprEnabled()) {
    return false;
  }
  KeyFrameId head = InvalidKeyFrameId;
  if (!map_.pose_graph_.GetHeadKeyframe(head) || head == InvalidKeyFrameId) {
    return false;
  }
  return !vpr_map_->HasImage(head);
}

bool LocalizerAndMapper::AddVprFrame(const vpr::VprImage& image, const int64_t timestamp_ns) {
  if (!IsVprEnabled()) {
    return false;
  }
  KeyFrameId head = InvalidKeyFrameId;
  if (!map_.pose_graph_.GetHeadKeyframe(head) || head == InvalidKeyFrameId) {
    // Nothing has been mapped yet. The frame is dropped rather than queued: the caller offers one
    // on every frame, so the next keyframe will get an equally good picture of the same place.
    return false;
  }
  if (!vpr_map_->AddFrame(head, timestamp_ns, image)) {
    return false;
  }
  const Isometry3T* pose = map_.pose_graph_hypothesis_.GetKeyframePose(head);
  if (pose != nullptr) {
    vpr_map_->UpdateNodePose(head, *pose);
  }
  return true;
}

vpr::VprPlace LocalizerAndMapper::RecognizePlace(const vpr::VprImage& image) {
  const std::vector<vpr::VprPlace> places = RecognizePlaces(image, 1);
  return places.empty() ? vpr::VprPlace{} : places.front();
}

std::vector<vpr::VprPlace> LocalizerAndMapper::RecognizePlaces(const vpr::VprImage& image, size_t max_results) {
  std::vector<vpr::VprPlace> places;
  if (!IsVprEnabled()) {
    return places;
  }

  places = vpr_map_->QueryTopK(image, max_results);
  for (auto it = places.begin(); it != places.end();) {
    if (it->imported) {
      // An imported entry keeps the pose stored with the map it came from; this session has no pose
      // graph node for it, which is exactly the relocalization case the map was saved for.
      ++it;
      continue;
    }
    // Everything else is resolved through the pose graph at query time rather than from a stored
    // copy, so a place recognized after a loop closure reports the corrected position.
    const Isometry3T* pose = map_.pose_graph_hypothesis_.GetKeyframePose(it->node_id);
    if (pose == nullptr) {
      it = places.erase(it);  // the node was merged away between the query and this lookup
      continue;
    }
    it->pose = *pose;
    ++it;
  }
  return places;
}

bool LocalizerAndMapper::SaveVprMap(const std::string& folder) {
  if (!IsVprEnabled()) {
    return false;
  }
  map_.pose_graph_.QueryKeyframes([this](const KeyFrameId keyframe_id) {
    const Isometry3T* pose = map_.pose_graph_hypothesis_.GetKeyframePose(keyframe_id);
    if (pose != nullptr) {
      vpr_map_->UpdateNodePose(keyframe_id, *pose);
    }
  });
  return vpr_map_->Save(folder);
}

bool LocalizerAndMapper::LoadVprMap(const std::string& folder, vpr::VprMap::LoadMode mode) {
  if (!IsVprEnabled()) {
    return false;
  }
  return vpr_map_->Load(folder, mode);
}

bool LocalizerAndMapper::GetKeepTrackPoses() const { return keep_track_poses_; }

void LocalizerAndMapper::SetActiveCameras(const std::vector<CameraId>& cameras) {
  if (active_cameras_) {
    throw std::runtime_error("Set active camera should be called once before map update.");
  }
  active_cameras_ = cameras;
}

std::optional<std::vector<CameraId>> LocalizerAndMapper::GetActiveCameras() const { return active_cameras_; }

bool LocalizerAndMapper::SelectHeadKeyframe(KeyFrameId const_slam_keyframe_id, int64_t timestamp_ns) {
  return map_.pose_graph_.SelectHeadKeyframe(const_slam_keyframe_id, timestamp_ns);
}

bool LocalizerAndMapper::CalcFramePose(FrameId frame_id, Isometry3T& pose) const {
  if (!keep_track_poses_) {
    return false;
  }

  KeyFrameId keyframe_id = InvalidKeyFrameId;
  Isometry3T from_keyframe_to_frame = Isometry3T::Identity();
  FindKeyframeByFrame(frame_id, keyframe_id, from_keyframe_to_frame);

  const Isometry3T* keyframe_pose = map_.pose_graph_hypothesis_.GetKeyframePose(keyframe_id);

  if (!keyframe_pose) {
    SlamStderr("Failed to calculate frame pose for keyframe_id %zd.\n", static_cast<uint64_t>(frame_id));
    return false;
  }

  pose = (*keyframe_pose) * from_keyframe_to_frame;
  return true;
}

bool LocalizerAndMapper::FindKeyframeByFrame(FrameId frame_id, KeyFrameId& keyframe_id,
                                             Isometry3T& from_keyframe_to_frame) const {
  keyframe_id = InvalidKeyFrameId;
  int diff = INT32_MAX;

  // sourced keyframes
  for (auto it : keyframe_sources_) {
    const auto& td = it.second;

    if (frame_id < td) {
      continue;
    }

    if (diff < static_cast<int>(frame_id) - static_cast<int>(td)) {
      continue;
    }

    diff = frame_id - td;
    keyframe_id = it.first;

    if (diff == 0) {
      break;
    }
  }

  // removed keyframes
  from_keyframe_to_frame = Isometry3T::Identity();
  for (;;) {
    auto it = keyframe_removed_.find(keyframe_id);
    if (it == keyframe_removed_.end()) {
      // key exists
      break;
    }
    auto& cki = it->second;
    keyframe_id = cki.new_node;
    from_keyframe_to_frame = cki.old_node_to_new * from_keyframe_to_frame;
  }

  return keyframe_id != InvalidKeyFrameId;
}

// current estimated pose
Isometry3T LocalizerAndMapper::GetCurrentPose() const {
  KeyFrameId head_keyframe;
  if (!map_.GetPoseGraph().GetHeadKeyframe(head_keyframe)) {
    return Isometry3T::Identity();
  }
  const Isometry3T* may_be_pose = map_.GetPoseGraphHypothesis().GetKeyframePose(head_keyframe);
  if (may_be_pose) {
    return *may_be_pose;
  }
  return Isometry3T::Identity();
}

bool LocalizerAndMapper::FlushActiveDatabase() const {
  if (!map_.database_) {
    return false;
  }
  TRACE_EVENT ev = profiler_domain_.trace_event("FlushActiveDatabase()", profiler_color_);
  map_.database_->SetSingleton(SlamDatabaseSingleton::SpatialIndex, 0, [&](BlobWriter& blob_writer) {
    return map_.landmarks_spatial_index_->ToBlob(blob_writer);
  });

  map_.database_->Flush();
  return true;
}

bool LocalizerAndMapper::AttachToExistingReadOnlyDatabase(const std::string& path) {
#ifdef USE_LMDB
  const auto lmdb = std::make_shared<LmdbSlamDatabase>();

  const char* url = path.c_str();
  if (!lmdb->Open(url, LmdbSlamDatabase::OpenMode::READ_ONLY_EXISTS)) {
    SlamStderr("Failed to open Slam Database %s.\n", url);
    return false;
  }
  SlamStdout("Successfully opened Slam Database %s.\n", url);
  if (!map_.AttachDatabase(lmdb, true)) {
    SlamStderr("Failed to attach to existing read only database \"%s\"", url);
    return false;
  }
  return true;
#else
  SlamStderr("AttachToExistingReadOnlyDatabase is not implemented for databases other than LMDB.");
  return false;
#endif
}

bool LocalizerAndMapper::AttachToNewDatabase(const std::string& path) {
#ifdef USE_LMDB
  auto lmdb = std::make_shared<LmdbSlamDatabase>();
  const char* url = path.c_str();
  if (!lmdb->Open(url, LmdbSlamDatabase::OpenMode::READ_WRITE_HARD_RESET)) {
    SlamStderr("Failed to open Slam Database %s.\n", url);
    return false;
  }
  if (!map_.AttachDatabase(lmdb, false)) {
    SlamStderr("Failed to attach database \"%s\".\n", url);
    return false;
  }
  SlamStdout("Successfully opened Slam Database %s for read-write.\n", url);
  return true;
#else
  SlamStderr("AttachToNewDatabase is not implemented for databases other than LMDB.");
  return false;
#endif
}

bool LocalizerAndMapper::AttachToNewDatabaseSaveMapAndDetach(const std::string& path) {
#ifdef USE_LMDB
  if (!AttachToNewDatabase(path)) {
    return false;
  }
  map_.DetachDatabase(true);
  SlamStdout("Successfully copied Slam Database %s.\n", path.c_str());
  return true;
#else
  SlamStderr("AttachToNewDatabaseSaveMapAndDetach is not implemented for databases other than LMDB.");
  return false;
#endif
}

void LocalizerAndMapper::DetachDatabase() { map_.DetachDatabase(false); }

void LocalizerAndMapper::DetectLoopClosure(const ILoopClosureSolver& loop_closure_solver, const Images& images,
                                           const Isometry3T& world_from_rig_guess, LoopClosureStatus& status) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("LC()", profiler_color_);

  status = LoopClosureStatus();
  status.success = false;

  SlamCheckTask task;
  task.loop_closure_task.current_images = images;
  task.loop_closure_task.pose_graph_hypothesis = map_.pose_graph_hypothesis_.MakeCopy();
  task.loop_closure_task.guess_world_from_rig = world_from_rig_guess;
  map_.pose_graph_.GetHeadKeyframe(task.loop_closure_task.pose_graph_head);
  const LSIGrid& lsi = *map_.landmarks_spatial_index_;
  SlamCheckHypothesis(lsi, &loop_closure_solver, rig_, map_.feature_descriptor_ops_.get(), task);

  status.keyframes_in_sight.assign(task.keyframes_in_sight.begin(), task.keyframes_in_sight.end());
  status.success = task.succesed;
  status.result_pose = task.result_pose;
  status.result_pose_covariance = task.result_pose_covariance;
  status.good_landmarks_count = task.landmarks.size();
  status.pnp_landmarks_count = task.landmarks.size() + task.probes_types[LP_PNP_FAILED];
  status.tracked_landmarks_count =
      task.landmarks.size() + task.probes_types[LP_PNP_FAILED] + task.probes_types[LP_RANSAC_FAILED];
  status.selected_landmarks_count = task.landmarks.size() + task.probes_types[LP_PNP_FAILED] +
                                    task.probes_types[LP_RANSAC_FAILED] + task.probes_types[LP_TRACKING_FAILED];
  status.landmarks = task.landmarks;
  status.discarded_landmarks = task.discarded_landmarks;
}

// Update Landmark Statistic in spatial index
void LocalizerAndMapper::UpdateLandmarkProbeStatistics(
    const std::vector<std::pair<LandmarkId, LandmarkProbe>>& discarded_landmarks) const {
  for (auto& discarded_landmark : discarded_landmarks) {
    map_.landmarks_spatial_index_->AddLandmarkProbeStatistic(discarded_landmark.first, discarded_landmark.second);
  }
}

bool LocalizerAndMapper::ApplyLoopClosureResult(const Isometry3T& world_from_lc, const Matrix6T& lc_pose_covariance,
                                                const std::vector<LandmarkInSolver>& lc_landmarks) {
  KeyFrameId headkf;
  if (!map_.pose_graph_.GetHeadKeyframe(headkf)) {
    return false;
  }
  PoseGraphEdgeStat edge_stat;
  const KeyFrameId lckf = FindKeyframeWithMostLandmarks(lc_landmarks, &edge_stat);
  if (lckf == InvalidKeyFrameId) {
    return false;
  }

  const Isometry3T* world_from_headkf = map_.pose_graph_hypothesis_.GetKeyframePose(headkf);
  if (!world_from_headkf) {
    return false;
  }
  const Isometry3T* world_from_lckf = map_.pose_graph_hypothesis_.GetKeyframePose(lckf);
  if (!world_from_lckf) {
    return false;
  }

  const Isometry3T headkf_from_world = world_from_headkf->inverse();
  const Isometry3T& world_from_estimate = GetCurrentPose();
  const Isometry3T headkf_from_estimate = headkf_from_world * world_from_estimate;
  const Isometry3T estimate_from_head = headkf_from_estimate.inverse();
  const Isometry3T world_from_correctedheadkf = world_from_lc * estimate_from_head;  // lc correction lc ~= estimate
  const Isometry3T lckf_from_world = world_from_lckf->inverse();
  const Isometry3T lckf_from_correctedheadkf = lckf_from_world * world_from_correctedheadkf;

  if (!AddEdgeToPoseGraph(lckf, headkf, lckf_from_correctedheadkf, lc_pose_covariance, edge_stat)) {
    return false;
  }
  // Add Landmark Relation to spatial index and pose graph
  for (auto& landmark_in_solver : lc_landmarks) {
    const bool valid = map_.pose_graph_.AddLandmarkRelation(landmark_in_solver.id, headkf);
    if (valid) {
      map_.landmarks_spatial_index_->AddLandmarkRelation(landmark_in_solver.id, headkf, nullptr,
                                                         landmark_in_solver.uv_norm, map_.pose_graph_hypothesis_);
    }
  }
  return true;
}

// Optimize
bool LocalizerAndMapper::OptimizePoseGraph(bool planar_constraints) {
  TRACE_EVENT ev1 = profiler_domain_.trace_event("Optimize()", profiler_color_);

  if (pose_graph_optimizer_.empty()) {
    return false;
  }

  Isometry3T vo_to_head;
  if (!map_.pose_graph_.Optimize(map_.pose_graph_hypothesis_, map_.pose_graph_hypothesis_for_swap_, planar_constraints,
                                 vo_to_head)) {
    return false;
  }

  // Update cells in pose graph
  {
    TRACE_EVENT ev2 = profiler_domain_.trace_event("update cells in pose graph", profiler_color_);
    map_.pose_graph_.QueryKeyframePoses([&](KeyFrameId keyframe_id, const Isometry3T& pose) {
      const auto* pose_src = map_.pose_graph_hypothesis_.GetKeyframePose(keyframe_id);
      const auto* pose_dst = map_.pose_graph_hypothesis_for_swap_.GetKeyframePose(keyframe_id);
      if (pose_src && pose_dst) {
        map_.landmarks_spatial_index_->OnUpdateKeyframePose(keyframe_id, (*pose_src) * pose, (*pose_dst) * pose);
      }
    });
    map_.landmarks_spatial_index_->OnUpdateKeyframePoseFinished();
  }

  // ok, so use new poses
  map_.pose_graph_hypothesis_.swap(map_.pose_graph_hypothesis_for_swap_);

  if (map_.database_) {
    map_.pose_graph_hypothesis_.PutToDatabase(map_.database_.get());
  }
  return true;
}

// callback for landmarks_spatial_index_->RemoveDeadLandmarks()
void LocalizerAndMapper::RemoveLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id) {
  // Remove pose graph links
  map_.pose_graph_.RemoveLandmarkRelation(landmark_id, keyframe_id);
}

void LocalizerAndMapper::RebuildSpatialIndex() {
  TRACE_EVENT ev = profiler_domain_.trace_event("RebuildSpatialIndex()", profiler_color_);

  map_.landmarks_spatial_index_->RebuildAllGridCells(map_.pose_graph_hypothesis_);

  std::function func_remove_from_keyframe = [&](LandmarkId landmark_id, KeyFrameId keyframe_id) {
    this->RemoveLandmarkRelation(landmark_id, keyframe_id);
  };
  map_.landmarks_spatial_index_->ReduceLandmarks();

  map_.landmarks_spatial_index_->RemoveDeadLandmarks(func_remove_from_keyframe);
}

KeyFrameId LocalizerAndMapper::FindKeyframeWithMostLandmarks(const std::vector<LandmarkInSolver>& landmarks,
                                                             PoseGraphEdgeStat* edge_stat) const {
  KeyFrameId pose_graph_head = InvalidKeyFrameId;
  map_.pose_graph_.GetHeadKeyframe(pose_graph_head);  // can set to InvalidKeyFrameId

  std::map<KeyFrameId, PoseGraphEdgeStat> keyframes_of_landmarks;

  for (const auto& landmark_in_solver : landmarks) {
    map_.landmarks_spatial_index_->QueryLandmarkRelations(landmark_in_solver.id,
                                                          [&](KeyFrameId kf_id, const Vector2T&) {
                                                            if (kf_id == pose_graph_head) {
                                                              return true;
                                                            }

                                                            auto& counter = keyframes_of_landmarks[kf_id];
                                                            counter.tracks3d_number++;
                                                            return true;
                                                          });
  }

  if (keyframes_of_landmarks.empty()) {
    return InvalidKeyFrameId;
  }

  using keyframesOfLandmarksKeyValue = std::pair<KeyFrameId, PoseGraphEdgeStat>;
  auto max_it = std::max_element(keyframes_of_landmarks.begin(), keyframes_of_landmarks.end(),
                                 [&](const keyframesOfLandmarksKeyValue& a1, const keyframesOfLandmarksKeyValue& a2) {
                                   float w1 = a1.second.tracks3d_number;
                                   float w2 = a2.second.tracks3d_number;

                                   // TODO: remove hack
                                   auto exist_stat1 = map_.pose_graph_.GetEdgeStatistic(a1.first, pose_graph_head);

                                   if (exist_stat1) {
                                     w1 = w1 * 0.5f;

                                     if (exist_stat1->Weight() < a1.second.Weight()) {
                                       w1 = w1 * 0.5f;
                                     }
                                   }

                                   auto exist_stat2 = map_.pose_graph_.GetEdgeStatistic(a2.first, pose_graph_head);

                                   if (exist_stat2) {
                                     w2 = w2 * 0.5f;

                                     if (exist_stat2->Weight() < a2.second.Weight()) {
                                       w2 = w2 * 0.5f;
                                     }
                                   }

                                   return w1 < w2;
                                 });
  const KeyFrameId max_landmarks_keyframe_id = max_it->first;

  if (edge_stat) {
    *edge_stat = max_it->second;
  }

  return max_landmarks_keyframe_id;
}

bool LocalizerAndMapper::AddEdgeToPoseGraph(const KeyFrameId start, KeyFrameId end, const Isometry3T& start_from_end,
                                            const Matrix6T& start_from_end_covariance, const PoseGraphEdgeStat& stat) {
  TRACE_EVENT ev = profiler_domain_.trace_event("AddEdgeToPoseGraph()", profiler_color_);

  // discard if existed edges has more weight
  const auto statistic_exists_edge = map_.pose_graph_.GetEdgeStatistic(start, end);

  if (statistic_exists_edge) {
    if (statistic_exists_edge->Weight() > stat.Weight()) {
      return false;
    }
  }

  // Add Edge to pose graph
  map_.pose_graph_.AddEdge(map_.pose_graph_hypothesis_, start, end, start_from_end, start_from_end_covariance, &stat);
  return true;
}

// Reduce landmarks number
bool LocalizerAndMapper::ReduceLandmarks() {
  TRACE_EVENT ev = profiler_domain_.trace_event("ReduceLandmarks()", profiler_color_);

  std::function<void(LandmarkId, KeyFrameId)> func_remove_from_keyframe =
      [&](LandmarkId landmark_id, KeyFrameId keyframe_id) { this->RemoveLandmarkRelation(landmark_id, keyframe_id); };
  map_.landmarks_spatial_index_->ReduceLandmarks();
  map_.landmarks_spatial_index_->RemoveDeadLandmarks(func_remove_from_keyframe);

  FlushActiveDatabase();
  return true;
}

// Reduce keyframes count
void LocalizerAndMapper::ReduceKeyframes() {
  if (max_keyframes_count_ == 0) {
    return;  // special value - unlimited pose graph
  }

  std::function update_landmark_keyframe = [&](KeyFrameId old, KeyFrameId neo, const std::vector<LandmarkId>& landmarks,
                                               const Isometry3T& transform_old_to_new) {
    // change keyframe of landmarks
    for (const auto landmark_id : landmarks) {
      map_.landmarks_spatial_index_->UpdateLandmarkRelation(landmark_id, old, neo, transform_old_to_new);
    }
  };

  while (map_.pose_graph_.GetKeyframeCount() > max_keyframes_count_) {
    // select not standalone most confident edge
    const EdgeId edge_id = map_.pose_graph_.GetSmallestVarianceEdgeId();
    if (edge_id != InvalidEdgeId) {
      map_.pose_graph_.ReduceSingleEdge(edge_id, map_.pose_graph_hypothesis_, update_landmark_keyframe);
    }
  }
  RERUN(map_.logPoseGraph);
}

const Map& LocalizerAndMapper::GetMap() const { return map_; }

bool LocalizerAndMapper::GetLastKeyframePoseAndTimestamp(Isometry3T& last_keyframe_pose,
                                                         int64_t& last_keyframe_ts) const {
  const PoseGraph& pg = map_.GetPoseGraph();

  KeyFrameId head_keyframe_id;
  if (!pg.GetHeadKeyframe(head_keyframe_id)) {
    return false;
  }

  const Isometry3T* p_head_keyframe_pose = map_.GetPoseGraphHypothesis().GetKeyframePose(head_keyframe_id);
  if (!p_head_keyframe_pose) {
    return false;
  }
  const PoseGraphKeyFrame& head_keyframe = pg.GetKeyframe(head_keyframe_id);
  last_keyframe_pose = *p_head_keyframe_pose;
  last_keyframe_ts = head_keyframe.keyframe_info.timestamp_ns;
  return true;
}

// calc covariation from prev VO keyframe
bool LocalizerAndMapper::CalcBetweenPose(
    KeyFrameId from, KeyFrameId to,
    const std::vector<VOFrameData::Track2DXY>& tracks2d_norm,  // normalized coordinates
    const std::map<TrackId, Vector3T>& tracks3d_rel,           // xyz in camera space
    Isometry3T& pose, Matrix6T& covariance) const {
  // xyz in "from" keyframe space
  // uv in "to" keyframe space
  // if keyframe==InvalidKeyframeId - use tracks2d_norm and tracks3d_rel

  std::unordered_map<TrackId, Vector3T> pnp_landmarks;
  std::vector<camera::Observation> pnp_observations;
  pnp_observations.reserve(tracks2d_norm.size());

  // extract tracks exists in staging3d_
  for (auto& track_xy : tracks2d_norm) {
    auto& id = track_xy.track_id;
    auto& last_uv_norm = track_xy.xy;
    auto it_3d = tracks3d_rel.find(id);

    if (it_3d == tracks3d_rel.end()) {
      continue;
    }

    auto& last_xyz = it_3d->second;

    auto it_staging = staging3d_.find(id);

    if (it_staging == staging3d_.end()) {
      continue;  // discard new tracks
    }

    auto& staging = it_staging->second;

    Vector3T xyz;

    if (from == InvalidKeyFrameId) {
      xyz = last_xyz;
    } else {
      auto kfd = staging.FindKeyframe(from);

      if (!kfd) {
        continue;
      }

      xyz = kfd->xyz_rel;
    }

    Vector2T uv_norm;

    if (to == InvalidKeyFrameId) {
      uv_norm = last_uv_norm;
    } else {
      auto kfd = staging.FindKeyframe(to);

      if (!kfd) {
        continue;
      }

      uv_norm = kfd->uv_norm;
    }

    TrackId track_id = pnp_landmarks.size();
    pnp_landmarks.insert({track_id, xyz});

    const camera::ICameraModel& intrinsics = *this->rig_.intrinsics[track_xy.cam_id];
    pnp_observations.emplace_back(track_xy.cam_id, track_id, uv_norm,
                                  camera::ObservationInfoUVToNormUV(intrinsics, camera::GetDefaultObservationInfoUV()));
  }

  Isometry3T rig_from_world_estimate = Isometry3T::Identity();
  Matrix6T precision;
  bool res =
      pnp_.solve(rig_from_world_estimate, precision, pnp_observations, pnp_landmarks, pnp::PNPSettings::LCSettings());

  if (!res) {
    return false;
  }

  pose = rig_from_world_estimate.inverse();
  // Eigen::CompleteOrthogonalDecomposition<Matrix6T> psInv(precision);
  covariance = precision.ldlt().solve(Matrix6T::Identity());

  return true;
}

const LocalizerAndMapper::TrackOnKeyframe* LocalizerAndMapper::StagingTrack3d::FindKeyframe(
    KeyFrameId keyframe_id) const {
  for (auto& kf : keyframes)
    if (kf.keyframe_id == keyframe_id) {
      return &kf;
    }
  return nullptr;
}

}  // namespace cuvslam::slam

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

#include "slam/map/spatial_index/lsi_grid.h"

#include <iterator>

#include "camera/rig.h"
#include "common/log_types.h"
#include "common/rerun.h"
#include "epipolar/camera_selection.h"

#include "slam/common/blob_eigen.h"
#include "slam/common/slam_log_types.h"

#ifdef USE_RERUN
#include "slam/map/spatial_index/visualizer.h"
#endif

namespace {
using namespace cuvslam;

bool IsLandmarkInFrustum(const camera::ICameraModel& intrinsics, const Isometry3T& cam_from_world,
                         const Vector3T& landmark_xyz) {
  const Vector3T point = cam_from_world * landmark_xyz;

  // Behind camera or too close for stable projection (+Z forward, OpenCV).
  if (!epipolar::IsDepthInFront(point.z())) {
    return false;
  }

  const auto one_over_z = 1.f / point.z();
  const Vector2T xy = point.topRows(2) * one_over_z;

  // if (xy.x() < -1 || xy.x() > 1) {
  //     return false;
  // }
  // if (xy.y() < -1 || xy.y() > 1) {
  //     return false;
  // }

  Vector2T uv;
  if (!intrinsics.denormalizePoint(xy, uv)) {
    return false;
  }

  auto res = intrinsics.getResolution();
  if (uv.x() < 0 || uv.x() > res.x()) {
    return false;
  }
  if (uv.y() < 0 || uv.y() > res.y()) {
    return false;
  }

  return true;
}

bool IsLandmarkInAnyFrustum(const camera::Rig& rig, const std::vector<CameraId>& cam_ids,
                            const std::unordered_map<CameraId, Isometry3T>& cams_from_world,
                            const Vector3T& landmark_xyz) {
  for (const CameraId cam_id : cam_ids) {
    const camera::ICameraModel& intrinsics = *rig.intrinsics[cam_id];
    const Isometry3T& cam_from_world = cams_from_world.at(cam_id);
    if (IsLandmarkInFrustum(intrinsics, cam_from_world, landmark_xyz)) {
      return true;
    }
  }
  return false;
}

}  // namespace

namespace cuvslam::slam {

LSIGrid::GridLandmark::GridLandmark() { probes_types.fill(0); }

bool LSIGrid::Cell::Add(LandmarkId landmark_id) {
  auto it = std::lower_bound(this->landmarks_in_cell.begin(), this->landmarks_in_cell.end(), landmark_id);
  if (it != this->landmarks_in_cell.end() && *it == landmark_id) {
    return false;  // already in list
  }
  // insert
  this->landmarks_in_cell.insert(it, landmark_id);
  return true;
}
void LSIGrid::Cell::Remove(LandmarkId landmark_id) {
  auto it = std::lower_bound(this->landmarks_in_cell.begin(), this->landmarks_in_cell.end(), landmark_id);
  if (it == this->landmarks_in_cell.end() || *it != landmark_id) {
    return;  // not in list
  }
  // delete
  this->landmarks_in_cell.erase(it);
}

LSIGrid::LSIGrid(const IFeatureDescriptorOps& feature_descriptor_ops, const camera::Rig& rig, float cell_size,
                 int cell_landmarks_limit)
    : feature_descriptor_ops_(feature_descriptor_ops),
      rig_(rig),
      cell_size_(cell_size),
      cell_landmarks_limit_(cell_landmarks_limit),
      landmark_quality_func_(MakeLandmarkQualityFunc()) {}

LSIGrid::~LSIGrid() = default;

float LSIGrid::GetCellSize() const { return cell_size_; }
size_t LSIGrid::GetMaxLandmarksInCell() const { return cell_landmarks_limit_; }

void LSIGrid::SetDatabase(std::shared_ptr<ISlamDatabase> database) { this->database_ = database; }

void LSIGrid::SyncDatabase(const PoseGraphHypothesis& pose_graph_hypothesis) {
  if (!this->database_) {
    return;
  }

  //
  std::sort(dead_landmarks_.begin(), dead_landmarks_.end());
  auto landmark_to_db = [&](LandmarkId landmark_id, GridLandmark& landmark) {
    // don't save the dead landmarks
    auto it_dead_landmarks = std::lower_bound(dead_landmarks_.begin(), dead_landmarks_.end(), landmark_id);
    if (it_dead_landmarks != dead_landmarks_.end() && *it_dead_landmarks == landmark_id) {
      return;
    }
    if (landmark.fd) {
      // move descriptor to DB
      this->database_->SetRecord(SlamDatabaseTable::Descriptors, landmark_id, 0, [&](BlobWriter& blob_writer) {
        feature_descriptor_ops_.ToBlob(landmark.fd, blob_writer);
        return true;
      });

      // Do test reading descriptor from DB
      if (false) {
        this->database_->GetRecord(SlamDatabaseTable::Descriptors, landmark_id, [&](const BlobReader& blob_reader) {
          auto fd = feature_descriptor_ops_.CreateFromBlob(blob_reader);
          if (fd) {
            if (feature_descriptor_ops_.Compare(fd, landmark.fd) != 0) {
              SlamStderr("Incorrectly written descriptor to database for landmark in LSI grid.\n");
            };
          }
          return true;
        });
      }
      landmark.have_to_get_descriptor_from_db = true;
    }
    // move landmark to DB
    this->database_->SetRecord(SlamDatabaseTable::Landmarks, landmark_id, 0, [&](BlobWriter& blob_writer) {
      LandmarkToBlob(landmark, blob_writer);
      return true;
    });
  };

  // move all landmarks to DB
  for (auto& it_landmarks : landmarks_) {
    const auto landmark_id = it_landmarks.first;
    landmark_to_db(landmark_id, it_landmarks.second);
  }
  TraceDebug("saving before: cells num = %d", cells_.size());
  std::set<HashCellId> cells_to_reduce;
  for (auto& [landmark_id, staged_landmark] : staged_landmarks_) {
    landmark_to_db(landmark_id, staged_landmark);

    if (staged_landmark.keyframes.empty()) {
      continue;
    }

    auto& landmark = landmarks_[landmark_id];
    landmark = staged_landmark;

    // add to cells
    std::vector<HashCellId> hash_cell_ids;
    GetKeyframeCells(landmark, InvalidKeyFrameId, hash_cell_ids, pose_graph_hypothesis);
    for (auto hash_cell_id : hash_cell_ids) {
      if (AddLandmarkToCell(landmark_id, hash_cell_id)) {
        cells_to_reduce.insert(hash_cell_id);
      }
    }
  }

  for (auto hash_cell_id : cells_to_reduce) {
    ReduceLandmarksInCell(hash_cell_id, landmark_quality_func_, cell_landmarks_limit_);
  }

  // move all cells to DB
  TraceDebug("saving after: cells num = %d", cells_.size());
  for (auto& it_cells : cells_) {
    database_->SetRecord(SlamDatabaseTable::SpatialCells, it_cells.first, 0, [&](BlobWriter& blob_writer) {
      const auto& cell = it_cells.second;
      blob_writer.write_std(cell.landmarks_in_cell);
      return true;
    });
  }

  landmarks_.clear();
  staged_landmarks_.clear();
  cells_.clear();

  database_->SetSingleton(SlamDatabaseSingleton::SpatialIndex, 0,
                          [&](BlobWriter& blob_writer) { return this->ToBlob(blob_writer); });
}

// read all data from DB
void LSIGrid::SyncDatabaseReverse() {
  if (!this->database_) {
    return;
  }
  // Copy all Landmarks from DB
  this->database_->ForEach(SlamDatabaseTable::Landmarks, [&](LandmarkId landmark_id, const BlobReader&) -> bool {
    this->GetLandmark(landmark_id);
    return true;
  });

  // read all descriptors from DB
  this->database_->ForEach(SlamDatabaseTable::Descriptors,
                           [&](LandmarkId landmark_id, const BlobReader& /* blob_reader */) -> bool {
                             LSIGrid::GridLandmark* landmark = this->GetLandmark(landmark_id);
                             if (landmark) {
                               const FeatureDescriptor fd = GetLandmarkFeatureDescriptor(landmark_id);
                               if (fd && fd.is_const_memory()) {
                                 const FeatureDescriptor fd_mem = feature_descriptor_ops_.Copy(fd);
                                 landmark->have_to_get_descriptor_from_db = false;
                                 landmark->fd = fd_mem;
                               }
                             }
                             return true;
                           });

  // read all cells
  this->database_->ForEach(SlamDatabaseTable::SpatialCells,
                           [&](HashCellId hash_cell_id, const BlobReader& blob_reader) -> bool {
                             auto it = cells_.find(hash_cell_id);
                             if (it == cells_.end()) {
                               // Add or read
                               auto& cell = cells_[hash_cell_id];
                               if (!blob_reader.read_std(cell.landmarks_in_cell)) {
                               };
                             }
                             return true;
                           });
}

// add or update landmark
LandmarkId LSIGrid::AddLandmark(const FeatureDescriptor& fd, int64_t timestamp_ns) {
  if (!fd) {
    return InvalidLandmarkId;
  }
  auto landmark_id = next_landmark_auto_id_++;
  auto& landmark = this->staged_landmarks_[landmark_id];
  landmark.fd = fd;
  landmark.timestamp_ns = timestamp_ns;
  return landmark_id;
}

// add landmark relation
void LSIGrid::AddLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id,
                                  const Vector3T* xyz_in_keyframe,  // can be null
                                  const Vector2T& uv_norm, const PoseGraphHypothesis& pose_graph_hypothesis) {
  auto landmark_ptr = GetLandmarkOrStaged(landmark_id);
  if (!landmark_ptr) {
    return;
  }
  auto& landmark = *landmark_ptr;

  LandmarkInKeyframe kp;
  kp.has_link = false;
  kp.keyframe_id = keyframe_id;
  if (xyz_in_keyframe) {
    kp.xyz_in_keyframe = *xyz_in_keyframe;
    kp.has_xyz = true;
    // temp: find good way to select main keyframe
    if (landmark.GetKeyFrame() == InvalidKeyFrameId) {
      kp.has_link = true;
    }
  }
  kp.uv_norm_in_keyframe = uv_norm;
  landmark.AddLandmarkInKeyframe(kp);

  // Fill landmark cells
  {
    std::vector<HashCellId> hash_cell_ids;
    this->GetKeyframeCells(landmark, kp.keyframe_id, hash_cell_ids, pose_graph_hypothesis);
    for (auto hash_cell_id : hash_cell_ids) {
      this->AddLandmarkToCell(landmark_id, hash_cell_id);
    }
  }
}

// On merge keyframes. Change keyframe id in landmark
void LSIGrid::UpdateLandmarkRelation(LandmarkId landmark_id, KeyFrameId old_keyframe_id, KeyFrameId new_keyframe_id,
                                     const Isometry3T& transform_old_to_new) {
  auto landmark_ptr = GetLandmarkOrStaged(landmark_id);
  if (!landmark_ptr) {
    return;
  }
  auto& landmark = *landmark_ptr;

  // change all old_keyframe_id to new_keyframe_id
  // Move xyz_in_keyframe and eye_in_keyframe to new space
  for (auto& landmark_in_keyframe : landmark.keyframes) {
    if (landmark_in_keyframe.keyframe_id == old_keyframe_id) {
      landmark_in_keyframe.xyz_in_keyframe = transform_old_to_new * landmark_in_keyframe.xyz_in_keyframe;
      landmark_in_keyframe.eye_in_keyframe = transform_old_to_new * landmark_in_keyframe.eye_in_keyframe;
      landmark_in_keyframe.keyframe_id = new_keyframe_id;
    }
  }
}

void LSIGrid::OnUpdateKeyframePose(KeyFrameId keyframe_id, const Isometry3T& pose_old, const Isometry3T& pose_new) {
  // calc "leaved cells" and "incoming cells"
  // all landmarks of keyframe add to "new cells"

  Vector3T eye_old = pose_old.translation();
  Vector3T eye_new = pose_new.translation();

  // get all cells for eye
  std::vector<HashCellId> hash_cell_id_old, hash_cell_id_new;
  this->HashCellIdFromEye(eye_old, hash_cell_id_old);
  this->HashCellIdFromEye(eye_new, hash_cell_id_new);

  if (hash_cell_id_old[0] == hash_cell_id_new[0]) {
    // nothing to do
    return;
  }

  // add record to copy landmarks from hash_cell_id_old to hash_cell_id_new
  size_t sz = std::min(hash_cell_id_old.size(), hash_cell_id_new.size());
  for (size_t i = 0; i < sz; i++) {
    update_keyframe_pose_items_.push_back(
        UpdateKeyframePoseItem{keyframe_id, hash_cell_id_old[i], hash_cell_id_new[i]});
  }
}
void LSIGrid::OnUpdateKeyframePoseFinished() {
  if (update_keyframe_pose_items_.empty()) {
    return;
  }
  log::Message<LogFrames>(log::kInfo, "OnUpdateKeyframePoseFinished %zd", update_keyframe_pose_items_.size());

  for (auto& it : update_keyframe_pose_items_) {
    // copy landmarks that is in this cell and is related to this keyframe to new cell
    // auto& cell_old = this->cells_[it.hash_cell_id_old];
    auto start_end_landmarks = GetCellLandmarks(it.hash_cell_id_old);
    auto& cell_new = this->GetCellForWrite(it.hash_cell_id_new);

    for (auto it_landmark_old_id = start_end_landmarks.first; it_landmark_old_id != start_end_landmarks.second;
         it_landmark_old_id++) {
      auto landmark_id = *it_landmark_old_id;
      auto landmark_it = this->landmarks_.find(landmark_id);
      if (landmark_it == this->landmarks_.end()) {
        continue;
      }
      auto lm_kf = landmark_it->second.GetLandmarkInKeyframe(it.keyframe_id);
      if (!lm_kf) {
        continue;
      }
      cell_new.Add(landmark_id);
    }
  }

  update_keyframe_pose_items_.clear();
};

void LSIGrid::AddLandmarkProbeStatistic(LandmarkId landmark_id, LandmarkProbe landmark_probe) {
  auto landmark_ptr = GetLandmark(landmark_id);
  if (!landmark_ptr) {
    return;
  }

  landmark_ptr->probes_types[landmark_probe]++;
};

// on each vo camera pose
void LSIGrid::MoveReadyStagedLandmarksToLSI(const Isometry3T& world_from_rig, const std::vector<CameraId>& cam_ids,
                                            const PoseGraphHypothesis& pose_graph_hypothesis,
                                            KeyFrameId pose_graph_head, const int64_t current_timestamp_ns) {
  TRACE_EVENT ev = profiler_domain_.trace_event("MoveReadyStagedLandmarksToLSI()", profiler_color_);

  std::unordered_map<CameraId, Isometry3T> cams_from_world;
  {
    cams_from_world.reserve(cam_ids.size());

    const Isometry3T rig_from_world = world_from_rig.inverse();
    for (CameraId cam_id : cam_ids) {
      cams_from_world[cam_id] = rig_.camera_from_rig[cam_id] * rig_from_world;
    }
  }

  // activate = move staged to worked if not in Frustum and not relate to pose_graph_head
  std::set<HashCellId> cells_to_reduce;
  for (auto it = staged_landmarks_.begin(); it != staged_landmarks_.end();) {
    const LandmarkId id = it->first;
    const GridLandmark& staged_landmark = it->second;

    const Vector3T staged_landmark_xyz = GetLandmarkCoords(pose_graph_hypothesis, staged_landmark);
    bool activate = true;
    activate &= !IsLandmarkInAnyFrustum(rig_, cam_ids, cams_from_world, staged_landmark_xyz);
    activate &= !GetLandmarkRelation(id, pose_graph_head);

    if (pose_graph_head == InvalidKeyFrameId) {
      // lost tracking: activate immediately
      activate = true;
    }
    // TODO: remove hack
    // activate &= !GetLandmarkRelation(id, pose_graph_head-1);

    if (current_timestamp_ns - staged_landmark.timestamp_ns > max_staged_landmark_lifetime_ns_) {
      // In stationary cases, there are only landmarks within the frustum,
      // force the landmarks activation in order to move them to the lsi_grid
      activate = true;
    }
    if (!activate) {
      ++it;
      continue;
    }

    if (staged_landmark.keyframes.empty()) {
      // ignore landmark without relations
      it = staged_landmarks_.erase(it);
      continue;
    }

    // activate
    GridLandmark& landmark = landmarks_[id];  // add new landmark
    landmark = staged_landmark;
    // move descriptor to DB
    if (database_) {
      database_->SetRecord(SlamDatabaseTable::Descriptors, id, 0, [&](BlobWriter& blob_writer) {
        feature_descriptor_ops_.ToBlob(landmark.fd, blob_writer);
        landmark.have_to_get_descriptor_from_db = true;
        landmark.fd.pull_memory = nullptr;  // free descriptor memory
        return true;
      });
    }

    it = staged_landmarks_.erase(it);

    // add to cells
    std::vector<HashCellId> hash_cell_ids;
    this->GetKeyframeCells(landmark, InvalidKeyFrameId, hash_cell_ids, pose_graph_hypothesis);
    for (auto hash_cell_id : hash_cell_ids) {
      if (AddLandmarkToCell(id, hash_cell_id)) {
        cells_to_reduce.insert(hash_cell_id);
      }
      RERUN(LSIGridVisualizer::LogCell, *this, hash_cell_id, pose_graph_hypothesis);
    }
  }

  // reducing cells
  for (const auto hash_cell_id : cells_to_reduce) {
    ReduceLandmarksInCell(hash_cell_id, this->landmark_quality_func_, this->cell_landmarks_limit_);
    RERUN(LSIGridVisualizer::LogCell, *this, hash_cell_id, pose_graph_hypothesis);
  }
}

// Remove landmarks
void LSIGrid::RemoveLandmarks(const std::vector<LandmarkId>& landmarks_to_remove,
                              std::function<void(LandmarkId, KeyFrameId)>& func_remove_from_keyframe) {
  for (auto landmark_id : landmarks_to_remove) {
    if (database_) {
      database_->TryDelRecord(SlamDatabaseTable::Landmarks, landmark_id);
    }

    auto it = this->landmarks_.find(landmark_id);
    if (it == this->landmarks_.end()) {
      continue;
    }
    auto& landmark = it->second;
    // remove from keyframe
    for (auto& it_kf : landmark.keyframes) {
      func_remove_from_keyframe(landmark_id, it_kf.keyframe_id);
    }
    // remove from cells
    for (auto& hash_cell_id : landmark.cells) {
      auto p_cell = this->GetExistsCellForWrite(hash_cell_id);
      if (!p_cell) {
        continue;  // (?) cell is not exists
      }
      p_cell->Remove(landmark_id);
    }

    landmarks_.erase(it);
  }
}

Vector3T LSIGrid::GetLandmarkOrStagedCoords(LandmarkId landmark_id, const PoseGraphHypothesis& pg_hypo) const {
  Vector3T xyz(0, 0, 0);
  if (!GetLandmarkOrStagedCoords(landmark_id, pg_hypo, xyz)) {
    return Vector3T(0, 0, 0);
  }
  return xyz;
}

FeatureDescriptor LSIGrid::GetLandmarkFeatureDescriptor(LandmarkId landmark_id) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("GetLandmarkFeatureDescriptor()", profiler_color_);
  const auto landmark_ptr = GetLandmark(landmark_id);
  if (!landmark_ptr) {
    return FeatureDescriptorEmpty;
  }
  auto& landmark = *landmark_ptr;
  if (!landmark.have_to_get_descriptor_from_db) {
    return landmark.fd;
  }
  if (!database_) {
    return FeatureDescriptorEmpty;
  }
  StopwatchScope ssw(sw_read_descriptor_);
  // read descriptor from DB
  FeatureDescriptor loaded_fd = FeatureDescriptorEmpty;
  auto read_cb = [&](const BlobReader& blob_reader) {
    const auto fd = feature_descriptor_ops_.CreateFromBlob(blob_reader);
    if (!fd) {
      return false;
    }
    loaded_fd = fd;

    return true;
  };
  if (database_->GetRecord(SlamDatabaseTable::Descriptors, landmark_id, read_cb)) {
    return loaded_fd;
  }
  return FeatureDescriptorEmpty;
}

bool LSIGrid::GetLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id, Vector2T* uv_norm_in_keyframe) const {
  auto landmark_ptr = GetLandmarkOrStaged(landmark_id);
  if (!landmark_ptr) {
    return false;
  }
  auto& landmark = *landmark_ptr;

  auto data = landmark.GetLandmarkInKeyframe(keyframe_id);
  if (!data) {
    return false;
  }
  if (uv_norm_in_keyframe) {
    *uv_norm_in_keyframe = data->uv_norm_in_keyframe;
  }

  return true;
}

bool LSIGrid::ToBlob(BlobWriter& blob_writer) const {
  uint64_t sz = this->format_and_version_.size();
  blob_writer.reserve(sizeof(sz) + sz + sizeof(this->cell_size_) + sizeof(this->next_landmark_auto_id_));

  blob_writer.write_str(this->format_and_version_);

  blob_writer.write(this->cell_size_);
  blob_writer.write(this->cell_landmarks_limit_);
  blob_writer.write(this->next_landmark_auto_id_);
  return true;
}
bool LSIGrid::FromBlob(const BlobReader& blob_reader) {
  std::string format_and_version;
  if (!blob_reader.read_str(format_and_version) || format_and_version != this->format_and_version_) {
    SlamStderr("Failed to read LSIGrid: wrong version %s.\n", format_and_version.c_str());
    SlamStderr("Current is %s.\n", this->format_and_version_.c_str());
    return false;
  }

  blob_reader.read(this->cell_size_);
  blob_reader.read(this->cell_landmarks_limit_);
  blob_reader.read(this->next_landmark_auto_id_);
  return true;
}

// return Landmark count
uint64_t LSIGrid::LandmarksCount() const {
  uint64_t count = landmarks_.size() + staged_landmarks_.size();
  if (database_) {
    count += database_->GetRecordsCount(SlamDatabaseTable::Landmarks);
  }

  return count;
}

// rebuild all grid
void LSIGrid::RebuildAllGridCells(const PoseGraphHypothesis& pose_graph_hypothesis) {
  TRACE_EVENT ev = profiler_domain_.trace_event("RebuildAllGridCells()", profiler_color_);

  for (auto& it : landmarks_) {
    // be sure what landmark set in all cells
    auto& landmark_id = it.first;
    auto& landmark = it.second;
    landmark.cells.clear();
    std::vector<HashCellId> hash_cell_ids;
    this->GetKeyframeCells(landmark, InvalidKeyFrameId, hash_cell_ids, pose_graph_hypothesis);
    for (auto hash_cell_id : hash_cell_ids) {
      this->AddLandmarkToCell(landmark_id, hash_cell_id);
    }
  }
}

// Don't discard cell before calling ReduceLandmarksInCell()
size_t LSIGrid::ReduceLandmarksInCell(HashCellId hash_cell_id, LandmarkQualityFunc& func,
                                      size_t max_landmarks_in_cell) {
  TRACE_EVENT ev = profiler_domain_.trace_event("ReduceLandmarksInCell()", profiler_color_);

  if (max_landmarks_in_cell == 0) {
    return 0;  // ignore
  }

  auto p_cell = this->GetExistsCellForWrite(hash_cell_id);
  if (!p_cell) {
    return 0;
  }
  auto& cell = *p_cell;

  if (cell.landmarks_in_cell.size() < max_landmarks_in_cell) {
    return 0;
  }

  TRACE_EVENT ev3_ = profiler_domain_.trace_event("weights");

  struct LandmarkWeight {
    LandmarkId landmark_id;
    GridLandmark* landmark = nullptr;
    float weight = 0;
  };
  std::vector<LandmarkWeight> landmark_weights;
  landmark_weights.clear();
  landmark_weights.resize(cell.landmarks_in_cell.size());
  for (size_t i = 0; i < cell.landmarks_in_cell.size(); i++) {
    auto& landmark_weight = landmark_weights[i];
    landmark_weight.landmark_id = cell.landmarks_in_cell[i];
    auto landmarks_it = this->landmarks_.find(landmark_weight.landmark_id);
    if (landmarks_it == this->landmarks_.end()) {
      // it is dead landmark
      landmark_weight.weight = -1;  // to remove fast
      landmark_weight.landmark = 0;
      continue;
    }

    auto& landmark = landmarks_it->second;

    // landmark_weight
    landmark_weight.weight = func(landmark);
    landmark_weight.landmark = &landmark;
  }

  ev3_.Pop();
  TRACE_EVENT ev3 = profiler_domain_.trace_event("sort by weights");

  // sort by weight_func
  std::sort(landmark_weights.begin(), landmark_weights.end(),
            [&](const LandmarkWeight& a0, const LandmarkWeight& a1) -> bool { return a0.weight > a1.weight; });
  ev3.Pop();

  // remove outlets
  auto from_it = landmark_weights.begin();
  std::advance(from_it, std::min(max_landmarks_in_cell, landmark_weights.size()));

  TRACE_EVENT ev4 = profiler_domain_.trace_event("remove this cell and keyframes from outlets landmarks");
  size_t removed_count = 0;
  // remove this cell and keyframes from outlets landmarks
  for (auto it_landmark_weights = from_it; it_landmark_weights != landmark_weights.end(); it_landmark_weights++) {
    auto& landmark_weight = *it_landmark_weights;
    const auto landmark_id = landmark_weight.landmark_id;

    cell.Remove(landmark_id);

    auto& landmark = landmark_weight.landmark;
    if (!landmark) {
      continue;
    }
    // remove cell
    auto it_cells = std::lower_bound(landmark->cells.begin(), landmark->cells.end(), hash_cell_id);
    if (it_cells != landmark->cells.end() && *it_cells == hash_cell_id) {
      landmark->cells.erase(it_cells, landmark->cells.end());
    }

    // to list of dead landmarks
    if (landmark->cells.empty()) {
      this->dead_landmarks_.push_back(landmark_id);
    }

    removed_count++;
  }

  ev4.Pop();
  return removed_count;
}

size_t LSIGrid::ReduceLandmarks() {
  TRACE_EVENT ev = profiler_domain_.trace_event("ReduceLandmarks()", profiler_color_);

  if (cell_landmarks_limit_ == 0) {
    return 0;
  }

  size_t max_landmarks_in_cell = cell_landmarks_limit_;

  // TODO: only in-memory cells will processed!
  // Don't discard cell before calling ReduceLandmarksInCell()
  size_t removed_count = 0;
  for (auto& it : cells_) {
    auto hash_cell_id = it.first;
    removed_count += this->ReduceLandmarksInCell(hash_cell_id, landmark_quality_func_, max_landmarks_in_cell);
  }

  {
    // remove dead landmarks
    for (auto& it : landmarks_) {
      auto landmark_id = it.first;
      auto& landmark = it.second;
      if (landmark.GetKeyFrame() == InvalidKeyFrameId) {
        this->dead_landmarks_.push_back(landmark_id);
      }
    }
  }

  return removed_count;
};

size_t LSIGrid::RemoveDeadLandmarks(std::function<void(LandmarkId, KeyFrameId)>& func_remove_from_keyframe) {
  TRACE_EVENT ev = profiler_domain_.trace_event("RemoveDeadLandmarks()", profiler_color_);

  if (this->dead_landmarks_.empty()) {
    return 0;
  }
  this->RemoveLandmarks(this->dead_landmarks_, func_remove_from_keyframe);
  size_t removed_count = this->dead_landmarks_.size();
  this->dead_landmarks_.clear();
  return removed_count;
}

bool LSIGrid::QueryV(const std::function<bool(LandmarkId)>& func) const {
  for (const auto& it : landmarks_) {
    if (!func(it.first)) {
      return false;
    }
  }
  for (const auto& it : staged_landmarks_) {
    if (!func(it.first)) {
      return false;
    }
  }
  // others from database
  if (!database_) {
    return false;
  }
  const auto for_each_landmark_cb = [this, &func](LandmarkId id, const BlobReader&) -> bool {
    // ignore landmarks stored in landmarks_ or staged_landmarks_
    if (landmarks_.find(id) != landmarks_.end()) {
      return true;  // continue ForEach to the next landmark
    }
    if (staged_landmarks_.find(id) != staged_landmarks_.end()) {
      return true;  // continue ForEach to the next landmark
    }
    if (!func(id)) {
      return false;  // stop ForEach processing
    }
    return true;  // continue ForEach to the next landmark
  };

  return database_->ForEach(SlamDatabaseTable::Landmarks, for_each_landmark_cb);
}

void LSIGrid::QueryLandmarksByCameraPoseV(const Isometry3T& pose, const std::vector<CameraId>& cam_ids,
                                          const PoseGraphHypothesis& pose_graph_hypothesis, const QueryOptions& option,
                                          const std::function<void(LandmarkId)>& func) const {
  const Vector3T pose_guess_tran = pose.translation();

  std::vector<LandmarkId> candidates;
  candidates.reserve(1024);

  std::vector<CellId> centers_ids;
  if (option.HasDirection()) {
    const auto cell_id = CellIdFromXYZ(pose_guess_tran, pose_guess_tran + option.direction);
    centers_ids.push_back(cell_id);
  } else {
    CellIdFromEye(pose_guess_tran, centers_ids);
  }

  int bias = 3;
  if (option.fetch_strategy == FetchStrategy::PointOnly) {
    // Point only
    bias = 0;
  }

  for (int x = -bias; x <= bias; x++) {
    for (int y = -bias; y <= bias; y++) {
      for (int z = -bias; z <= bias; z++) {
        for (auto center_id : centers_ids) {
          CellId cell_id = center_id;
          cell_id.x_ += x;
          cell_id.y_ += y;
          cell_id.z_ += z;

          auto landmarks_in_cell = GetCellLandmarks(HashCellIdFromCellId(cell_id));
          if (!landmarks_in_cell.first) {
            continue;
          }

          log::Value<LogSlamHypothesis>("cell", HashCellIdFromCellId(cell_id));
          candidates.insert(candidates.end(), landmarks_in_cell.first, landmarks_in_cell.second);
        }
      }
    }
  }
  std::sort(candidates.begin(), candidates.end());
  const auto it_last = std::unique(candidates.begin(), candidates.end());

  std::unordered_map<CameraId, Isometry3T> cams_from_world;
  cams_from_world.reserve(cam_ids.size());
  for (CameraId cam_id : cam_ids) {
    cams_from_world[cam_id] = rig_.camera_from_rig[cam_id] * pose.inverse();
  }

  for (auto it_id = candidates.begin(); it_id != it_last; ++it_id) {
    const LandmarkId landmark_id = *it_id;
    Vector3T landmark_xyz;
    if (!GetLandmarkOrStagedCoords(landmark_id, pose_graph_hypothesis, landmark_xyz)) {
      continue;
    }
    if (IsLandmarkInAnyFrustum(rig_, cam_ids, cams_from_world, landmark_xyz)) {
      func(landmark_id);
    }
  }
}

void LSIGrid::QueryLandmarkRelationsV(LandmarkId id,
                                      std::function<bool(KeyFrameId, const Vector2T& uv_norm)>& func) const {
  auto landmark_ptr = GetLandmark(id);
  if (landmark_ptr) {
    for (auto& kf : landmark_ptr->keyframes) {
      bool res = func(kf.keyframe_id, kf.uv_norm_in_keyframe);
      if (!res) {
        return;
      }
    }
  }
}

void LSIGrid::GetKeyframeCells(const GridLandmark& landmark, KeyFrameId keyframe_id,
                               std::vector<HashCellId>& hash_cell_ids,
                               const PoseGraphHypothesis& pose_graph_hypothesis) const {
  hash_cell_ids.clear();

  auto xyz = landmark.xyz_world([&](KeyFrameId kf) { return pose_graph_hypothesis.GetKeyframePose(kf); });

  for (auto& landmark_in_keyframe : landmark.keyframes) {
    if (keyframe_id != InvalidKeyFrameId && keyframe_id != landmark_in_keyframe.keyframe_id) {
      continue;  // filter
    }
    auto pose = pose_graph_hypothesis.GetKeyframePose(landmark_in_keyframe.keyframe_id);
    if (pose) {
      Vector3T eye = landmark_in_keyframe.eye_in_keyframe;

      auto hash_cell_id = this->HashCellIdFromXYZ((*pose) * eye, xyz);

      // add to sorted hash_cell_ids
      auto it = std::lower_bound(hash_cell_ids.begin(), hash_cell_ids.end(), hash_cell_id);
      if (it != hash_cell_ids.end() && *it == hash_cell_id) {
        continue;  // already in list
      }
      // insert
      hash_cell_ids.insert(it, hash_cell_id);
    }
  }
}

bool LSIGrid::AddLandmarkToCell(LandmarkId landmark_id, HashCellId hash_cell_id) {
  auto it = landmarks_.find(landmark_id);
  if (it == landmarks_.end()) {
    // fail: landmark_id not in landmarks_
    return false;
  }
  auto& landmark = it->second;

  auto& cell = GetCellForWrite(hash_cell_id);
  bool is_cell_changed = cell.Add(landmark_id);

  // add to sorted fill GridLandmark::cells
  auto it_cells = std::lower_bound(landmark.cells.begin(), landmark.cells.end(), hash_cell_id);
  if (it_cells == landmark.cells.end() || *it_cells != hash_cell_id) {
    landmark.cells.insert(it_cells, hash_cell_id);
  }

  return is_cell_changed;
}

std::pair<const LandmarkId*, const LandmarkId*> LSIGrid::GetCellLandmarks(const HashCellId& hash_cell_id) const {
  auto it = cells_.find(hash_cell_id);
  if (it == cells_.end()) {
    if (database_) {
      // read from DB
      std::pair<const LandmarkId*, const LandmarkId*> res(nullptr, nullptr);
      database_->TryGetRecord(SlamDatabaseTable::SpatialCells, hash_cell_id, [&](const BlobReader& blob_reader) {
        size_t sz;
        if (!blob_reader.read(sz)) {
          return false;
        };
        if (sz == 0) {
          return false;
        }
        auto ptr = blob_reader.feed_forward(0);
        auto begin = reinterpret_cast<const LandmarkId*>(ptr);
        res.first = begin;
        res.second = begin + sz;
        return true;
      });
      return res;
    }
    return std::pair<const LandmarkId*, const LandmarkId*>(nullptr, nullptr);
  }

  const Cell& cell = it->second;
  const LandmarkId* begin = &(*cell.landmarks_in_cell.begin());
  const LandmarkId* end = &(*cell.landmarks_in_cell.end());
  return std::pair<const LandmarkId*, const LandmarkId*>(begin, end);
}

LSIGrid::Cell& LSIGrid::GetCellForWrite(const HashCellId& hash_cell_id) {
  auto it = cells_.find(hash_cell_id);
  if (it != cells_.end()) {
    return it->second;
  }
  // Add or read
  auto& cell = cells_[hash_cell_id];
  if (database_) {
    database_->TryGetRecord(SlamDatabaseTable::SpatialCells, hash_cell_id, [&](const BlobReader& blob_reader) {
      if (!blob_reader.read_std(cell.landmarks_in_cell)) {
        return false;
      };
      return true;
    });
  }
  return cell;
}
LSIGrid::Cell* LSIGrid::GetExistsCellForWrite(const HashCellId& hash_cell_id) {
  auto it = cells_.find(hash_cell_id);
  if (it != cells_.end()) {
    return &it->second;
  }
  // read if exists
  if (database_) {
    Cell* p_cell = nullptr;
    database_->GetRecord(SlamDatabaseTable::SpatialCells, hash_cell_id, [&](const BlobReader& blob_reader) {
      auto& cell = cells_[hash_cell_id];
      if (!blob_reader.read_std(cell.landmarks_in_cell)) {
        return false;
      };
      p_cell = &cell;
      return true;
    });
    return p_cell;
  }
  return nullptr;
}

void LSIGrid::LandmarkToBlob(const GridLandmark& landmark, BlobWriter& blob_writer) const {
  uint32_t keyframes_size = static_cast<uint32_t>(landmark.keyframes.size());
  uint32_t cells_size = static_cast<uint32_t>(landmark.cells.size());
  size_t size = sizeof(this->grid_landmark_version_) + sizeof(keyframes_size) +
                keyframes_size * (sizeof(Vector3T) + sizeof(Vector2T) + sizeof(KeyFrameId) + sizeof(bool)) +
                sizeof(cells_size) + cells_size * sizeof(HashCellId) + sizeof(GridLandmark::probes_types);
  blob_writer.reserve(size);

  blob_writer.write(this->grid_landmark_version_);
  blob_writer.write(keyframes_size);
  for (auto& kfl : landmark.keyframes) {
    blob_writer.write(kfl.keyframe_id);
    blob_writer.write(kfl.has_link);
    blob_writer.write(kfl.has_xyz);
    write(blob_writer, kfl.xyz_in_keyframe);
    write(blob_writer, kfl.uv_norm_in_keyframe);
    write(blob_writer, kfl.eye_in_keyframe);
  }
  blob_writer.write(cells_size);
  for (auto& cell_id : landmark.cells) {
    blob_writer.write(cell_id);
  }

  blob_writer.write(landmark.probes_types.data(), sizeof(*landmark.probes_types.data()) * landmark.probes_types.size());
}

bool LSIGrid::LandmarkFromBlob(const BlobReader& blob_reader, GridLandmark& landmark) const {
  uint32_t version;
  if (!blob_reader.read(version)) {
    return false;
  }
  if (version != this->grid_landmark_version_) {
    return false;
  }

  uint32_t keyframes_size;
  if (!blob_reader.read(keyframes_size)) {
    return false;
  }

  landmark.keyframes.resize(keyframes_size);
  for (auto& kfl : landmark.keyframes) {
    if (!blob_reader.read(kfl.keyframe_id)) {
      return false;
    }
    if (!blob_reader.read(kfl.has_link)) {
      return false;
    }
    if (!blob_reader.read(kfl.has_xyz)) {
      return false;
    }
    if (!read(blob_reader, kfl.xyz_in_keyframe)) {
      return false;
    }
    if (!read(blob_reader, kfl.uv_norm_in_keyframe)) {
      return false;
    }
    if (!read(blob_reader, kfl.eye_in_keyframe)) {
      return false;
    }
  }

  uint32_t cells_size;
  if (!blob_reader.read(cells_size)) {
    return false;
  }
  landmark.cells.resize(cells_size);
  for (auto& cell_id : landmark.cells) {
    blob_reader.read(cell_id);
  }
  blob_reader.read(landmark.probes_types.data(), sizeof(*landmark.probes_types.data()) * landmark.probes_types.size());

  return true;
}

LSIGrid::LandmarkQualityFunc LSIGrid::MakeLandmarkQualityFunc() {
  return [](const GridLandmark& landmark) -> float {
    float p = 0;
    p += landmark.probes_types[LP_SOLVER_OK];
    p += landmark.probes_types[LP_PNP_FAILED] * 0.75f;
    p += -landmark.probes_types[LP_RANSAC_FAILED] * 0.1f;
    p += -landmark.probes_types[LP_TRACKING_FAILED] * 0.25f;
    return p;
  };
}

Vector3T LSIGrid::GetLandmarkCoords(const PoseGraphHypothesis& pg_hypo, const GridLandmark& landmark) {
  return landmark.xyz_world([&](KeyFrameId kf) { return pg_hypo.GetKeyframePose(kf); });
}

bool LSIGrid::GetLandmarkOrStagedCoords(LandmarkId landmark_id, const PoseGraphHypothesis& pg_hypo,
                                        Vector3T& xyz) const {
  const auto landmark_ptr = GetLandmarkOrStaged(landmark_id);
  if (!landmark_ptr) {
    return false;
  }
  xyz = GetLandmarkCoords(pg_hypo, *landmark_ptr);
  return true;
}

}  // namespace cuvslam::slam

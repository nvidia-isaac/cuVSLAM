
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

#include <unordered_map>
#include <unordered_set>

#include "camera/rig.h"
#include "common/stopwatch.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "slam/map/database/slam_database.h"
#include "slam/map/descriptor/feature_descriptor.h"
#include "slam/map/pose_graph/pose_graph_hypothesis.h"
#include "slam/map/spatial_index/landmark.h"
#include "slam/map/spatial_index/lsi_grid_cell_id.h"

namespace cuvslam::slam {

enum LandmarkProbe { LP_SOLVER_OK = 0, LP_TRACKING_FAILED = 1, LP_RANSAC_FAILED = 2, LP_PNP_FAILED = 3, LP_MAX = 4 };

using CellId = grid::CellId;
using HashCellId = grid::HashCellId;

class LSIGrid {
  friend class LSIGridVisualizer;  // visualizer should have access to protected implementation

  const std::string format_and_version_ = "LSIGrid v0.03";
  const uint32_t grid_landmark_version_ = 1;

  struct Cell {
    // sorted id list
    std::vector<LandmarkId> landmarks_in_cell;

    bool Add(LandmarkId landmark);
    void Remove(LandmarkId landmark);
  };

  struct GridLandmark : Landmark {
    GridLandmark();

    mutable bool have_to_get_descriptor_from_db = false;
    // was added to cells
    std::vector<HashCellId> cells;
    // probes
    std::array<int, LP_MAX> probes_types;  // sum all probes

    int64_t timestamp_ns;
  };

public:
  LSIGrid(const IFeatureDescriptorOps& feature_descriptor_ops, const camera::Rig& rig, float cell_size = 0.25f,
          int cell_landmarks_limit = 0);
  ~LSIGrid();

  LSIGrid(const LSIGrid&) = delete;
  LSIGrid& operator=(const LSIGrid&) = delete;

public:
  float GetCellSize() const;
  size_t GetMaxLandmarksInCell() const;

  void SetDatabase(std::shared_ptr<ISlamDatabase> database);

  void SyncDatabase(const PoseGraphHypothesis& pose_graph_hypothesis);

  void SyncDatabaseReverse();

  // Writes the in-memory state to the attached database or a copy of it, leaving the index as it is.
  bool SaveToDatabase(ISlamDatabase& database, const PoseGraphHypothesis& pose_graph_hypothesis) const;

  // SaveToDatabase() into the attached database, which then also loses what was removed while it was read-only.
  bool FlushToDatabase(const PoseGraphHypothesis& pose_graph_hypothesis);

  // add landmark
  LandmarkId AddLandmark(const FeatureDescriptor& fd, int64_t timestamp_ns);
  // add landmark relation
  void AddLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id,
                           const Vector3T* xyz_in_keyframe,  // can be null
                           const Vector2T& uv_norm, const PoseGraphHypothesis& pose_graph_hypothesis);

  // On merge keyframes. Change keyframe id in landmark
  void UpdateLandmarkRelation(LandmarkId landmark_id, KeyFrameId old_keyframe_id, KeyFrameId new_keyframe_id,
                              const Isometry3T& transform_old_to_new);

  void AddLandmarkProbeStatistic(LandmarkId landmark_id, LandmarkProbe landmark_probe);

  Vector3T GetLandmarkOrStagedCoords(LandmarkId id, const PoseGraphHypothesis& pg_hypo) const;

  // Like GetLandmarkOrStagedCoords(), but a landmark read from the database is not kept in memory.
  bool PeekLandmarkOrStagedCoords(LandmarkId id, const PoseGraphHypothesis& pg_hypo, Vector3T& xyz) const;

  FeatureDescriptor GetLandmarkFeatureDescriptor(LandmarkId id) const;

  bool GetLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id,
                           Vector2T* uv_norm_in_keyframe = nullptr) const;

public:
  // called after PGO.
  void OnUpdateKeyframePose(KeyFrameId keyframe_id, const Isometry3T& pose_old, const Isometry3T& pose_new);
  // must be called after all OnUpdateKeyframePose() to update cells
  void OnUpdateKeyframePoseFinished();

  // on each vo camera pose
  void MoveReadyStagedLandmarksToLSI(const Isometry3T& world_from_rig, const std::vector<CameraId>& cam_ids,
                                     const PoseGraphHypothesis& pose_graph_hypothesis, KeyFrameId pose_graph_head,
                                     int64_t current_timestamp_ns);

  // Remove landmarks
  void RemoveLandmarks(const std::vector<LandmarkId>& landmarks_to_remove,
                       std::function<void(LandmarkId, KeyFrameId)>& func_remove_from_keyframe);

public:
  bool ToBlob(BlobWriter& blob) const;
  bool FromBlob(const BlobReader& blob);

  // return Landmark count
  uint64_t LandmarksCount() const;

public:
  // query all landmarks
  // example: lsi.Query([&](LandmarkId id) {return true;}
  // if lambda return false -> stop query else continue
  // return false if database error occurs.
  template <typename FUNC>
  bool Query(FUNC&& lambda) const {
    const std::function<bool(LandmarkId)> func(lambda);
    return QueryV(func);
  }

  // QueryOptions
  enum class FetchStrategy { PointOnly, Volume };
  struct QueryOptions {
    FetchStrategy fetch_strategy = FetchStrategy::Volume;
    Vector3T direction = Vector3T::Zero();
    bool HasDirection() const { return direction != Vector3T::Zero(); }
  };
  // query landmarks by camera pose
  // landmarks_spatial_index.Query(task.guess_pose, radius_scale, radius_bias, *task.pose_graph_hypothesis,
  // [&](LandmarkId id) {return;}
  template <typename FUNC>
  void QueryLandmarksByCameraPose(const Isometry3T& pose, const std::vector<CameraId>& cam_ids,
                                  const PoseGraphHypothesis& pose_graph_hypothesis, QueryOptions option,
                                  FUNC&& lambda) const {
    const std::function<void(LandmarkId)> func(lambda);
    QueryLandmarksByCameraPoseV(pose, cam_ids, pose_graph_hypothesis, option, func);
  }
  // query relations (keyframe) of landmark
  template <typename FUNC>
  void QueryLandmarkRelations(LandmarkId id, FUNC&& lambda) const {
    std::function<bool(KeyFrameId, const Vector2T& uv_norm)> func(lambda);
    QueryLandmarkRelationsV(id, func);
  }

  // rebuild all grid
  void RebuildAllGridCells(const PoseGraphHypothesis& pose_graph_hypothesis);
  size_t ReduceLandmarks();
  size_t RemoveDeadLandmarks(std::function<void(LandmarkId, KeyFrameId)>& func_remove_from_keyframe);

protected:
  const IFeatureDescriptorOps& feature_descriptor_ops_;  // abstraction for "operations over descriptors"

  // Landmarks
  LandmarkId next_landmark_auto_id_ = 0;
  mutable std::map<LandmarkId, GridLandmark> staged_landmarks_;
  mutable std::map<LandmarkId, GridLandmark> landmarks_;

  // maximum lifetime of a landmark in the staged_landmarks_ map,
  // after this duration, the landmark will be written into the lsi_grid
  int64_t max_staged_landmark_lifetime_ns_ = 60e9;

  camera::Rig rig_;

  float cell_size_;
  size_t cell_landmarks_limit_ = 0;         // default=0: limit is off
  std::vector<LandmarkId> dead_landmarks_;  // landmarks to remove
  // removed while the database was read-only: hidden from reads, deleted by SaveToDatabase()
  std::unordered_set<LandmarkId> read_only_removals_;

  std::unordered_map<HashCellId, Cell> cells_;

  // database
  std::shared_ptr<ISlamDatabase> database_;

  // profiler
  profiler::SLAMProfiler::DomainHelper profiler_domain_ = profiler::SLAMProfiler::DomainHelper("SLAM");
  uint32_t profiler_color_ = 0x00FFFF;

  // stopwatches
  mutable Stopwatch sw_read_landmark_;
  mutable Stopwatch sw_read_descriptor_;

protected:
  using LandmarkQualityFunc = std::function<float(const GridLandmark& landmark)>;
  LandmarkQualityFunc landmark_quality_func_;

  bool QueryV(const std::function<bool(LandmarkId)>& func) const;
  void QueryLandmarksByCameraPoseV(const Isometry3T& pose, const std::vector<CameraId>& cam_ids,
                                   const PoseGraphHypothesis& pose_graph_hypothesis, const QueryOptions& option,
                                   const std::function<void(LandmarkId)>& func) const;

  void QueryLandmarkRelationsV(LandmarkId id, std::function<bool(KeyFrameId, const Vector2T& uv_norm)>& func) const;

  bool AddLandmarkToCell(LandmarkId landmark_id, HashCellId hash_cell_id);

  // return begin/end of landmarks vector for cell_id
  std::pair<const LandmarkId*, const LandmarkId*> GetCellLandmarks(const HashCellId& hash_cell_id) const;
  // Return exists or add new cell
  Cell& GetCellForWrite(const HashCellId& hash_cell_id);
  // Return exists cell
  Cell* GetExistsCellForWrite(const HashCellId& hash_cell_id);

protected:
  void GetKeyframeCells(const GridLandmark& landmark,
                        KeyFrameId keyframe_id,  // ==InvalidKeyFrameId
                        std::vector<HashCellId>& hash_cell_ids, const PoseGraphHypothesis& pose_graph_hypothesis) const;

  void LandmarkToBlob(const GridLandmark& landmark, BlobWriter& blob_writer) const;
  bool LandmarkFromBlob(const BlobReader& blob, GridLandmark& landmark) const;

  static LandmarkQualityFunc MakeLandmarkQualityFunc();

  size_t ReduceLandmarksInCell(HashCellId hash_cell_id, LandmarkQualityFunc& func, size_t max_landmarks_in_cell);

protected:
  // for OnUpdateKeyframePoseFinished() and OnUpdateKeyframePose()
  // add record to copy landmarks from hash_cell_id_old to hash_cell_id_new
  struct UpdateKeyframePoseItem {
    KeyFrameId keyframe_id;
    HashCellId hash_cell_id_old;
    HashCellId hash_cell_id_new;
  };
  std::vector<UpdateKeyframePoseItem> update_keyframe_pose_items_;

  // Access lo landmark
protected:
  inline GridLandmark* GetLandmark(LandmarkId landmark_id);
  inline const GridLandmark* GetLandmark(LandmarkId landmark_id) const;

  inline GridLandmark* GetLandmarkOrStaged(LandmarkId landmark_id);
  inline const GridLandmark* GetLandmarkOrStaged(LandmarkId landmark_id) const;

  inline GridLandmark* TryToReadLandmark(LandmarkId landmark_id) const;

  static Vector3T GetLandmarkCoords(const PoseGraphHypothesis& pg_hypo, const GridLandmark& landmark);
  bool GetLandmarkOrStagedCoords(LandmarkId landmark_id, const PoseGraphHypothesis& pose_graph_hypothesis,
                                 Vector3T& xyz) const;

  // CellId and HashCellId helpers
protected:
  CellId CellIdFromXYZ(const Vector3T& eye, const Vector3T& landmark) const {
    return CellId::CellIdFromXYZ(eye, landmark, cell_size_);
  }
  void CellIdFromEye(const Vector3T& eye, std::vector<CellId>& cell_ids) const {
    return CellId::CellIdFromEye(eye, cell_ids, cell_size_);
  }
  HashCellId HashCellIdFromCellId(CellId id) const { return CellId::HashCellIdFromCellId(id); }
  CellId HashCellIdToCellId(HashCellId hash_cell_id) const { return CellId::HashCellIdToCellId(hash_cell_id); }
  HashCellId HashCellIdFromXYZ(const Vector3T& eye, const Vector3T& landmark) const {
    return CellId::HashCellIdFromXYZ(eye, landmark, cell_size_);
  }
  void HashCellIdFromEye(const Vector3T& eye, std::vector<HashCellId>& hashcell_ids) const {
    return CellId::HashCellIdFromEye(eye, hashcell_ids, cell_size_);
  }
  Vector3T CellIdToXYZ(const CellId& id) const { return CellId::CellIdToXYZ(id, cell_size_); }
  std::string CellIdToString(const CellId& cell_id) const { return CellId::CellIdToString(cell_id); }
};

// GetLandmark()
inline LSIGrid::GridLandmark* LSIGrid::GetLandmark(LandmarkId landmark_id) {
  auto it = this->landmarks_.find(landmark_id);
  if (it == this->landmarks_.end()) {
    GridLandmark* readed = TryToReadLandmark(landmark_id);
    if (readed) {
      return readed;
    }
    return nullptr;
  }
  return &it->second;
}
inline const LSIGrid::GridLandmark* LSIGrid::GetLandmark(LandmarkId landmark_id) const {
  auto it = this->landmarks_.find(landmark_id);
  if (it == this->landmarks_.end()) {
    const GridLandmark* readed = TryToReadLandmark(landmark_id);
    if (readed) {
      return readed;
    }
    return nullptr;
  }
  return &it->second;
}

// GetLandmarkOrStaged()
inline LSIGrid::GridLandmark* LSIGrid::GetLandmarkOrStaged(LandmarkId landmark_id) {
  GridLandmark* landmark = GetLandmark(landmark_id);
  if (landmark) {
    return landmark;
  }
  auto it = this->staged_landmarks_.find(landmark_id);
  if (it == this->staged_landmarks_.end()) {
    return nullptr;
  }
  return &it->second;
}
inline const LSIGrid::GridLandmark* LSIGrid::GetLandmarkOrStaged(LandmarkId landmark_id) const {
  const GridLandmark* landmark = GetLandmark(landmark_id);
  if (landmark) {
    return landmark;
  }
  const auto it = this->staged_landmarks_.find(landmark_id);
  if (it == this->staged_landmarks_.end()) {
    return nullptr;
  }
  return &it->second;
}
inline LSIGrid::GridLandmark* LSIGrid::TryToReadLandmark(LandmarkId landmark_id) const {
  // FlushMap writes staged landmarks too, and a read-only database keeps the removed ones.
  if (!database_ || staged_landmarks_.count(landmark_id) != 0 || read_only_removals_.count(landmark_id) != 0) {
    return nullptr;
  }
  StopwatchScope ssw(sw_read_landmark_);
  GridLandmark* readed = nullptr;
  database_->TryGetRecord(SlamDatabaseTable::Landmarks, landmark_id, [&](const BlobReader& blob_reader) {
    auto& landmark = this->landmarks_[landmark_id];
    bool res = LandmarkFromBlob(blob_reader, landmark);
    if (!res) {
      this->landmarks_.erase(landmark_id);
      return false;
    }
    landmark.have_to_get_descriptor_from_db = true;
    readed = &landmark;
    return true;
  });
  return readed;
}

}  // namespace cuvslam::slam

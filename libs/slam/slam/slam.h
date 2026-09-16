
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

#include <optional>
#include <random>

#include "camera/rig.h"
#include "common/camera_id.h"
#include "common/image.h"
#include "common/isometry.h"
#include "common/stopwatch.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "pnp/multicam_pnp.h"
#include "profiler/profiler.h"

#include "slam/common/slam_common.h"
#include "slam/map/map.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"
#include "slam/vpr/vpr_map.h"

namespace cuvslam::slam {

enum PoseGraphOptimizerType { Dummy, Simple };

struct PoseGraphOptimizerOptions {
  PoseGraphOptimizerType type = Simple;
};
struct SpatialIndexOptions {
  float cell_size = 0;
  int max_landmarks_in_cell = 100;
};
struct VOFrameData {
  FrameId frame_id;
  uint64_t timestamp_ns;          // timestamp of image (in microseconds)
  std::string frame_information;  // frame information for log

  struct Track2DXY {
    CameraId cam_id;
    TrackId track_id;
    Vector2T xy;
  };
  std::vector<Track2DXY> tracks2d_norm;
  std::map<TrackId, Vector3T> tracks3d_rel;
};

// Core SLAM system that performs visual localization and mapping.
//
// LocalizerAndMapper is the main class for visual SLAM operations, managing:
// - Map:
//   * Persistent storage of keyframes and landmarks and features.
//   * Spatial index: Efficient spatial querying of landmarks for loop closure.
//   * Pose graph: Graph structure for keyframe poses and constraints.
// - Pose estimation: Current camera rig pose tracking.
// - Landmark staging: Temporary storage for track-to-landmark conversion.
//
// Key workflows:
//
// 1. Mapping:
//    - AddKeyframe() incrementally builds the map using visual odometry data.
//
// 2. Loop Closure:
//    - DetectLoopClosure() localizes current pose against existing map
//    - ApplyLoopClosureResult() adds loop closure constraints to pose graph
//    - OptimizePoseGraph() globally optimizes all poses to correct drift
//
// Thread safety: Not thread-safe. External synchronization required.
class LocalizerAndMapper {
public:
  struct LoopClosureStatus {
    bool success = false;
    Isometry3T result_pose = Isometry3T::Identity();
    Matrix6T result_pose_covariance = Matrix6T::Constant(std::numeric_limits<float>::infinity());
    uint32_t selected_landmarks_count = 0;       // Count of selected Landmarks for LC
    uint32_t tracked_landmarks_count = 0;        // Count of Tracked Landmarks in LC
    uint32_t pnp_landmarks_count = 0;            // Count of Landmarks after filtering
    uint32_t good_landmarks_count = 0;           // Count of Good Landmarks in LC if OK
    std::vector<KeyFrameId> keyframes_in_sight;  // Keyframes of processed landmarks
    std::vector<LandmarkInSolver> landmarks;
    std::vector<std::pair<LandmarkId, LandmarkProbe>> discarded_landmarks;
  };

  // ----- Initialization functions -----
  LocalizerAndMapper(const camera::Rig& rig, FeatureDescriptorType descriptor_type, bool use_gpu);
  ~LocalizerAndMapper();

  void SetReproduceMode(bool reproduce_mode);
  bool GetReproduceMode() const;

  void SetLandmarksSpatialIndex(const SpatialIndexOptions& options);
  SpatialIndexOptions GetLandmarksSpatialIndexOptions() const;

  void SetKeyframesLimit(int max_keyframes_count);
  int GetKeyframesLimit() const;

  bool SetPoseGraphOptimizerOptions(const PoseGraphOptimizerOptions& options);
  PoseGraphOptimizerOptions GetPoseGraphOptimizerOptions() const;

  // if keep_track_poses true - CalcFramePose() will work
  void SetKeepTrackPoses(bool keep_track_poses);
  bool GetKeepTrackPoses() const;

  // should be called once before any call of AddKeyframe or SetLost
  void SetActiveCameras(const std::vector<CameraId>& cameras);
  std::optional<std::vector<CameraId>> GetActiveCameras() const;

  // ----- Map building -----

  // Process and integrate a new keyframe from visual odometry tracking.
  // This is the main entry point for incremental map building.
  //
  // Operations performed:
  // - Updates current pose estimate from odometry delta
  // - Adds new keyframe to pose graph with covariance
  // - Moves ready staged landmarks to LSI (Landmarks Spatial Index) grid
  // - Creates feature descriptors for new tracks
  // - Removes dead landmarks from spatial index
  // - Persists changes to database if attached
  //
  // Parameters:
  //   from_last_keyframe - Relative pose transform (delta) from previous keyframe
  //   frame_data - Visual odometry frame data (tracks uv/xyz, timestamp, frame_id)
  //   images - Input images from cameras for descriptor extraction
  void AddKeyframe(const Isometry3T& from_last_keyframe, const VOFrameData& frame_data, const Images& images);

  // ----- Data query -----

  // Returns the current estimated pose of the camera rig in the world coordinate frame.
  //
  // The pose is updated by:
  // - AddKeyframe(): incremental update from relative odometry
  // - OptimizePoseGraph(): pose graph optimization refinements (after LoopClosure)
  //
  // Note: This is the instantaneous pose estimate, not necessarily a keyframe pose.
  Isometry3T GetCurrentPose() const;

  // For visualizations only.
  // TODO: return cost MapView
  const Map& GetMap() const;

  // Get the pose and timestamp of the last keyframe.
  bool GetLastKeyframePoseAndTimestamp(Isometry3T& last_keyframe_pose, int64_t& last_keyframe_ts) const;

  // ----- Database operations -----
  bool AttachToExistingReadOnlyDatabase(const std::string& path);  // allow query data
  bool AttachToNewDatabase(const std::string& path);
  bool AttachToNewDatabaseSaveMapAndDetach(const std::string& path);  // copy Map from memory to disk

  // Flush current in-memory map updates to the active database.
  //
  // This method writes the current state of the landmarks spatial index to the database
  // and performs a flush operation to ensure all pending writes are persisted to disk.
  // The database must be attached and writable for this operation to succeed.
  //
  // Returns:
  //   true if the flush succeeded, false if no database is attached
  bool FlushActiveDatabase() const;

  void DetachDatabase();  // flush data and close connection

  // ----- Loop closure search -----

  // Attempt to detect loop closure from provided guess pose against the existing map.
  //
  // This method searches for previously mapped landmarks visible in the current images,
  // then computes the rig pose using PnP and RANSAC. If successful, it provides a corrected
  // pose estimate that can be used to close loops and improve map consistency.
  //
  // Operations performed:
  // - Selects landmark candidates from spatial index based on initial pose guess
  // - Tracks landmarks to the current images using feature descriptors
  // - Computes pose using landmarks through PnP and RANSAC
  // - Collects statistics about landmark matching quality
  //
  // Parameters:
  //   loop_closure_solver - Solver implementation for loop closure detection
  //   images - Current camera images
  //   world_from_rig_guess - Initial pose guess to search around
  //   status - Output structure containing success flag, computed pose, and statistics
  //
  // Returns:
  //   true if loop closure detected, false otherwise
  void DetectLoopClosure(const ILoopClosureSolver& loop_closure_solver, const Images& images,
                         const Isometry3T& world_from_rig_guess, LoopClosureStatus& status) const;

  // ----- Loop closure result processing -----

  // Find the keyframe that observes the most landmarks from the provided set.
  //
  // This method can be called after successful loop closure detection to identify
  // the optimal existing keyframe for creating a loop closure edge. The selected keyframe
  // is the one with maximum covisibility with the detected landmarks.
  //
  // Parameters:
  //   landmarks - Set of landmarks detected during loop closure
  //   edge_stat - Optional output for statistics of the best match (landmark count, etc.)
  //
  // Returns:
  //   KeyFrameId of the keyframe observing the most landmarks, or InvalidKeyFrameId if none found
  KeyFrameId FindKeyframeWithMostLandmarks(const std::vector<LandmarkInSolver>& landmarks,
                                           PoseGraphEdgeStat* edge_stat = nullptr) const;

  // Update landmark probe statistics for landmarks that failed during loop closure detection.
  //
  // Records failure information (tracking failed, PnP failed, RANSAC failed, etc.) for
  // landmarks that were selected but discarded during loop closure. These statistics help
  // evaluate landmark quality and reliability over time, which can be used to improve
  // future loop closure attempts by avoiding unreliable landmarks.
  //
  // Parameters:
  //   discarded_landmarks - Pairs of (landmark_id, probe_result) for landmarks that failed
  void UpdateLandmarkProbeStatistics(
      const std::vector<std::pair<LandmarkId, LandmarkProbe>>& discarded_landmarks) const;

  // Apply a successful loop closure result by integrating it into the map.
  //
  // This method can be called after DetectLoopClosure() succeeds to commit the loop closure
  // to the map. It performs the following operations:
  // - Finds the keyframe with maximum covisibility with the provided landmarks
  // - Adds a loop closure edge in the pose graph between that keyframe and the current head
  // - Updates landmark relations in both the pose graph and spatial index
  //
  // The added edge and landmark relations can later be refined by pose graph optimization.
  //
  // Parameters:
  //   world_from_lc - Loop closure pose result (world frame)
  //   lc_pose_covariance - Covariance matrix for the loop closure pose
  //   lc_landmarks - Set of landmarks matched during loop closure detection
  //
  // Returns:
  //   true if loop closure successfully applied, false if keyframes not found or edge cannot be added
  bool ApplyLoopClosureResult(const Isometry3T& world_from_lc, const Matrix6T& lc_pose_covariance,
                              const std::vector<LandmarkInSolver>& lc_landmarks);

  // Perform pose graph optimization to refine all keyframe poses.
  //
  // The optimization updates all keyframe poses and propagates changes to the spatial index
  // grid to maintain consistency between pose graph and landmark locations.
  bool OptimizePoseGraph(bool planar_constraints);

  // Rebuild the landmark spatial index grid and perform landmark cleanup.
  //
  // This method performs maintenance on the spatial index structure by:
  // - Rebuilding all spatial grid cells with current keyframe poses
  // - Reducing/filtering landmarks based on quality metrics (probe statistics)
  // - Removing dead landmarks that no longer meet quality thresholds
  //
  // This is typically called after significant pose graph changes (e.g., after optimization)
  // to ensure the spatial index remains consistent with updated keyframe poses.
  void RebuildSpatialIndex();

  // Remove excess keyframes to enforce the configured keyframe limit.
  //
  // This method reduces the number of keyframes in the pose graph by merging keyframes
  // along edges with the smallest covariance (highest confidence). When two keyframes
  // are merged, one is removed and its landmark observations are transferred to the other.
  //
  // The reduction continues until the keyframe count is within the configured limit
  // (set via SetKeyframesLimit). If no limit is set (max_keyframes_count_ == 0), this
  // method does nothing.
  void ReduceKeyframes();

  // ----- Only if keep_track_poses true -----
  // calc the pose for the past frame
  bool CalcFramePose(FrameId frame_id, Isometry3T& pose) const;
  // Find Keyframe and transform By FrameId
  bool FindKeyframeByFrame(FrameId frame_id, KeyFrameId& keyframe_id, Isometry3T& from_keyframe_to_frame) const;

  [[nodiscard]] bool SelectHeadKeyframe(KeyFrameId const_slam_keyframe_id, int64_t timestamp_ns);

  // ----- Visual place recognition -----

  // Install the place recognition backend. Must be called before the first keyframe.
  // Passing VprType::kNone leaves place recognition off, which is the default.
  void SetVprOptions(const vpr::VprOptions& options);

  // True when a place recognition backend is installed.
  bool IsVprEnabled() const;

  // Number of frames in the place recognition map.
  size_t GetVprMapSize() const;

  // True when the newest pose graph node has no image in the place recognition map yet, so
  // handing it one would do something. Lets a caller skip converting pixels it does not need.
  bool VprHeadNeedsImage() const;

  // Attach `image` to the newest pose graph node, unless that node already has one, so a caller may
  // offer every frame and still store exactly one image per node. True when the image was stored.
  bool AddVprFrame(const vpr::VprImage& image, int64_t timestamp_ns);

  // Find the pose graph node from which `image` was observed, and its current world pose.
  vpr::VprPlace RecognizePlace(const vpr::VprImage& image);

  // The `max_results` best places for `image`, best first, for a relocalizer to verify in turn.
  std::vector<vpr::VprPlace> RecognizePlaces(const vpr::VprImage& image, size_t max_results);

  // Write the place recognition map into `folder` alongside the SLAM map, refreshing the stored node
  // poses first so a session that has no pose graph for these nodes can still use it.
  bool SaveVprMap(const std::string& folder);

  // Load a place recognition map written by SaveVprMap.
  // kWithPoseGraph is for a map whose pose graph this instance also loaded, so the node ids in the
  // file are its own; kStandalone is for a map that belongs to another session.
  bool LoadVprMap(const std::string& folder, vpr::VprMap::LoadMode mode = vpr::VprMap::LoadMode::kStandalone);

private:
  // ----- internal types -----
  struct TrackOnKeyframe {
    KeyFrameId keyframe_id;
    Vector3T xyz_rel;  // relative to corresponded keyframe
    Vector2T uv_norm;  // uv in corresponded keyframe
    Vector2T uv_pix;
  };

  // staging 3d
  struct StagingTrack3d {
    FeatureDescriptor fd;
    CameraId cam_id;
    std::vector<TrackOnKeyframe> keyframes;
    size_t num_frames_not_tracked = 0;

    [[nodiscard]] const TrackOnKeyframe* FindKeyframe(KeyFrameId keyframe_id) const;
  };

  // ----- Config -----
  bool reproduce_mode_ = true;
  PoseGraphOptimizerOptions pg_options_;
  std::string pose_graph_optimizer_;
  size_t max_keyframes_count_ = 0;  // 0 means no limit

  // Before merged into the map, landmarks wait in 'staging3d_' until
  // 'staging_keyframes_thresh' keyframes pass without their tracking.
  const size_t staging_keyframes_thresh_ = 0;

  // ----- Internal state -----
  const camera::Rig rig_;
  std::optional<std::vector<CameraId>> active_cameras_;  // should be set once before map updates
  Map map_;
  std::map<TrackId, StagingTrack3d> staging3d_;

  // ----- Solvers -----
  pnp::PNPSolver pnp_;

  // ----- keep_track_poses_ functionality -----
  bool keep_track_poses_ = false;
  std::map<KeyFrameId, FrameId> keyframe_sources_;
  struct ChangeKeyframeInfo {
    KeyFrameId new_node;
    storage::Isometry3<float> old_node_to_new;
  };
  std::map<KeyFrameId, ChangeKeyframeInfo> keyframe_removed_;

  // random guess pose
  mutable std::random_device random_device_;
  mutable std::mt19937 random_generator_;

  // ----- Visual place recognition -----
  // Lives here rather than in Map so that LocalizeInMap, which swaps the whole
  // LocalizerAndMapper, swaps the place recognition map with the pose graph its node ids refer to.
  std::unique_ptr<vpr::VprMap> vpr_map_;

  // ----- Cache to reallocation -----
  std::vector<TrackId> to_remove_;

  // ----- Profiler -----
  profiler::SLAMProfiler::DomainHelper profiler_domain_ = profiler::SLAMProfiler::DomainHelper("SLAM");
  uint32_t profiler_color_ = 0xFF0000;

  // ----- Internal methods -----
  // Install the single PoseGraph removal callback that keeps keyframe_removed_ and the place
  // recognition map in step with keyframe merges.
  void RegisterRemoveNodeCB();

  // Attempts to add an edge between two keyframes in the pose graph.
  // Returns false if an existing edge has higher weight or if the 'start' keyframe pose is not found.
  bool AddEdgeToPoseGraph(KeyFrameId start, KeyFrameId end, const Isometry3T& start_from_end,
                          const Matrix6T& start_from_end_covariance, const PoseGraphEdgeStat& stat);

  // calc covariation from prev VO keyframe
  bool CalcBetweenPose(KeyFrameId from, KeyFrameId to,
                       const std::vector<VOFrameData::Track2DXY>& tracks2d_norm,  // normalized coordinates
                       const std::map<TrackId, Vector3T>& tracks3d_rel,           // xyz in camera space
                       Isometry3T& pose, Matrix6T& covariance) const;

  // callback for landmarks_spatial_index_->RemoveDeadLandmarks()
  void RemoveLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id);

  void MoveReadyStagedLandmarksToLSI(uint64_t timestamp_ns = 0);
  bool ReduceLandmarks();
};

}  // namespace cuvslam::slam

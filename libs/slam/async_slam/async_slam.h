
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

#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "common/camera_id.h"
#include "common/thread_safe_queue.h"
#include "cuvslam/cuvslam2.h"
#include "odometry/ivisual_odometry.h"
#include "slam/async_slam/tail.h"
#include "slam/localizer/localizer.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"
#include "slam/slam/slam.h"
#include "slam/view/view_landmarks.h"
#include "slam/view/view_manager.h"
#include "slam/view/view_pose_graph.h"

namespace cuvslam::slam {

struct AsyncSlamOptions {
  std::string map_cache_path;  // if non-empty, connect to LMDB at this path
  bool use_gpu = true;
  bool reproduce_mode = true;            // allow to repeat results: ransac.seed(0), sync=true
  bool pose_for_frame_required = false;  // set true for calling GetPoseForFrame()
  int max_pose_graph_nodes = 0;          // SLAM: limit of the node count in the pose graph
  uint64_t throttling_time_ms = 0;
  uint64_t retention_time_ms = 5000;
  // Length of the SLAM input queue at which a warning is printed because SLAM falls behind odometry.
  // Diagnostic only. Default: 10 commands.
  uint32_t delay_warning_queue_size = 10;
  PoseGraphOptimizerOptions pgo_options;
  SpatialIndexOptions spatial_index_options;
  float max_landmarks_distance = std::numeric_limits<float>::max();
  LoopClosureSolverType loop_closure_solver_type = LoopClosureSolverType::kTwoStepsEasy;
  bool planar_constraints = false;
};

struct AsyncSlamLCTelemetry {
  uint64_t timestamp_ns = 0;                 // timestamp of these measurements (in microseconds)
  bool lc_status = false;                    // 0 - failed, 1 - succeed
  bool pgo_status = false;                   // 0 - failed, 1 - succeed
  uint32_t lc_selected_landmarks_count = 0;  // Count of Selected Landmarks
  uint32_t lc_tracked_landmarks_count = 0;   // Count of Tracked Landmarks
  uint32_t lc_pnp_landmarks_count = 0;       // Count of Landmarks after filtering
  uint32_t lc_good_landmarks_count = 0;      // Count of Good Landmarks
};

struct LoopClosureStamped {
  FrameId frame_id = InvalidFrameId;
  uint64_t timestamp_ns = 0;
  Isometry3T pose;
};

class AsyncSlam {
public:
  // cameras - list of camera indexes from rig to be used in slam
  AsyncSlam(const camera::Rig& rig, const std::vector<CameraId>& cameras, const AsyncSlamOptions& options);
  ~AsyncSlam();

  void ProcessInputSynchronously();  // for "blocking" = reproduce_mode without launching background thread
  void Stop();

  void TrackResult(FrameId frameId, int64_t timestamp_ns, const odom::IVisualOdometry::VOFrameStat& stat,
                   const sof::Images& images, const Isometry3T& delta);

  void LocalizeInMap(const std::string_view& folder_name, int64_t timestamp_ns, const Isometry3T& guess_pose,
                     const sof::Images& images, const Slam::LocalizationSettings& settings,
                     Slam::LocalizeStartCB start_cb, Slam::LocalizeFinishCB finish_cb);

  Isometry3T GetSlamPose() const;

  // could be blocked by slam thread
  bool GetPoseForFrame(FrameId frameId, Isometry3T& pose) const;
  // could be blocked by slam thread
  bool GetPosesForAllFrames(std::map<uint64_t, storage::Isometry3<float>>& frames) const;

  bool GetLastTelemetry(AsyncSlamLCTelemetry& telemetry) const;
  const std::list<LoopClosureStamped>& GetLastLoopClosuresStamped();

  void CopyToDatabase(const std::string& path, const std::function<void(bool)>& callback = nullptr);

  // Set landmarks view
  void SetLandmarksView(std::shared_ptr<ViewManager<ViewLandmarks>> view);
  // Set loop closure view
  void SetLoopClosureView(std::shared_ptr<ViewManager<ViewLandmarks>> view);
  // Set pose graph view
  void SetPoseGraphView(std::shared_ptr<ViewManager<ViewPoseGraph>> view);

private:
  struct VOTrackData {
    FrameId end_frame_id = InvalidFrameId;
    uint64_t timestamp_ns = 0;
    Isometry3T from_keyframe = Isometry3T::Identity();
  };

  class ICommand {
  public:
    virtual ~ICommand() = default;
    virtual void Execute(AsyncSlam& async_slam, FrameId frame_id, const Isometry3T& vo_pose_at_that_frame) = 0;
  };

  // VO data or command
  struct VOKeyframeInfo {
    VOTrackData track_data;
    VOFrameData frame_data;
    Isometry3T vo_pose_at_this_frame;
    std::shared_ptr<ICommand> command;
  };

  class LocalizeInMapCmd : public ICommand {
  public:
    LocalizeInMapCmd(const std::string_view& folder_name, int64_t timestamp_ns, const Isometry3T& guess_pose,
                     const sof::Images& images, const Slam::LocalizationSettings& settings,
                     Slam::LocalizeStartCB start_cb, Slam::LocalizeFinishCB finish_cb);

    ~LocalizeInMapCmd() override = default;
    void Execute(AsyncSlam& async_slam, FrameId, const Isometry3T&) override;

  private:
    const std::string folder_name_;
    const int64_t timestamp_ns_;
    const Isometry3T guess_pose_;
    const sof::Images images_;
    const Slam::LocalizationSettings settings_;
    Slam::LocalizeStartCB start_cb_;
    Slam::LocalizeFinishCB finish_cb_;
  };

  class CopyToDatabaseCmd : public ICommand {
  public:
    std::string path_;
    explicit CopyToDatabaseCmd(std::string path) : path_(std::move(path)) {}
    ~CopyToDatabaseCmd() override = default;

    void Execute(AsyncSlam& async_slam, FrameId, const Isometry3T&) override;
  };

  // --- Immutable after construction; read by both the caller's thread and the background worker thread ---
  const camera::Rig rig_;
  const std::vector<CameraId> cameras_;
  const AsyncSlamOptions options_;

  // --- Accessed only from the caller's thread (e.g. TrackResult(), GetPoseForFrame() callers) ---
  bool reproduce_mode_ = false;
  std::thread thread_;
  bool is_first_frame_ = true;
  VOTrackData track_data_;
  std::map<FrameId, VOTrackData> trajectory_;
  AsyncSlamLCTelemetry last_telemetry_;  // cache of the telemetry last popped from telemetry_queue_

  // --- Shared between the caller's thread and the background worker thread (see async_slam_worker.cpp) ---
  mutable std::mutex slam_mutex_;
  std::unique_ptr<LocalizerAndMapper> slam_;  // protected by slam_mutex_
  Tail tail_;                                 // thread-safe
  std::mutex processing_images_mutex_;
  Images processing_images_;  // protected by processing_images_mutex_
  ThreadSafeQueue<std::shared_ptr<VOKeyframeInfo>> input_queue_;
  ThreadSafeQueue<std::shared_ptr<AsyncSlamLCTelemetry>> telemetry_queue_;
  std::list<LoopClosureStamped> last_loop_closures_stamped_;     // TODO: check for raise condition
  std::shared_ptr<ViewManager<ViewLandmarks>> landmarks_view_;   // TODO: check for raise condition
  std::shared_ptr<ViewManager<ViewLandmarks>> loop_close_view_;  // TODO: check for raise condition
  std::shared_ptr<ViewManager<ViewPoseGraph>> pose_graph_view_;  // TODO: check for raise condition
  std::function<void(bool)> copy_to_database_callback_;          // TODO: check for raise condition
  const profiler::SLAMProfiler::DomainHelper profiler_domain_ = profiler::SLAMProfiler::DomainHelper("SLAM");
  const uint32_t profiler_color_ = 0x00FF00;

  // --- Accessed only from the background worker thread ---
  const std::unique_ptr<ILoopClosureSolver> loop_closure_solver_;
  // max size for the list of last loop closure poses with timestamps/frame_ids
  static constexpr uint32_t max_num_last_lcs_ = 10;
  const uint64_t throttling_time_ns_;  // min time interval allowed between successive loop closures

  void Shutdown();

  // Background thread entry point (launched via thread_ = std::thread{&AsyncSlam::Run_worker, this}).
  void Run_worker();

  void ProcessInput_worker();

  // Pops all pending items from input_queue_, executing commands and adding VO keyframes to slam_.
  // Writes the last processed keyframe's frame_id/timestamp_ns/world_from_rig_guess (used by the caller
  // to run loop closure / pose graph optimization once on the freshest state of the batch).
  // Returns true if at least one VO keyframe was added.
  bool AddKeyframesAndRunCommands_worker(FrameId& frame_id, uint64_t& timestamp_ns, Isometry3T& world_from_rig_guess);

  // Returns the current processing_images_ if they match frame_id, false otherwise.
  bool GetLatestProcessingImages_worker(FrameId frame_id, Images& current_images);

  // Loop closure detection + pose graph optimization for the given batch's latest state.
  void DetectLoopClosure_worker(FrameId frame_id, uint64_t timestamp_ns, const Isometry3T& world_from_rig_guess,
                                const Images& current_images);

  // Runs pose graph optimization if lc_found or planar_constraints. Returns true if optimization ran.
  bool OptimizePoseGraph_worker(bool lc_found);

  // Publishes landmarks and pose graph to their respective views, if attached.
  void PublishViews_worker(uint64_t timestamp_ns);

  // Copy to database
  bool CopyToDatabase_worker(const std::string& path);

  // reset VOFrameData (if data was post to ProcessVOFrameData)
  static void VO_ResetFrameData(FrameId frame_id, uint64_t timestamp_ns, VOTrackData& track_data);

  // Increment FrameData value after tracker.Solve()
  // pose_estimate_rel - relative to previous position
  static void VO_IncrementFrameData(FrameId frame_id, uint64_t timestamp_ns, const Isometry3T& pose_estimate_rel,
                                    VOTrackData& track_data);

  static std::string FrameInformationString(const sof::Images& images);
};

}  // namespace cuvslam::slam

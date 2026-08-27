
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

#include "slam/async_slam/async_slam.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "common/coordinate_system.h"
#include "common/rerun.h"
#include "common/thread_safe_queue.h"
#include "common/unaligned_types.h"
#include "cuvslam/internal.h"
#include "pipelines/visualizer.h"
#include "profiler/profiler.h"

#include "slam/map/database/lmdb_slam_database.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"
#include "slam/slam/slam.h"
#include "slam/view/view_landmarks.h"
#include "sof/image_manager.h"
#include "visualizer/visualizer.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define CALLBACK_AND_RETURN_IF(condition, callback, type, message) \
  if (condition) {                                                 \
    callback(Result<type>::Error(message));                        \
    return;                                                        \
  }

namespace {
using namespace cuvslam;

#ifdef USE_RERUN
void LogSlamPose(const Isometry3T& slam_pose) {
  const Vector3T& p = slam_pose.translation();
  thread_local std::vector<rerun::Position3D> poses(1);
  poses.clear();
  poses.emplace_back(p.x(), p.y(), p.z());
  visualizer::RerunVisualizer::getInstance().getRecordingStream().log(
      "world/slam_pose", rerun::Points3D(poses).with_colors(Color(0, 255, 0)).with_radii(5.f));
}
#endif

}  // namespace

namespace cuvslam::slam {

AsyncSlam::AsyncSlam(const camera::Rig& rig, const std::vector<CameraId>& cameras, const AsyncSlamOptions& options)
    : rig_(rig),
      cameras_(cameras),
      options_(options),
      slam_(std::make_unique<LocalizerAndMapper>(rig, FeatureDescriptorType::kShiTomasi6, options.use_gpu)),
      tail_(options.retention_time_ms * 1'000'000ULL),
      loop_closure_solver_(
          CreateLoopClosureSolver(options.loop_closure_solver_type, RansacType::kPnP, !options.reproduce_mode, rig_)),
      throttling_time_ns_(options.throttling_time_ms * 1'000'000) {
  reproduce_mode_ = options.reproduce_mode;

  // no second thread here - safe access to all data
  slam_->SetReproduceMode(reproduce_mode_);
  slam_->SetLandmarksSpatialIndex(options.spatial_index_options);
  slam_->SetPoseGraphOptimizerOptions(options.pgo_options);
  slam_->SetActiveCameras(cameras_);

  if (options.max_pose_graph_nodes) {
    slam_->SetKeyframesLimit(options.max_pose_graph_nodes);
  }
  if (options.pose_for_frame_required) {
    slam_->SetKeepTrackPoses(true);
  }

  tail_.Clear();

  if (!options.map_cache_path.empty()) {
    if (!slam_->AttachToNewDatabase(options.map_cache_path)) {
      throw std::runtime_error("Failed to connect SLAM to map at " + options.map_cache_path);
    }
  }

  if (!reproduce_mode_) {
    SlamStdout("Starting background thread");
    thread_ = std::thread{&AsyncSlam::Run_worker, this};
  } else {
    SlamStdout("Do not start background thread because reproduce_mode is set.");
  }
}
AsyncSlam::~AsyncSlam() {
  Shutdown();
  SlamStdout("Destroyed AsyncSlam instance. ");
}

void AsyncSlam::TrackResult(const FrameId frameId, const int64_t timestamp_ns,
                            const odom::IVisualOdometry::VOFrameStat& stat, const sof::Images& images,
                            const Isometry3T& delta) {
  TRACE_EVENT ev = profiler_domain_.trace_event("AsyncSlam::TrackResult()", profiler_color_);

  assert(track_data_.from_keyframe.matrix().allFinite());

  // extract telemetry
  {
    std::shared_ptr<AsyncSlamLCTelemetry> async_slam_telemetry;
    while (telemetry_queue_.TryPop(async_slam_telemetry)) {
      if (async_slam_telemetry) {
        last_telemetry_ = *async_slam_telemetry;
      }
    }
  }

  const bool is_keyframe = stat.keyframe;

  // first frame
  if (is_first_frame_) {
    VO_ResetFrameData(frameId, timestamp_ns, track_data_);
    tail_.Clear();
    is_first_frame_ = false;
  }

  VO_IncrementFrameData(frameId, timestamp_ns, delta, track_data_);

  const bool has_images = std::any_of(images.begin(), images.end(), [](const auto& image) { return image != nullptr; });
  if (is_keyframe && has_images) {
    const auto vo_keyframe = std::make_shared<VOKeyframeInfo>(VOKeyframeInfo());
    const Isometry3T current_pose = GetSlamPose();

    vo_keyframe->vo_pose_at_this_frame = current_pose;
    vo_keyframe->track_data = track_data_;
    vo_keyframe->frame_data.frame_id = frameId;
    vo_keyframe->frame_data.timestamp_ns = timestamp_ns;
    vo_keyframe->frame_data.frame_information = FrameInformationString(images);

    {
      std::lock_guard image_guard(processing_images_mutex_);
      processing_images_ = images;
    }
    {
      vo_keyframe->frame_data.tracks2d_norm.reserve(stat.tracks2d.size());

      std::unordered_set<TrackId> added_tracks;
      int invalid_cam_id_count = 0;
      for (const auto& [cam_id, track_id, uv] : stat.tracks2d) {
        if (stat.tracks3d.find(track_id) == stat.tracks3d.end()) {
          continue;  // remove landmarks without 3d
        }
        const Vector3T& v = stat.tracks3d.at(track_id);
        if (v.norm() > options_.max_landmarks_distance) {
          continue;  // remove landmarks outside the max distance
        }

        if (cam_id >= images.size() || images[cam_id] == nullptr) {
          continue;
        }
        if (cam_id >= rig_.num_cameras) {
          ++invalid_cam_id_count;
          continue;  // skip invalid camera ID
        }
        const auto& intrinsics = rig_.intrinsics[cam_id];
        if (intrinsics == nullptr) {
          SlamStderr("Intrinsics for camera %zu should not be null", static_cast<size_t>(cam_id));
          continue;
        }
        Vector2T uv_norm;
        if (!intrinsics->normalizePoint(uv, uv_norm)) {
          continue;
        }
        vo_keyframe->frame_data.tracks2d_norm.emplace_back(VOFrameData::Track2DXY{cam_id, track_id, uv_norm});
        added_tracks.insert(track_id);
      }
      if (invalid_cam_id_count > 0) {
        SlamStdout("Skipped %d track(s) with invalid camera id", invalid_cam_id_count);
      }
      // xyz to camera space
      for (const auto& [track_id, xyz_rel] : stat.tracks3d) {
        if (added_tracks.find(track_id) == added_tracks.end()) {
          continue;
        }
        vo_keyframe->frame_data.tracks3d_rel[track_id] = xyz_rel;
      }
    }
    if (tail_.UpdateTimeByOdometry(timestamp_ns, current_pose)) {
      input_queue_.Push(vo_keyframe);
      const size_t queue_size = input_queue_.Size();
      TraceWarningIf(queue_size > options_.delay_warning_queue_size,
                     "SLAM is behind odometry: %zu commands are queued to the SLAM thread that is more than desired "
                     "%u. Check SLAM settings: reduce max_map_size or increase throttling_time_ms.",
                     queue_size, options_.delay_warning_queue_size);
    }
    VO_ResetFrameData(frameId, timestamp_ns, track_data_);
  }
  if (options_.pose_for_frame_required) {
    trajectory_[frameId] = track_data_;
  }

#ifdef USE_RERUN
  {
    const Isometry3T current_pose = GetSlamPose();
    LogSlamPose(current_pose);
    RERUN(pipelines::logTrajectory, current_pose.inverse(), "world/trajectories/slam", Color(255, 255, 255),
          TrajectoryType::GT);
  }
#endif
}

Isometry3T AsyncSlam::GetSlamPose() const {
  const auto may_be_tip = tail_.GetTip();  // thread-safe
  if (!may_be_tip) {
    return Isometry3T::Identity();
  }
  const Isometry3T& tail_tip = may_be_tip->second;
  Isometry3T slam_pose = tail_tip * track_data_.from_keyframe;  // track_data_ touched only in main thread

  if (options_.planar_constraints) {
    slam_pose.translation().y() = 0;
  }
  return slam_pose;
}

void AsyncSlam::ProcessInputSynchronously() {
  if (reproduce_mode_) {
    ProcessInput_worker();
  }
}

// stop thread
void AsyncSlam::Stop() {
  Shutdown();
  reproduce_mode_ = true;
}

bool AsyncSlam::GetPoseForFrame(const FrameId frameId, Isometry3T& pose) const {
  pose.setIdentity();

  if (!options_.pose_for_frame_required) {
    return false;
  }

  const auto it = trajectory_.find(frameId);
  if (it == trajectory_.end()) {
    SlamStderr("Pose not found for frame %zd.\n", static_cast<uint64_t>(frameId));
    return false;
  }
  auto& track_data = it->second;
  assert(track_data.from_keyframe.matrix().allFinite());

  Isometry3T m;
  std::lock_guard slam_guard(slam_mutex_);
  if (slam_->CalcFramePose(track_data.end_frame_id, m)) {
    pose = m * track_data.from_keyframe;
    return true;
  }

  return false;
}

bool AsyncSlam::GetPosesForAllFrames(std::map<uint64_t, storage::Isometry3<float>>& frames) const {
  if (!options_.pose_for_frame_required) {
    return false;
  }

  std::lock_guard slam_guard(slam_mutex_);
  for (const auto& [frame_id, track_data] : trajectory_) {
    (void)frame_id;
    assert(track_data.from_keyframe.matrix().allFinite());

    Isometry3T m;
    if (slam_->CalcFramePose(track_data.end_frame_id, m)) {
      const Isometry3T pose = m * track_data.from_keyframe;
      frames[track_data.timestamp_ns] = pose;
    }
  }
  return true;
}

bool AsyncSlam::GetLastTelemetry(AsyncSlamLCTelemetry& telemetry) const {
  telemetry = last_telemetry_;
  return true;
}

const std::list<LoopClosureStamped>& AsyncSlam::GetLastLoopClosuresStamped() { return last_loop_closures_stamped_; }

void AsyncSlam::CopyToDatabase(const std::string& path, const std::function<void(bool)>& callback) {
  const auto copy_to_database_cmd = std::make_shared<CopyToDatabaseCmd>(path);
  assert(!copy_to_database_callback_);
  copy_to_database_callback_ = callback;
  TraceDebug("AttachToNewDatabaseSaveMapAndDetach reproduce_mode_=%d", reproduce_mode_ ? 1 : 0);
  if (reproduce_mode_) {
    // sync
    constexpr FrameId end_frame_id = ~0UL;
    const Isometry3T vo_pose_at_that_frame = Isometry3T::Identity();
    copy_to_database_cmd->Execute(*this, end_frame_id, vo_pose_at_that_frame);
  } else {
    // async
    const auto vo_keyframe = std::make_shared<VOKeyframeInfo>(VOKeyframeInfo());
    vo_keyframe->command = copy_to_database_cmd;
    input_queue_.Push(vo_keyframe);
  }
}

void AsyncSlam::CopyToDatabaseCmd::Execute(AsyncSlam& async_slam, FrameId, const Isometry3T&) {
  async_slam.CopyToDatabase_worker(path_);
}

// Set landmarks view
void AsyncSlam::SetLandmarksView(std::shared_ptr<ViewManager<ViewLandmarks>> view) {
  this->landmarks_view_ = std::move(view);
}

// Set loop closure view
void AsyncSlam::SetLoopClosureView(std::shared_ptr<ViewManager<ViewLandmarks>> view) {
  this->loop_close_view_ = std::move(view);
}

// Set pose graph view
void AsyncSlam::SetPoseGraphView(std::shared_ptr<ViewManager<ViewPoseGraph>> view) {
  this->pose_graph_view_ = std::move(view);
}

void AsyncSlam::Shutdown() {
  input_queue_.Abort();
  telemetry_queue_.Abort();

  if (thread_.joinable()) {
    thread_.join();
  }
  // Copy to database
  std::lock_guard slam_guard(slam_mutex_);
  slam_->DetachDatabase();
}

std::string AsyncSlam::FrameInformationString(const sof::Images& images) {
#ifdef CUVSLAM_LOG_ENABLE
  Json::Value json;

  const auto first_image =
      std::find_if(images.begin(), images.end(), [](const auto& image) { return image != nullptr; });
  if (first_image != images.end()) {
    auto& meta_0 = (*first_image)->get_image_meta();
    json["frame_id"] = static_cast<Json::UInt64>(meta_0.frame_id);
    json["timestamp"] = static_cast<Json::UInt64>(meta_0.timestamp);
    json["frame_number"] = meta_0.frame_number;
    for (CameraId cam_id = 0; cam_id < images.size(); ++cam_id) {
      const auto& img = images[cam_id];
      if (img != nullptr) {
        json["image_file" + std::to_string(cam_id)] = img->get_image_meta().filename;
      }
    }
  }

  std::string jsonStr = writeString(Json::StreamWriterBuilder(), json);
  return jsonStr;
#else
  return "";
#endif
}

// reset VOFrameData (if data was post to ProcessVOFrameData)
void AsyncSlam::VO_ResetFrameData(const FrameId frame_id, const uint64_t timestamp_ns, VOTrackData& track_data) {
  track_data.end_frame_id = frame_id;
  track_data.timestamp_ns = timestamp_ns;
  track_data.from_keyframe = Isometry3T::Identity();
}

void AsyncSlam::VO_IncrementFrameData(const FrameId frame_id, const uint64_t timestamp_ns,
                                      const Isometry3T& pose_estimate_rel, VOTrackData& track_data) {
  track_data.end_frame_id = frame_id;
  track_data.timestamp_ns = timestamp_ns;
  track_data.from_keyframe = track_data.from_keyframe * pose_estimate_rel;
}

}  // namespace cuvslam::slam

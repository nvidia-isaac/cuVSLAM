
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

#include "common/log_types.h"
#include "common/rerun.h"
#include "common/thread_safe_queue.h"
#include "cuvslam/internal.h"
#include "profiler/profiler.h"
#include "slam/slam/slam.h"
#include "slam/view/map_to_view.h"
#include "slam/view/view_landmarks.h"
#include "visualizer/visualizer.hpp"

namespace cuvslam::slam {

// Run_worker() is the background thread entry point. The other methods below also run on that thread
// in async mode, or synchronously on the caller's thread via ProcessInputSynchronously() in reproduce_mode.

void AsyncSlam::Run_worker() {
#ifdef USE_CUDA
  if (options_.use_gpu) {
    cudaSetDevice(0);
  }
#endif
  for (;;) {
    // wait for news in input_queue
    if (!input_queue_.Wait()) {
      // aborted
      return;
    }
    ProcessInput_worker();
  }
}

bool AsyncSlam::AddKeyframesAndRunCommands_worker(FrameId& frame_id, uint64_t& timestamp_ns,
                                                  Isometry3T& world_from_rig_guess) {
  FrameId end_frame_id = InvalidFrameId;
  Isometry3T vo_pose_at_that_frame = Isometry3T::Identity();
  bool has_input = false;

  for (;;) {
    TRACE_EVENT ev = profiler_domain_.trace_event("process vo data", profiler_color_);

    // extract all from input_queue
    std::shared_ptr<VOKeyframeInfo> vo_kf;
    {
      TRACE_EVENT ev_input = profiler_domain_.trace_event("input", profiler_color_);
      if (!input_queue_.TryPop(vo_kf)) {
        break;  // input_queue_ is empty
      }
    }

    const auto& frame_data = vo_kf->frame_data;
    RERUN(visualizer::RerunVisualizer::getInstance().setupTimeline, frame_data.frame_id, frame_data.timestamp_ns);

    // execute command from input_queue_
    if (vo_kf->command) {
      std::shared_ptr<ICommand>& cmd = vo_kf->command;
      cmd->Execute(*this, end_frame_id, vo_pose_at_that_frame);
      continue;
    }

    Images current_images;
    const bool is_valid_image = GetLatestProcessingImages_worker(frame_data.frame_id, current_images);
    Isometry3T pose_estimate_slam;
    {
      const VOTrackData& track_data = vo_kf->track_data;

      Isometry3T last_keyframe_pose;
      int64_t last_keyframe_ts;
      std::lock_guard slam_guard(slam_mutex_);
      if (slam_->GetLastKeyframePoseAndTimestamp(last_keyframe_pose, last_keyframe_ts) && last_keyframe_ts > 0 &&
          track_data.timestamp_ns < static_cast<uint64_t>(last_keyframe_ts)) {
        continue;  // no need to add keyframe in past if slam in a future
      }
      slam_->AddKeyframe(track_data.from_keyframe, frame_data, is_valid_image ? current_images : Images());
      pose_estimate_slam = slam_->GetCurrentPose();
    }
    SlamStdout("'");

    {
      TRACE_EVENT ev_cd = profiler_domain_.trace_event("copy data", profiler_color_);

      frame_id = frame_data.frame_id;
      timestamp_ns = frame_data.timestamp_ns;
      end_frame_id = vo_kf->track_data.end_frame_id;
      vo_pose_at_that_frame = vo_kf->vo_pose_at_this_frame;
      world_from_rig_guess = pose_estimate_slam;
    }

    vo_kf.reset();
    has_input = true;
  }
  return has_input;
}

bool AsyncSlam::GetLatestProcessingImages_worker(FrameId frame_id, Images& current_images) {
  {
    std::lock_guard image_guard(processing_images_mutex_);
    current_images = processing_images_;
  }
  const auto current_image =
      std::find_if(current_images.begin(), current_images.end(), [](const auto& image) { return image != nullptr; });
  return current_image != current_images.end() && ((*current_image)->get_image_meta().frame_id == frame_id);
}

void AsyncSlam::DetectLoopClosure_worker(FrameId frame_id, uint64_t timestamp_ns,
                                         const Isometry3T& world_from_rig_guess, const Images& current_images) {
  // init last_step_telemetry_
  AsyncSlamLCTelemetry last_step_telemetry;
  last_step_telemetry.timestamp_ns = timestamp_ns;

  TRACE_EVENT ev = profiler_domain_.trace_event("LC & optimization", profiler_color_);

  bool skip_loop_closure = false;
  if (!last_loop_closures_stamped_.empty()) {
    // no need for LC if previous LC was recent; treat out-of-order timestamps as within the window
    const uint64_t last_lc_ts = last_loop_closures_stamped_.back().timestamp_ns;
    if (timestamp_ns < last_lc_ts || (timestamp_ns - last_lc_ts) < throttling_time_ns_) {
      skip_loop_closure = true;  // loop closure not needed
    }
  }

  //--- Loop Closure detection ---
  if (loop_closure_solver_ == nullptr || skip_loop_closure) {
    return;
  }

  LocalizerAndMapper::LoopClosureStatus lc_status;
  bool lc_found = false;
  {
    std::lock_guard slam_guard(slam_mutex_);
    slam_->DetectLoopClosure(*loop_closure_solver_, current_images, world_from_rig_guess, lc_status);
    lc_found = lc_status.success;
  }

  // view loop closure
  std::shared_ptr<ViewLandmarks> lc_view = loop_close_view_ ? loop_close_view_->acquire_earliest() : nullptr;
  if (lc_view) {
    std::lock_guard slam_guard(slam_mutex_);
    PublishLoopClosureToView(slam_->GetMap(), lc_status.landmarks, *lc_view);
    lc_view->timestamp_ns = timestamp_ns;
    lc_view.reset();
  }

  // Update Landmark Statistic in spatial index
  {
    std::lock_guard slam_guard(slam_mutex_);
    slam_->UpdateLandmarkProbeStatistics(lc_status.discarded_landmarks);
  }
  // Add LC edge and Add Landmark Relation to spatial index and pose graph
  if (lc_found) {
    std::lock_guard slam_guard(slam_mutex_);
    if (slam_->ApplyLoopClosureResult(lc_status.result_pose, lc_status.result_pose_covariance, lc_status.landmarks)) {
      const LoopClosureStamped lc_pose_stamped = {frame_id, timestamp_ns, lc_status.result_pose};
      last_loop_closures_stamped_.push_back(lc_pose_stamped);
      while (last_loop_closures_stamped_.size() > max_num_last_lcs_) {
        last_loop_closures_stamped_.pop_front();
      }
    } else {
      SlamStdout("Can't apply loop closure result");
    }
  }

  last_step_telemetry.lc_status = lc_found;
  last_step_telemetry.lc_selected_landmarks_count = lc_status.selected_landmarks_count;
  last_step_telemetry.lc_tracked_landmarks_count = lc_status.tracked_landmarks_count;
  last_step_telemetry.lc_pnp_landmarks_count = lc_status.pnp_landmarks_count;
  last_step_telemetry.lc_good_landmarks_count = lc_status.good_landmarks_count;
  if (lc_found) {
    SlamStdout("S");  // Successful LC
  }
  last_step_telemetry.pgo_status = OptimizePoseGraph_worker(lc_found);
  telemetry_queue_.Push(std::make_shared<AsyncSlamLCTelemetry>(last_step_telemetry));
}

bool AsyncSlam::OptimizePoseGraph_worker(bool lc_found) {
  bool optimization_happens = false;
  if (lc_found || options_.planar_constraints) {
    // TODO: ? optimize_options.keyframes_in_sight = loop_closure_status.keyframes_in_sight;
    std::lock_guard slam_guard(slam_mutex_);
    optimization_happens = slam_->OptimizePoseGraph(options_.planar_constraints);
  }
  if (!optimization_happens) {
    return false;
  }
  {
    std::lock_guard slam_guard(slam_mutex_);
    const Isometry3T pose_estimate_slam = slam_->GetCurrentPose();
    TRACE_EVENT ev1 = profiler_domain_.trace_event("post optimization", profiler_color_);
    log::Value<LogFrames>("pose_slam", pose_estimate_slam);

    int64_t last_keyframe_ts;
    Isometry3T last_keyframe_pose;
    if (slam_->GetLastKeyframePoseAndTimestamp(last_keyframe_pose, last_keyframe_ts)) {
      if (!tail_.UpdatePoseBySLAM(last_keyframe_ts, last_keyframe_pose)) {
        TraceWarning(
            "Failed to update SLAM tail after pose graph optimization: keyframe timestamp is outside retention.");
      }
    }
  }
  SlamStdout(":");
  return true;
}

void AsyncSlam::PublishViews_worker(uint64_t timestamp_ns) {
  // view landmarks
  std::shared_ptr<ViewLandmarks> landmarks_view = landmarks_view_ ? landmarks_view_->acquire_earliest() : nullptr;
  if (landmarks_view) {
    std::lock_guard slam_guard(slam_mutex_);
    landmarks_view->landmarks.clear();
    PublishAllLandmarksToView(slam_->GetMap(), timestamp_ns, *landmarks_view);
    landmarks_view.reset();
  }

  // view pose graph
  std::shared_ptr<ViewPoseGraph> pose_graph_view = pose_graph_view_ ? pose_graph_view_->acquire_earliest() : nullptr;
  if (pose_graph_view) {
    std::lock_guard slam_guard(slam_mutex_);
    PublishPoseGraphToView(slam_->GetMap(), timestamp_ns, *pose_graph_view);
    pose_graph_view.reset();
  }
}

void AsyncSlam::ProcessInput_worker() {
  FrameId frame_id = InvalidFrameId;
  uint64_t timestamp_ns = 0;
  Isometry3T world_from_rig_guess;

  const bool has_input = AddKeyframesAndRunCommands_worker(frame_id, timestamp_ns, world_from_rig_guess);
  {
    std::lock_guard slam_guard(slam_mutex_);
    slam_->ReduceKeyframes();
  }
  if (!has_input) {
    return;
  }

  Images current_images;
  const bool is_valid_image = GetLatestProcessingImages_worker(frame_id, current_images);
  if (is_valid_image) {
    DetectLoopClosure_worker(frame_id, timestamp_ns, world_from_rig_guess, current_images);
  }

  PublishViews_worker(timestamp_ns);

  if (input_queue_.IsEmpty()) {
    std::lock_guard slam_guard(slam_mutex_);
    slam_->FlushActiveDatabase();
  }
}

bool AsyncSlam::CopyToDatabase_worker(const std::string& path) {
  bool status;
  {
    std::lock_guard slam_guard(slam_mutex_);
    status = slam_->AttachToNewDatabaseSaveMapAndDetach(path);
  }
  if (copy_to_database_callback_) {
    copy_to_database_callback_(status);
  }
  copy_to_database_callback_ = nullptr;
  return status;
}

}  // namespace cuvslam::slam

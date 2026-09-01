
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
#include <string>

#include "common/coordinate_system.h"
#include "common/unaligned_types.h"
#include "cuvslam/internal.h"

#include "slam/localizer/localizer.h"

#define CALLBACK_AND_RETURN_IF(condition, callback, type, message) \
  if (condition) {                                                 \
    if (callback) {                                                \
      callback(Result<type>::Error(message));                      \
    }                                                              \
    return;                                                        \
  }

namespace cuvslam::slam {

namespace {

void AddFakeKeyframeForLocalizedPose(LocalizerAndMapper& slam, KeyFrameId from_keyframe_id,
                                     const Isometry3T& pose_in_slam, int64_t timestamp_ns, const sof::Images& images,
                                     const std::string& frame_information) {
  const auto first_image =
      std::find_if(images.begin(), images.end(), [](const auto& image) { return image != nullptr; });
  if (first_image == images.end()) {
    return;
  }
  const Isometry3T* maybe_from_keyframe_pose = slam.GetMap().GetPoseGraphHypothesis().GetKeyframePose(from_keyframe_id);

  if (!maybe_from_keyframe_pose) {
    return;
  }
  const Isometry3T& from_keyframe_pose = *maybe_from_keyframe_pose;
  const Isometry3T delta = from_keyframe_pose.inverse() * pose_in_slam;

  VOFrameData empty_frame_data;
  // all frame_ids are guaranteed to be the same
  empty_frame_data.frame_id = (*first_image)->get_image_meta().frame_id;
  empty_frame_data.timestamp_ns = timestamp_ns;
  empty_frame_data.frame_information = frame_information;
  slam.AddKeyframe(delta, empty_frame_data, images);
}

void AddFakeKeyframeForLastTailPose(LocalizerAndMapper& slam, const Tail& tail) {
  const auto may_be_tip = tail.GetTip();
  if (!may_be_tip) {
    return;
  }
  const int64_t tail_timestamp_ns = may_be_tip->first;
  const Isometry3T& tail_pose = may_be_tip->second;

  const Isometry3T delta = slam.GetCurrentPose().inverse() * tail_pose;

  VOFrameData empty_frame_data;
  empty_frame_data.frame_id = 0;
  empty_frame_data.timestamp_ns = tail_timestamp_ns;
  empty_frame_data.frame_information = "";
  slam.AddKeyframe(delta, empty_frame_data, {});
}

}  // namespace

// Should be very fast. Just put task to the queue to not block main thread.
void AsyncSlam::LocalizeInMap(const std::string_view& folder_name, int64_t timestamp_ns, const Isometry3T& guess_pose,
                              const sof::Images& images, const Slam::LocalizationSettings& settings,
                              Slam::LocalizeStartCB start_cb, Slam::LocalizeFinishCB finish_cb) {
  const auto cmd =
      std::make_shared<LocalizeInMapCmd>(folder_name, timestamp_ns, guess_pose, images, settings, start_cb, finish_cb);
  const auto vo_keyframe = std::make_shared<VOKeyframeInfo>(VOKeyframeInfo());
  vo_keyframe->command = cmd;
  input_queue_.Push(vo_keyframe);
  if (reproduce_mode_) {
    ProcessInputSynchronously();
  }
}

// This constructor should be very fast - all work should be offloaded to ::Execute
AsyncSlam::LocalizeInMapCmd::LocalizeInMapCmd(const std::string_view& folder_name, int64_t timestamp_ns,
                                              const Isometry3T& guess_pose, const sof::Images& images,
                                              const Slam::LocalizationSettings& settings,
                                              Slam::LocalizeStartCB start_cb, Slam::LocalizeFinishCB finish_cb)
    : folder_name_(folder_name),
      timestamp_ns_(timestamp_ns),
      guess_pose_(guess_pose),
      images_(images),
      settings_(settings),
      start_cb_(start_cb),
      finish_cb_(finish_cb) {}

void AsyncSlam::LocalizeInMapCmd::Execute(AsyncSlam& async_slam, FrameId, const Isometry3T&) {
  if (start_cb_) {
    start_cb_();
  }

  LocalizerOptions options;
  {
    options.use_gpu = async_slam.options_.use_gpu;
    options.reproduce_mode = async_slam.options_.reproduce_mode;
    options.horizontal_search_radius = settings_.horizontal_search_radius;
    options.vertical_search_radius = settings_.vertical_search_radius;
    options.horizontal_step = settings_.horizontal_step;
    options.vertical_step = settings_.vertical_step;
    options.angle_step_rads = settings_.angular_step_rads;
  }

  // TODO: keep localizer to safe memory
  const auto localizer = std::make_unique<Localizer>();  // only for the duration of the localization
  localizer->Init(async_slam.rig_, options);
  CALLBACK_AND_RETURN_IF(!localizer->OpenDatabase(std::string{folder_name_}), finish_cb_, Pose,
                         "Failed to open database.");

  LocalizationResult result;
  CALLBACK_AND_RETURN_IF(!localizer->Localize(guess_pose_, images_, result), finish_cb_, Pose,
                         "Can't localize in map using provided image and guess");

  CALLBACK_AND_RETURN_IF(!result.slam_from->SelectHeadKeyframe(result.from_keyframe_id, timestamp_ns_), finish_cb_,
                         Pose, "Internal error during set head keyframe.");

  PoseGraphOptimizerOptions pgo_opt;
  std::optional<std::vector<CameraId>> maybe_active_cameras;
  {
    std::lock_guard slam_guard(async_slam.slam_mutex_);
    pgo_opt = async_slam.slam_->GetPoseGraphOptimizerOptions();
    result.slam_from->SetReproduceMode(async_slam.slam_->GetReproduceMode());
    result.slam_from->SetLandmarksSpatialIndex(async_slam.slam_->GetLandmarksSpatialIndexOptions());
    result.slam_from->SetKeyframesLimit(async_slam.slam_->GetKeyframesLimit());
    result.slam_from->SetKeepTrackPoses(async_slam.slam_->GetKeepTrackPoses());
    maybe_active_cameras = async_slam.slam_->GetActiveCameras();
  }
  CALLBACK_AND_RETURN_IF(!result.slam_from->SetPoseGraphOptimizerOptions(pgo_opt), finish_cb_, Pose,
                         "Internal error during set PoseGraphOptimizerOptions.");

  if (maybe_active_cameras) {
    result.slam_from->SetActiveCameras(maybe_active_cameras.value());
  }
  result.slam_from->RebuildSpatialIndex();
  result.slam_from->ReduceKeyframes();

  bool tail_updated = false;
  {
    std::lock_guard slam_guard(async_slam.slam_mutex_);
    tail_updated = async_slam.tail_.UpdatePoseBySLAM(timestamp_ns_, result.pose_in_slam);
    if (tail_updated) {
      std::swap(async_slam.slam_, result.slam_from);
      AddFakeKeyframeForLocalizedPose(*async_slam.slam_, result.from_keyframe_id, result.pose_in_slam, timestamp_ns_,
                                      images_, FrameInformationString(images_));
      AddFakeKeyframeForLastTailPose(*async_slam.slam_, async_slam.tail_);
    }
  }
  CALLBACK_AND_RETURN_IF(!tail_updated, finish_cb_, Pose,
                         "Localization timestamp is outside the SLAM retention window. LocalizeInMap can initialize "
                         "an empty SLAM state, but once SLAM has retained poses the timestamp must be within that "
                         "window.");

  if (finish_cb_) {
    finish_cb_(Result<Pose>::Success(ConvertIsometryToPose(result.pose_in_slam)));
  }
}

}  // namespace cuvslam::slam

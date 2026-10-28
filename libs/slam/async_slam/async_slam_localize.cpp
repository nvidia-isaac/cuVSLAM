
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
void AsyncSlam::LocalizeInMap(const std::string_view& folder_name, int64_t timestamp_ns,
                              const std::optional<Isometry3T>& guess_pose, const sof::Images& images,
                              const Slam::LocalizationSettings& settings, bool attach, Slam::LocalizeStartCB start_cb,
                              Slam::LocalizeFinishCB finish_cb) {
  const auto cmd = std::make_shared<LocalizeInMapCmd>(folder_name, timestamp_ns, guess_pose, images, settings, attach,
                                                      start_cb, finish_cb);
  const auto vo_keyframe = std::make_shared<VOKeyframeInfo>(VOKeyframeInfo());
  vo_keyframe->command = cmd;
  input_queue_.Push(vo_keyframe);
  if (reproduce_mode_) {
    ProcessInputSynchronously();
  }
}

// This constructor should be very fast - all work should be offloaded to ::Execute
AsyncSlam::LocalizeInMapCmd::LocalizeInMapCmd(const std::string_view& folder_name, int64_t timestamp_ns,
                                              const std::optional<Isometry3T>& guess_pose, const sof::Images& images,
                                              const Slam::LocalizationSettings& settings, bool attach,
                                              Slam::LocalizeStartCB start_cb, Slam::LocalizeFinishCB finish_cb)
    : folder_name_(folder_name),
      timestamp_ns_(timestamp_ns),
      guess_pose_(guess_pose),
      images_(images),
      settings_(settings),
      attach_(attach),
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
    options.vpr_options = async_slam.options_.vpr_options;
  }

  bool own_map = false;
  {
    std::lock_guard slam_guard(async_slam.slam_mutex_);
    own_map = async_slam.slam_->HasWritableDatabase() && async_slam.slam_->IsDatabaseAt(folder_name_);
  }
  CALLBACK_AND_RETURN_IF(own_map, finish_cb_, Pose,
                         "The folder holds the map this session is writing; FlushMap saves it.");

  // TODO: keep localizer to safe memory
  const auto localizer = std::make_unique<Localizer>();  // only for the duration of the localization
  // Init reads the VPR engine and vocabulary files again, which may be gone since Slam was constructed.
  std::string init_error;
  try {
    localizer->Init(async_slam.rig_, options);
  } catch (const std::exception& error) {
    init_error = error.what();
  }
  CALLBACK_AND_RETURN_IF(!init_error.empty(), finish_cb_, Pose, init_error);
  CALLBACK_AND_RETURN_IF(!localizer->OpenDatabase(std::string{folder_name_}), finish_cb_, Pose,
                         "Failed to open database.");

  // Without a guess, place recognition proposes the candidate poses to verify.
  std::vector<Isometry3T> guess_poses;
  if (guess_pose_.has_value()) {
    guess_poses.push_back(*guess_pose_);
  } else {
    guess_poses = localizer->RecognizePlaces(images_, settings_.vpr_candidates);
    localizer->UseRecognizedPlaceProbes();
    CALLBACK_AND_RETURN_IF(guess_poses.empty(), finish_cb_, Pose,
                           "No guess pose was given and the map's place recognition index did not recognize this "
                           "place. The map has to have been saved with use_vpr, and with vpr_vocabulary_path's "
                           "vocabulary when that is set.");
  }

  LocalizationResult result;
  CALLBACK_AND_RETURN_IF(!localizer->Localize(guess_poses, images_, result), finish_cb_, Pose,
                         "Can't localize in map using provided image and guess");

  CALLBACK_AND_RETURN_IF(!result.slam_from->SelectHeadKeyframe(result.from_keyframe_id, timestamp_ns_), finish_cb_,
                         Pose, "Internal error during set head keyframe.");

  PoseGraphOptimizerOptions pgo_opt;
  std::optional<std::vector<CameraId>> maybe_active_cameras;
  {
    std::lock_guard slam_guard(async_slam.slam_mutex_);
    pgo_opt = async_slam.slam_->GetPoseGraphOptimizerOptions();
    result.slam_from->SetReproduceMode(async_slam.slam_->GetReproduceMode());
    result.slam_from->SetKeyframesLimit(async_slam.slam_->GetKeyframesLimit());
    result.slam_from->SetKeepTrackPoses(async_slam.slam_->GetKeepTrackPoses());
    maybe_active_cameras = async_slam.slam_->GetActiveCameras();
  }
  CALLBACK_AND_RETURN_IF(!result.slam_from->SetPoseGraphOptimizerOptions(pgo_opt), finish_cb_, Pose,
                         "Internal error during set PoseGraphOptimizerOptions.");

  if (maybe_active_cameras) {
    result.slam_from->SetActiveCameras(maybe_active_cameras.value());
  }
  // The session trims its map to max_map_size, which would drop parts of the loaded map, so it has to fit already.
  const size_t keyframes_limit = static_cast<size_t>(result.slam_from->GetKeyframesLimit());
  const size_t keyframe_count = result.slam_from->GetMap().GetPoseGraph().GetKeyframeCount();
  CALLBACK_AND_RETURN_IF(keyframes_limit != 0 && keyframe_count > keyframes_limit, finish_cb_, Pose,
                         "The map has " + std::to_string(keyframe_count) + " nodes, more than max_map_size " +
                             std::to_string(keyframes_limit) + "; set max_map_size to 0 or at least " +
                             std::to_string(keyframe_count) + ".");
  result.slam_from->RebuildSpatialIndex();
  // Nothing writes before the swap below, so a rejected localization leaves the folder as it was.
  CALLBACK_AND_RETURN_IF(attach_ && !result.slam_from->ReopenDatabaseForWriting(folder_name_), finish_cb_, Pose,
                         "Can't open the map folder for writing, or it changed since it was loaded, so it can't be "
                         "attached.");

  bool flushed = true;
  bool tail_updated = false;
  {
    std::lock_guard slam_guard(async_slam.slam_mutex_);
    // The session's map file gets everything mapped before the swap, also what Track added meanwhile in sync mode.
    if (async_slam.slam_->HasWritableDatabase()) {
      flushed = async_slam.slam_->FlushMapToDatabase();
    }
    tail_updated = flushed && async_slam.tail_.UpdatePoseBySLAM(timestamp_ns_, result.pose_in_slam);
    if (tail_updated) {
      std::swap(async_slam.slam_, result.slam_from);
      AddFakeKeyframeForLocalizedPose(*async_slam.slam_, result.from_keyframe_id, result.pose_in_slam, timestamp_ns_,
                                      images_, FrameInformationString(images_));
      AddFakeKeyframeForLastTailPose(*async_slam.slam_, async_slam.tail_);
    }
  }
  CALLBACK_AND_RETURN_IF(!flushed, finish_cb_, Pose, "Failed to write the session's map file; the session keeps it.");
  CALLBACK_AND_RETURN_IF(!tail_updated, finish_cb_, Pose,
                         "Localization timestamp is outside the SLAM retention window. LocalizeInMap can initialize "
                         "an empty SLAM state, but once SLAM has retained poses the timestamp must be within that "
                         "window.");

  if (finish_cb_) {
    finish_cb_(Result<Pose>::Success(ConvertIsometryToPose(result.pose_in_slam)));
  }
}

}  // namespace cuvslam::slam

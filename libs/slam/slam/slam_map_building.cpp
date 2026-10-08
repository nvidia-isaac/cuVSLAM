
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

#include <algorithm>
#include <functional>
#include <vector>

#include "common/isometry.h"
#include "common/rerun.h"
#include "common/types.h"
#include "common/vector_2t.h"

#include "slam/common/slam_common.h"
#include "slam/map/descriptor/feature_descriptor.h"
#include "slam/map/pose_graph/pose_graph.h"
#include "slam/slam/slam.h"
#include "slam/vpr/vpr_sof_image.h"

namespace cuvslam::slam {

void LocalizerAndMapper::AddKeyframe(const Isometry3T& from_last_keyframe, const VOFrameData& frame_data,
                                     const Images& images) {
  Isometry3T pose_estimate = GetCurrentPose() * from_last_keyframe;
  RemoveScaleFromTransform(pose_estimate);

  const uint64_t timestamp_ns = frame_data.timestamp_ns;

  // add new keyframe to keyframes
  KeyFrameId keyFrameId = InvalidKeyFrameId;
  {
    KeyFrameId head_keyframe;
    const bool has_head_pose = map_.pose_graph_.GetHeadKeyframe(head_keyframe);
    bool has_head_pose_covariance = false;

    // calc covariation from prev keyframe
    Isometry3T from_head_keyframe;
    Matrix6T head_pose_covariance;

    if (has_head_pose) {
      has_head_pose_covariance = CalcBetweenPose(head_keyframe, InvalidKeyFrameId, frame_data.tracks2d_norm,
                                                 frame_data.tracks3d_rel, from_head_keyframe, head_pose_covariance);
      if (!has_head_pose_covariance) {
        // use default covariation
        head_pose_covariance.setZero();
        Vector6T d;
        d << 1000, 1000, 1000, 1000, 1000, 1000;
        head_pose_covariance.diagonal() = d;
        has_head_pose_covariance = true;
      }
    }

    PoseGraphEdgeStat edge_stat;
    edge_stat.tracks3d_number = frame_data.tracks3d_rel.size();

    PoseGraphKeyframeInfo extra_keyframe_info;
    extra_keyframe_info.timestamp_ns = timestamp_ns;
    if (has_head_pose_covariance) {
      keyFrameId = map_.pose_graph_.AddKeyframe(map_.pose_graph_hypothesis_, &from_last_keyframe, &head_pose_covariance,
                                                frame_data.frame_information, extra_keyframe_info, &edge_stat);
    } else {
      keyFrameId = map_.pose_graph_.AddKeyframe(map_.pose_graph_hypothesis_, nullptr, nullptr,
                                                frame_data.frame_information, extra_keyframe_info, &edge_stat);
    }
    map_.pose_graph_hypothesis_.SetKeyframePose(keyFrameId, pose_estimate);

    // required for CalcFramePose()
    if (keep_track_poses_) {
      keyframe_sources_[keyFrameId] = frame_data.frame_id;
    }

    // Register the node with the place recognition map and, when this frame's pixels are still
    // reachable, give the node its picture straight away. Pixels are unreachable when SLAM has
    // fallen behind odometry and the frame's image context was recycled before the worker got to
    // it; Slam::AddFrameToVprMap covers that case from the caller's thread.
    if (vpr_map_ && vpr_map_->Enabled()) {
      vpr_map_->OnKeyframeAdded(keyFrameId, static_cast<int64_t>(timestamp_ns));
      if (!vpr_map_->HasImage(keyFrameId)) {
        const vpr::VprImage vpr_image = vpr::MakeVprImageFromImages(images);
        if (!vpr_image.Empty() && vpr_map_->AddFrame(keyFrameId, static_cast<int64_t>(timestamp_ns), vpr_image)) {
          vpr_map_->UpdateNodePose(keyFrameId, pose_estimate);
        }
      }
    }

    MoveReadyStagedLandmarksToLSI(timestamp_ns);
  }

  // add new track_id to staging3d_ and add record to StagingTrack3d::keyframes
  struct CreateDescriptorInfo {
    TrackId id;
    Vector2T desc_input;
  };

  std::unordered_map<CameraId, std::vector<CreateDescriptorInfo>> info;

  for (const auto& tracks2d : frame_data.tracks2d_norm) {
    const auto id = tracks2d.track_id;
    const auto uv_norm = tracks2d.xy;
    const auto it = frame_data.tracks3d_rel.find(id);

    if (it == frame_data.tracks3d_rel.end()) {
      continue;
    }

    Vector2T uv_pix;
    const auto& intrinsics = *rig_.intrinsics[tracks2d.cam_id];
    if (!intrinsics.denormalizePoint(uv_norm, uv_pix)) {
      continue;
    }

    TrackOnKeyframe ttk;
    ttk.keyframe_id = keyFrameId;
    ttk.xyz_rel = it->second;
    ttk.uv_norm = uv_norm;
    ttk.uv_pix = uv_pix;

    auto& new_track = staging3d_[id];
    new_track.cam_id = tracks2d.cam_id;
    new_track.keyframes.push_back(ttk);

    // prepare create_descriptor_input & new_tracks_id
    if (!new_track.fd) {
      info[tracks2d.cam_id].push_back({id, uv_pix});
    }
  }

  const bool has_images = std::any_of(images.begin(), images.end(), [](const auto& image) { return image != nullptr; });
  if (!info.empty() && has_images) {
    std::vector<TrackId> track_ids;
    std::vector<Vector2T> desc_inputs;
    for (const auto& [cam_id, desc_info_vec] : info) {
      track_ids.clear();
      desc_inputs.clear();
      track_ids.reserve(desc_info_vec.size());
      desc_inputs.reserve(desc_info_vec.size());

      for (const auto& [id, input] : desc_info_vec) {
        track_ids.push_back(id);
        desc_inputs.push_back(input);
      }

      // Create new descriptors
      if (cam_id >= images.size() || images[cam_id] == nullptr) {
        continue;
      }
      std::vector<FeatureDescriptor> create_descriptor_output;
      map_.feature_descriptor_ops_->CreateDescriptors(images[cam_id], desc_inputs, create_descriptor_output);
      // copy descriptors to staging3d_
      for (size_t i = 0; i < track_ids.size(); i++) {
        if (i >= create_descriptor_output.size()) {
          break;
        }
        const auto id = track_ids[i];
        auto& new_track = staging3d_[id];
        new_track.fd = create_descriptor_output[i];
        new_track.cam_id = cam_id;
      }
    }
  }

  // create landmarks
  to_remove_.clear();
  to_remove_.reserve(staging3d_.size());

  for (auto& [track_id, staging] : staging3d_) {
    if (staging.cam_id >= images.size() || images[staging.cam_id] == nullptr) {
      continue;
    }

    if (frame_data.tracks3d_rel.find(track_id) != frame_data.tracks3d_rel.end()) {
      continue;  // track is not lost yet
    }

    if (staging.keyframes.empty()) {
      // wrong staging
      to_remove_.push_back(track_id);
      continue;
    }

    if (staging.num_frames_not_tracked < staging_keyframes_thresh_) {
      staging.num_frames_not_tracked++;
      continue;
    }

    // have to remove from staging3d_
    to_remove_.push_back(track_id);

    const LandmarkId landmark_id = map_.landmarks_spatial_index_->AddLandmark(staging.fd, timestamp_ns);

    if (landmark_id == InvalidLandmarkId) {
      continue;
    }

    // add relations
    // If landmark has no relation it will be ignored in MoveReadyStagedLandmarksToLSI
    for (const auto& kfd : staging.keyframes) {
      const bool valid_keyframe = map_.pose_graph_.AddLandmarkRelation(landmark_id, kfd.keyframe_id);
      if (valid_keyframe) {
        Vector3T xyz_rel = kfd.xyz_rel;
        map_.landmarks_spatial_index_->AddLandmarkRelation(landmark_id, kfd.keyframe_id, &xyz_rel, kfd.uv_norm,
                                                           map_.pose_graph_hypothesis_);
      }
    }
  }

  // cleanup staging3d
  for (const auto id : to_remove_) {
    staging3d_.erase(id);
  }

  if (map_.database_) {
    map_.pose_graph_hypothesis_.PutToDatabase(map_.database_.get());
  }
  RERUN(map_.logPoseGraph);
}

void LocalizerAndMapper::MoveReadyStagedLandmarksToLSI(uint64_t timestamp_ns) {
  if (!active_cameras_) {
    throw std::runtime_error("No active cameras found. Consider to call SetActiveCameras");
  }

  KeyFrameId head_keyframe;
  map_.pose_graph_.GetHeadKeyframe(head_keyframe);  // can be InvalidKeyFrameId is Lost

  map_.landmarks_spatial_index_->MoveReadyStagedLandmarksToLSI(
      GetCurrentPose(), active_cameras_.value(), map_.pose_graph_hypothesis_, head_keyframe, timestamp_ns);

  std::function func_remove_from_keyframe = [&](LandmarkId landmark_id, KeyFrameId keyframe_id) {
    RemoveLandmarkRelation(landmark_id, keyframe_id);
  };
  map_.landmarks_spatial_index_->RemoveDeadLandmarks(func_remove_from_keyframe);
}

}  // namespace cuvslam::slam

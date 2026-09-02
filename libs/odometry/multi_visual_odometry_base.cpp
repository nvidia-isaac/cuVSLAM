
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

#include "odometry/multi_visual_odometry_base.h"

#include <algorithm>

#include "math/twist.h"
#include "pipelines/feature_predictor.h"
#include "sof/sof_create.h"

namespace cuvslam::odom {

namespace {

bool IsAllZeroHostU8(const ImageSource& source, const ImageShape& shape) {
  if (source.data == nullptr || source.memory_type != ImageSource::Host || source.type != ImageSource::U8) {
    return false;
  }

  const size_t num_channels = source.image_encoding == ImageEncoding::RGB8 ? 3 : 1;
  const auto* data = static_cast<const uint8_t*>(source.data);
  const size_t num_values = static_cast<size_t>(shape.width) * static_cast<size_t>(shape.height) * num_channels;
  return std::all_of(data, data + num_values, [](uint8_t value) { return value == 0; });
}

bool AreAllAvailableImagesBlack(const Sources& sources, const sof::Images& images) {
  bool saw_image = false;
  for (size_t cam_id = 0; cam_id < images.size(); ++cam_id) {
    if (images[cam_id] == nullptr || cam_id >= sources.size() || sources[cam_id].data == nullptr) {
      continue;
    }
    saw_image = true;
    if (!IsAllZeroHostU8(sources[cam_id], images[cam_id]->get_image_meta().shape)) {
      return false;
    }
  }
  return saw_image;
}

void DropCurrentImages(sof::Images& images) {
  for (auto& image : images) {
    image = nullptr;
  }
}

void ClearFrameStat(IVisualOdometry::VOFrameStat* stat) {
  if (!stat) {
    return;
  }
  stat->keyframe = false;
  stat->heating = false;
  stat->tracks2d.clear();
  stat->tracks3d.clear();
}

}  // namespace

MultiVisualOdometryBase::MultiVisualOdometryBase(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fig,
                                                 const Settings& settings, bool use_gpu)

    : rig_(rig),
      fig_(fig),
      settings_(settings),
      map_(20),
      feature_predictor_(std::make_shared<pipelines::FeaturePredictor>(map_, rig)) {
  observations_.resize(rig.num_cameras);

  sof::Implementation implementation = sof::Implementation::kCPU;
  if (use_gpu) {
#ifdef USE_CUDA
    implementation = sof::Implementation::kGPU;
#else
    TraceError("To use GPU SOF one must use USE_CUDA=true cmake option");
#endif
  }
  feature_tracker_ =
      sof::CreateMultiSOF(implementation, rig, fig_, feature_predictor_, settings.sof_settings, settings.kf_settings);
}

void MultiVisualOdometryBase::reset() {
  prediction_model_.reset();  // don't do any prediction until two frames tracked successfully
  feature_tracker_->reset();
  pipelines::ISFMSolver& solver = get_solver();
  solver.reset();
  map_.clear();
}

bool MultiVisualOdometryBase::track(const Sources& curr_sources, [[maybe_unused]] const DepthSources& depth_sources,
                                    sof::Images& curr_images, const sof::Images& prev_images,
                                    const Sources& masks_sources, Isometry3T& delta, Matrix6T& static_info_exp,
                                    const TrackPerFrameSettings& per_frame_setting) {
  assert(std::none_of(depth_sources.begin(), depth_sources.end(),
                      [](const ImageSource& source) { return source.data != nullptr; }));
  const auto first_image = std::find_if(curr_images.begin(), curr_images.end(),
                                        [](const sof::ImageContextPtr& image) { return image != nullptr; });
  if (first_image == curr_images.end()) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track, images are not available");
    return false;
  }
  TRACE_EVENT ev = profiler_domain_.trace_event("MultiVisualOdometryBase::track()", profiler_color_);
  const int64_t timestamp = (*first_image)->get_image_meta().timestamp;  // current frame timestamp
  Isometry3T predicted_world_from_rig = prev_world_from_rig_;
  Isometry3T world_from_rig;
  pipelines::ISFMSolver& solver = get_solver();

  if (settings_.use_prediction) {
    do_predict(&prediction_model_, timestamp, predicted_world_from_rig);
  }

  if (can_track_visual_blackout() && AreAllAvailableImagesBlack(curr_sources, curr_images)) {
    for (auto& cam_observations : observations_) {
      cam_observations.clear();
    }

    const bool have_pose = solver.solveNextFrame(
        timestamp, sof::FrameState::None, observations_, world_from_rig, static_info_exp,
        {per_frame_setting.sba, per_frame_setting.sm, per_frame_setting.vo_pnp, per_frame_setting.inertial_stereo_pnp,
         per_frame_setting.imu_pnp, per_frame_setting.icp});

    DropCurrentImages(curr_images);
    if (!have_pose) {
      ClearFrameStat(last_frame_stat_.get());
      delta = Isometry3T::Identity();
      static_info_exp.setZero();
      return false;
    }

    ClearFrameStat(last_frame_stat_.get());
    prediction_model_.add_known_pose(world_from_rig, timestamp);
    delta = prev_world_from_rig_.inverse() * world_from_rig;
    prev_world_from_rig_ = world_from_rig;
    return true;
  }

  sof::FrameState frame_type;
  for (auto& cam_observations : observations_) {
    cam_observations.clear();
  }

  const bool track_result =
      feature_tracker_->trackNextFrame(curr_sources, curr_images, prev_images, masks_sources, predicted_world_from_rig,
                                       observations_, frame_type, per_frame_setting);
  if (!track_result) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track on the 2D tracking stage");
    return false;
  }

  IVisualOdometry::VOFrameStat* stat = last_frame_stat_.get();
  std::vector<Track2D>* tracks2d = stat ? &(stat->tracks2d) : nullptr;
  Tracks3DMap* tracks3d = stat ? &(stat->tracks3d) : nullptr;

  const bool have_pose =
      solver.solveNextFrame(timestamp, frame_type, observations_, world_from_rig, static_info_exp,
                            {per_frame_setting.sba, per_frame_setting.sm, per_frame_setting.vo_pnp,
                             per_frame_setting.inertial_stereo_pnp, per_frame_setting.imu_pnp, per_frame_setting.icp},
                            tracks2d, tracks3d);

  if (stat) {
    stat->keyframe = frame_type == sof::FrameState::Key;
    stat->heating = false;
  }

  if (!have_pose) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track on the PnP stage");
    return false;
  }

  prediction_model_.add_known_pose(world_from_rig, timestamp);
  delta = prev_world_from_rig_.inverse() * world_from_rig;
  prev_world_from_rig_ = world_from_rig;

  return true;
}

void MultiVisualOdometryBase::enable_stat(bool enable) {
  const bool current_state_is_enable = last_frame_stat_ != nullptr;
  if (current_state_is_enable == enable) {
    return;  // if nothing is changed do nothing
  }
  last_frame_stat_ = enable ? std::make_unique<VOFrameStat>() : nullptr;
}

const std::unique_ptr<IVisualOdometry::VOFrameStat>& MultiVisualOdometryBase::get_last_stat() const {
  return last_frame_stat_;
}

}  // namespace cuvslam::odom

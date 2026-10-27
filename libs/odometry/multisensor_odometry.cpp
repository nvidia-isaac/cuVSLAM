
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

#include "odometry/multisensor_odometry.h"

#include <algorithm>

#include "camera/camera.h"
#include "common/log.h"
#include "cuda_modules/cuda_helper.h"
#include "pipelines/feature_predictor.h"
#include "sof/sof_create.h"

namespace cuvslam::odom {

namespace {

// Depth lifting, ICP and point-to-plane factors project into depth images with pinhole intrinsics only.
// Cameras without depth only feed normalized (undistorted) observations to the solver, so any model works.
void WarnIfNonPinholeDepthCamera(const camera::Rig& rig, const std::vector<int32_t>& depth_camera_ids) {
  for (int32_t cam_id : depth_camera_ids) {
    if (dynamic_cast<const camera::PinholeCameraModel*>(rig.intrinsics[cam_id]) == nullptr) {
      TraceWarning(
          "Multisensor depth cameras support only the pinhole model. Depth camera %d uses a different model; "
          "its depth measurements will be misaligned.",
          cam_id);
    }
  }
}

pipelines::SolverSfMMultisensor MakeSolver(map::UnifiedMap& map, const camera::Rig& rig, const Settings& settings) {
  if (settings.multisensor_settings.with_imu) {
    return pipelines::SolverSfMMultisensor(map, rig, settings.sba_settings, settings.imu_calibration);
  }
  return pipelines::SolverSfMMultisensor(map, rig, settings.sba_settings);
}

}  // namespace

MultisensorOdometry::MultisensorOdometry(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fig,
                                         const Settings& settings, bool use_gpu)

    : fig_(fig),
      settings_(settings),
      map_(20),
      feature_predictor_(std::make_shared<pipelines::FeaturePredictor>(map_, rig)),
      solver_(MakeSolver(map_, rig, settings)) {
  WarnIfNonPinholeDepthCamera(rig, settings.multisensor_settings.depth_camera_ids);
  observations_.resize(rig.num_cameras);
  depth_cam_ids_.reserve(rig.num_cameras);
  depth_info_storage_.reserve(rig.num_cameras);
  depth_infos_.resize(rig.num_cameras, nullptr);

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

void MultisensorOdometry::reset() {
  prediction_model_.reset();
  feature_tracker_->reset();
  solver_.reset();
  map_.clear();
}

bool MultisensorOdometry::track(const Sources& curr_sources, const DepthSources& depth_sources,
                                sof::Images& curr_images, const sof::Images& prev_images, const Sources& masks_sources,
                                Isometry3T& delta, Matrix6T& static_info_exp,
                                const TrackPerFrameSettings& per_frame_setting) {
  const auto first_image = std::find_if(curr_images.begin(), curr_images.end(),
                                        [](const sof::ImageContextPtr& image) { return image != nullptr; });
  if (first_image == curr_images.end()) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track, images are not available");
    return false;
  }
  TRACE_EVENT ev = profiler_domain_.trace_event("MultisensorOdometry::track()", profiler_color_);
  const int64_t timestamp = (*first_image)->get_image_meta().timestamp;
  Isometry3T predicted_world_from_rig = prev_world_from_rig_;

  {
    TRACE_EVENT predict_ev = profiler_domain_.trace_event("predict");
    if (settings_.use_prediction) {
      do_predict(&prediction_model_, timestamp, predicted_world_from_rig);
    }
  }

  sof::FrameState frame_type;
  for (auto& cam_observations : observations_) {
    cam_observations.clear();
  }

  bool track_result;
  {
    TRACE_EVENT track_ev = profiler_domain_.trace_event("trackNextFrame");
    track_result =
        feature_tracker_->trackNextFrame(curr_sources, curr_images, prev_images, masks_sources,
                                         predicted_world_from_rig, observations_, frame_type, per_frame_setting);
  }
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
  Isometry3T world_from_rig;

  depth_cam_ids_.clear();
  {
    TRACE_EVENT depth_ev = profiler_domain_.trace_event("build_depth_pyramids");
    for (CameraId cam_id = 0; cam_id < curr_images.size(); ++cam_id) {
      const auto& image = curr_images[cam_id];
      if (image == nullptr) {
        continue;
      }
      if (image->support_depth()) {
        const auto& depthptr = depth_sources[cam_id];

        const ImageSource* mask_source = nullptr;
        if (masks_sources[cam_id].data != nullptr) {
          mask_source = &masks_sources[cam_id];
        }

        auto depth_pyramids = image->build_gpu_depth_pyramid(depthptr, stream.get_stream(), mask_source);
        if (depth_pyramids) {
          depth_cam_ids_.push_back(cam_id);
        }
      }
    }
  }

  {
    TRACE_EVENT sync_ev = profiler_domain_.trace_event("depth_stream_sync");
    CUDA_CHECK(cudaStreamSynchronize(stream.get_stream()));
  }

  depth_info_storage_.clear();
  depth_info_storage_.reserve(depth_cam_ids_.size());
  std::fill(depth_infos_.begin(), depth_infos_.end(), nullptr);
  for (CameraId cam_id : depth_cam_ids_) {
    depth_info_storage_.push_back({cam_id, curr_images[cam_id]->gpu_depth_pyramid()->get()});
    depth_infos_[cam_id] = &depth_info_storage_.back();
  }

  bool have_pose;
  {
    TRACE_EVENT solve_ev = profiler_domain_.trace_event("solveNextFrame");
    have_pose = solver_.solveNextFrame(timestamp, frame_type, observations_, depth_infos_, world_from_rig,
                                       static_info_exp, tracks2d, tracks3d);
  }

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

void MultisensorOdometry::enable_stat(bool enable) {
  const bool current_state_is_enable = last_frame_stat_ != nullptr;
  if (current_state_is_enable == enable) {
    return;
  }
  last_frame_stat_ = enable ? std::make_unique<IVisualOdometry::VOFrameStat>() : nullptr;
}

const std::unique_ptr<IVisualOdometry::VOFrameStat>& MultisensorOdometry::get_last_stat() const {
  return last_frame_stat_;
}

bool MultisensorOdometry::do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction) {
  Isometry3T update;
  const int64_t prev_abs_timestamp = prediction_model_.last_timestamp_ns();
  if (predictor->predict(prev_abs_timestamp, timestamp, update)) {
    sof_prediction = update * sof_prediction;
    return true;
  }
  return false;
}

void MultisensorOdometry::add_imu_measurement(const cuvslam::imu::ImuMeasurement& m) { solver_.add_imu_measurement(m); }

std::optional<Vector3T> MultisensorOdometry::get_gravity() const { return solver_.get_gravity(); }

std::optional<pipelines::SolverSfMMultisensor::ImuState> MultisensorOdometry::GetImuState() const {
  return solver_.GetImuState();
}

}  // namespace cuvslam::odom

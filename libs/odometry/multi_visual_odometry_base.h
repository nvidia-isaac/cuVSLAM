
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

#include <memory>

#include "camera/frustum_intersection_graph.h"
#include "map/map.h"
#include "pipelines/sfm_solver_interface.h"
#include "profiler/profiler.h"
#include "sof/feature_prediction_interface.h"
#include "sof/kf_selector.h"
#include "sof/sof_multicamera_interface.h"

#include "odometry/ivisual_odometry.h"
#include "odometry/pose_prediction.h"
#include "odometry/svo_config.h"

namespace cuvslam::odom {

class MultiVisualOdometryBase : public IVisualOdometry {
public:
  MultiVisualOdometryBase(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fig,
                          const Settings& svo_settings, bool use_gpu);
  ~MultiVisualOdometryBase() override = default;

  bool track(const Sources& curr_sources, const DepthSources& depth_sources, sof::Images& curr_images,
             const sof::Images& prev_images, const Sources& masks_sources, Isometry3T& delta, Matrix6T& static_info_exp,
             const TrackPerFrameSettings& per_frame_setting) final;

  void enable_stat(bool enable) final;
  const std::unique_ptr<VOFrameStat>& get_last_stat() const final;

  virtual bool do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction) = 0;
  virtual pipelines::ISFMSolver& get_solver() = 0;

protected:
  virtual bool can_track_visual_blackout() const { return false; }

  void reset();
  camera::Rig rig_;
  camera::FrustumIntersectionGraph fig_;
  Settings settings_;
  // This is the internal prediction model.
  PosePredictionModel prediction_model_;
  map::UnifiedMap map_;
  sof::FeaturePredictorPtr feature_predictor_;

private:
  Isometry3T prev_world_from_rig_ = Isometry3T::Identity();

  std::unique_ptr<VOFrameStat> last_frame_stat_;
  std::unique_ptr<sof::IMultiSOF> feature_tracker_;
  pipelines::MulticamObservations observations_;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("Multi VO");
  const uint32_t profiler_color_ = 0xFFFF00;
};

}  // namespace cuvslam::odom


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

#include "pipelines/feature_predictor.h"

#include "epipolar/camera_selection.h"

namespace cuvslam::pipelines {

FeaturePredictor::FeaturePredictor(const map::UnifiedMap& map, const camera::Rig& rig) : map_(map), rig_(rig){};

void FeaturePredictor::predictObservations(const Isometry3T& world_from_rig, int cameraIndex,
                                           const std::vector<TrackId>& ids, sof::Prediction& uvs) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("predictObservations");
  const size_t n = ids.size();

  assert(uvs.size() == ids.size());
  assert(cameraIndex >= 0);
  assert(cameraIndex < rig_.num_cameras);

  Isometry3T cameraFromWorld = rig_.camera_from_rig[cameraIndex] * world_from_rig.inverse();
  auto& cameraModel = rig_.intrinsics[cameraIndex];
  map_.get_recent_landmarks(cameraIndex, map_landmarks_);

  for (size_t i = 0; i < n; ++i) {
    uvs[i].reset();
    auto track = map_landmarks_.find(ids[i]);
    if (track == map_landmarks_.end()) {
      continue;
    }

    const auto& point_3d = track->second;

    const Vector3T camPoint = cameraFromWorld * point_3d;

    if (epipolar::IsDepthInFront(camPoint.z())) {
      // Project into the camera (OpenCV +Z forward). "denormalize" = project to pixels, "normalize" = unproject.
      Vector2T xy(camPoint.x() / camPoint.z(), camPoint.y() / camPoint.z());
      Vector2T uv;
      if (cameraModel->denormalizePoint(xy, uv)) {
        uvs[i].emplace(uv);
      }
    }
  }
}

}  // namespace cuvslam::pipelines

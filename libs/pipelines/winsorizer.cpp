
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

#include "pipelines/winsorizer.h"

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/track_id.h"
#include "common/vector_3t.h"
#include "epipolar/camera_selection.h"

namespace cuvslam::pipelines {
void winsorize(const camera::Rig& rig, const std::vector<KeyframeLandmarkObs>& observations_from_last_keyframe) {
  const size_t num_tracks = observations_from_last_keyframe.size();

  constexpr float kPercent = 5.f;
  constexpr float kMinSize = 0.0025f;
  constexpr int kSafe = 50;

  std::vector<std::pair<float, LandmarkPtr>> errors;
  errors.resize(num_tracks);
  size_t i = 0;

  for (const auto& obs : observations_from_last_keyframe) {
    if (!obs.landmark->get_pose()) {
      continue;
    }

    const Isometry3T& cam_from_rig = rig.camera_from_rig[obs.observation.cam_id];
    Isometry3T rig_from_w = obs.keyframe->get_pose();
    Vector3T point = *obs.landmark->get_pose();
    const Vector3T pos = cam_from_rig * rig_from_w * point;
    if (!epipolar::IsDepthInFront(pos.z())) {
      continue;
    }

    const Vector2T projected = pos.head(2) / pos.z();
    const float repr2 = (projected - obs.observation.xy).squaredNorm();
    assert(i < num_tracks);
    errors[i] = {repr2, obs.landmark};
    ++i;
  }
  if (i < kSafe) {
    return;
  }
  errors.resize(i);
  std::sort(errors.begin(), errors.end());  // TODO: use partial sort

  constexpr float kMinSize2 = kMinSize * kMinSize;

  const int to_delete = static_cast<int>(kPercent * i / 100.f);
  for (int j = 0; j < to_delete; ++j) {
    const auto& [error_value, l] = errors[i - 1 - j];
    if (error_value > kMinSize2) {
      l->reset();
    }
  }
}

}  // namespace cuvslam::pipelines


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

#include <vector>

#include "common/camera_id.h"
#include "common/track_id.h"
#include "common/types.h"
#include "common/unaligned_types.h"

#include "camera/camera.h"

namespace cuvslam::camera {

struct Observation {
  Observation() = default;
  Observation(CameraId cam_id_, TrackId id_, const Vector2T& xy_, const Matrix2T& xy_info_)
      : cam_id(cam_id_), id(id_), xy(xy_), xy_info(xy_info_){};

  CameraId cam_id;
  TrackId id;
  Vector2T xy;
  Matrix2T xy_info;
};

// observations should be sorted by track_id
// O(log(n))
bool FindObservation(const std::vector<Observation>& observations, TrackId track_id, size_t* index = nullptr);

// Creates a physically meaningful information matrix for 2D observations.
Matrix2T GetDefaultObservationInfoUV();

Matrix2T ObservationInfoUVToNormUV(const ICameraModel& intrinsics, const Matrix2T& info_uv);
// Fails when a sample of the numeric jacobian falls outside the camera model's validity — the
// resulting matrix would be silently wrong. `info_xy` is left untouched in that case.
[[nodiscard]] bool ObservationInfoUVToXY(const ICameraModel& intrinsics, const Vector2T& uv, const Vector2T& xy,
                                         const Matrix2T& info_uv, Matrix2T& info_xy);

}  // namespace cuvslam::camera

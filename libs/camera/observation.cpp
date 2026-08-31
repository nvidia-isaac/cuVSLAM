
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

#include "camera/observation.h"

#include "math/sigmapoint.h"

namespace {

bool ObservationsCompareById(const cuvslam::camera::Observation& lhs, const cuvslam::camera::Observation& rhs) {
  return lhs.id < rhs.id;
}

}  // namespace

namespace cuvslam::camera {

bool FindObservation(const std::vector<Observation>& observations, TrackId track_id, size_t* index) {
  Observation x;
  x.id = track_id;
  const auto res_iter = std::lower_bound(std::begin(observations), std::end(observations), x, ObservationsCompareById);
  if (res_iter == std::end(observations) || res_iter->id != track_id) {
    return false;
  }
  if (index != NULL) {
    *index = res_iter - std::begin(observations);
  }
  return true;
}

// Creates a physically meaningful information matrix for 2D observations.
Matrix2T GetDefaultObservationInfoUV() {
  // conservative estimate for the tracking noise on undistorted frame
  constexpr auto stdPx = float{3};

  // information matrix in pixel space
  Matrix2T invSigma;
  invSigma << 1.f / (stdPx * stdPx), 0.f, 0.f, 1.f / (stdPx * stdPx);

  return invSigma;
}

Matrix2T ObservationInfoUVToNormUV(const ICameraModel& intrinsics, const Matrix2T& info_uv) {
  const Vector2T& focal = intrinsics.getFocal();
  const Matrix2T invJacobian = focal.asDiagonal();

  // if y = f(x), j^-1 = f'(x), then
  // cov(y) ~ j^-1 cov(x) j^-t
  // info(y) ~ j^t info(x) j

  // invJacobian - diagonal matrix => invJacobian^T = invJacobian
  return invJacobian * info_uv * invJacobian;
}

// xy must be equal to output of intrinsics.normalizePoint(uv)
bool ObservationInfoUVToXY(const ICameraModel& intrinsics, const Vector2T& uv, const Vector2T& xy,
                           const Matrix2T& info_uv, Matrix2T& info_xy) {
  // The jacobian is sampled around uv, so a sample can land outside the model even when uv itself
  // is fine. Such a sample contributes a zero instead of a real value, which would quietly skew
  // the result — report it rather than returning a plausible-looking matrix.
  bool all_samples_valid = true;
  const Matrix2T cov = info_uv.inverse();
  const Matrix2T cov_xy = math::approximate<2, 2>(
      uv, xy, cov,
      [&intrinsics, &all_samples_valid](const Vector2T& in) {
        Vector2T out = Vector2T::Zero();
        if (!intrinsics.normalizePoint(in, out)) {
          all_samples_valid = false;
        }
        return out;
      },
      true /* is_lambda_affine */);
  if (!all_samples_valid) {
    return false;
  }
  info_xy = cov_xy.inverse();
  return true;
}

}  // namespace cuvslam::camera

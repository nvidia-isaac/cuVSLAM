
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
#include <stdexcept>

#include "common/affine.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "common/vector_2t.h"
#include "cuvslam/cuvslam2.h"

namespace cuvslam::camera {

// there are three coordinate spaces:
// 1) pixel coordinates (top left is zero)
// 2) uv - float image coordinates in pixels.
//         left-top pixel corner coordinates are
//             (0.f, 0.f),
//             (0.f, 1.f),
//             (1.f, 1.f),
//             (1.f, 0.f)
//         right-bottom pixel has coordinates
//             (resolution.x - 1.f, resolution.y - 1.f),
//             (resolution.x - 1.f, resolution.y),
//             (resolution.x      , resolution.y),
//             (resolution.x      , resolution.y - 1.f)
// 3) normalized_uv = inv_calibration * uv
//    translated with principal point and scaled with focal.
//    Focal is measured in pixels unit. For example for 1024x1024 focal can be 512.f
//    (0.f, 0.f) can be considered as lens center.
//    no more resolutions/pixels.
//    Min/max coordinates can be positive/negative.
//    [-1.f, 1.f] is an example of good mapping.
//    Maximum radius is limited by max_normalized_uv_radius2.
// 4) xy - undistorted normalized image coordinates (OpenCV-style): x = (u-cx)/fx, y = (v-cy)/fy after undistortion.
//    (0.f, 0.f) is the lens center in normalized space. max_xy_radius limits validity.
class ICameraModel {
public:
  ICameraModel(const Vector2T& resolution,  // Not used in calculations. see focal
               const Vector2T& focal,       // in "pixels" unit
               const Vector2T& principal,   // in "pixels" unit
               float max_normalized_uv_radius = 10000.f, float max_xy_radius = 10000.f);
  virtual ~ICameraModel() = default;

  [[nodiscard]] bool normalizePoint(const Vector2T& uv, Vector2T& xy) const;
  [[nodiscard]] bool denormalizePoint(const Vector2T& xy, Vector2T& uv) const;

  [[nodiscard]] const Vector2T& getPrincipal() const;
  [[nodiscard]] const Vector2T& getFocal() const;
  [[nodiscard]] const Vector2T& getResolution() const;

protected:  // for debug purpose, more specific implementation may want to compare result with generic one
  virtual bool distort(const Vector2T& xy, Vector2T& normalized_uv) const = 0;
  virtual bool undistort(const Vector2T& normalized_uv, Vector2T& xy) const = 0;

private:
  Vector2T resolution_;
  Vector2T focal_;      // camera focal distance expressed in pixels
  Vector2T principal_;  // camera principal point expressed in pixels

  const float max_normalized_uv_radius2_;  // square of radius of validity for rejecting denormalizing outside points
  const float max_xy_radius2_;             // square of radius of validity for rejecting normalizing outside points

  Affine2T calibration_;
  Affine2T inv_calibration_;
};

class PinholeCameraModel final : public ICameraModel {
public:
  PinholeCameraModel(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal,
                     float max_normalized_uv_radius = 10000.f, float max_xy_radius = 10000.f);

private:
  bool distort(const Vector2T& xy, Vector2T& normalized_uv) const override;
  bool undistort(const Vector2T& normalized_uv, Vector2T& xy) const override;
};

class FisheyeCameraModel final : public ICameraModel {
public:
  FisheyeCameraModel(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal, float K1, float K2,
                     float K3, float K4, float max_normalized_uv_radius = 10000.f, float max_xy_radius = 10000.f);

private:
  bool distort(const Vector2T& xy, Vector2T& normalized_uv) const override;
  bool undistort(const Vector2T& normalized_uv, Vector2T& xy) const override;

  float compute_distorted_radius(float undistorted_radius) const;
  bool compute_distorted_radius_derivative(float undistorted_radius, float& derivative) const;

  float K1_;
  float K2_;
  float K3_;
  float K4_;
};

// Implements iterative undistort algorithm using Jacobian.
// Jacobian computation must be provided by a derived class.
class CameraModelWithJacobianBasedUndistortion : public ICameraModel {
public:
  CameraModelWithJacobianBasedUndistortion(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal,
                                           float max_normalized_uv_radius = 10000.f, float max_xy_radius = 10000.f);

private:
  bool undistort(const Vector2T& normalized_uv, Vector2T& xy) const override;
  virtual void compute_distort_jacobian(const Vector2T& xy, Matrix2T& jacobian) const = 0;
};

// Brown(aka Brown - Conrady) distortion model with 5 coefficients:
// 3 for radial distortion and 2 for tangential.
// https://www.control.isy.liu.se/student/graduate/DynVis/Lectures/le2.pdf
// http://robots.stanford.edu/cs223b04/JeanYvesCalib/htmls/parameters.html (Caltechs calibration toolbox)
// This model is a special case of rational function that OpenCV uses.
class Brown5KCameraModel final : public CameraModelWithJacobianBasedUndistortion {
public:
  Brown5KCameraModel(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal, float K1, float K2,
                     float K3, float P1, float P2, float max_normalized_uv_radius = 10000.f,
                     float max_xy_radius = 10000.f);

private:
  bool distort(const Vector2T& xy, Vector2T& normalized_uv) const override;

  void compute_distort_jacobian(const Vector2T& xy, Matrix2T& jacobian) const override;

  float K1_;
  float K2_;
  float K3_;
  float P1_;
  float P2_;
};

// Rational polynomial model with 6 radial distortion coefficients and 2 tangential distortion coefficients.
// https://docs.nvidia.com/vpi/group__VPI__LDC.html#structVPIPolynomialLensDistortionModel
class PolynomialCameraModel final : public CameraModelWithJacobianBasedUndistortion {
public:
  PolynomialCameraModel(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal, float K1,
                        float K2, float K3, float K4, float K5, float K6, float P1, float P2,
                        float max_normalized_uv_radius = 10000.f, float max_xy_radius = 10000.f);

private:
  bool distort(const Vector2T& xy, Vector2T& normalized_uv) const override;

  void compute_distort_jacobian(const Vector2T& xy, Matrix2T& jacobian) const override;

  float K1_;
  float K2_;
  float K3_;
  float K4_;
  float K5_;
  float K6_;
  float P1_;
  float P2_;
};

/// Parse distortion model name to enum.
inline Distortion::Model StringToDistortionModel(std::string_view str) {
  if (str == "polynomial") return Distortion::Model::Polynomial;
  if (str == "brown5k") return Distortion::Model::Brown;
  if (str == "pinhole") return Distortion::Model::Pinhole;
  if (str == "fisheye4" || str == "fisheye") return Distortion::Model::Fisheye;
  throw std::invalid_argument{"Incorrect distortion model: " + std::string{str}};
}

// Factory function for camera models.
// Instead of accepting named distortion parameters it accepts parameters array.
// Order of parameters in array is the same as in appropriate constructors except for PolynomialCameraModel,
// where parameters order in array is K1, K2, P1, P2, K3, K4, K5, K6 as in cv::initUndistortRectifyMap and in ROS
// messages. Returns nullptr if distortion model is unsupported or number of parameters is wrong.
std::unique_ptr<ICameraModel> CreateCameraModel(const Vector2T& resolution, const Vector2T& focal,
                                                const Vector2T& principal, Distortion::Model model,
                                                const float* parameters, int32_t num_parameters);

}  // namespace cuvslam::camera

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

#include "camera/camera.h"

#include "common/unaligned_types.h"

namespace {
using namespace cuvslam;

// Returns true iff the 3x3 matrix has bottom row (0, 0, 1), i.e. is stored in canonical
// 2D affine form. If false, consider calling .makeAffine() on the transform.
[[maybe_unused]] bool IsBottomRowCanonicalAffine(const Eigen::Matrix3f& m) {
  const Eigen::RowVector3f expected{0.f, 0.f, 1.f};

  return m.matrix().row(2).isApprox(expected, 0);  // zero epsilon
}
}  // namespace

namespace cuvslam::camera {
ICameraModel::ICameraModel(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal,
                           float max_normalized_uv_radius, float max_xy_radius)
    : resolution_(resolution),
      focal_(focal),
      principal_(principal),
      max_normalized_uv_radius2_(max_normalized_uv_radius * max_normalized_uv_radius),
      max_xy_radius2_(max_xy_radius * max_xy_radius) {
  calibration_.setIdentity();
  calibration_.affine().row(0) << focal_.x(), 0, principal_.x();
  calibration_.affine().row(1) << 0, focal_.y(), principal_.y();
  assert(IsBottomRowCanonicalAffine(calibration_.matrix()));

  const bool is_invertible = std::abs(focal_.x()) > epsilon() && std::abs(focal_.y()) > epsilon();

  if (!is_invertible || focal_.x() < 0.f || focal_.y() < 0.f || resolution_.x() < 0.f || resolution_.y() < 0.f) {
    std::ostringstream oss;
    oss << "Wrong camera intrinsics. resolution=(" << resolution_.x() << ", " << resolution_.y() << ") "
        << "focal=(" << focal_.x() << ", " << focal_.y() << ") "
        << "principal=(" << principal_.x() << ", " << principal_.y() << ")";
    throw std::runtime_error(oss.str());
  }
  inv_calibration_ = calibration_.inverse();
  assert(IsBottomRowCanonicalAffine(inv_calibration_.matrix()));
  assert(calibration_.isApprox(inv_calibration_.inverse()));  // epsilon = Eigen::NumTraits::dummy_precision = 1e-5f
}

bool ICameraModel::normalizePoint(const Vector2T& uv, Vector2T& xy) const {
  const Vector2T normalized_uv = inv_calibration_ * uv;

  if (normalized_uv.squaredNorm() > max_normalized_uv_radius2_) {
    return false;  // it isn't safe to use the model
  }

  if (!undistort(normalized_uv, xy)) {
    return false;
  }

  return xy.squaredNorm() <= max_xy_radius2_;
}

bool ICameraModel::denormalizePoint(const Vector2T& xy, Vector2T& uv) const {
  if (xy.squaredNorm() > max_xy_radius2_) {
    return false;  // it isn't safe to use the model
  }

  Vector2T normalized_uv;

  if (!distort(xy, normalized_uv)) {
    return false;
  }

  if (normalized_uv.squaredNorm() > max_normalized_uv_radius2_) {
    return false;
  }

  uv = calibration_ * normalized_uv;
  return true;
}

const Vector2T& ICameraModel::getPrincipal() const { return principal_; }

const Vector2T& ICameraModel::getFocal() const { return focal_; }

const Vector2T& ICameraModel::getResolution() const { return resolution_; }

PinholeCameraModel::PinholeCameraModel(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal,
                                       float max_normalized_uv_radius, float max_xy_radius)
    : ICameraModel(resolution, focal, principal, max_normalized_uv_radius, max_xy_radius) {}

bool PinholeCameraModel::distort(const Vector2T& xy, Vector2T& normalized_uv) const {
  normalized_uv = xy;
  return true;
}

bool PinholeCameraModel::undistort(const Vector2T& normalized_uv, Vector2T& xy) const {
  xy = normalized_uv;
  return true;
}

FisheyeCameraModel::FisheyeCameraModel(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal,
                                       float K1, float K2, float K3, float K4, float max_normalized_uv_radius,
                                       float max_xy_radius)
    : ICameraModel(resolution, focal, principal, max_normalized_uv_radius, max_xy_radius),
      K1_(K1),
      K2_(K2),
      K3_(K3),
      K4_(K4) {}

// undistorted_radius - radius in xy
float FisheyeCameraModel::compute_distorted_radius(float undistorted_radius) const {
  if (undistorted_radius <= std::numeric_limits<float>::epsilon()) {
    return undistorted_radius;  // no distortion in the screen center
  }

  const float theta = std::atan(undistorted_radius);
  const float theta2 = theta * theta;
  const float theta4 = theta2 * theta2;
  const float theta6 = theta4 * theta2;
  const float theta8 = theta6 * theta2;

  return theta * (1 + theta2 * K1_ + theta4 * K2_ + theta6 * K3_ + theta8 * K4_);
}

bool FisheyeCameraModel::compute_distorted_radius_derivative(float undistorted_radius, float& derivative) const {
  // f(r) = atan(r) * (1 + k1 * atan(r)^2 + k2 * atan(r)^4 + k3 * atan(r)^6 + k4 * atan(r)^8)
  // f'(r) =  (1 + 3 * k1 * atan(r)^2  + 5 * k2 * atan(r)^4 + 7 * k3 * atan(r)^6 + 9 * k4 * atan(r)^8) / (r^2 + 1)

  if (std::abs(undistorted_radius) <= std::numeric_limits<float>::epsilon()) {
    return false;
  }

  const float r = undistorted_radius;
  const float r2 = r * r;
  const float theta = std::atan(r);
  const float theta2 = theta * theta;
  const float theta4 = theta2 * theta2;
  const float theta6 = theta4 * theta2;
  const float theta8 = theta6 * theta2;

  derivative = (1.f + 3.f * K1_ * theta2 + 5.f * K2_ * theta4 + 7.f * K3_ * theta6 + 9.f * K4_ * theta8) / (1.f + r2);
  return true;
}

bool FisheyeCameraModel::distort(const Vector2T& xy, Vector2T& normalized_uv) const {
  const float undistorted_radius = xy.norm();

  normalized_uv = compute_distorted_radius(undistorted_radius) * xy.normalized();

  return true;
}

bool FisheyeCameraModel::undistort(const Vector2T& normalized_uv, Vector2T& xy) const {
  const float distorted_radius = normalized_uv.norm();
  // Newton-Raphson method for finding successively better approximations to the roots of a real-valued function.
  // let f(r) = compute_distorted_radius(r) - distorted_radius =>
  //     f'(r) = compute_distorted_radius_derivative(r) =>
  //  we need to find r: f(r) = 0     //

  constexpr int max_n_iters = 10;                      // maximum number of iterations
  constexpr float precision = (1.f / 1000.f) / 100.f;  // about 1/100 of the pixel for F = 1000px

  float r = distorted_radius;  // undistorted_radius - initial guess
  int n_iters = 0;
  float previous_step_size = std::numeric_limits<float>::max();
  float df;  // = f'(r)
  constexpr float eps = std::numeric_limits<float>::epsilon();

  while ((previous_step_size > precision) && (n_iters < max_n_iters)) {
    // STEP 1: check current solution
    const float f = compute_distorted_radius(r) - distorted_radius;
    if (std::abs(f) <= eps) {
      break;  // solution was found
    }

    // STEP 2: compute derivatives
    if (!compute_distorted_radius_derivative(r, df)) {
      return false;
    }
    if (std::abs(df) < eps) {
      // local maximum/minimum but solution wasn't find
      assert(std::abs(compute_distorted_radius(r) - distorted_radius) > eps);  // solution is not found
      return false;
    }

    // STEP 3: make new step
    const float prev_r = r;
    r -= f / df;
    previous_step_size = std::abs(r - prev_r);
    ++n_iters;
  }
  xy = r * normalized_uv.normalized();
  return n_iters < max_n_iters;
}

CameraModelWithJacobianBasedUndistortion::CameraModelWithJacobianBasedUndistortion(const Vector2T& resolution,
                                                                                   const Vector2T& focal,
                                                                                   const Vector2T& principal,
                                                                                   float max_normalized_uv_radius,
                                                                                   float max_xy_radius)
    : ICameraModel(resolution, focal, principal, max_normalized_uv_radius, max_xy_radius) {}

bool CameraModelWithJacobianBasedUndistortion::undistort(const Vector2T& normalized_uv, Vector2T& xy) const {
  // Newton-Raphson method for finding successively better approximations to the roots of a real-valued function.
  constexpr int max_n_iters = 10;                      // maximum number of iterations
  constexpr float precision = (1.f / 1000.f) / 100.f;  // about 1/100 of the pixel for F = 1000px

  xy = normalized_uv;  // initial guess
  int n_iters = 0;
  float previous_step_size = std::numeric_limits<float>::max();
  Matrix2T df;
  constexpr float eps = std::numeric_limits<float>::epsilon();

  while ((previous_step_size > precision) && (n_iters < max_n_iters)) {
    // STEP 1: check current solution
    Vector2T estimate_normalized_uv;
    if (!distort(xy, estimate_normalized_uv)) {
      assert(false);  // distort is always true (right?)
      return false;
    }
    const Vector2T f = estimate_normalized_uv - normalized_uv;
    if (f.norm() <= eps) {
      return true;  // solution was found
    }

    // STEP 2: compute derivatives
    compute_distort_jacobian(xy, df);
    // We need to check that df if invertible
    // According to Eigen documentation for small matrices we need to use computeInverseAndDetWithCheck()
    // See manual to Eigen::Matrix<>::inverse()
    Matrix2T df_inverse;
    bool invertible;
    float determinant;
    df.computeInverseAndDetWithCheck(df_inverse, determinant, invertible);
    if (!invertible) {
      // local maximum/minimum but solution wasn't find
      assert((estimate_normalized_uv - normalized_uv).norm() > eps);
      return false;
    }

    // STEP 3: make new step
    Vector2T prev_xy = xy;
    xy -= df_inverse * f;
    previous_step_size = (xy - prev_xy).norm();
    ++n_iters;
  }
  return n_iters < max_n_iters;
}

Brown5KCameraModel::Brown5KCameraModel(const Vector2T& resolution, const Vector2T& focal, const Vector2T& principal,
                                       float K1, float K2, float K3, float P1, float P2, float max_normalized_uv_radius,
                                       float max_xy_radius)
    : CameraModelWithJacobianBasedUndistortion(resolution, focal, principal, max_normalized_uv_radius, max_xy_radius),
      K1_(K1),
      K2_(K2),
      K3_(K3),
      P1_(P1),
      P2_(P2) {}

bool Brown5KCameraModel::distort(const Vector2T& xy, Vector2T& normalized_uv) const {
  const float x = xy.x();
  const float y = xy.y();
  const float x2 = x * x;
  const float y2 = y * y;
  const float x_y = x * y;
  const float r2 = x2 + y2;
  const float r4 = r2 * r2;
  const float r6 = r4 * r2;

  const Vector2T radial = (1 + K1_ * r2 + K2_ * r4 + K3_ * r6) * xy;
  const Vector2T tangential(2 * P1_ * x_y + P2_ * (r2 + 2 * x2), 2 * P2_ * x_y + P1_ * (r2 + 2 * y2));

  normalized_uv = radial + tangential;

  return true;
}

void Brown5KCameraModel::compute_distort_jacobian(const Vector2T& xy, Matrix2T& jacobian) const {
  // You can use expression below on https://www.wolframalpha.com/
  // jacobian matrix of (x*R+2p1xy+p2*(y^2+3x^2),y*R+p1*(x^2+3y^2)+2p2xy) where
  // R=(1+k1*(x^2+y^2)+k2*(x^2+y^2)^2+k3*(x^2+y^2)^3)
  const float x = xy.x();
  const float y = xy.y();
  const float x2 = x * x;
  const float y2 = y * y;
  const float r2 = x2 + y2;
  const float r4 = r2 * r2;
  const float r6 = r4 * r2;

  const float t1 = 1.f + K3_ * r6 + K2_ * r4 + K1_ * r2;
  const float t3 = 6 * K3_ * r4 + 4 * K2_ * r2 + 2 * K1_;
  jacobian.row(0)(0) = t1 + 6 * P2_ * x + 2 * P1_ * y + t3 * x2;
  jacobian.row(1)(1) = t1 + 2 * P2_ * x + 6 * P1_ * y + t3 * y2;

  const float t2 = 2 * (P1_ * x + P2_ * y) + x * y * t3;
  jacobian.row(0)(1) = t2;
  jacobian.row(1)(0) = t2;
}

PolynomialCameraModel::PolynomialCameraModel(const Vector2T& resolution, const Vector2T& focal,
                                             const Vector2T& principal, float K1, float K2, float K3, float K4,
                                             float K5, float K6, float P1, float P2, float max_normalized_uv_radius,
                                             float max_xy_radius)
    : CameraModelWithJacobianBasedUndistortion(resolution, focal, principal, max_normalized_uv_radius, max_xy_radius),
      K1_(K1),
      K2_(K2),
      K3_(K3),
      K4_(K4),
      K5_(K5),
      K6_(K6),
      P1_(P1),
      P2_(P2) {}

bool PolynomialCameraModel::distort(const Vector2T& xy, Vector2T& normalized_uv) const {
  const float x2 = xy.x() * xy.x(), y2 = xy.y() * xy.y();
  const float r2 = x2 + y2;
  const float kr = (1 + ((K3_ * r2 + K2_) * r2 + K1_) * r2) / (1 + ((K6_ * r2 + K5_) * r2 + K4_) * r2);

  normalized_uv.x() = xy.x() * kr + P1_ * (2 * xy.x() * xy.y()) + P2_ * (r2 + 2 * x2);
  normalized_uv.y() = xy.y() * kr + P1_ * (r2 + 2 * y2) + P2_ * (2 * xy.x() * xy.y());

  return true;
}

void PolynomialCameraModel::compute_distort_jacobian(const Vector2T& xy, Matrix2T& jacobian) const {
  // You can use expression below on https://www.wolframalpha.com/
  // jacobian matrix of (x*k+p1*(2*x*y)+p2*(r2+2*x*x), y*k+p1*(r2+2*y*y)+p2*(2*x*y))
  // where k = (1+((k3*r2+k2)*r2+k1)*r2) / (1+((k6*r2+k5)*r2+k4)*r2) and r2 = x*x+y*y
  const float x = xy.x();
  const float y = xy.y();
  const float x2 = x * x;
  const float y2 = y * y;
  const float r2 = x2 + y2;

  const float s = 1 + ((K3_ * r2 + K2_) * r2 + K1_) * r2;
  const float t = 1 + ((K6_ * r2 + K5_) * r2 + K4_) * r2;
  const float t2 = t * t;

  const float a = (3 * K3_ * r2 + 2 * K2_) * r2 + K1_;
  const float b = (3 * K6_ * r2 + 2 * K5_) * r2 + K4_;

  const float c = 2 * (a * t - s * b) / t2;

  jacobian.row(0)(0) = x2 * c + s / t + 2 * P1_ * y + 6 * P2_ * x;
  jacobian.row(1)(1) = y2 * c + s / t + 6 * P1_ * y + 2 * P2_ * x;

  const float j01 = x * y * c + 2 * P1_ * x + 2 * P2_ * y;

  jacobian.row(0)(1) = j01;
  jacobian.row(1)(0) = j01;
}

std::unique_ptr<ICameraModel> CreateCameraModel(const Vector2T& resolution, const Vector2T& focal,
                                                const Vector2T& principal, Distortion::Model model,
                                                const float* parameters, int32_t num_parameters) {
  switch (model) {
    case Distortion::Model::Polynomial:
      if (num_parameters != 8) return {};
      return std::make_unique<PolynomialCameraModel>(resolution, focal, principal, parameters[0], parameters[1],
                                                     parameters[4], parameters[5], parameters[6], parameters[7],
                                                     parameters[2],   // P1!
                                                     parameters[3]);  // P2!
    case Distortion::Model::Brown:
      if (num_parameters != 5) return {};
      // Brown parameters are already (k1, k2, k3, p1, p2), so no reordering here unlike Polynomial above.
      return std::make_unique<Brown5KCameraModel>(resolution, focal, principal, parameters[0], parameters[1],
                                                  parameters[2], parameters[3], parameters[4]);
    case Distortion::Model::Fisheye:
      if (num_parameters != 4) return {};
      return std::make_unique<FisheyeCameraModel>(resolution, focal, principal, parameters[0], parameters[1],
                                                  parameters[2], parameters[3]);
    case Distortion::Model::Pinhole:
      if (num_parameters != 0) return {};
      return std::make_unique<PinholeCameraModel>(resolution, focal, principal);
  }
  return {};
}
}  // namespace cuvslam::camera

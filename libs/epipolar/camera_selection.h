
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

#include <array>

#include "Eigen/Eigenvalues"

#include "common/isometry_utils.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "epipolar/near_plane.h"

namespace cuvslam::epipolar {

enum class TriangulationState { None, Triangulated, AlmostParallel };

/* TODO: review OptimalTriangulation limits
 * With 0.005 a lot of good tracks reported as non triangulated, that cause
 * tracking lost in KITTI12 and LINK00 sequences.
 * With 0.0005 tracking robustness improved, but average KITTI translation
 * error increases a bit. Maximum z distance reported in KITTI is 1200 meters,
 * for LINK00 is 120 meters that prove this limit is safe.
 * There is a possible mistake in OptimalTriangulation math in the
 * calculation of the triangulation quality. */
const float MINIMUM_TRIANGULATION_MEASURE = 0.0005f;

template <typename _Vector, typename _VectorOther, typename _Scalar = typename _Vector::Scalar>
_Scalar VectorsParallelMeasure(const _Vector& v1, const _VectorOther& v2) {
  const _Scalar norm = v1.norm() * v2.norm();
  return (norm > epsilon()) ? (v1.cross(v2)).norm() / norm : 0;
}

template <typename _Vector, typename _VectorOther, typename _Scalar = typename _Vector::Scalar>
bool IsVectorsParallel(const _Vector& v1, const _VectorOther& v2, _Scalar& measure) {
  return (measure = VectorsParallelMeasure(v1, v2)) < MINIMUM_TRIANGULATION_MEASURE;
}

template <typename _Scalar>
class Ray3 {
  Vector3<_Scalar> origin_;
  Vector3<_Scalar> direction_;  /// The direction of the ray is normalized.

public:
  Ray3() = default;
  Ray3(const Ray3&) = default;
  Ray3(const Vector3T& o, const Vector3T& d) : origin_(o), direction_(d) {
    assert(d.norm() > epsilon());
    direction_.normalize();
  }

  static Ray3 GenerateRayFrom2Points(const Vector3<_Scalar>& origin, const Vector3<_Scalar>& otherPoint) {
    return Ray3(origin, otherPoint - origin);
  }

  const Vector3<_Scalar>& getOrigin() const { return origin_; }
  const Vector3<_Scalar>& getDirection() const { return direction_; }

  bool Intersect(const Ray3& ray, Vector3<_Scalar>& intersection) const {
    const _Scalar r12 = direction_.dot(ray.direction_);
    const _Scalar r11 = direction_.dot(direction_);
    const _Scalar r22 = ray.direction_.dot(ray.direction_);
    const _Scalar denom = r22 * r11 - r12 * r12;

    // If the denominator is too small, we are parallel
    if (std::abs(denom) < 20.0f * epsilon()) {
      return false;
    }

    const Vector3<_Scalar> origin_delta = ray.origin_ - origin_;
    const _Scalar r1d = direction_.dot(origin_delta);
    const _Scalar r2d = ray.direction_.dot(origin_delta);

    const _Scalar mu1 = (r22 * r1d - r12 * r2d) / denom;
    const _Scalar mu2 = (r12 * r1d - r11 * r2d) / denom;

    const Vector3<_Scalar> p1 = origin_ + mu1 * direction_;
    const Vector3<_Scalar> p2 = ray.origin_ + mu2 * ray.direction_;

    intersection = (p1 + p2) / 2;

    const float scaleFactor = 1e3f;
    const float atInfinity = 1.0f / epsilon();

    if ((intersection / scaleFactor).norm() > atInfinity / scaleFactor) {
      intersection = direction_ * -atInfinity;
    }

    return true;
  }
};

using Ray3T = Ray3<float>;

inline bool IsPointInFrontInLocalSpace(const Vector3T& point, const Isometry3T& toLocalSpace) {
  return IsDepthInFront((toLocalSpace * point).z());
}

inline bool IntersectRaysInReferenceSpace(const Isometry3T& transform, const Vector3T& direction1,
                                          const Vector3T& direction2, Vector3T& point3DInReferenceSpace) {
  // Camera1 is the frame of reference, i.e. camera1 is the world coordinate space. All inputs
  // are expressed in camera1 frame of reference. Transform direction2 from the camera1 frame of
  // reference to the hypothetical camera2 frame of reference.
  const Ray3T ray2(transform.translation(), transform.linear() * direction2);
  return ray2.Intersect(Ray3T(Vector3T::Zero(), direction1), point3DInReferenceSpace) &&
         IsPointInFrontInLocalSpace(point3DInReferenceSpace, Isometry3T::Identity()) &&
         IsPointInFrontInLocalSpace(point3DInReferenceSpace, transform.inverse());
}

inline size_t CountPointsInFrontOfCameras(const Vector2TPairVectorCIt start, const Vector2TPairVectorCIt end,
                                          const Isometry3T& t) {
  return std::count_if(start, end, [&](const Vector2TPair& v) -> bool {
    Vector3T ignore3DPoint;
    return IntersectRaysInReferenceSpace(t, v.first.homogeneous(), v.second.homogeneous(), ignore3DPoint);
  });
}

// Method below is 'niter' methods from https://e-reports-ext.llnl.gov/pdf/384387.pdf ("Triangulation Made Easy",
// Listing 3, left column)
inline bool OptimalTriangulation(const Isometry3T& transform, const Vector2T& x1, const Vector2T& x2, Vector3T& loc3d,
                                 float& parallelMeasure, TriangulationState& ts) {
  ts = TriangulationState::None;

  // `transform` is cam1_from_cam2 (same as rig: camera_from_rig[c1]*camera_from_rig[c2].inverse()): p_c1 = T * p_c2.
  // x1,x2 are normalized coords (OpenCV pinhole: +Z forward). E = [t]_× R from essential(T); epipolar c = x1^T E x2.
  // loc3d is expressed in camera-1 coordinates; cheirality uses z>0 in cam1 and in cam2 (transform.inverse()*loc3d).

  const Vector3T x1h = x1.homogeneous();
  const Vector3T x2h = x2.homogeneous();

  const Matrix3T& rotation = transform.linear();
  if (IsVectorsParallel(x1h, rotation * x2h, parallelMeasure)) {
    return false;
  }

  const Matrix3T e = essential(transform);

  MatrixMN<float, 2, 3> S;
  S.setZero();
  S(0, 0) = 1.0f;
  S(1, 1) = 1.0f;

  Vector2T n1 = S * e * x2h;
  Vector2T n2 = S * e.transpose() * x1h;

  Matrix2T essential2by2 = S * e * S.transpose();

  const float a = n1.dot(essential2by2 * n2);
  const float b = 0.5f * (n1.squaredNorm() + n2.squaredNorm());
  const float c = x1h.dot(e * x2h);
  const float bbac = b * b - a * c;

  if (bbac < -float(sqrt_epsilon<float>())) {
    // this can happen if 'c', measure of non-epipolarity, is quite high
    return false;
  }

  const float d = (bbac > 0) ? std::sqrt(bbac) : 0;

  const float failureThreshold = 2.0f * epsilon();
  const float bd = std::max(b + d, failureThreshold);

  const float lambda = c / bd;
  Vector2T dx1 = lambda * n1;
  Vector2T dx2 = lambda * n2;

  n1 = n1 - essential2by2 * dx2;
  n2 = n2 - essential2by2.transpose() * dx1;

  const float n1sqN = n1.squaredNorm();
  const float n2sqN = n2.squaredNorm();

#if 0

    if (n1sqN + n2sqN < failureThreshold)
    {
        return false;
    }

    // niter2 - faster but lower quality
    lambda = lambda * 2.0f * d / (n1sqN + n2sqN);
    dx1 = lambda * n1;
    dx2 = lambda * n2;
#else

  if (n1sqN < failureThreshold || n2sqN < failureThreshold) {
    return false;
  }

  // niter1 - slower but higher quality
  dx1 = dx1.dot(n1) / n1sqN * n1;
  dx2 = dx2.dot(n2) / n2sqN * n2;
#endif

  const Vector3T x1m = x1h - S.transpose() * dx1;
  const Vector3T x2m = x2h - S.transpose() * dx2;

  if (IsVectorsParallel(x1m, rotation * x2m, parallelMeasure)) {
    return false;
  }

  const Vector3T z = x1m.cross(rotation * x2m);
  const float zn2 = z.squaredNorm();

  if (zn2 < failureThreshold) {
    return false;
  }

  // we are getting 3D location in C1 local coords, so in world coords 3D location will be Loc3d = C1*loc3d
  loc3d = z.dot(e * x2m) * x1m / zn2;

  // const float scaleFactor = 1e3f;
  // const float atInfinity = 1.0f / epsilon();
  //(void)scaleFactor;
  //(void)atInfinity;
  // assert((loc3d / scaleFactor).norm() < atInfinity / scaleFactor);

  ts = (IsDepthInFront(loc3d.z()) && IsPointInFrontInLocalSpace(loc3d, transform.inverse()) && loc3d.norm() > epsilon())
           ? ((IsVectorsParallel(x1m, rotation * x2m, parallelMeasure)) ? TriangulationState::AlmostParallel
                                                                        : TriangulationState::Triangulated)
           : TriangulationState::None;

  return (ts != TriangulationState::None);
}

// @TODO: need to find better home for this class
class OptimalPlanarTriangulation {
  using Matrix9x4T = MatrixMN<float, 9, 4>;

public:
  OptimalPlanarTriangulation() : t_{{Matrix9x4T::Zero(), Matrix9x4T::Zero(), Matrix9x4T::Zero()}} {
    t_[0](3, 0) = -1;
    t_[0](4, 1) = -1;
    t_[0](8, 3) = 1;
    t_[1](0, 0) = 1;
    t_[1](1, 1) = 1;
    t_[1](8, 2) = -1;
    t_[2](2, 3) = -1;
    t_[2](5, 2) = 1;

    xi_[0].head(3).setZero();
    xi_[0](5) = -1;
    xi_[1].segment(3, 3).setZero();
    xi_[1](2) = 1;
    xi_[2].segment(6, 3).setZero();
  }

  // @TODO: implementation of this method is not optimal. It will benefit from algorithmic optimization.
  bool triangulate(const Matrix3T& /*h*/, const Isometry3T& trans, const Vector2T& p1, const Vector2T& p2,
                   Vector3T& loc3d, float& parallelMeasure, TriangulationState& ts) {
    const Vector3T p1h = p1.homogeneous();
    const Vector3T p2h = p2.homogeneous();

    ts = IsVectorsParallel(p1h, trans.linear() * p2h, parallelMeasure) ? TriangulationState::AlmostParallel
                                                                       : TriangulationState::Triangulated;
#if 1

    if (!IntersectRaysInReferenceSpace(trans, p1h, p2h, loc3d) || loc3d.norm() < epsilon()) {
      ts = TriangulationState::None;
    }

    return (ts != TriangulationState::None);
#else
    const Vector9T u(h.transpose().eval().data());
    const Vector4T p0((Vector4T() << p1, p2).finished());
    Vector4T p(p0);
    Vector4T delta(Vector4T::Zero());
    float error = std::numeric_limits<float>::max();
    int iters = 30;  // max iter count
    bool isConverged = false;

    // Vector3T p2H = h * p1h;
    // TraceMessage("discrepancy = %f\n",(p2h - p2H / p2H[2]).norm());

    do {
      // update Ts
      t_[0](6, 0) = p(3);
      t_[0](6, 3) = p(0);
      t_[0](7, 1) = p(3);
      t_[0](7, 3) = p(1);
      t_[1](6, 0) = -p(2);
      t_[1](6, 2) = -p(0);
      t_[1](7, 1) = -p(2);
      t_[1](7, 2) = -p(1);
      t_[2](0, 0) = -p(3);
      t_[2](1, 1) = -p(3);
      t_[2](3, 0) = p(2);
      t_[2](4, 1) = p(2);
      t_[2](0, 3) = -p(0);
      t_[2](1, 3) = -p(1);
      t_[2](4, 2) = p(1);
      t_[2](3, 2) = p(0);

      // update Xis
      xi_[0](3) = -p(0);
      xi_[0](4) = -p(1);
      xi_[0](6) = p(0) * p(3);
      xi_[0](7) = p(1) * p(3);
      xi_[0](8) = p(3);
      xi_[1](0) = p(0);
      xi_[1](1) = p(1);
      xi_[1](6) = -p(0) * p(2);
      xi_[1](7) = -p(1) * p(2);
      xi_[1](8) = -p(2);
      xi_[2](0) = -p(0) * p(3);
      xi_[2](1) = -p(1) * p(3);
      xi_[2](2) = -p(3);
      xi_[2](3) = p(0) * p(2);
      xi_[2](4) = p(1) * p(2);
      xi_[2](5) = p(2);

      // @TODO: this step is no-op on very first iteration
      xi_[0] += t_[0] * delta;
      xi_[1] += t_[1] * delta;
      xi_[2] += t_[2] * delta;

      //// compute weight matrix
      auto item = [&](const int i, const int j) -> float { return u.dot(t_[i] * t_[j].transpose() * u); };
      Eigen::SelfAdjointEigenSolver<Matrix3T> es((Matrix3T() << item(0, 0), item(0, 1), item(0, 2), item(1, 0),
                                                  item(1, 1), item(1, 2), item(2, 0), item(2, 1), item(2, 2))
                                                     .finished());

      const Vector3T& ev = es.eigenvalues();

      if (ev(1) < sqrt_epsilon() || ev(2) < sqrt_epsilon()) {
        TraceError("ill-formed weight matrix, ev1 = %f, ev2 = %f", ev(1), ev(2));
        return false;
      }

      const Matrix3T& evc = es.eigenvectors();
      const Matrix3T w(((evc.col(1) * evc.col(1).transpose()) / ev(1) + (evc.col(2) * evc.col(2).transpose()) / ev(2)));

      // compute new deltas (as in reference, rewrite to be more efficient, use Eigen matrix ops)
      const std::array<Vector4T, 3> vs{t_[0].transpose() * u, t_[1].transpose() * u, t_[2].transpose() * u};

      delta = w(0, 0) * xi_[0].dot(u) * vs[0] + w(0, 1) * xi_[1].dot(u) * vs[0] + w(0, 2) * xi_[2].dot(u) * vs[0];
      delta += w(1, 0) * xi_[0].dot(u) * vs[1] + w(1, 1) * xi_[1].dot(u) * vs[1] + w(1, 2) * xi_[2].dot(u) * vs[1];
      delta += w(2, 0) * xi_[0].dot(u) * vs[2] + w(2, 1) * xi_[1].dot(u) * vs[2] + w(2, 2) * xi_[2].dot(u) * vs[2];

      p = p0 - delta;
      const float e = delta.squaredNorm();
      isConverged = std::abs(e - error) < sqrt_epsilon();
      error = e;
    } while (!isConverged && --iters > 0);

    if (!isConverged) {
      TraceMessage("Failed to converge!");
      return false;
    }

    // Vector3T p2Hd = h * p.head(2).homogeneous();
    // TraceMessage("discrepancy = %f\n", (p.tail(2).homogeneous() - p2Hd / p2Hd[2]).norm());

    const Vector3T ray1 = p.head(2).homogeneous();
    const Vector3T ray2 = trans.linear() * p.tail(2).homogeneous();

    if (IsVectorsParallel(ray1, ray2, parallelMeasure)) {
      return false;
    }

    return IntersectRaysInReferenceSpace(trans, p.head(2).homogeneous(), p.tail(2).homogeneous(), loc3d);
#endif
  }

private:
  std::array<Matrix9x4T, 3> t_;
  std::array<Vector9T, 3> xi_;
};

}  // namespace cuvslam::epipolar

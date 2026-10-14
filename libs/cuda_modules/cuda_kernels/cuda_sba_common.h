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

#include <cfloat>
#include <cstdio>

#include <float.h>

#include "cuda_modules/cuda_kernels/cuda_matrix.h"

namespace {

using namespace cuvslam::cuda;

template <typename T>
__device__ __forceinline__ T max(T a, T b) {
  return (a < b) ? b : a;
}

template <typename T>
__device__ __forceinline__ void swap(T& a, T& b) {
  const T t = a;
  a = b;
  b = t;
}

template <typename T, int N>
__device__ __forceinline__ void dampen(Mat<T, N, N>& m, T lambda) {
  const T f = static_cast<T>(1) + lambda;
  for (int i = 0; i < N; ++i) m.d_[i][i] *= f;
}

template <int N>
__device__ __forceinline__ void copy_fd(const Mat<float, N, N>& a, Mat<double, N, N>& b) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) b.d_[i][j] = a.d_[i][j];
}

template <typename T, int M, int N>
__device__ __forceinline__ void transp(const Mat<T, M, N>& a, Mat<T, N, M>& b) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) b.d_[i][j] = a.d_[j][i];
}

template <typename T, int M, int N>
__device__ __forceinline__ Mat<T, N, M> transp(const Mat<T, M, N>& a) {
  Mat<T, N, M> res;
  transp(a, res);
  return res;
}

template <typename T, int N>
__device__ __forceinline__ Mat<T, 1, N> transp(const Vec<T, N>& a) {
  Mat<T, 1, N> res;
  for (int i = 0; i < N; ++i) res.d_[0][i] = a.d_[i];
  return res;
}

template <typename T, int N>
__device__ __forceinline__ Vec<T, N> to_vector(const Mat<T, N, 1>& a) {
  Vec<T, N> res;
  for (int i = 0; i < N; ++i) res.d_[i] = a.d_[i][0];
  return res;
}

template <typename T, int M, int N, int K>
__device__ __forceinline__ void mul(const Mat<T, M, K>& a, const Mat<T, K, N>& b, Mat<T, M, N>& c) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      c.d_[i][j] = a.d_[i][0] * b.d_[0][j];
      for (int k = 1; k < K; ++k) c.d_[i][j] += a.d_[i][k] * b.d_[k][j];
    }
}

template <typename T, int M, int N, int K>
__device__ __forceinline__ Mat<T, M, N> operator*(const Mat<T, M, K>& a, const Mat<T, K, N>& b) {
  Mat<T, M, N> res;
  mul(a, b, res);
  return res;
}

template <typename T, int M, int N>
__device__ __forceinline__ void mul(const Mat<T, M, N>& a, const Vec<T, N>& b, Vec<T, M>& c) {
  for (int i = 0; i < M; ++i) {
    c.d_[i] = a.d_[i][0] * b.d_[0];
    for (int j = 1; j < N; ++j) {
      c.d_[i] += a.d_[i][j] * b.d_[j];
    }
  }
}

template <typename T, int M, int N>
__device__ __forceinline__ Vec<T, M> operator*(const Mat<T, M, N>& a, const Vec<T, N>& b) {
  Vec<T, M> res;
  mul(a, b, res);
  return res;
}

template <typename T, int N>
__device__ __forceinline__ void mul_add(const Mat<T, N, N>& a, const Vec<T, N>& b, const Vec<T, N>& c, Vec<T, N>& d) {
  for (int i = 0; i < N; ++i) {
    d.d_[i] = c.d_[i] + a.d_[i][0] * b.d_[0];
    for (int j = 1; j < N; ++j) d.d_[i] += a.d_[i][j] * b.d_[j];
  }
}

template <typename T, int N>
__device__ __forceinline__ void add(const Vec<T, N>& a, const Vec<T, N>& b, Vec<T, N>& c) {
  for (int i = 0; i < N; ++i) c.d_[i] = a.d_[i] + b.d_[i];
}

template <typename T, int N>
__device__ __forceinline__ Vec<T, N> operator+(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> res;
  add(a, b, res);
  return res;
}

template <typename T, int N>
__device__ __forceinline__ void sub(const Vec<T, N>& a, const Vec<T, N>& b, Vec<T, N>& c) {
  for (int i = 0; i < N; ++i) c.d_[i] = a.d_[i] - b.d_[i];
}

template <typename T, int N>
__device__ __forceinline__ Vec<T, N> operator-(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> res;
  sub(a, b, res);
  return res;
}

template <typename T, int N>
__device__ __forceinline__ void mul(T a, const Vec<T, N>& b, Vec<T, N>& c) {
  for (int i = 0; i < N; ++i) c.d_[i] = a * b.d_[i];
}

template <typename T, int N>
__device__ __forceinline__ Vec<T, N> operator*(T a, const Vec<T, N>& b) {
  Vec<T, N> res;
  mul(a, b, res);
  return res;
}

template <typename T, int N>
__device__ __forceinline__ T dot(const Vec<T, N>& a, const Vec<T, N>& b) {
  T res = a.d_[0] * b.d_[0];
  for (int i = 1; i < N; ++i) {
    res += a.d_[i] * b.d_[i];
  }
  return res;
}

template <typename T>
__device__ __forceinline__ Vec<T, 3> cross(const Vec<T, 3>& a, const Vec<T, 3>& b) {
  Vec<T, 3> res;
  res.d_[0] = a.d_[1] * b.d_[2] - a.d_[2] * b.d_[1];
  res.d_[1] = a.d_[2] * b.d_[0] - a.d_[0] * b.d_[2];
  res.d_[2] = a.d_[0] * b.d_[1] - a.d_[1] * b.d_[0];
  return res;
}

template <typename T, int N>
__device__ __forceinline__ T trace(const Mat<T, N, N>& a) {
  T res = a.d_[0][0];
  for (int i = 1; i < N; ++i) res += a.d_[i][i];
  return res;
}

template <typename T, int M, int N>
__device__ __forceinline__ Mat<T, M, N> operator*(T a, const Mat<T, M, N>& b) {
  Mat<T, M, N> res;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) res.d_[i][j] = a * b.d_[i][j];
  return res;
}

template <typename T, int M, int N>
__device__ __forceinline__ Mat<T, M, N> operator-(const Mat<T, M, N>& a, const Mat<T, M, N>& b) {
  Mat<T, M, N> res;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) res.d_[i][j] = a.d_[i][j] - b.d_[i][j];
  return res;
}

template <typename T, int M, int N>
__device__ __forceinline__ Mat<T, M, N> operator+(const Mat<T, M, N>& a, const Mat<T, M, N>& b) {
  Mat<T, M, N> res;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) res.d_[i][j] = a.d_[i][j] + b.d_[i][j];
  return res;
}

template <typename T, int N>
__device__ __forceinline__ Mat<T, N, N> identity() {
  Mat<T, N, N> res;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      res.d_[i][j] = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
    }
  }
  return res;
}

template <typename T, int N>
__device__ __forceinline__ T extract(const Vec<T, N>& a, int index) {
  T res = a.d_[0];
  for (int i = 1; i < N; ++i)
    if (i == index) res = a.d_[i];
  return res;
}

template <typename T>
__device__ __forceinline__ T extract(const Vec<T, 3>& a, int index) {
  return (index == 0) ? a.d_[0] : ((index == 1) ? a.d_[1] : a.d_[2]);
}

template <typename T, int M, int N>
__device__ __forceinline__ T extract(const Mat<T, M, N>& a, int row_id, int col_id) {
  T res = a.d_[0][0];
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if ((i == row_id) && (j == col_id)) {
        res = a.d_[i][j];
      }
    }
  }
  return res;
}

template <typename T>
__device__ __forceinline__ Mat<T, 3, 3> SkewSymmetric(const Vec<T, 3>& v, const T d = 0) {
  Mat<T, 3, 3> res;
  res.d_[0][0] = d;
  res.d_[0][1] = -v.d_[2];
  res.d_[0][2] = v.d_[1];
  res.d_[1][0] = v.d_[2];
  res.d_[1][1] = d;
  res.d_[1][2] = -v.d_[0];
  res.d_[2][0] = -v.d_[1];
  res.d_[2][1] = v.d_[0];
  res.d_[2][2] = d;
  return res;
}

template <typename T>
__device__ __forceinline__ Mat<T, 1, 3> SkewSymmetric_row(const Vec<T, 3>& v, int row_id, const T d = 0) {
  Mat<T, 1, 3> res;
  res.d_[0][0] = (row_id == 0) ? d : ((row_id == 1) ? v.d_[2] : -v.d_[1]);
  res.d_[0][1] = (row_id == 0) ? -v.d_[2] : ((row_id == 1) ? d : v.d_[0]);
  res.d_[0][2] = (row_id == 0) ? v.d_[1] : ((row_id == 1) ? -v.d_[0] : d);
  return res;
}

template <typename T>
__device__ __forceinline__ T SkewSymmetric(const Vec<T, 3>& v, int row_id, int col_id, const T d = 0) {
  int line = col_id + row_id;
  float res = (line == 1) ? v.d_[2] : ((line == 2) ? v.d_[1] : v.d_[0]);
  if (row_id == col_id) res = d;
  if (((row_id + 1) == col_id) || (row_id == (col_id + 2))) res = -res;
  return res;
}

template <typename T>
__device__ __forceinline__ T fast_div(T a, T b) {
  if (sizeof(T) == 4)
    return __fdividef(a, b);
  else
    return a / b;
}

template <typename T>
__device__ __forceinline__ T fast_sqrt(T a) {
  if (sizeof(T) == 4)
    return sqrtf(a);
  else
    return sqrt(a);
}

template <typename T>
__device__ __forceinline__ T fast_rsqrt(T a) {
  if (sizeof(T) == 4)
    return rsqrtf(a);
  else
    return rsqrt(a);
}

template <typename T>
__device__ __forceinline__ T fast_acos(T a) {
  if (sizeof(T) == 4)
    return acosf(a);
  else
    return acos(a);
}

template <typename T>
__device__ __forceinline__ T fast_cos(T a) {
  if (sizeof(T) == 4)
    return __cosf(a);
  else
    return cos(a);
}

template <typename T>
__device__ __forceinline__ T fast_sin(T a) {
  if (sizeof(T) == 4)
    return __sinf(a);
  else
    return sin(a);
}

template <typename T>
__device__ __forceinline__ T fast_tan(T a) {
  if (sizeof(T) == 4)
    return __tanf(a);
  else
    return tan(a);
}

template <typename T>
__device__ __forceinline__ T fast_cotan(T a) {
  if (sizeof(T) == 4)
    return __fdividef(__cosf(a), __sinf(a));
  else
    return cos(a) / sin(a);
}

__device__ __forceinline__ float ComputeHuberLoss(float x_squared, float delta) {
  if (delta <= 0.f) return 0.5f * x_squared;

  float delta_squared = delta * delta;

  if (x_squared < delta_squared) {
    return 0.5f * x_squared;
  }

  return delta * fast_sqrt(x_squared) - 0.5f * delta_squared;
}

__device__ __forceinline__ float ComputeDHuberLoss(float x_squared, float delta) {
  if (delta <= 0.f) return 0.5f;

  float delta_squared = delta * delta;

  if (x_squared < delta_squared) {
    return 0.5f;
  }

  return 0.5f * delta * fast_rsqrt(x_squared);
}

template <typename T>
class LDLT {
public:
  __device__ __forceinline__ LDLT(const Mat<T, 3, 3>& m) {
    l_.d_[0][0] = l_.d_[1][1] = l_.d_[2][2] = (T)1.;
    l_.d_[0][1] = l_.d_[0][2] = l_.d_[1][2] = (T)0.;

    Mat<T, 3, 3> a = m;
    perm_0_swap01 = false;
    perm_0_swap02 = false;
    perm_1_swap12 = false;

    T d0 = abs(a.d_[0][0]), d1 = abs(a.d_[1][1]), d2 = abs(a.d_[2][2]);
    if (max(d1, d2) > d0) {
      if (d1 > d2) {
        swap(a.d_[2][0], a.d_[2][1]);
        swap(a.d_[0][0], a.d_[1][1]);
        perm_0_swap01 = true;
      } else {
        swap(a.d_[0][0], a.d_[2][2]);
        swap(a.d_[1][0], a.d_[2][1]);
        perm_0_swap02 = true;
      }
    }
    a.d_[0][0] = max(a.d_[0][0], (T)0.);
    a.d_[1][0] = (a.d_[0][0] == (T)0.) ? (T)0. : fast_div(a.d_[1][0], a.d_[0][0]);
    a.d_[2][0] = (a.d_[0][0] == (T)0.) ? (T)0. : fast_div(a.d_[2][0], a.d_[0][0]);

    d1 = abs(a.d_[1][1]);
    d2 = abs(a.d_[2][2]);
    if (d2 > d1) {
      swap(a.d_[1][0], a.d_[2][0]);
      swap(a.d_[1][1], a.d_[2][2]);
      perm_1_swap12 = true;
    }
    a.d_[1][1] = a.d_[1][1] - a.d_[1][0] * a.d_[1][0] * a.d_[0][0];
    a.d_[1][1] = max(a.d_[1][1], (T)0.);
    a.d_[2][1] =
        (a.d_[1][1] == (T)0.) ? (T)0. : fast_div(a.d_[2][1] - a.d_[2][0] * a.d_[1][0] * a.d_[0][0], a.d_[1][1]);

    a.d_[2][2] = a.d_[2][2] - a.d_[2][0] * a.d_[2][0] * a.d_[0][0] - a.d_[2][1] * a.d_[2][1] * a.d_[1][1];
    a.d_[2][2] = max(a.d_[2][2], (T)0.);

    l_.d_[1][0] = a.d_[1][0];
    l_.d_[2][0] = a.d_[2][0];
    l_.d_[2][1] = a.d_[2][1];
    d_.d_[0] = a.d_[0][0];
    d_.d_[1] = a.d_[1][1];
    d_.d_[2] = a.d_[2][2];
    d_inv_.d_[0] = (d_.d_[0] == (T)0.) ? (T)0. : fast_div((T)1., d_.d_[0]);
    d_inv_.d_[1] = (d_.d_[1] == (T)0.) ? (T)0. : fast_div((T)1., d_.d_[1]);
    d_inv_.d_[2] = (d_.d_[2] == (T)0.) ? (T)0. : fast_div((T)1., d_.d_[2]);
  }

  __device__ __forceinline__ Vec<T, 3> solve(const Vec<T, 3>& b) const {
    // Solving P * bp = b
    Vec<T, 3> bp = b;
    if (perm_0_swap01) swap(bp.d_[0], bp.d_[1]);
    if (perm_0_swap02) swap(bp.d_[0], bp.d_[2]);
    if (perm_1_swap12) swap(bp.d_[1], bp.d_[2]);

    // Solving L * x1 = bp
    Vec<T, 3> x1;
    x1.d_[0] = bp.d_[0];
    x1.d_[1] = bp.d_[1] - x1.d_[0] * l_.d_[1][0];
    x1.d_[2] = bp.d_[2] - x1.d_[0] * l_.d_[2][0] - x1.d_[1] * l_.d_[2][1];

    // Solving D * x2 = x1
    Vec<T, 3> x2;
    x2.d_[0] = x1.d_[0] * d_inv_.d_[0];
    x2.d_[1] = x1.d_[1] * d_inv_.d_[1];
    x2.d_[2] = x1.d_[2] * d_inv_.d_[2];

    // Solving LT * x3 = x2
    Vec<T, 3> x3;
    x3.d_[2] = x2.d_[2];
    x3.d_[1] = x2.d_[1] - x3.d_[2] * l_.d_[2][1];
    x3.d_[0] = x2.d_[0] - x3.d_[1] * l_.d_[1][0] - x3.d_[2] * l_.d_[2][0];

    // Solving P * x3_new = x3
    if (perm_1_swap12) swap(x3.d_[1], x3.d_[2]);
    if (perm_0_swap02) swap(x3.d_[0], x3.d_[2]);
    if (perm_0_swap01) swap(x3.d_[0], x3.d_[1]);

    return x3;
  }

  __device__ __forceinline__ Mat<T, 3, 3> inv() const {
    Mat<T, 3, 3> res;

    Vec<T, 3> u1{(T)1., (T)0., (T)0.};
    Vec<T, 3> x1 = solve(u1);
    res.d_[0][0] = x1.d_[0];
    res.d_[1][0] = x1.d_[1];
    res.d_[2][0] = x1.d_[2];

    Vec<T, 3> u2{(T)0., (T)1., (T)0.};
    Vec<T, 3> x2 = solve(u2);
    res.d_[0][1] = x2.d_[0];
    res.d_[1][1] = x2.d_[1];
    res.d_[2][1] = x2.d_[2];

    Vec<T, 3> u3{(T)0., (T)0., (T)1.};
    Vec<T, 3> x3 = solve(u3);
    res.d_[0][2] = x3.d_[0];
    res.d_[1][2] = x3.d_[1];
    res.d_[2][2] = x3.d_[2];

    return res;
  }

  __device__ __forceinline__ const Mat<T, 3, 3>& l() const { return l_; }
  __device__ __forceinline__ const Vec<T, 3>& d() const { return d_; }
  __device__ __forceinline__ const Matd33 p() const {
    Mat<T, 3, 3> res{(T)1., (T)0., (T)0., (T)0., (T)1., (T)0., (T)0., (T)0., (T)1.};
    if (perm_0_swap01) {
      swap(res.d_[0][0], res.d_[0][1]);
      swap(res.d_[1][0], res.d_[1][1]);
      swap(res.d_[2][0], res.d_[2][1]);
    }
    if (perm_0_swap02) {
      swap(res.d_[0][0], res.d_[0][2]);
      swap(res.d_[1][0], res.d_[1][2]);
      swap(res.d_[2][0], res.d_[2][2]);
    }
    if (perm_1_swap12) {
      swap(res.d_[0][1], res.d_[0][2]);
      swap(res.d_[1][1], res.d_[1][2]);
      swap(res.d_[2][1], res.d_[2][2]);
    }
    return res;
  }

private:
  Mat<T, 3, 3> l_;
  Vec<T, 3> d_;
  Vec<T, 3> d_inv_;
  bool perm_0_swap01;
  bool perm_0_swap02;
  bool perm_1_swap12;
};

template <typename T>
class SVD {
public:
  __device__ __forceinline__ SVD(const Mat<T, 3, 3>& m, T threshold) {
    Mat<T, 3, 3> ata = transp(m) * m;
    auto eigenvalues = compute_eigenvalues(ata);

    s_.d_[0] = fast_sqrt(eigenvalues.d_[0]);
    s_.d_[1] = fast_sqrt(eigenvalues.d_[1]);
    s_.d_[2] = fast_sqrt(eigenvalues.d_[2]);

    rank_ = get_rank_(threshold);

    auto eigenvector2 = compute_eigenvector(eigenvalues.d_[2], ata);
    v_.d_[0][2] = eigenvector2.d_[0];
    v_.d_[1][2] = eigenvector2.d_[1];
    v_.d_[2][2] = eigenvector2.d_[2];
    auto eigenvector0 = compute_eigenvector(eigenvalues.d_[0], ata);
    v_.d_[0][0] = eigenvector0.d_[0];
    v_.d_[1][0] = eigenvector0.d_[1];
    v_.d_[2][0] = eigenvector0.d_[2];
    auto eigenvector1 = cross(eigenvector0, eigenvector2);
    eigenvector1 = fast_rsqrt(dot(eigenvector1, eigenvector1)) * eigenvector1;
    v_.d_[0][1] = eigenvector1.d_[0];
    v_.d_[1][1] = eigenvector1.d_[1];
    v_.d_[2][1] = eigenvector1.d_[2];

    auto u0 = m * eigenvector0;
    u0 = fast_rsqrt(dot(u0, u0)) * u0;
    u_.d_[0][0] = u0.d_[0];
    u_.d_[1][0] = u0.d_[1];
    u_.d_[2][0] = u0.d_[2];
    auto u1 = m * eigenvector1;
    u1 = fast_rsqrt(dot(u1, u1)) * u1;
    u_.d_[0][1] = u1.d_[0];
    u_.d_[1][1] = u1.d_[1];
    u_.d_[2][1] = u1.d_[2];
    auto u2_direction = m * eigenvector2;
    auto u2 = cross(u0, u1);
    T mult = fast_rsqrt(dot(u2, u2));
    if (dot(u2, u2_direction) < (T)0.0) mult = -mult;
    u2 = mult * u2;
    u_.d_[0][2] = u2.d_[0];
    u_.d_[1][2] = u2.d_[1];
    u_.d_[2][2] = u2.d_[2];
  }

  __device__ __forceinline__ int rank() const { return rank_; }
  __device__ __forceinline__ const Mat<T, 3, 3>& u() const { return u_; }
  __device__ __forceinline__ const Vec<T, 3>& s() const { return s_; }
  __device__ __forceinline__ const Mat<T, 3, 3>& v() const { return v_; }

private:
  Mat<T, 3, 3> u_;
  Vec<T, 3> s_;
  Mat<T, 3, 3> v_;
  int rank_;

  // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
  // Matrix a should be symmetric
  __device__ __forceinline__ Vec<T, 3> compute_eigenvalues(const Mat<T, 3, 3>& a) {
    Vec<T, 3> res;
    T A0 = a.d_[0][0], A1 = a.d_[0][1], A2 = a.d_[0][2], A4 = a.d_[1][1], A5 = a.d_[1][2], A8 = a.d_[2][2];

    const T ONE_THIRD_PI = (T)1.0471975511965977461542144610931676280657231331250352736583148641;
    const T TWO_THIRDS_PI = (T)2.0943951023931954923084289221863352561314462662500705473166297282;

    T p1 = A1 * A1 + A2 * A2 + A5 * A5;
    T trace = A0 + A4 + A8;
    bool isDiagonal = (p1 <= (T)FLT_MIN);
    if (isDiagonal) {
      res.d_[0] = (T)max(max(A0, A4), A8);
      res.d_[2] = (T)min(min(A0, A4), A8);
    } else {
      T q = trace * (T)(1. / 3.);
      T p2 = (A0 - q) * (A0 - q) + (A4 - q) * (A4 - q) + (A8 - q) * (A8 - q) + ((T)2.) * p1;
      T p = fast_sqrt(p2 * (T)(1. / 6.));
      T r = fast_div(A0 * A4 * A8 - A0 * A4 * q - A0 * A8 * q - A0 * A5 * A5 + A0 * q * q - A4 * A8 * q - A4 * A2 * A2 +
                         A4 * q * q - A8 * A1 * A1 + A8 * q * q + A1 * A1 * q + (T)(2.) * A1 * A2 * A5 + A2 * A2 * q +
                         A5 * A5 * q - q * q * q,
                     (T)(2.) * p * p * p);

      T phi;
      if (r <= (T)(-1.))
        phi = (T)ONE_THIRD_PI;
      else if (r >= (T)(1.))
        phi = (T)(0.);
      else
        phi = fast_acos(r) * (T)(1. / 3.);

      res.d_[0] = q + (T)2. * p * fast_cos(phi);
      res.d_[2] = q + (T)2. * p * fast_cos(phi + (T)TWO_THIRDS_PI);
    }
    res.d_[1] = trace - res.d_[0] - res.d_[2];

    return res;
  }

  __device__ __forceinline__ int get_rank_(T threshold) const {
    const T premultiplied_threshold = max(s_.d_[0] * threshold, static_cast<T>(FLT_MIN));
    for (int i = 0; i < 3; ++i) {
      if (s_.d_[i] < premultiplied_threshold) {
        return i;
      }
    }
    return 3;
  }

  // Matrix a should be symmetric
  __device__ __forceinline__ Vec<T, 3> compute_eigenvector(T eigenvalue, const Mat<T, 3, 3>& a) {
    Vec<T, 3> r0{a.d_[0][0] - eigenvalue, a.d_[0][1], a.d_[0][2]};
    Vec<T, 3> r1{a.d_[0][1], a.d_[1][1] - eigenvalue, a.d_[1][2]};
    Vec<T, 3> r2{a.d_[0][2], a.d_[1][2], a.d_[2][2] - eigenvalue};
    Vec<T, 3> r0r1 = cross(r0, r1);
    Vec<T, 3> r1r2 = cross(r1, r2);
    Vec<T, 3> r0r2 = cross(r0, r2);
    T normsq01 = dot(r0r1, r0r1);
    T normsq12 = dot(r1r2, r1r2);
    T normsq02 = dot(r0r2, r0r2);

    Vec<T, 3> res_v;
    T res_normsq;
    bool b1 = (normsq12 > normsq01);
    res_normsq = b1 ? normsq12 : normsq01;
    res_v.d_[0] = b1 ? r1r2.d_[0] : r0r1.d_[0];
    res_v.d_[1] = b1 ? r1r2.d_[1] : r0r1.d_[1];
    res_v.d_[2] = b1 ? r1r2.d_[2] : r0r1.d_[2];
    if (normsq02 > res_normsq) {
      res_normsq = normsq02;
      res_v.d_[0] = r0r2.d_[0];
      res_v.d_[1] = r0r2.d_[1];
      res_v.d_[2] = r0r2.d_[2];
    }

    T res_norm_inv = fast_rsqrt(res_normsq);
    res_v.d_[0] *= res_norm_inv;
    res_v.d_[1] *= res_norm_inv;
    res_v.d_[2] *= res_norm_inv;

    return res_v;
  }
};
}  // namespace

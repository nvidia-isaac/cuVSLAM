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

#include <cassert>
#include <cfloat>

#include <stdio.h>

#include "cuda_modules/cuda_kernels/cuda_matrix.h"
#include "cuda_modules/cuda_kernels/cuda_sba_v1.h"
#include "epipolar/near_plane.h"

#define USE_XAVIER_OPTIMIZATION true
/*
 * This number was obtained experimentally
 * We use this constant to enable some smart optimization on xavier for matrix multiplication.
 * This constant is related to the amount of allocated shared memory, which is hardware limmited.
 * Please have a look at 2 kernels, that exploit such optimization:
 * 1. calc_point_update_kernel_xavier_special
 * 2. v1T_x_M_T_x_v2_kernel
 * */
#define MAX_OPTIMAL_POSES 22

namespace {

using namespace cuvslam::cuda;

template <typename T>
__device__ T max(T a, T b) {
  return (a < b) ? b : a;
}

template <typename T>
__device__ void swap(T &a, T &b) {
  const T t = a;
  a = b;
  b = t;
}

class JacobiSVD {
public:
  __device__ JacobiSVD(const Matf33 &m, float threshold);
  __device__ int rank() const;
  __device__ const Matf33 &u() const;
  __device__ const Vecf3 &s() const;
  __device__ const Matf33 &v() const;

private:
  Matf33 u_;
  Vecf3 s_;
  Matf33 v_;
  int rank_;
  __device__ static Matf33 get_identity_();
  __device__ void symm_(Matf33 &s);
  __device__ void rotate_(Matf33 &s, int i, int j);
  __device__ void abs_(int i);
  __device__ void order_(int i, int j);
  __device__ int get_rank_(float threshold) const;
};

__device__ JacobiSVD::JacobiSVD(const Matf33 &m, float threshold)
    : u_(get_identity_()), s_(), v_(get_identity_()), rank_() {
  Matf33 s = m;
  symm_(s);
  auto v = [&s](int i, int j) { return std::abs(s.d_[i][j] + s.d_[j][i]) / 2; };
  for (int k = 0; k < 15; k++) {
    const float max_diagonal = max(v(0, 0), max(v(1, 1), v(2, 2)));
    // search for max non diagonal item
    float max_v = v(0, 2);
    int max_i = 0;
    int max_j = 2;
    const float v01 = v(0, 1);
    if (v01 > max_v) {
      max_v = v01;
      max_i = 0;
      max_j = 1;
    }
    const float v12 = v(1, 2);
    if (v12 > max_v) {
      max_v = v12;
      max_i = 1;
      max_j = 2;
    }
    if (max_v < max_diagonal * threshold) {
      break;
    }
    rotate_(s, max_i, max_j);
    symm_(s);
  }

  // make singular values positive
  for (int i = 0; i < 3; ++i) {
    s_.d_[i] = s.d_[i][i];
    abs_(i);
  }

  // sort singular values descending
  order_(0, (s_.d_[1] < s_.d_[2]) ? 2 : 1);
  order_(1, 2);

  rank_ = get_rank_(threshold);
}

__device__ int JacobiSVD::rank() const { return rank_; }
__device__ const Vecf3 &JacobiSVD::s() const { return s_; }
__device__ const Matf33 &JacobiSVD::u() const { return u_; }
__device__ const Matf33 &JacobiSVD::v() const { return v_; }

__device__ Matf33 JacobiSVD::get_identity_() {
  Matf33 m = {0};
  m.d_[0][0] = 1;
  m.d_[1][1] = 1;
  m.d_[2][2] = 1;
  return m;
}

__device__ void JacobiSVD::symm_(Matf33 &s) {
  auto symm = [&s](int i, int j) { s.d_[i][j] = s.d_[j][i] = (s.d_[i][j] + s.d_[j][i]) / 2; };
  symm(0, 1);
  symm(0, 2);
  symm(1, 2);
}

__device__ void JacobiSVD::rotate_(Matf33 &s, int i, int j) {
  const float a = s.d_[i][j];  // (s.d_[i][j] + s.d_[j][i]) / 2;
  const float b = (s.d_[j][j] - s.d_[i][i]) / 2;
  const float d = hypot(a, b);
  const float cos2 = b / d;
  const float cos = sqrt((1 + cos2) / 2);
  const float sin = copysign(sqrt((1 - cos2) / 2), a);

  // we apply here givens rotations to U, V, S matrices. We adjust only involved rows/cols.

  {
    float i_col[3] = {u_.d_[0][i], u_.d_[1][i], u_.d_[2][i]};
    float j_col[3] = {u_.d_[0][j], u_.d_[1][j], u_.d_[2][j]};

    float i_row[3] = {v_.d_[i][0], v_.d_[i][1], v_.d_[i][2]};
    float j_row[3] = {v_.d_[j][0], v_.d_[j][1], v_.d_[j][2]};
    for (int k = 0; k < 3; k++) {
      u_.d_[k][i] = cos * i_col[k] - sin * j_col[k];
      u_.d_[k][j] = cos * j_col[k] + sin * i_col[k];

      v_.d_[i][k] = cos * i_row[k] - sin * j_row[k];
      v_.d_[j][k] = cos * j_row[k] + sin * i_row[k];
    }
  }

  {
    float i_row[3] = {s.d_[i][0], s.d_[i][1], s.d_[i][2]};
    float j_row[3] = {s.d_[j][0], s.d_[j][1], s.d_[j][2]};
    for (int k = 0; k < 3; k++) {
      s.d_[i][k] = cos * i_row[k] - sin * j_row[k];
      s.d_[j][k] = cos * j_row[k] + sin * i_row[k];
    }
  }

  {
    float i_col[3] = {s.d_[0][i], s.d_[1][i], s.d_[2][i]};
    float j_col[3] = {s.d_[0][j], s.d_[1][j], s.d_[2][j]};
    for (int k = 0; k < 3; k++) {
      s.d_[k][i] = cos * i_col[k] - sin * j_col[k];
      s.d_[k][j] = cos * j_col[k] + sin * i_col[k];
    }
  }
}

__device__ void JacobiSVD::abs_(int i) {
  if (s_.d_[i] < 0) {
    s_.d_[i] = -s_.d_[i];
    u_.d_[0][i] = -u_.d_[0][i];
    u_.d_[1][i] = -u_.d_[1][i];
    u_.d_[2][i] = -u_.d_[2][i];
  }
}

__device__ void JacobiSVD::order_(int i, int j) {
  if (s_.d_[i] < s_.d_[j]) {
    swap(s_.d_[i], s_.d_[j]);

    swap(u_.d_[0][i], u_.d_[0][j]);
    swap(u_.d_[1][i], u_.d_[1][j]);
    swap(u_.d_[2][i], u_.d_[2][j]);

    swap(v_.d_[i][0], v_.d_[j][0]);
    swap(v_.d_[i][1], v_.d_[j][1]);
    swap(v_.d_[i][2], v_.d_[j][2]);
  }
}

__device__ int JacobiSVD::get_rank_(float threshold) const {
  const float premultiplied_threshold = max(s_.d_[0] * threshold, FLT_MIN);
  for (int i = 0; i < 3; ++i) {
    if (s_.d_[i] < premultiplied_threshold) {
      return i;
    }
  }
  return 3;
}

}  // namespace

namespace cuvslam::cuda::sba {

__device__ __forceinline__ void Adjoint(const Pose &pose, Mat<float, 6, 6> &adj) {
  float3 translation = {pose.d_[0][3], pose.d_[1][3], pose.d_[2][3]};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      adj.d_[i + 3][j + 3] = adj.d_[i][j] = pose.d_[i][j];  // rotation part
      adj.d_[i][j + 3] = 0;
    }
  }

  for (int i = 0; i < 3; i++) {
    adj.d_[3][i] = pose.d_[2][i] * translation.y - pose.d_[1][i] * translation.z;
    adj.d_[4][i] = pose.d_[0][i] * translation.z - pose.d_[2][i] * translation.x;
    adj.d_[5][i] = pose.d_[1][i] * translation.x - pose.d_[0][i] * translation.y;
  }
}

template <typename Scalar, int InRows, int InCols, int OutCols>
__device__ __forceinline__ void MatMul(const Mat<Scalar, InRows, InCols> &A, const Mat<Scalar, InCols, OutCols> &B,
                                       Mat<Scalar, InRows, OutCols> &result) {
  float temp;

  for (int i = 0; i < InRows; i++) {
    for (int k = 0; k < OutCols; k++) {
      temp = 0;
      for (int j = 0; j < InCols; j++) {
        temp += A.d_[i][j] * B.d_[j][k];
      }
      result.d_[i][k] = temp;
    }
  }
}

template <typename Scalar, int InRows, int InCols, int OutCols>
__device__ __forceinline__ void ATxB(const Mat<Scalar, InRows, InCols> &A, const Mat<Scalar, InRows, OutCols> &B,
                                     Mat<Scalar, InCols, OutCols> &result) {
  float temp;

  for (int i = 0; i < InCols; i++) {
    for (int k = 0; k < OutCols; k++) {
      temp = 0;
      for (int j = 0; j < InRows; j++) {
        temp += A.d_[j][i] * B.d_[j][k];
      }
      result.d_[i][k] = temp;
    }
  }
}

template <typename Scalar, int InRows, int InCols, int BCols, int outCols>
__device__ __forceinline__ void ATxBxC(const Mat<Scalar, InRows, InCols> &A, const Mat<Scalar, InRows, BCols> &B,
                                       const Mat<Scalar, BCols, outCols> &C, Mat<Scalar, InCols, outCols> &result) {
  Mat<Scalar, InCols, BCols> temp_mat;
  float temp;

  ATxB(A, B, temp_mat);

  for (int i = 0; i < InCols; i++) {
    for (int k = 0; k < outCols; k++) {
      temp = 0;
      for (int j = 0; j < BCols; j++) {
        temp += temp_mat.d_[i][j] * C.d_[j][k];
      }
      result.d_[i][k] = temp;
    }
  }
}

template <typename Scalar, int InCols>
__device__ __forceinline__ void ATxBxfloat2(const Mat<Scalar, 2, InCols> &A, const Mat<Scalar, 2, 2> &B,
                                            const float2 &vec, float *result) {
  Mat<Scalar, InCols, 2> temp_mat;
  ATxB(A, B, temp_mat);
  for (int i = 0; i < InCols; i++) {
    result[i] = temp_mat.d_[i][0] * vec.x + temp_mat.d_[i][1] * vec.y;
  }
}

__device__ __forceinline__ void Transform(const Pose &A, const float3 &B, float3 &result) {
  result = {A.d_[0][0] * B.x + A.d_[0][1] * B.y + A.d_[0][2] * B.z + A.d_[0][3],
            A.d_[1][0] * B.x + A.d_[1][1] * B.y + A.d_[1][2] * B.z + A.d_[1][3],
            A.d_[2][0] * B.x + A.d_[2][1] * B.y + A.d_[2][2] * B.z + A.d_[2][3]};
}

template <typename Scalar, int NROWS_, int NCOLS_>
__device__ __forceinline__ void setZero(Mat<Scalar, NROWS_, NCOLS_> &mat) {
  for (int i = 0; i < NROWS_; i++) {
    for (int j = 0; j < NCOLS_; j++) {
      mat.d_[i][j] = 0;
    }
  }
}

__device__ __forceinline__ float ComputeStudentLoss(float x_squared, const float delta, const float nu = 1.f) {
  return (delta + nu) * log1pf(x_squared / delta);
}

__device__ __forceinline__ float ComputeDStudentLoss(float x_squared, const float delta, const float nu = 1.f) {
  return (delta + nu) / (delta + x_squared);
}

__device__ __forceinline__ void Exp(Pose &result, const float6 &twist) {
  float3 w = {twist.x1, twist.x2, twist.x3};
  float w_norm = sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
  float theta = max(w_norm, kFloatEpsilon);
  float c = cos(theta);
  float s_theta = sin(theta) / theta;
  float3 n = {w.x / theta, w.y / theta, w.z / theta};

  result.d_[0][0] = c;
  result.d_[0][1] = -w.z * s_theta;
  result.d_[0][2] = w.y * s_theta;
  result.d_[1][0] = w.z * s_theta;
  result.d_[1][1] = c;
  result.d_[1][2] = -w.x * s_theta;
  result.d_[2][0] = -w.y * s_theta;
  result.d_[2][1] = w.x * s_theta;
  result.d_[2][2] = c;

  result.d_[0][0] += (1 - c) * n.x * n.x;
  result.d_[0][1] += (1 - c) * n.y * n.x;
  result.d_[0][2] += (1 - c) * n.z * n.x;

  result.d_[1][0] += (1 - c) * n.x * n.y;
  result.d_[1][1] += (1 - c) * n.y * n.y;
  result.d_[1][2] += (1 - c) * n.z * n.y;

  result.d_[2][0] += (1 - c) * n.x * n.z;
  result.d_[2][1] += (1 - c) * n.y * n.z;
  result.d_[2][2] += (1 - c) * n.z * n.z;

  float3 v = {twist.x4, twist.x5, twist.x6};
  float3 nxv = {v.z * n.y - v.y * n.z, v.x * n.z - v.z * n.x, v.y * n.x - v.x * n.y};
  float3 nxnxv = {nxv.z * n.y - nxv.y * n.z, nxv.x * n.z - nxv.z * n.x, nxv.y * n.x - nxv.x * n.y};

  // result.translation() =  v + ((1 - c) / theta * nxv + (1 - s_theta) * n.cross(nxv));
  result.d_[0][3] = v.x + ((1 - c) / theta * nxv.x + (1 - s_theta) * nxnxv.x);
  result.d_[1][3] = v.y + ((1 - c) / theta * nxv.y + (1 - s_theta) * nxnxv.y);
  result.d_[2][3] = v.z + ((1 - c) / theta * nxv.z + (1 - s_theta) * nxnxv.z);

  result.d_[3][0] = result.d_[3][1] = result.d_[3][2] = 0;
  result.d_[3][3] = 1;
}

__global__ void calc_jacobians_kernel(GPUModelFunctionMeta function_meta, GPUBundleAdjustmentProblemMeta problem_meta,
                                      float robustifier_scale) {
  extern __shared__ Pose dynamic_sh_mem[];
  Pose *sh_poses = dynamic_sh_mem;
  Pose *cams_from_rig = sh_poses + problem_meta.num_poses;

  for (int i = threadIdx.x; i < problem_meta.num_poses; i += blockDim.x) {
    sh_poses[i] = problem_meta.rig_from_world[i];
  }
  for (size_t i = threadIdx.x; i < problem_meta.rig->num_cameras; i += blockDim.x) {
    cams_from_rig[i] = problem_meta.rig->camera_from_rig[i];
  }

  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= problem_meta.num_observations) {
    return;
  }

  Observation obs = problem_meta.observations[idx];
  Point p_w = problem_meta.points[obs.point_id];
  const Pose &camera_from_rig = cams_from_rig[obs.camera_id];
  const Pose &pose = sh_poses[obs.pose_id];

  Mat<float, 6, 6> adj;
  Adjoint(cams_from_rig[obs.camera_id], adj);

  Pose camera_from_world;
  MatMul(camera_from_rig, pose, camera_from_world);

  // Vec3 p_c = camera_from_world * p_w;
  float3 p_c;
  Transform(camera_from_world, p_w.coords, p_c);

  if (cuvslam::epipolar::IsDepthInFront(p_c.z)) {
    float2 prediction = {p_c.x / p_c.z, p_c.y / p_c.z};
    float2 r = {obs.xy.x - prediction.x, obs.xy.y - prediction.y};

    float inv_z = 1.f / p_c.z;

    Matf23 dproj;
    dproj.d_[0][0] = inv_z;
    dproj.d_[0][1] = 0;
    dproj.d_[0][2] = -prediction.x * inv_z;
    dproj.d_[1][0] = 0;
    dproj.d_[1][1] = inv_z;
    dproj.d_[1][2] = -prediction.y * inv_z;

    // camera_from_world.rotation = camera_from_world.matrix().block<3, 3>(0, 0)
    Matf33 cam_rotation;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        cam_rotation.d_[i][j] = camera_from_world.d_[i][j];
      }
    }

    Matf23 dp;
    MatMul(dproj, cam_rotation, dp);

    Mat<float, 2, 6> dc;

    // dproj * Skew(-p_c);
    dc.d_[0][0] = dproj.d_[0][2] * p_c.y - dproj.d_[0][1] * p_c.z;
    dc.d_[1][0] = dproj.d_[1][2] * p_c.y - dproj.d_[1][1] * p_c.z;

    dc.d_[0][1] = dproj.d_[0][0] * p_c.z - dproj.d_[0][2] * p_c.x;
    dc.d_[1][1] = dproj.d_[1][0] * p_c.z - dproj.d_[1][2] * p_c.x;

    dc.d_[0][2] = dproj.d_[0][1] * p_c.x - dproj.d_[0][0] * p_c.y;
    dc.d_[1][2] = dproj.d_[1][1] * p_c.x - dproj.d_[1][0] * p_c.y;

    // dproj;
    dc.d_[0][3] = dproj.d_[0][0];
    dc.d_[0][4] = dproj.d_[0][1];
    dc.d_[0][5] = dproj.d_[0][2];

    dc.d_[1][3] = dproj.d_[1][0];
    dc.d_[1][4] = dproj.d_[1][1];
    dc.d_[1][5] = dproj.d_[1][2];

    function_meta.residuals[idx] = r;
    function_meta.point_jacobians[idx] = dp;
    // we have computed jacobian for the camera update,
    // transform back to rig update
    MatMul(dc, adj, function_meta.pose_jacobians[idx]);
    // as robustifier_weights will be recalculated in CalculateInformationMatrices
    // we just set them to non 0 to indicate that we want them to be calculated

    float r_sq_norm = (obs.info.d_[0][0] * r.x + obs.info.d_[0][1] * r.y) * r.x +
                      (obs.info.d_[1][0] * r.x + obs.info.d_[1][1] * r.y) * r.y;

    function_meta.robustifier_weights[idx] = ComputeDStudentLoss(r_sq_norm, robustifier_scale);
  } else {
    function_meta.residuals[idx] = {0, 0};
    setZero(function_meta.point_jacobians[idx]);
    setZero(function_meta.pose_jacobians[idx]);
    function_meta.robustifier_weights[idx] = 0.f;
  }
}

#define GET_ELEMENT(ptr, pitch, row, col) ((float *)((char *)ptr + (row)*pitch) + (col))

__global__ void clear_full_system_stage_1_kernel(GPULinearSystemMeta system, int num_points, int num_poses) {
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
  if (idx_x < 6 * num_poses && idx_y < 3 * num_points) {
    float *ptr =
        GET_ELEMENT(system.point_pose_block_transposed, system.point_pose_block_transposed_pitch, idx_x, idx_y);
    *ptr = 0;
  }

  if (idx_x < 6 * num_poses && idx_y < 6 * num_poses) {
    float *ptr = GET_ELEMENT(system.pose_block, system.pose_block_pitch, idx_x, idx_y);
    *ptr = 0;
  }
}

__global__ void clear_full_system_stage_2_kernel(GPULinearSystemMeta system, int num_poses) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 6 * num_poses) {
    system.pose_rhs[idx] = 0;
  }
}

__global__ void build_full_system_1_kernel(GPULinearSystemMeta system, GPUModelFunctionMeta function_meta,
                                           GPUBundleAdjustmentProblemMeta problem_meta) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= problem_meta.num_points) {
    return;
  }

  system.point_rhs[3 * idx] = 0;
  system.point_rhs[3 * idx + 1] = 0;
  system.point_rhs[3 * idx + 2] = 0;

  setZero(system.point_block[idx]);

  const Point &point = problem_meta.points[idx];
  for (int k = 0; k < point.num_observations; ++k) {
    const int obs_idx = point.first_observation_id + k;

    const int pose_id = problem_meta.observations[obs_idx].pose_id;
    const Matf22 &info = problem_meta.observations[obs_idx].info;
    const Mat<float, 2, 6> &pose_jacobian = function_meta.pose_jacobians[obs_idx];
    const Mat<float, 2, 3> &point_jacobian = function_meta.point_jacobians[obs_idx];
    const float2 &residual = function_meta.residuals[obs_idx];
    const float weight = function_meta.robustifier_weights[obs_idx];

    Matf33 hpp;
    Matf36 hpc;
    ATxBxC(point_jacobian, info, point_jacobian, hpp);
    ATxBxC(point_jacobian, info, pose_jacobian, hpc);

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 6; j++) {
        float *ptr = GET_ELEMENT(system.point_pose_block_transposed, system.point_pose_block_transposed_pitch,
                                 6 * pose_id + j, 3 * idx + i);

        *ptr += hpc.d_[i][j] * weight;
      }
    }

    float temp[3];
    ATxBxfloat2(point_jacobian, info, residual, temp);

    for (int i = 0; i < 3; i++) {
      system.point_rhs[3 * idx + i] += temp[i] * weight;
      for (int j = 0; j < 3; j++) {
        system.point_block[idx].d_[i][j] += hpp.d_[i][j] * weight;
      }
    }
  }
}

// one block per pose
__global__ void build_full_system_2_kernel(GPULinearSystemMeta system, GPUModelFunctionMeta function_meta,
                                           GPUBundleAdjustmentProblemMeta problem_meta) {
  int pose_id = blockIdx.x;

  Mat<float, 2, 6> pose_jacobian_;
  float2 residual_;
  float weight_;
  Matf66 hcc_;
  float rhs_[6];

  Matf66 sum_hcc_;
  float sum_rhs_[6];
  setZero(sum_hcc_);
  for (int i = 0; i < 6; i++) {
    sum_rhs_[i] = 0;
  }

  for (int i = threadIdx.x; i < problem_meta.num_observations; i += blockDim.x) {
    int ob_pose_id = problem_meta.observations[i].pose_id;
    if (ob_pose_id != pose_id) {
      continue;
    }

    pose_jacobian_ = function_meta.pose_jacobians[i];
    residual_ = function_meta.residuals[i];
    weight_ = function_meta.robustifier_weights[i];

    const Matf22 &info = problem_meta.observations[i].info;

    ATxBxC(pose_jacobian_, info, pose_jacobian_, hcc_);
    ATxBxfloat2(pose_jacobian_, info, residual_, rhs_);

    for (int j = 0; j < 6; j++) {
      sum_rhs_[j] += rhs_[j] * weight_;
      for (int k = 0; k < 6; k++) {
        sum_hcc_.d_[j][k] += hcc_.d_[j][k] * weight_;
      }
    }
  }
  __syncthreads();

  // reduce
  static __shared__ Matf66 shared_hcc[8];    // Shared mem for 8 partial sums
  static __shared__ float shared_rhs[8][6];  // Shared mem for 8 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    for (int i = 0; i < 6; i++) {
      sum_rhs_[i] += __shfl_down_sync(0xffffffff, sum_rhs_[i], offset);
      for (int j = 0; j < 6; j++) {
        sum_hcc_.d_[i][j] += __shfl_down_sync(0xffffffff, sum_hcc_.d_[i][j], offset);
      }
    }
  }
  if (lane == 0) {
    shared_hcc[wid] = sum_hcc_;
    for (int i = 0; i < 6; i++) {
      shared_rhs[wid][i] = sum_rhs_[i];
    }
  }
  __syncthreads();
  if (lane < 8) {
    sum_hcc_ = shared_hcc[lane];
    for (int i = 0; i < 6; i++) {
      sum_rhs_[i] = shared_rhs[lane][i];
    }
  } else {
    setZero(sum_hcc_);
    for (int i = 0; i < 6; i++) {
      sum_rhs_[i] = 0;
    }
  }

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      for (int i = 0; i < 6; i++) {
        sum_rhs_[i] += __shfl_down_sync(0xffffffff, sum_rhs_[i], offset);
        for (int j = 0; j < 6; j++) {
          sum_hcc_.d_[i][j] += __shfl_down_sync(0xffffffff, sum_hcc_.d_[i][j], offset);
        }
      }
    }
    if (lane == 0) {
      for (int i = 0; i < 6; i++) {
        atomicAdd(&system.pose_rhs[6 * pose_id + i], sum_rhs_[i]);
        for (int j = 0; j < 6; j++) {
          float *ptr = GET_ELEMENT(system.pose_block, system.pose_block_pitch, 6 * pose_id + i, 6 * pose_id + j);
          atomicAdd(ptr, sum_hcc_.d_[i][j]);
        }
      }
    }
  }
}

__global__ void evaluate_cost_kernel(float *cost, int *num_skipped, GPUBundleAdjustmentProblemMeta problem,
                                     GPUParameterUpdateMeta update, float robustifier_scale) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float value = 0;
  int skipped_value = 0;

  extern __shared__ Pose cam_poses[];
  for (int i = threadIdx.x; i < problem.rig->num_cameras; i += blockDim.x) {
    cam_poses[i] = problem.rig->camera_from_rig[i];
  }
  __syncthreads();

  if (idx < problem.num_observations) {
    Observation obs = problem.observations[idx];
    int point_id = obs.point_id;
    int pose_id = obs.pose_id;
    int camera_id = obs.camera_id;

    float3 u_point = update.point[point_id];
    Pose u_pose = update.pose[pose_id];

    Point p_point = problem.points[point_id];
    const float3 &p_coords = p_point.coords;
    Pose p_pose = problem.rig_from_world[pose_id];

    float3 p_w = {p_coords.x + u_point.x, p_coords.y + u_point.y, p_coords.z + u_point.z};
    float3 p_r;
    {
      float3 temp;
      Transform(p_pose, p_w, temp);
      Transform(u_pose, temp, p_r);
    }

    float3 p_c;
    Transform(cam_poses[camera_id], p_r, p_c);

    if (cuvslam::epipolar::IsDepthInFront(p_c.z)) {
      float2 r = {
          obs.xy.x - p_c.x / p_c.z,
          obs.xy.y - p_c.y / p_c.z,
      };
      const Matf22 &info = obs.info;
      float error =
          (info.d_[0][0] * r.x + info.d_[0][1] * r.y) * r.x + (info.d_[1][0] * r.x + info.d_[1][1] * r.y) * r.y;
      value = ComputeStudentLoss(error, robustifier_scale);
    } else {
      skipped_value = 1;
    }
  }

  static __shared__ float shared_cost[8];   // Shared mem for 8 partial sums
  static __shared__ int shared_skipped[8];  // Shared mem for 8 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
    skipped_value += __shfl_down_sync(0xffffffff, skipped_value, offset);
  }
  if (lane == 0) {
    shared_cost[wid] = value;
    shared_skipped[wid] = skipped_value;
  }
  __syncthreads();

  value = lane < 8 ? shared_cost[lane] : 0;
  skipped_value = lane < 8 ? shared_skipped[lane] : 0;

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      value += __shfl_down_sync(0xffffffff, value, offset);
      skipped_value += __shfl_down_sync(0xffffffff, skipped_value, offset);
    }
    if (lane == 0) {
      atomicAdd(cost, value);
      atomicAdd(num_skipped, skipped_value);
    }
  }
}

#define CALC_POINT_SHARED_WIDTH 8
__global__ void calc_point_update_kernel_xavier_special(float *point, float *camera_backsub_block_, size_t pitch,
                                                        float *dpose, float *point_rhs, int num_points, int num_poses) {
  // blockDim.y = 6 * num_poses
  // blockDim.x = CALC_POINT_SHARED_WIDTH

  // shared_size = CALC_POINT_SHARED_WIDTH * (6 * num_poses)

  extern __shared__ float dynamic_shared[];
  float *shared_dpose = dynamic_shared;
  float *shared_matrix_T = shared_dpose + 6 * num_poses;

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y;  // one Y-axis block with 6 * num_poses dim

  if (col >= 3 * num_points || row >= 6 * num_poses) {
    return;
  }

  if (threadIdx.x == 0) {
    shared_dpose[threadIdx.y] = dpose[threadIdx.y];
  }
  shared_matrix_T[threadIdx.x * (6 * num_poses) + threadIdx.y] = *GET_ELEMENT(camera_backsub_block_, pitch, row, col);
  __syncthreads();

  if (threadIdx.y == 0) {
    float out = point_rhs[col];
    for (int i = 0; i < 6 * num_poses; i++) {
      out -= shared_matrix_T[threadIdx.x * (6 * num_poses) + i] * shared_dpose[i];
    }
    point[col] = out;
  }
}

// blocks == 3 * num_points -> one block per row;
__global__ void calc_point_update_kernel(float *point, float *camera_backsub_block_, size_t pitch, float *dpose,
                                         float *point_rhs, int num_points, int num_poses) {
  int row_id = blockIdx.x;
  if (row_id >= 3 * num_points) {
    return;
  }

  float row_dot = 0;
  for (int i = threadIdx.x; i < 6 * num_poses; i += blockDim.x) {
    float *ptr = GET_ELEMENT(camera_backsub_block_, pitch, i, row_id);
    row_dot += *ptr * dpose[i];
  }

  // reduce
  static __shared__ float shared[8];  // Shared mem for 8 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    row_dot += __shfl_down_sync(0xffffffff, row_dot, offset);
  }
  if (lane == 0) {
    shared[wid] = row_dot;
  }
  __syncthreads();

  row_dot = lane < 8 ? shared[lane] : 0;

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      row_dot += __shfl_down_sync(0xffffffff, row_dot, offset);
    }
    if (lane == 0) {
      point[row_id] = point_rhs[row_id] - row_dot;
    }
  }
}

// one block
__global__ void reduce_abs_max_kernel(float *array, size_t size, float *result) {
  float value = 0;
  for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
    value = max(abs(array[i]), value);
  }

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  float neigh;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    neigh = __shfl_down_sync(0xffffffff, value, offset);
    value = max(neigh, value);
  }

  static __shared__ float shared[8];  // Shared mem for 8 partial maximums
  if (lane == 0) {
    shared[wid] = value;
  }
  __syncthreads();

  value = lane < 8 ? shared[lane] : 0;

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      neigh = __shfl_down_sync(0xffffffff, value, offset);
      value = max(neigh, value);
    }
    if (lane == 0) {
      *result = value;
    }
  }
}

__global__ void calc_pose_update_kernel(Pose *poses, float6 *twists, int num_poses) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_poses) {
    return;
  }
  Exp(poses[idx], twists[idx]);
}

__global__ void update_points_kernel(Point *points, float3 *updates, int num_points) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_points) {
    return;
  }
  float3 p = points[idx].coords;
  float3 u = updates[idx];
  points[idx].coords = {p.x + u.x, p.y + u.y, p.z + u.z};
}

__global__ void update_poses_kernel(Pose *poses, Pose *updates, int num_poses) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_poses) {
    return;
  }
  Pose temp;
  MatMul(updates[idx], poses[idx], temp);
  temp.d_[3][0] = 0;
  temp.d_[3][1] = 0;
  temp.d_[3][2] = 0;
  temp.d_[3][3] = 1;
  poses[idx] = temp;
}

// 1 block per row
__global__ void v1T_x_M_x_v2_kernel(float *v1, float *matrix, size_t pitch, int rows, int cols, float *v2,
                                    float *result, bool use_M_transposed) {
  int row_id = blockIdx.x;
  if (row_id >= rows) {
    return;
  }
  float row_dot = 0;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    if (use_M_transposed) {
      row_dot += *GET_ELEMENT(matrix, pitch, i, row_id) * v2[i];
    } else {
      row_dot += *GET_ELEMENT(matrix, pitch, row_id, i) * v2[i];
    }
  }
  // reduce
  static __shared__ float shared[8];  // Shared mem for 8 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    row_dot += __shfl_down_sync(0xffffffff, row_dot, offset);
  }
  if (lane == 0) {
    shared[wid] = row_dot;
  }
  __syncthreads();

  row_dot = lane < 8 ? shared[lane] : 0;

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      row_dot += __shfl_down_sync(0xffffffff, row_dot, offset);
    }
    if (lane == 0) {
      atomicAdd(result, row_dot * v1[row_id]);
    }
  }
}

__global__ void v1T_x_M_T_x_v2_kernel(float *v1, float *matrix, size_t pitch, int rows /*6 * num_poses*/,
                                      int cols /*3 * num_points */, float *v2, float *result) {
  // blockDim.y = rows
  // blockDim.x = CALC_POINT_SHARED_WIDTH

  // shared_size = CALC_POINT_SHARED_WIDTH * (rows)
  extern __shared__ float shared_v2[];
  float *shared_matrix_T = shared_v2 + rows;

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y;  // one Y-axis block with 6 * num_poses dim

  if (col >= cols || row >= rows) {
    return;
  }

  if (threadIdx.x == 0) {
    shared_v2[threadIdx.y] = v2[threadIdx.y];
  }
  shared_matrix_T[threadIdx.x * rows + threadIdx.y] = *GET_ELEMENT(matrix, pitch, row, col);
  __syncthreads();

  if (threadIdx.y == 0) {
    float out = 0;
    for (int i = 0; i < rows; i++) {
      out += shared_matrix_T[threadIdx.x * rows + i] * shared_v2[i];
    }
    atomicAdd(result, out * v1[col]);
  }
}

__global__ void pose_scaling_term_kernel(float *vector, float *matrix, size_t matrix_pitch, int vector_size,
                                         float *res) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float partial_res = 0;
  if (idx < vector_size) {
    float v_value = vector[idx];
    float m_diag_value = *GET_ELEMENT(matrix, matrix_pitch, idx, idx);
    partial_res = v_value * m_diag_value * v_value;
  }

  // reduce
  static __shared__ float shared[8];  // Shared mem for 8 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    partial_res += __shfl_down_sync(0xffffffff, partial_res, offset);
  }
  if (lane == 0) {
    shared[wid] = partial_res;
  }
  __syncthreads();

  partial_res = lane < 8 ? shared[lane] : 0;

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      partial_res += __shfl_down_sync(0xffffffff, partial_res, offset);
    }
    if (lane == 0) {
      atomicAdd(res, partial_res);
    }
  }
}

__global__ void point_term_kernel(float3 *vector, Matf33 *blocks, int num_blocks, bool is_scaling, float *res) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float partial_res = 0;
  if (idx < num_blocks) {
    float3 v_value = vector[idx];
    Matf33 mat = blocks[idx];
    if (is_scaling) {
      partial_res = v_value.x * mat.d_[0][0] * v_value.x + v_value.y * mat.d_[1][1] * v_value.y +
                    v_value.z * mat.d_[2][2] * v_value.z;
    } else {
      partial_res = v_value.x * (mat.d_[0][0] * v_value.x + mat.d_[0][1] * v_value.y + mat.d_[0][2] * v_value.z) +
                    v_value.y * (mat.d_[1][0] * v_value.x + mat.d_[1][1] * v_value.y + mat.d_[1][2] * v_value.z) +
                    v_value.z * (mat.d_[2][0] * v_value.x + mat.d_[2][1] * v_value.y + mat.d_[2][2] * v_value.z);
    }
  }

  // reduce
  static __shared__ float shared[8];  // Shared mem for 8 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    partial_res += __shfl_down_sync(0xffffffff, partial_res, offset);
  }
  if (lane == 0) {
    shared[wid] = partial_res;
  }
  __syncthreads();

  partial_res = lane < 8 ? shared[lane] : 0;

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      partial_res += __shfl_down_sync(0xffffffff, partial_res, offset);
    }
    if (lane == 0) {
      atomicAdd(res, partial_res);
    }
  }
}

__global__ void make_prediction(float current_cost, float lambda, float *pose_hessian_term_, float *point_hessian_term_,
                                float *pose_scaling_term_, float *point_scaling_term_, float *prediction) {
  if (threadIdx.x > 0) {
    return;
  }

  *prediction =
      (*pose_hessian_term_ + *point_hessian_term_ + (*pose_scaling_term_ + *point_scaling_term_) * 2.f * lambda) /
      current_cost;
}

__global__ void reduced_system_stage_1_kernel(const GPULinearSystemMeta full_system,
                                              const GPULinearSystemMeta reduced_system, int num_points, int num_poses,
                                              float lambda, float threshold) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_points) {
    return;
  }

  const Matf33 &mf = full_system.point_block[i];
  Matf33 md;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      md.d_[i][j] = mf.d_[i][j];
      if (i == j) {
        md.d_[i][j] *= 1.f + lambda;
      }
    }
  }

  Matf33 m_inv = {0};

  JacobiSVD usv(md, threshold);

  if (usv.rank() >= 2) {
    const Vecf3 &s = usv.s();
    const Matf33 &v = usv.v();
    const Matf33 &u = usv.u();

    const Vecf3 diagonal = {1.f / s.d_[0], 1.f / s.d_[1], (usv.rank() == 3) ? 1.f / s.d_[2] : 0};

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        float sum = 0;
        for (int k = 0; k < 3; k++) {
          sum += v.d_[k][i] * diagonal.d_[k] * u.d_[j][k];
        }
        m_inv.d_[i][j] = sum;
      }
    }
  }
  reduced_system.point_block[i] = m_inv;

  // HZ: inv(V*) * e_b_i
  {
    const Vecf3 a = {full_system.point_rhs[3 * i], full_system.point_rhs[3 * i + 1], full_system.point_rhs[3 * i + 2]};

    reduced_system.point_rhs[3 * i] = m_inv.d_[0][0] * a.d_[0] + m_inv.d_[0][1] * a.d_[1] + m_inv.d_[0][2] * a.d_[2];
    reduced_system.point_rhs[3 * i + 1] =
        m_inv.d_[1][0] * a.d_[0] + m_inv.d_[1][1] * a.d_[1] + m_inv.d_[1][2] * a.d_[2];
    reduced_system.point_rhs[3 * i + 2] =
        m_inv.d_[2][0] * a.d_[0] + m_inv.d_[2][1] * a.d_[1] + m_inv.d_[2][2] * a.d_[2];
  }
}

__global__ void reduced_system_stage_12_kernel(const GPULinearSystemMeta full_system,
                                               const GPULinearSystemMeta reduced_system, int num_points,
                                               int num_poses) {
  const int point_id = blockIdx.x;
  if (point_id >= num_points) {
    return;
  }

  extern __shared__ float dyn_sh_mem[];

  Matf33 &shared_m_inv = *(Matf33 *)dyn_sh_mem;
  float *shared_point_subblock = (float *)((char *)dyn_sh_mem + sizeof(Matf33));

  if (threadIdx.x == 0) {
    shared_m_inv = reduced_system.point_block[point_id];
  }

  const int N = 3 * 6 * num_poses;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int row = i / 3;
    int col = i % 3;

    shared_point_subblock[3 * row + col] =
        *GET_ELEMENT(full_system.point_pose_block_transposed, full_system.point_pose_block_transposed_pitch, row,
                     3 * point_id + col);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int row = i / 3;
    int col = i % 3;

    float res = 0;

    for (int j = 0; j < 3; j++) {
      res += shared_point_subblock[3 * row + j] * shared_m_inv.d_[j][col];
    }

    float &a = *GET_ELEMENT(reduced_system.point_pose_block_transposed,
                            reduced_system.point_pose_block_transposed_pitch, row, 3 * point_id + col);
    a = res;
  }
}

// HZ: S_jj = U_j - Y_ij * transp(W)
// reduced_system.pose_block = full_system.pose_block - temp * full_system.point_pose_block;
// grid_size = (6 * num_poses, 6 * num_poses)
__global__ void reduced_system_stage_2_kernel(GPULinearSystemMeta full_system, GPULinearSystemMeta reduced_system,
                                              int num_points, float lambda) {
  int out_row = blockIdx.x;
  int out_col = blockIdx.y;

  float row_dot_col = 0;
  for (int i = threadIdx.x; i < 3 * num_points; i += blockDim.x) {
    float Y_ij = *GET_ELEMENT(reduced_system.point_pose_block_transposed,
                              reduced_system.point_pose_block_transposed_pitch, out_row, i);
    float transp_W = *GET_ELEMENT(full_system.point_pose_block_transposed,
                                  full_system.point_pose_block_transposed_pitch, out_col, i);
    row_dot_col += Y_ij * transp_W;
  }

  // reduce
  static __shared__ float shared[8];  // Shared mem for 8 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    row_dot_col += __shfl_down_sync(0xffffffff, row_dot_col, offset);
  }
  if (lane == 0) {
    shared[wid] = row_dot_col;
  }
  __syncthreads();

  row_dot_col = lane < 8 ? shared[lane] : 0;

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      row_dot_col += __shfl_down_sync(0xffffffff, row_dot_col, offset);
    }
    if (lane == 0) {
      float full_sys_pb = *GET_ELEMENT(full_system.pose_block, full_system.pose_block_pitch, out_row, out_col);
      float out = full_sys_pb - row_dot_col;
      if (out_row == out_col) {  // on diagonal
        // dampening
        // HZ: S_jj* = S_jj + lambda * U_j
        out += lambda * full_sys_pb;
      }
      *GET_ELEMENT(reduced_system.pose_block, reduced_system.pose_block_pitch, out_row, out_col) = out;
    }
  }
}

// HZ: e_j = e_a_j - Y_ij * e_b_i
// reduced_system.pose_rhs = full_system.pose_rhs - temp * full_system.point_rhs;
// grid = (6 * num_poses)
__global__ void reduced_system_stage_3_kernel(GPULinearSystemMeta full_system, GPULinearSystemMeta reduced_system,
                                              int num_points) {
  int out_row = blockIdx.x;

  float row_dot_col = 0;
  for (int i = threadIdx.x; i < 3 * num_points; i += blockDim.x) {
    float Y_ij = *GET_ELEMENT(reduced_system.point_pose_block_transposed,
                              reduced_system.point_pose_block_transposed_pitch, out_row, i);
    row_dot_col += Y_ij * full_system.point_rhs[i];
  }

  // reduce
  static __shared__ float shared[8];  // Shared mem for 8 partial sums

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    row_dot_col += __shfl_down_sync(0xffffffff, row_dot_col, offset);
  }
  if (lane == 0) {
    shared[wid] = row_dot_col;
  }
  __syncthreads();

  row_dot_col = lane < 8 ? shared[lane] : 0;

  if (wid == 0) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      row_dot_col += __shfl_down_sync(0xffffffff, row_dot_col, offset);
    }
    if (lane == 0) {
      reduced_system.pose_rhs[out_row] = full_system.pose_rhs[out_row] - row_dot_col;
    }
  }
}

cudaError_t update_model(const GPUModelFunctionMeta &function_meta, const GPUBundleAdjustmentProblemMeta &problem_meta,
                         float robustifier_scale, cudaStream_t s) {
  size_t num_threads = MAX_THREADS;
  size_t num_blocks_obs = (problem_meta.num_observations + num_threads - 1) / num_threads;

  // TODO: check max num poses allowed by sh mem size on jetson nano
  assert(problem_meta.num_poses < 10);
  size_t shared_size = (problem_meta.num_poses + problem_meta.num_cameras) * sizeof(Pose);

  cudaError_t error = cudaSuccess;

  calc_jacobians_kernel<<<num_blocks_obs, num_threads, shared_size, s>>>(function_meta, problem_meta,
                                                                         robustifier_scale);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  return cudaGetLastError();
}

cudaError_t build_full_system(GPULinearSystemMeta system, GPUModelFunctionMeta function_meta,
                              GPUBundleAdjustmentProblemMeta problem_meta, int num_fixed_points,
                              int num_fixed_key_frames, cudaStream_t s) {
  int num_points = problem_meta.num_points - num_fixed_points;
  int num_poses = problem_meta.num_poses - num_fixed_key_frames;

  dim3 threads_2d(WARP_SIZE, WARP_SIZE);
  dim3 blocks_2d((6 * num_poses + threads_2d.x - 1) / threads_2d.x, (3 * num_points + threads_2d.y - 1) / threads_2d.y);

  const size_t num_blocks_2_stage = (6 * num_poses + MAX_THREADS - 1) / MAX_THREADS;

  const size_t num_blocks_3_stage = (num_points + MAX_THREADS - 1) / MAX_THREADS;

  clear_full_system_stage_1_kernel<<<blocks_2d, threads_2d, 0, s>>>(system, num_points, num_poses);
  clear_full_system_stage_2_kernel<<<num_blocks_2_stage, MAX_THREADS, 0, s>>>(system, num_poses);
  build_full_system_1_kernel<<<num_blocks_3_stage, MAX_THREADS, 0, s>>>(system, function_meta, problem_meta);
  build_full_system_2_kernel<<<num_poses, MAX_THREADS, 0, s>>>(system, function_meta, problem_meta);
  return cudaGetLastError();
}

cudaError_t build_reduced_system(GPULinearSystemMeta full_system, GPULinearSystemMeta reduced_system, int num_points,
                                 int num_poses, float lambda, float threshold, cudaStream_t s) {
  {
    const size_t block_size = WARP_SIZE;
    const size_t nblocks = (num_points + block_size - 1) / block_size;

    reduced_system_stage_1_kernel<<<nblocks, block_size, 0, s>>>(full_system, reduced_system, num_points, num_poses,
                                                                 lambda, threshold);

    size_t shared_size = sizeof(Matf33) + 3 * 6 * num_poses * sizeof(float);
    reduced_system_stage_12_kernel<<<num_points, MAX_THREADS, shared_size, s>>>(full_system, reduced_system, num_points,
                                                                                num_poses);

    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
  }

  {
    dim3 blocks_stage_2(6 * num_poses, 6 * num_poses);
    reduced_system_stage_2_kernel<<<blocks_stage_2, MAX_THREADS, 0, s>>>(full_system, reduced_system, num_points,
                                                                         lambda);

    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
  }

  {
    size_t blocks_stage_3 = 6 * num_poses;
    reduced_system_stage_3_kernel<<<blocks_stage_3, MAX_THREADS, 0, s>>>(full_system, reduced_system, num_points);
  }

  return cudaGetLastError();
}

cudaError_t evaluate_cost(float *cost, int *num_skipped, GPUBundleAdjustmentProblemMeta problem_meta,
                          GPUParameterUpdateMeta update, float robustifier_scale, cudaStream_t s) {
  size_t num_blocks = (problem_meta.num_observations + MAX_THREADS - 1) / MAX_THREADS;

  cudaError_t error = cudaMemsetAsync(cost, 0, sizeof(float), s);
  if (error != cudaSuccess) {
    return error;
  }
  error = cudaMemsetAsync(num_skipped, 0, sizeof(int), s);
  if (error != cudaSuccess) {
    return error;
  }

  size_t shared_size = problem_meta.num_cameras * sizeof(Pose);
  evaluate_cost_kernel<<<num_blocks, MAX_THREADS, shared_size, s>>>(cost, num_skipped, problem_meta, update,
                                                                    robustifier_scale);
  return cudaGetLastError();
}

cudaError_t calc_update(float *point, float *camera_backsub_block_T, size_t pitch, float *point_rhs, Pose *poses,
                        float6 *twists, int num_points, int num_poses, cudaStream_t s) {
  size_t num_blocks_poses = (num_poses + MAX_THREADS - 1) / MAX_THREADS;

  if (USE_XAVIER_OPTIMIZATION && num_poses <= MAX_OPTIMAL_POSES) {
    dim3 threads(CALC_POINT_SHARED_WIDTH, 6 * num_poses);
    dim3 blocks((3 * num_points + CALC_POINT_SHARED_WIDTH - 1) / CALC_POINT_SHARED_WIDTH, 1);
    size_t shared_size = 6 * num_poses * (CALC_POINT_SHARED_WIDTH + 1) * sizeof(float);

    calc_point_update_kernel_xavier_special<<<blocks, threads, shared_size, s>>>(
        point, camera_backsub_block_T, pitch, (float *)twists, point_rhs, num_points, num_poses);
  } else {
    size_t num_blocks_points = 3 * num_points;
    calc_point_update_kernel<<<num_blocks_points, MAX_THREADS, 0, s>>>(
        point, camera_backsub_block_T, pitch, (float *)twists, point_rhs, num_points, num_poses);
  }

  calc_pose_update_kernel<<<num_blocks_poses, MAX_THREADS, 0, s>>>(poses, twists, num_poses);
  return cudaGetLastError();
}

cudaError_t reduce_abs_max(float *array, size_t size, float *result, cudaStream_t s) {
  reduce_abs_max_kernel<<<1, MAX_THREADS, 0, s>>>(array, size, result);
  return cudaGetLastError();
}

cudaError_t update_parameters(Point *points, float3 *point_update, Pose *poses, Pose *pose_update, int num_points,
                              int num_poses, cudaStream_t s) {
  size_t num_blocks_points = (num_points + MAX_THREADS - 1) / MAX_THREADS;
  size_t num_blocks_poses = (num_poses + MAX_THREADS - 1) / MAX_THREADS;

  update_points_kernel<<<num_blocks_points, MAX_THREADS, 0, s>>>(points, point_update, num_points);
  update_poses_kernel<<<num_blocks_poses, MAX_THREADS, 0, s>>>(poses, pose_update, num_poses);
  return cudaGetLastError();
}

cudaError_t v1T_x_M_x_v2(float *v1, float *matrix, size_t pitch, int rows, int cols, float *v2, float *result,
                         bool use_M_transposed, cudaStream_t s) {
  if (!use_M_transposed) {
    v1T_x_M_x_v2_kernel<<<rows, MAX_THREADS, 0, s>>>(v1, matrix, pitch, rows, cols, v2, result, use_M_transposed);
    return cudaGetLastError();
  }

  if (USE_XAVIER_OPTIMIZATION && rows <= 6 * MAX_OPTIMAL_POSES) {
    dim3 threads(CALC_POINT_SHARED_WIDTH, rows);
    dim3 blocks((cols + CALC_POINT_SHARED_WIDTH - 1) / CALC_POINT_SHARED_WIDTH, 1);
    size_t shared_size = rows * (CALC_POINT_SHARED_WIDTH + 1) * sizeof(float);

    v1T_x_M_T_x_v2_kernel<<<blocks, threads, shared_size, s>>>(v1, matrix, pitch, rows, cols, v2, result);
  } else {
    v1T_x_M_x_v2_kernel<<<cols, MAX_THREADS, 0, s>>>(v1, matrix, pitch, cols, rows, v2, result, use_M_transposed);
  }
  return cudaGetLastError();
}

cudaError_t pose_scaling_term(float *vector, float *matrix, size_t matrix_pitch, int vector_size, float *res,
                              cudaStream_t s) {
  size_t num_blocks = (vector_size + MAX_THREADS - 1) / MAX_THREADS;
  pose_scaling_term_kernel<<<num_blocks, MAX_THREADS, 0, s>>>(vector, matrix, matrix_pitch, vector_size, res);
  return cudaGetLastError();
}

cudaError_t point_term(float3 *vector, Matf33 *blocks, int num_blocks, int is_scaling, float *res, cudaStream_t s) {
  size_t cuda_blocks = (num_blocks + MAX_THREADS - 1) / MAX_THREADS;

  point_term_kernel<<<cuda_blocks, MAX_THREADS, 0, s>>>(vector, blocks, num_blocks, is_scaling, res);
  return cudaGetLastError();
}

cudaError_t make_prediction(float current_cost, float lambda, float *pose_hessian_term_, float *point_hessian_term_,
                            float *pose_scaling_term_, float *point_scaling_term_, float *prediction, cudaStream_t s) {
  make_prediction<<<1, 1, 0, s>>>(current_cost, lambda, pose_hessian_term_, point_hessian_term_, pose_scaling_term_,
                                  point_scaling_term_, prediction);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda::sba

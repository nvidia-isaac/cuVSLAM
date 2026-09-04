
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

#include "common/isometry.h"
#include "common/log.h"
#include "common/rotation_utils.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "common/vector_3t.h"
#ifdef USE_CUDA
#include "cuda_modules/cuda_helper.h"
#endif
#include "sof/image_context.h"

#include "cuvslam/cuvslam2.h"

namespace cuvslam {

struct Odometry::State::Context : public sof::ImageContext {};

inline Isometry3T ConvertPoseToIsometry(const Pose& pose) {
  Isometry3T ret = Isometry3T::Identity();
  // Convert quaternion (x, y, z, w) to rotation matrix
  Eigen::Quaternionf quat(pose.rotation[3], pose.rotation[0], pose.rotation[1],
                          pose.rotation[2]);  // w, x, y, z order for Eigen constructor
  ret.linear() = quat.toRotationMatrix();
  ret.translation() = vec<3>(pose.translation);
  return ret;
}

inline Pose ConvertIsometryToPose(const Isometry3T& isometry) {
  Pose pose;
  // Use our local SVD-based rotation extraction instead of the problematic .rotation() method
  Matrix3T rot_matrix = common::CalculateRotationFromSVD(isometry.matrix());
  QuaternionT quat(rot_matrix);
  Vector3T trans = isometry.translation();

  // Convert rotation matrix to quaternion and store in (x, y, z, w) order
  pose.rotation[0] = quat.x();
  pose.rotation[1] = quat.y();
  pose.rotation[2] = quat.z();
  pose.rotation[3] = quat.w();

  // Convert translation vector
  pose.translation[0] = trans(0);  // x
  pose.translation[1] = trans(1);  // y
  pose.translation[2] = trans(2);  // z

  return pose;
}

// TODO(C++23): replace with std::to_underlying
template <typename T, std::enable_if_t<std::is_enum_v<T>, bool> = true>
constexpr std::underlying_type_t<T> ToUnderlying(T val) {
  return static_cast<std::underlying_type_t<T>>(val);
}

inline bool CheckCudaCompatibility([[maybe_unused]] std::string& message) {
#ifdef USE_CUDA
  return cuda::CheckCompatibility(message);
#else
  return true;
#endif
}

inline void WarmUpGpuImpl() {
#ifdef USE_CUDA
  cuda::WarmUpGpu();
#endif
}

}  // namespace cuvslam

// If condition fails, log info (__VA_ARGS__) and return error code.
#define LOG_AND_RETURN_IF_FALSE(condition, code, ...) \
  do {                                                \
    if (!(condition)) {                               \
      TraceError(__VA_ARGS__);                        \
      return (code);                                  \
    }                                                 \
  } while (0)

// Log a failed condition and return error. Sadly, there is no easy way to make __VA_ARGS__ macro work with zero args.
// TODO: get rid of this macro with C++20's __VA_OPT__
#define LOG_AND_RETURN_IF_FALSE0(condition, code)                                                  \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      TraceError(#code "(" _CRT_STRINGIZE(code) ") Condition failed: " _CRT_STRINGIZE(condition)); \
      return (code);                                                                               \
    }                                                                                              \
  } while (0)

#define RETURN_IF_FALSE(condition, code) \
  do {                                   \
    if (!(condition)) {                  \
      return (code);                     \
    }                                    \
  } while (0)

#define THROW_IF(condition, ex_type, what) \
  do {                                     \
    if (condition) {                       \
      throw ex_type{(what)};               \
    }                                      \
  } while (0)

#define THROW_INVALID_ARG_IF(condition, what) THROW_IF((condition), std::invalid_argument, (what))

#define THROW_RUNTIME_ERROR_IF(condition, what) THROW_IF((condition), std::runtime_error, (what))

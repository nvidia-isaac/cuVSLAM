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

#include "Eigen/Geometry"
#include "common/include_gtest.h"
#include "cuvslam/cuvslam2.h"

TEST(QuaternionTest, MemoryLayout) {
  // Create our quaternion in (x,y,z,w) order as we store it in Pose
  cuvslam::Array<4> quat_xyzw = {0.1f, 0.2f, 0.3f, 0.4f};  // Not normalized for clear testing

  // IMPORTANT: Eigen's Quaternion constructors have different parameter orders!
  // 1. The scalar constructor takes (w,x,y,z) order: Quaternion(w,x,y,z)
  // 2. The raw pointer constructor and Eigen::Map interpret memory as (x,y,z,w)

  // 1. Test direct use of the raw pointer constructor
  Eigen::Quaternionf quat_direct(quat_xyzw.data());

  EXPECT_FLOAT_EQ(quat_direct.x(), quat_xyzw[0]);
  EXPECT_FLOAT_EQ(quat_direct.y(), quat_xyzw[1]);
  EXPECT_FLOAT_EQ(quat_direct.z(), quat_xyzw[2]);
  EXPECT_FLOAT_EQ(quat_direct.w(), quat_xyzw[3]);

  // 2. Test explicit constructor approach
  // Eigen's scalar constructor takes (w,x,y,z) order, so we need to rearrange
  Eigen::Quaternionf quat_explicit(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]);

  EXPECT_FLOAT_EQ(quat_explicit.x(), quat_xyzw[0]);
  EXPECT_FLOAT_EQ(quat_explicit.y(), quat_xyzw[1]);
  EXPECT_FLOAT_EQ(quat_explicit.z(), quat_xyzw[2]);
  EXPECT_FLOAT_EQ(quat_explicit.w(), quat_xyzw[3]);

  // 3. Test using Eigen::Map
  Eigen::Map<Eigen::Quaternionf> quat_map(quat_xyzw.data());

  EXPECT_FLOAT_EQ(quat_map.x(), quat_xyzw[0]);
  EXPECT_FLOAT_EQ(quat_map.y(), quat_xyzw[1]);
  EXPECT_FLOAT_EQ(quat_map.z(), quat_xyzw[2]);
  EXPECT_FLOAT_EQ(quat_map.w(), quat_xyzw[3]);
}

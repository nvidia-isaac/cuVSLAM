
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

#include <map>
#include <memory>

#include "common/isometry.h"
#include "common/types.h"

#include "slam/common/slam_common.h"
#include "slam/map/database/slam_database.h"

namespace cuvslam::slam {

class PoseGraphHypothesis {
public:
  PoseGraphHypothesis() = default;

  PoseGraphHypothesis(const PoseGraphHypothesis&) = delete;
  PoseGraphHypothesis& operator=(const PoseGraphHypothesis&) = delete;

  // keyframe poses: pose is world_from_keyframe
  const Isometry3T* GetKeyframePose(KeyFrameId keyframe) const;
  void SetKeyframePose(KeyFrameId keyframe, const Isometry3T& m);

  // make copy
  [[nodiscard]] std::shared_ptr<PoseGraphHypothesis> MakeCopy() const;

  bool PutToDatabase(ISlamDatabase* database) const;
  bool GetFromDatabase(ISlamDatabase* database);

  void CopyTo(PoseGraphHypothesis& pose_graph_hypothesis_dst) const;

  void swap(PoseGraphHypothesis& to_swap) noexcept;

protected:
  std::map<KeyFrameId, Isometry3T> pose_graph_verts_;
  const std::string format_and_version_ = "PoseGraphHypothesis v0.0";
};

}  // namespace cuvslam::slam

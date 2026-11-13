
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

#include "slam/map/pose_graph/pose_graph_hypothesis.h"

#include "slam/common/blob_eigen.h"

namespace cuvslam::slam {

const Isometry3T* PoseGraphHypothesis::GetKeyframePose(KeyFrameId keyframe) const {
  const auto it = pose_graph_verts_.find(keyframe);
  if (it == pose_graph_verts_.end()) {
    return nullptr;
  }
  return &it->second;
}
void PoseGraphHypothesis::SetKeyframePose(KeyFrameId keyframe, const Isometry3T& m) { pose_graph_verts_[keyframe] = m; }
std::shared_ptr<PoseGraphHypothesis> PoseGraphHypothesis::MakeCopy() const {
  auto cpy = std::make_shared<PoseGraphHypothesis>();
  cpy->pose_graph_verts_ = this->pose_graph_verts_;

  return cpy;
}

bool PoseGraphHypothesis::PutToDatabase(ISlamDatabase* database) const {
  uint64_t sz0 = pose_graph_verts_.size();
  uint64_t sz = this->format_and_version_.size();
  size_t reserve = sizeof(sz) + sz + sizeof(sz0) + sz0 * (sizeof(KeyFrameId) + sizeof(Isometry3T));

  return database->SetSingleton(SlamDatabaseSingleton::PoseGraphHypothesis, reserve, [&](BlobWriter& blob_writer) {
    blob_writer.write_str(this->format_and_version_);
    blob_writer.write(sz0);
    for (auto& it : pose_graph_verts_) {
      blob_writer.write(it.first);
      blob_writer.write_eigen(it.second);
    }
    return true;
  });
}

bool PoseGraphHypothesis::GetFromDatabase(ISlamDatabase* database) {
  return database->GetSingleton(SlamDatabaseSingleton::PoseGraphHypothesis, [&](const BlobReader& blob_reader) {
    std::string format_and_version;
    if (!blob_reader.read_str(format_and_version) || format_and_version != this->format_and_version_) {
      return false;
    };

    uint64_t sz;
    if (!blob_reader.read(sz)) {
      return false;
    }

    pose_graph_verts_.clear();
    for (uint64_t i = 0; i < sz; i++) {
      KeyFrameId key;
      Isometry3T value;
      blob_reader.read(key);
      blob_reader.read_eigen(value);
      pose_graph_verts_[key] = value;
    }
    return true;
  });
}

void PoseGraphHypothesis::CopyTo(PoseGraphHypothesis& pose_graph_hypothesis_dst) const {
  pose_graph_hypothesis_dst.pose_graph_verts_ = this->pose_graph_verts_;
}

void PoseGraphHypothesis::swap(PoseGraphHypothesis& to_swap) noexcept {
  pose_graph_verts_.swap(to_swap.pose_graph_verts_);
}

}  // namespace cuvslam::slam


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

#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "map/map.h"
#include "map/service.h"
#include "pnp/multicam_pnp.h"
#include "pnp/visual_icp.h"
#include "profiler/profiler.h"
#include "sba/sba_config.h"

#include "pipelines/sfm_solver_interface.h"
#include "pipelines/triangulator.h"

namespace cuvslam::pipelines {

struct SFMInputs {
  const MulticamObservations& observations;
  const pnp::IcpInfo* depth_info;
};

class SolverSfMRGBD {
public:
  SolverSfMRGBD(map::UnifiedMap& map, const camera::Rig& rig, const sba::Settings& sba_settings);

  // set parameters of the camera rig
  virtual const camera::Rig& getRig() const;

  // @see base class for comment
  bool solveNextFrame(int64_t time_ns, const sof::FrameState& frameState, const SFMInputs& inputs,
                      Isometry3T& world_from_rig, Matrix6T& static_info_exp,
                      const SolverPerFrameSettings& solver_settings, std::vector<Track2D>* tracks2d = nullptr,
                      Tracks3DMap* tracks3d = nullptr);

  void reset();

private:
  camera::Rig rig_;
  Isometry3T prev_rig_from_world_{Isometry3T::Identity()};
  Matrix6T prev_static_info_exp_{Matrix6T::Zero()};

  map::UnifiedMap& map_;

  std::unique_ptr<map::ServiceBase> sba_service_;

  MulticamTriangulator triangulator;

  pnp::VisualICP visual_icp_;
  std::vector<camera::Observation> obs_vector_;
  map::Map<TrackId, Vector3T> recent_landmarks_;

  // Exports observations in left camera along with corresponding 3d points
  // out_tracks2d - output 2d track coordinates in pixels
  // out_tracks3d - in rig space
  void exportTracks(const std::vector<camera::Observation>& observations, std::vector<Track2D>& out_tracks2d,
                    Tracks3DMap& out_tracks3d, const Isometry3T& camera_from_world) const;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0x00FF00;
};

}  // namespace cuvslam::pipelines

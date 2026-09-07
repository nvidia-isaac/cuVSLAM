
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

#include "camera/rig.h"
#include "common/isometry.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "pipelines/inertial_pnp.h"
#include "pipelines/tracker_state_machine.h"
#include "pnp/multicam_pnp.h"
#include "sba/sba_config.h"
#include "sof/sof_multicamera_interface.h"
#ifdef USE_CUDA
#include "pnp/visual_icp.h"
#endif

namespace cuvslam::pipelines {

using MulticamObservations = sof::MulticamObservations;

// Solver-level per-frame settings. Contains the subset of per-frame overrides
// that are consumed inside solveNextFrame. Extend this struct (rather than
// adding more parameters to solveNextFrame) when new solver-level knobs are needed.
struct SolverPerFrameSettings {
  sba::Settings sba;
  StateMachineSettings sm;
  pnp::PNPSettings vo_pnp;
  pnp::PNPSettings inertial_stereo_pnp = pnp::PNPSettings::InertialSettings();
  InertialPnPSettings imu_pnp;
#ifdef USE_CUDA
  pnp::ICPSettings icp;
#endif
};

class ISFMSolver {
public:
  virtual ~ISFMSolver() = default;

  virtual const camera::Rig& getRig() const = 0;

  // IN:  time_ns, frameState,
  //      observations - observations from all cameras
  // OUT: world_from_rig - if return code is true - estimated world_from_rig otherwise - previous pose.
  //                       Start pose is Identity.
  //      static_info_exp - information matrix for static frame in exponential mapping form
  //      tracks2d        - optional output 2d track coordinates in pixels
  //      tracks3d        - in camera space
  // return true if accurate solution was found
  virtual bool solveNextFrame(int64_t time_ns, const sof::FrameState& frameState,
                              const MulticamObservations& observations, Isometry3T& world_from_rig,
                              Matrix6T& static_info_exp, const SolverPerFrameSettings& solver_settings,
                              std::vector<Track2D>* tracks2d = nullptr, Tracks3DMap* tracks3d = nullptr) = 0;

  virtual void reset() = 0;
};

}  // namespace cuvslam::pipelines

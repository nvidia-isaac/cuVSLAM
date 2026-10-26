
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

#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"

namespace cuvslam::slam {

ILoopClosureSolver* CreateLoopClosureSolver(LoopClosureSolverType solver_type, RansacType ransac_type, bool randomized,
                                            const camera::Rig& rig) {
  switch (solver_type) {
    case LoopClosureSolverType::kDummy:
      return CreateLoopClosureSolverDummy();
    case LoopClosureSolverType::kSimple:
      return CreateLoopClosureSolverSimple(rig, ransac_type, randomized, LSIGrid::FetchStrategy::Volume);
    case LoopClosureSolverType::kSimplePoint:
      return CreateLoopClosureSolverSimple(rig, ransac_type, randomized, LSIGrid::FetchStrategy::PointOnly);
    case LoopClosureSolverType::kTwoStepsEasy:
      return CreateLoopClosureSolverTwoStepsEasy(rig, ransac_type, randomized);
    case LoopClosureSolverType::kVpr:
      return CreateLoopClosureSolverVpr(rig, ransac_type, randomized);
    default:
      SlamStderr("Unsupported LoopClosureSolverType in LocalizerAndMapper::SetLoopClosureSolver.\n");
      return nullptr;
  }
}

}  // namespace cuvslam::slam

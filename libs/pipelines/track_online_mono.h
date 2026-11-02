
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

#include <future>
#include <list>

#include "camera/rig.h"
#include "common/error.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "sof/sof.h"

#include "pipelines/internal/mono/sba_mono_wrapper.h"
#include "pipelines/internal/mono/workthread.h"
#include "pipelines/track.h"

namespace cuvslam::pipelines {

class SolverSfMMono {
public:
  SolverSfMMono(const camera::Rig& rig);
  ~SolverSfMMono();

  const camera::Rig& getRig() const;

  // During tracking we keep only small 3d point cloud of tracks visible in last few frames.
  // This attribute returns exactly what tracker has "in mind" for current moment.
  // To keep all 3d tracks from the beginning you need to call this function each frame
  // and merge result. This function return only tracks that was triangulated.
  // predicted_pose can be null
  // out_tracks2d - optional 2d tracks in pixels
  ErrorCode monoSolveNextFrame(const std::vector<camera::Observation>& frame_tracks, const FrameId frameId,
                               const bool isKeyFrame, const cuvslam::Isometry3T* predicted_pose,
                               storage::Isometry3<float>& cameraExtrinsics, std::vector<Track2D>* tracks2d,
                               Tracks3DMap& tracks3d, Matrix6T& covariance);
  bool resectioningStarted() const;

private:
  // constants
  const float nonaccuracy_ = 0.002f;

  // keep setup
  camera::Rig rig_;

  TrackingMap map_;
  TrackingMap sba_map_;

  // temporary state
  bool isHomography_ = false;
  storage::Mat3<float> homography_;

  storage::Isometry3<float> prevCamMat_{storage::Isometry3<float>::Identity()};

  FrameId lastKeyframe_ = 0;

  storage::Isometry3<float> last_kf_predicted_pose_;
  bool last_kf_predicted_pose_valid_{false};

  storage::Mat3<float> essential_;

  // C2inv -> C1inv, = C1inv*C2, i.e. C2 = C1 * relativeTransform_, CV camera = Inverse(cuVSLAM camera) for MONO
  storage::Isometry3<float> relativeTransform_;

  std::map<TrackId, Vector2T> tracksForEpipolar_1_;  // for 1st of 2 frames (for Epipolar: H or F, evaluation)
  std::map<TrackId, Vector2T> tracksForEpipolar_2_;  // for 2nd of 2 frames (for Epipolar: H or F, evaluation)

  bool nodalCamera_ = false;
  bool startResectioning_ = false;
  FrameVector keyframesForSBA_;

  std::future<FrameId> bgFrameProcResult_;
  // After the parallel thread finish map optimization, we need to do quick
  // re-resectioning all frames that have been tracked using non-optimal map.
  std::list<std::vector<camera::Observation>> ahead_sba_observations_;
  WorkThread<FrameId> kfThread_{"cuvslam:sba"};

  /***
   * Implementation
   ***/
  void getTracksForEpipolar(const std::vector<camera::Observation>& tracks1,
                            const std::vector<camera::Observation>& tracks2);

  void mergeDataForEpipolar(Vector2TPairVector& sampleSequence);

  void getLastFewFramesTracks3DPosition(const Isometry3T& camera_from_world, Tracks3DMap& posMap) const;

  bool presetResectioning(const Isometry3T& camMat, const std::vector<camera::Observation>& sofTracks,
                          Vector3TVector& points3D, Vector2TVector& points2D) const;

  bool resectioning(const camera::ICameraModel& intrinsics, Isometry3T& inoutCamMat,
                    const std::vector<camera::Observation>& sofTracks,
                    Eigen::Ref<storage::Mat6<float>> covariance) const;

  bool resectioningScale(const storage::Pose<float>& start_pos, const std::vector<camera::Observation>& observations,
                         Isometry3T* pos) const;

  bool findRelativeTransformation();

  void prepareDataForComputeFundamental(Vector2TVector& points2dImage1, Vector2TVector& points2dImage2);

  bool findHomography();

  // triangulate tracks using last key frame and closest to last key frame, where rays parallelism > threshold
  void triangulateTracks(FrameId frame1, FrameId frame2, const std::vector<camera::Observation>& frame2_tracks,
                         std::map<TrackId, Track>& tracks) const;

  // triangulate() called by triangulateTracks() for every trackId in a loop
  bool triangulate(FrameId frame1, FrameId frame2, const std::vector<camera::Observation>& frame2_tracks,
                   const TrackId trackId, Vector3T& loc3D, TrackState& ts) const;

  void sparseBApair(FrameId key1, FrameId key2);

  float calcFrameResidual(const std::vector<camera::Observation>& tracks2d, const Tracks3DMap& tracks3d,
                          const Isometry3T& camExtrinsics) const;

  void processBGFrameProcResult(const camera::ICameraModel& intrinsics, Isometry3T& final_pos,
                                Eigen::Ref<storage::Mat6<float>> covariance);
  void eradicatePast();

  void purgeNonEpipolarTracks(Vector2TPairVector& sampleSequence, float threshold);

  bool solveFrameRange(FrameId frame1, FrameId frame2, const storage::Pose<float>& cam1, Isometry3T& cam2,
                       const std::vector<camera::Observation>& frame2_tracks, const Isometry3T* prediction_frame1,
                       const Isometry3T* prediction_frame2, const camera::ICameraModel& intrinsics,
                       Eigen::Ref<storage::Mat6<float>> covariance, bool& predicted_scale);

  void frameRangeSBA(const FrameVector& frame_range);
  void sparseBAchain(int chDepth);

  void copyWorldToSBA();
  void updateWorldFromSBA();
};

}  // namespace cuvslam::pipelines

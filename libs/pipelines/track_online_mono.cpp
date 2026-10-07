
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

#include "pipelines/track_online_mono.h"

#include "camera/observation.h"
#include "common/log.h"
#include "common/rotation_utils.h"
#include "epipolar/fundamental_ransac.h"
#include "epipolar/homography.h"
#include "epipolar/homography_ransac.h"
#include "epipolar/resectioning_utils.h"
#include "pnp/mono_pnp.h"

#include "pipelines/internal/mono/sba_mono_wrapper.h"

// define to 0 to use synchronous key frame processing
#ifndef USE_ASYNC_SBA
#define USE_ASYNC_SBA 1
#endif

// RESECTION_ONLY tries to minimize need for Fundamental Matrix or Homography, both wrapped in RANSAC,
// calculations - both points of possible failures and slowdown
// #define RESECTION_ONLY 1

const size_t MAXIMUM_NUM_OF_KEYFRAMES = 10;

namespace {
using namespace cuvslam;

void add_frame_to_map(pipelines::TrackingMap& map, FrameId frame, const std::vector<camera::Observation>& observations,
                      const Isometry3T& world_from_rig) {
  map.observations[0][frame] = observations;
  map.tracking_states[frame].world_from_rig = world_from_rig;

  for (const auto& td : observations) {
    map.tracks[td.id].setLastFrameOfVisibility(frame);
  }
}

}  // namespace

namespace cuvslam::pipelines {

SolverSfMMono::SolverSfMMono(const camera::Rig& rig) : rig_(rig) {
  map_.n_cameras = 1;
  sba_map_.n_cameras = 1;
}

SolverSfMMono::~SolverSfMMono() {
  if (bgFrameProcResult_.valid())  // is it was whenever started?
  {
    bgFrameProcResult_.get();  // block until ready then return the stored result
  }
}

const camera::Rig& SolverSfMMono::getRig() const { return rig_; }

bool SolverSfMMono::resectioningStarted() const { return startResectioning_; }

void SolverSfMMono::getTracksForEpipolar(const std::vector<camera::Observation>& tracks1,
                                         const std::vector<camera::Observation>& tracks2) {
  tracksForEpipolar_1_.clear();
  tracksForEpipolar_2_.clear();

  for (size_t tv_index1 = 0; tv_index1 < tracks1.size(); ++tv_index1) {
    size_t tv_index2;
    const camera::Observation& observation1 = tracks1[tv_index1];
    const TrackId id = observation1.id;

    if (FindObservation(tracks2, id, &tv_index2)) {
      tracksForEpipolar_1_[id] = observation1.xy;
      tracksForEpipolar_2_[id] = tracks2[tv_index2].xy;
    }
  }
}

void SolverSfMMono::mergeDataForEpipolar(Vector2TPairVector& sampleSequence) {
  sampleSequence.resize(tracksForEpipolar_1_.size());

  size_t si = 0;
  Tracks2DCIt p2_iter = tracksForEpipolar_2_.cbegin();

  for (const auto& p1 : tracksForEpipolar_1_) {
    Vector2TPair pr(p1.second, p2_iter->second);
    sampleSequence[si++] = pr;
    p2_iter++;
  }
}

// posMap in camera 0 space, which mono requires to be the rig origin
void SolverSfMMono::getLastFewFramesTracks3DPosition(const Isometry3T& camera_from_world, Tracks3DMap& posMap) const {
  posMap.clear();

  for (std::map<TrackId, Track>::const_iterator i = map_.tracks.begin(); i != map_.tracks.end(); i++) {
    const TrackId& trackId = i->first;
    const Track& track = i->second;

    if (track.hasLocation()) {
      posMap[trackId] = camera_from_world * track.getLocation3D();
    }
  }
}

bool SolverSfMMono::presetResectioning(const Isometry3T& camMat, const std::vector<camera::Observation>& sofTracks,
                                       Vector3TVector& points3D, Vector2TVector& points2D) const {
  Tracks3DMap tempLoc3d;

  for (const auto& track : sofTracks) {
    // map_.tracks are populated in key frames only, so:
    if (map_.tracks.find(track.id) == map_.tracks.end()) {
      continue;
    }

    // if track was triangulated than it has already lifetime long enough
    if (map_.tracks.at(track.id).hasLocation()) {
      tempLoc3d[track.id] = map_.tracks.at(track.id).getLocation3D();
    } else if (nodalCamera_) {
      if (tracksForEpipolar_1_.find(track.id) != tracksForEpipolar_1_.end()) {
        const float atInfinity = -1.0f / epsilon();
        tempLoc3d[track.id] = camMat * (atInfinity * tracksForEpipolar_1_.at(track.id).homogeneous());
      }
    }
  }

  points3D.clear();
  points2D.clear();

  for (const auto& track : sofTracks) {
    if (tempLoc3d.find(track.id) == tempLoc3d.end()) {
      continue;
    }

    points3D.push_back(tempLoc3d.at(track.id));
    points2D.push_back(track.xy);
  }

  assert(points3D.size() == points2D.size());
  return (!points3D.empty());
}

bool SolverSfMMono::resectioning(const camera::ICameraModel& intrinsics, Isometry3T& inoutCamMat,
                                 const std::vector<camera::Observation>& sofTracks,
                                 Eigen::Ref<storage::Mat6<float>> covariance) const {
  Vector3TVector points3D;
  Vector2TVector points2D;
#if 0
    currentFrame_ = frame;
#endif
  covariance.setZero();
  covariance.diagonal().setConstant(std::numeric_limits<float>::infinity());

  if (!presetResectioning(inoutCamMat, sofTracks, points3D, points2D)) {
    return false;
  }

  std::vector<storage::Vec3<float>> points(points3D.begin(), points3D.end());
  std::vector<storage::Vec2<float>> observations(points2D.begin(), points2D.end());
  std::vector<storage::Mat2<float>> infos;  // information matrices

  auto n = static_cast<int32_t>(points3D.size());
  infos.resize(n, camera::ObservationInfoUVToNormUV(intrinsics, camera::GetDefaultObservationInfoUV()));

  pnp::PoseRefinementInput input[2];
  input[1].n = 0;

  input[0].huber_delta = 0.05f;  // this values gives good results on our test data
  input[0].n = n;

  // vectors with unaligned types have data laid out contiguously in memory,
  // so it is safe to obtain pointers this way.
  input[0].infos = infos.front().data();
  input[0].uvs = observations.front().data();
  input[0].xyzs = points.front().data();

  bool status = false;

  Isometry3T camera_from_world = inoutCamMat.inverse();
  if (nodalCamera_) {
    Matrix3T precision;
    status = RefineRotation(camera_from_world, precision, input, {}, Isometry3T::Identity());
    if (status) {
      Matrix3T covariance3 = precision.ldlt().solve(Matrix3T::Identity());
      covariance.block<3, 3>(0, 0) = covariance3;
    }
  } else {
    Matrix6T precision;
    status = pnp::RefinePose(camera_from_world, precision, input, {}, Isometry3T::Identity());

    if (status) {
      Eigen::CompleteOrthogonalDecomposition<Matrix6T> psInv(precision);
      covariance = psInv.pseudoInverse();
    }
  }
  if (status) {
    inoutCamMat = camera_from_world.inverse();
  }

  return status;
}

bool SolverSfMMono::resectioningScale(const storage::Pose<float>& start_pos,
                                      const std::vector<camera::Observation>& observations, Isometry3T* pos) const {
  Vector3TVector points3D;
  Vector2TVector points2D;

  if (!presetResectioning(start_pos, observations, points3D, points2D)) {
    return false;
  }

  const Vector3T constraint = pos->translation() - start_pos.translation();

  return epipolar::OptimizeCameraExtrinsicsExpMapConstrained<false>(
      points3D.cbegin(), points3D.cend(), points2D.cbegin(), points2D.cend(), *pos, constraint);
}

bool SolverSfMMono::findRelativeTransformation() {
  Vector2TPairVector sampleSequence;
  mergeDataForEpipolar(sampleSequence);

  float avMinMotion = 0.002f;
  math::Ransac<epipolar::Fundamental> fr;

  // last parameter below is thresholdAuxiliary_ of math::Ransac<Fundamental> telling minimum average
  // motion of tracks population between 2 frames to consider it substantial to calculate Essential reliably
  fr.setOptions(math::Ransac<epipolar::Fundamental>::Epipolar, nonaccuracy_, avMinMotion);
  {
    Matrix3T essential{essential_};  // aligned copy
    size_t numIterations = fr(essential, sampleSequence.begin(), sampleSequence.end());
    essential_ = essential;  // store result to non-aligned memory
    if (numIterations == 0) {
      return false;
    }
  }

  Isometry3T relTransform;
#if 0

    // use LM to improve RANSAC results
    if (FindOptimalCameraMatrixFromEssential(sampleSequence.cbegin(), sampleSequence.cend(), essential_, relTransform))
    {
        relativeTransform_ = relTransform.inverse();

        if (RLM_EssentialEstimate(sampleSequence, relativeTransform_))
        {
            // this is 2D transfer of 2nd frame to 1st and we need other way around => transpose
            essential_ = relativeTransform_.essential().transpose();
        }
    }

#endif
  // Homography purge outliers inside findHomography using its own measure of residual, ||x' - Hx||
  purgeNonEpipolarTracks(sampleSequence, fr.getTheshold());

  Vector2TVector points2dImage1;
  Vector2TVector points2dImage2;
  prepareDataForComputeFundamental(points2dImage1, points2dImage2);
  epipolar::ComputeFundamental fm(points2dImage1, points2dImage2);

  if (fm.isPotentialHomography()) {
    if (findHomography()) {
      // in findHomography we purged outliers
      mergeDataForEpipolar(sampleSequence);

      Matrix3T homography{homography_};  // make aligment copy
      if (!epipolar::GoldStandardHomographyEstimate(sampleSequence, homography)) {
        return false;
      }
      homography_ = homography;  // copy result to non-aligned memory

      relativeTransform_ =
          epipolar::ComputeHomography::DecomposeHomography(sampleSequence.cbegin(), sampleSequence.cend(), homography_);

      const auto& rotation = relativeTransform_.linear();
      if ((homography_ - storage::Mat3<float>(rotation)).norm() < epipolar::ROTATIONAL_THRESHOLD) {
        nodalCamera_ = true;
      }

      relativeTransform_ = relativeTransform_.inverse();
      isHomography_ = true;

      return true;
    }

    return false;
  }

  float accScalar = 2.0f;

  for (int ii = 0; ii < 2; ii++) {
    fr.setOptions(math::Ransac<epipolar::Fundamental>::Epipolar, nonaccuracy_ / accScalar, avMinMotion / accScalar);

    Matrix3T essential{essential_};  // aligned copy
    if (fr(essential, sampleSequence.begin(), sampleSequence.end())) {
      essential_ = essential;  // store result to non-aligned memory
      purgeNonEpipolarTracks(sampleSequence, fr.getTheshold());
    }

    accScalar += 2.0f;
  }

  // at this moment we work on purged of outliers population of tracks
  if (!epipolar::FindOptimalCameraMatrixFromEssential(sampleSequence.cbegin(), sampleSequence.cend(), essential_,
                                                      relTransform)) {
    // unable to find relative transform. Usually due to very small motion causing
    // all points triangulated closer than camera hither.
    return false;
  }

  // to make relativeTransform_ =  C1inverse * C2 - to be ready for use in triangulation: C2inv -> C1inv
  relativeTransform_ = relTransform.inverse();

  Isometry3T relativeTransform{relativeTransform_};  // copy to aliged memory
  if (epipolar::GoldStandardEssentialEstimate(sampleSequence, relativeTransform)) {
    // this is 2D transfer of 2nd frame to 1st and we need other way around => transpose
    const Matrix3T e = essential(relativeTransform);
    const storage::Mat3<float> essential = e.transpose();

    // to pick solution in front of both cameras
    if (epipolar::FindOptimalCameraMatrixFromEssential(sampleSequence.cbegin(), sampleSequence.cend(), essential,
                                                       relTransform)) {
      relativeTransform_ = relTransform.inverse();
      essential_ = essential;
    }
  }

  isHomography_ = false;
  return true;
}

void SolverSfMMono::prepareDataForComputeFundamental(Vector2TVector& points2dImage1, Vector2TVector& points2dImage2) {
  points2dImage1.clear();
  points2dImage1.reserve(tracksForEpipolar_1_.size());

  for (const auto& te : tracksForEpipolar_1_) {
    points2dImage1.push_back(te.second);
  }

  points2dImage2.clear();
  points2dImage2.reserve(tracksForEpipolar_2_.size());

  for (const auto& te : tracksForEpipolar_2_) {
    points2dImage2.push_back(te.second);
  }
}

bool SolverSfMMono::findHomography() {
  Vector2TPairVector sampleSequence;
  mergeDataForEpipolar(sampleSequence);

  // in normalized coordinates, i.e. different than nonaccuracy_ = 0.002f for Fundamental which does not have
  // physical meaning
  float nonaccuracyHomographyThreshold = 0.0005f;
  epipolar::HomographyRansac homographyRansac;
  homographyRansac.setThreshold(nonaccuracyHomographyThreshold);

  Matrix3T homograpy{homography_};  // aligned copy
  size_t numIterations = homographyRansac(homograpy, sampleSequence.begin(), sampleSequence.end());
  homography_ = homograpy;  // cstore result to non-aligned memory

  if (numIterations) {
    std::vector<TrackId> inlierIDs;
    std::set<TrackId> outlierIDs;

    auto tid = tracksForEpipolar_1_.cbegin();

    for (size_t i = 0; i < sampleSequence.size(); i++, tid++) {
      const float residual =
          epipolar::ComputeHomographyResidual(sampleSequence[i].first, sampleSequence[i].second, homography_);

      if (residual < nonaccuracyHomographyThreshold) {
        inlierIDs.push_back(i);
      } else {
        outlierIDs.insert(tid->first);
      }
    }

    Vector2TVector points1, points2;
    points1.reserve(inlierIDs.size());
    points2.reserve(inlierIDs.size());

    for (const auto& inId : inlierIDs) {
      points1.push_back(sampleSequence[inId].first);
      points2.push_back(sampleSequence[inId].second);
    }

    epipolar::ComputeHomography fh(points1, points2);

    // ComputeHomographyResidual has physical sense, while x'Fx does not => recalculate homography_ on
    // cleaned population of tracks (doing the same in general (F) case worsened solution. Switch to Sampson
    // distance, rather than x'Fx, in F case may resolve this non-uniformity (F vs. H) of approach to calculate
    // transformation on filtered of outliers data.
    Matrix3T homography{homography_};  // aligned copy
    const bool is_homography = epipolar::ComputeHomography::ReturnCode::Success == fh.findHomography(homography);
    homography_ = homography;  // store result to non-aligned memory

    if (is_homography) {
      for (const auto& outId : outlierIDs) {
        tracksForEpipolar_1_.erase(outId);
        tracksForEpipolar_2_.erase(outId);
      }

      return true;
    }
  }

  return false;
}

// triangulate tracks using last key frame and closest to last key frame, where rays parallelism > threshold
void SolverSfMMono::triangulateTracks(FrameId frame1, FrameId frame2,
                                      const std::vector<camera::Observation>& frame2_tracks,
                                      std::map<TrackId, Track>& tracks) const {
  assert(tracksForEpipolar_1_.size() == tracksForEpipolar_2_.size());

  Vector3T loc3D;

  for (const auto& track_pair : tracksForEpipolar_1_) {
    const TrackId track_id = track_pair.first;
    Track& track = tracks.at(track_id);
    if (track.hasLocation()) {
      continue;
    }

    TrackState ts = TrackState::kNone;
    loc3D.setZero();  // for debug purposes

    if (triangulate(frame1, frame2, frame2_tracks, track_id, loc3D, ts)) {
      track.setLocation3D(loc3D, ts);
    }
  }
}

// triangulate() called by triangulateTracks() for every trackId in a loop
bool SolverSfMMono::triangulate(FrameId frame1, FrameId frame2, const std::vector<camera::Observation>& frame2_tracks,
                                const TrackId trackId, Vector3T& loc3D, TrackState& ts) const {
  FrameVector::const_reverse_iterator kf = keyframesForSBA_.crbegin();

  // to distinguish triangulation with last frame already appended to keyframesForSBA_ (solveFrameRange() case)
  // (if (true) below) vs. not - immediately after sparseBApair()
  if (!keyframesForSBA_.empty() && *keyframesForSBA_.crbegin() == frame2) {
    kf++;
  }

  size_t track_id_ind_frame1;
  size_t track_id_ind_frame2;
  if (!FindObservation(frame2_tracks, trackId, &track_id_ind_frame2)) {
    return false;
  }
  const Vector2T v2 = frame2_tracks[track_id_ind_frame2].xy;
  if (!FindObservation(map_.observations[0].at(frame1), trackId, &track_id_ind_frame1)) {
    return false;
  }

  size_t numKeyframes = 0;
  Vector3T point3DInReferenceSpace;
  Isometry3T relativeTransform = relativeTransform_;
  const size_t maxIterations = std::min(keyframesForSBA_.size(), (size_t)25);
  float parallelism = -1.0f;
  Vector3T loc3Dopt(Vector3T::Zero());

  do {
    if (!keyframesForSBA_.empty()) {
      if (kf == keyframesForSBA_.crend()) {
        break;
      }

      frame1 = *kf++;

      // checking below only needed if we erase data beyond some frame in eradicatePast()
      if (map_.observations[0].find(frame1) == map_.observations[0].end() ||
          map_.tracking_states.find(frame2) == map_.tracking_states.end()) {
        break;
      }

      // track is not present in the (key)frame that we use for potential triangulation
      if (!FindObservation(map_.observations[0].at(frame1), trackId, &track_id_ind_frame1)) {
        break;
      }
      relativeTransform =
          map_.tracking_states.at(frame1).world_from_rig.inverse() * map_.tracking_states.at(frame2).world_from_rig;
    }

    const Vector2T v1 = map_.observations[0].at(frame1)[track_id_ind_frame1].xy;

    float parallelMeasure = -1.0f;
    epipolar::OptimalPlanarTriangulation opt;
    epipolar::TriangulationState tsTemp = epipolar::TriangulationState::None;

    if ((!isHomography_ &&
         OptimalTriangulation(relativeTransform, v1, v2, point3DInReferenceSpace, parallelMeasure, tsTemp)) ||
        (isHomography_ && opt.triangulate(homography_.inverse(), relativeTransform, v1, v2, point3DInReferenceSpace,
                                          parallelMeasure, tsTemp))) {
      loc3D = map_.tracking_states.at(frame1).world_from_rig * point3DInReferenceSpace;

      if (parallelMeasure > parallelism) {
        if (tsTemp == epipolar::TriangulationState::Triangulated) {
          ts = TrackState::kTriangulated;
        } else {
          ts = TrackState::kNone;
        }
        loc3Dopt = loc3D;
        parallelism = parallelMeasure;
        continue;
      } else {
        loc3D = loc3Dopt;
        assert(loc3Dopt.norm() > epsilon());
        break;
      }
    }
  } while (++numKeyframes < maxIterations);

  if (ts != TrackState::kNone) {
    assert(loc3Dopt.norm() > epsilon());
    loc3D = loc3Dopt;
  }

  return ts != TrackState::kNone;
}

float SolverSfMMono::calcFrameResidual(const std::vector<camera::Observation>& tracks2d, const Tracks3DMap& tracks3d,
                                       const Isometry3T& camExtrinsics) const {
  Isometry3T camGlobal2Local = camExtrinsics.inverse();
  float totalFrameResidual = 0;
  size_t numTriangulatedFrames = 0;

  for (const auto& track2d : tracks2d) {
    auto itv3 = tracks3d.find(track2d.id);

    if (itv3 != tracks3d.end()) {
      Vector2T track2d_reproj;
      epipolar::Project3DPointInLocalCoordinates(camGlobal2Local, itv3->second, track2d_reproj);

      float trackResidual = (track2d.xy - track2d_reproj).stableNorm();

      totalFrameResidual += trackResidual;

      ++numTriangulatedFrames;
    }
  }

  if (numTriangulatedFrames == 0) {
    return 0;
  } else {
    return totalFrameResidual / numTriangulatedFrames;
  }
}

void SolverSfMMono::processBGFrameProcResult(const camera::ICameraModel& intrinsics, Isometry3T& final_pos,
                                             Eigen::Ref<storage::Mat6<float>> covariance) {
  FrameId last_sba_frame;

  try {
    // block until ready then return the stored result
    last_sba_frame = bgFrameProcResult_.get();
  } catch (...) {
    TracePrint("\nSBA thread crashed\n");
    ahead_sba_observations_.clear();
    return;
  }
  updateWorldFromSBA();

  final_pos = map_.tracking_states.at(last_sba_frame).world_from_rig;

  for (const auto& observations : ahead_sba_observations_) {
    if (!resectioning(intrinsics, final_pos, observations, covariance)) {
      TracePrint(
          "Can't do resectioning in frame %zu on "
          "processBGFrameProcResult\n",
          static_cast<size_t>(last_sba_frame));
    }
  }
  ahead_sba_observations_.clear();
}

// get rid of the tracks (and containing them structures) that gone out of visibility before the first
// of the key frames that we consider in the sliding time window of SfM
void SolverSfMMono::eradicatePast() {
  bool eradicateHappened = false;

  while (keyframesForSBA_.size() > MAXIMUM_NUM_OF_KEYFRAMES) {
    FrameId frameToRemove = *keyframesForSBA_.begin();

    map_.tracking_states.erase(frameToRemove);
    map_.observations[0].erase(frameToRemove);
    keyframesForSBA_.erase(keyframesForSBA_.begin());

    eradicateHappened = true;
  }

  if (eradicateHappened) {
    std::map<TrackId, Track>::iterator tit = map_.tracks.begin();
    const FrameId firstFrameOfTracksRequiredVisibility = *keyframesForSBA_.begin();

    while (tit != map_.tracks.end()) {
      assert(keyframesForSBA_.size() == MAXIMUM_NUM_OF_KEYFRAMES);
      assert(keyframesForSBA_[0] < keyframesForSBA_[1]);  // tracking forward (frame1 < frame2) or backward
      if (tit->second.getLastFrameOfVisibility() < firstFrameOfTracksRequiredVisibility) {
        tit = map_.tracks.erase(tit);
      } else {
        ++tit;
      }
    }
  }
}

void SolverSfMMono::purgeNonEpipolarTracks(Vector2TPairVector& sampleSequence, float threshold) {
  assert(tracksForEpipolar_1_.size() == sampleSequence.size());

  std::vector<TrackId> inlierIDs;
  std::set<TrackId> outlierIDs;

  auto tid = tracksForEpipolar_1_.cbegin();

  for (size_t i = 0; i < sampleSequence.size(); i++, tid++) {
    float resid = epipolar::ComputeQuadraticResidual(sampleSequence[i].first, sampleSequence[i].second, essential_);

    if (resid < threshold) {
      inlierIDs.push_back(i);
    } else {
      outlierIDs.insert(tid->first);
    }
  }

  Vector2TPairVector sampleInliers;

  for (const auto& inId : inlierIDs) {
    sampleInliers.push_back(sampleSequence[inId]);
  }

  sampleSequence.swap(sampleInliers);
  assert(tracksForEpipolar_1_.size() == sampleSequence.size() + outlierIDs.size());

  if (!outlierIDs.empty()) {
    for (const auto& outId : outlierIDs) {
      tracksForEpipolar_1_.erase(outId);
      tracksForEpipolar_2_.erase(outId);
    }

    assert(tracksForEpipolar_1_.size() == sampleSequence.size());
    assert(tracksForEpipolar_2_.size() == sampleSequence.size());
  }
}

// prediction_frame* can be null
static bool ExtractScaleFromPrediction(const Isometry3T* prediction_frame1, const Isometry3T* prediction_frame2,
                                       float* length) {
  assert(length != NULL);

  if (prediction_frame1 == NULL || prediction_frame2 == NULL) {
    return false;
  }
  *length = ((*prediction_frame2).translation() - (*prediction_frame1).translation()).norm();

  return true;
}

// prediction_frame* can be null if prediction doesn't exist
bool SolverSfMMono::solveFrameRange(FrameId frame1, FrameId frame2, const storage::Pose<float>& cam1, Isometry3T& cam2,
                                    const std::vector<camera::Observation>& frame2_tracks,
                                    const Isometry3T* predict_frame1, const Isometry3T* predict_frame2,
                                    const camera::ICameraModel& intrinsics, Eigen::Ref<storage::Mat6<float>> covariance,
                                    bool& predicted_scale) {
  nodalCamera_ = false;  // might be reset in isCameraNodal()

#ifdef RESECTION_ONLY
  bool relativeTransformSucceed = (!m_startResectioning)
                                      ? findRelativeTransformation()
                                      : resectioning(intrinsics, camera_[frame2], sofTracks, covariance);

  // we have already 3D points (Landmarks) available (m_startResectioning = true)
  // but resectioning() above failed
  if (m_startResectioning && !relativeTransformSucceed) {
    relativeTransformSucceed = findRelativeTransformation();
    if (relativeTransformSucceed) {
      camera_[frame2] = camera_.at(frame1) * relativeTransform_;
    }
  }
#else
  const bool relativeTransformSucceed = findRelativeTransformation();  // can change nodalCamera_
#endif
  // at this moment we should know
  // 1) if we succeeded to get relativeTransform_
  // 2) if we're nodal or not

  // if we failed in findRelativeTransformation() on first ever pair of key frames we should bail
  if (!relativeTransformSucceed) {
    return false;
  }
  if (nodalCamera_) {
    return true;
  }

  float scale = 1.f;
  predicted_scale = ExtractScaleFromPrediction(predict_frame1, predict_frame2, &scale);
  if (relativeTransformSucceed) {
    relativeTransform_.translation() *= scale;

    cam2 = cam1 * relativeTransform_;

    if (frame1 != 0)  // TODO: track lost recover
    {
#ifdef RESECTION_ONLY
      relativeTransform_ = camera_.at(frame1).inverse() * camera_.at(frame2);
      Isometry3T relativeTransform{m_relativeTransform};  // copy to aligned memory
      essential_ = relativeTransform.essential().transpose();
#else
      if (!predicted_scale) {
        if (resectioningScale(cam1, frame2_tracks, &cam2)) {
          relativeTransform_ = cam1.inverse() * cam2;
        }
      }
#endif
    }
  } else {
    cam2 = cam1;

    // TraceMessage("relativeTransformFailed I: %d, %d", it_start->first, it_end->first);

    Isometry3T cam2_iso = cam2;
    if (resectioning(intrinsics, cam2_iso, frame2_tracks, covariance)) {
      cam2 = cam2_iso;
      relativeTransform_ = cam1.inverse() * cam2_iso;
    } else {
      TraceMessage("relativeTransformFailed II: %zu, %zu", (size_t)frame1, (size_t)frame2);
      return false;
    }
  }

  if (!startResectioning_) {
    cam2 = cam1 * relativeTransform_;
    // TODO: decide sparseBApair(frame1, frame2);
  }

  return true;
}

void SolverSfMMono::frameRangeSBA(const FrameVector& frame_range) {
  // prepare map
  TrackingMap map;

  map.n_cameras = 1;
  map.observations[0].clear();
  map.tracking_states.clear();

  for (FrameId frame_id : frame_range) {
    map.observations[0][frame_id] = map_.observations[0].at(frame_id);
    map.tracking_states[frame_id] = map_.tracking_states.at(frame_id);
  }
  map.tracks = map_.tracks;

  // do sba
  assert(map.tracking_states.size() == map.observations[0].size());
  sparseBA(map, 20, false);

  // copy result back
  for (FrameId frame_id : frame_range) {
    map_.tracking_states.at(frame_id) = map.tracking_states.at(frame_id);
  }
  // copy triangulated back
  map_.tracks = map.tracks;
}

void SolverSfMMono::sparseBApair(FrameId key1, FrameId key2) {
  FrameVector frame_range;

  frame_range.push_back(key1);
  frame_range.push_back(key2);

  frameRangeSBA(frame_range);
}

void SolverSfMMono::sparseBAchain(int chDepth) {
  FrameVector frange;
  {
    FrameVector::reverse_iterator existing_keys_iter = keyframesForSBA_.rbegin();

    for (int i = 0; i < chDepth; i++) {
      if (existing_keys_iter == keyframesForSBA_.rend()) {
        break;
      }

      frange.push_back(*existing_keys_iter++);
    }

    std::reverse(frange.begin(), frange.end());
  }

  frameRangeSBA(frange);
}

void SolverSfMMono::copyWorldToSBA() {
  sba_map_.observations[0].clear();
  sba_map_.tracking_states.clear();
  for (FrameId frame_id : keyframesForSBA_) {
    sba_map_.observations[0][frame_id] = map_.observations[0].at(frame_id);
    sba_map_.tracking_states[frame_id] = map_.tracking_states.at(frame_id);
  }
  sba_map_.tracks = map_.tracks;
}

void SolverSfMMono::updateWorldFromSBA() {
  map_.tracks = sba_map_.tracks;
  for (const auto& t : sba_map_.tracking_states) {
    map_.tracking_states[t.first] = t.second;
  }
}

// out_tracks2d - optional 2d tracks in pixels
ErrorCode SolverSfMMono::monoSolveNextFrame(const std::vector<camera::Observation>& frame_tracks,
                                            const FrameId curFrame, const bool isKeyFrame,
                                            const Isometry3T* predicted_pose,
                                            storage::Isometry3<float>& out_cameraExtrinsics,
                                            std::vector<Track2D>* out_tracks2d,
                                            Tracks3DMap& out_tracks3d,  // in camera 0 space, which mono
                                                                        // requires to be the rig origin
                                            Matrix6T& covariance) {
  out_cameraExtrinsics = Isometry3T();
  if (out_tracks2d != nullptr) {
    out_tracks2d->clear();
  }
  out_tracks3d.clear();

  assert(rig_.num_cameras > 0);
  const camera::ICameraModel& intrinsics = *(rig_.intrinsics[0]);
  // Background process has ended or we must wait for it to end?
  if (bgFrameProcResult_.valid()
#if USE_ASYNC_SBA
      && (bgFrameProcResult_.wait_for(std::chrono::seconds(0)) == std::future_status::ready || isKeyFrame)
#endif
  ) {
    Isometry3T last_cam;
    processBGFrameProcResult(intrinsics, last_cam, covariance);
    prevCamMat_ = last_cam;
  }

  Isometry3T curCamMat = prevCamMat_;
  if (startResectioning_) {
    getTracksForEpipolar(map_.observations[0].at(lastKeyframe_), frame_tracks);

    if (!resectioning(intrinsics, curCamMat, frame_tracks, covariance)) {
      startResectioning_ = false;
      TracePrint("Can't do first resectioning in frame %zu\n", static_cast<size_t>(curFrame));
    }
  }
  if (isKeyFrame) {
    // We need two keyframes for SBA, the last one and the current one,
    // and they must (they should actually) be different.
    if (curFrame == 0) {
      lastKeyframe_ = curFrame;
      add_frame_to_map(map_, curFrame, frame_tracks, curCamMat);
      keyframesForSBA_.push_back(curFrame);
    } else {
      getTracksForEpipolar(map_.observations[0].at(lastKeyframe_), frame_tracks);

      const Isometry3T last_kf_predicted_pose = last_kf_predicted_pose_;
      const Isometry3T* p_last_kf_predicted_pose = last_kf_predicted_pose_valid_ ? &last_kf_predicted_pose : NULL;

      const Isometry3T& last_kf_pose = map_.tracking_states.at(lastKeyframe_).world_from_rig;

      bool predicted_scale;
      if (solveFrameRange(lastKeyframe_, curFrame, last_kf_pose, curCamMat, frame_tracks, p_last_kf_predicted_pose,
                          predicted_pose, intrinsics, covariance, predicted_scale)) {
        add_frame_to_map(map_, curFrame, frame_tracks, curCamMat);

        if (lastKeyframe_ == 0) {  // TODO: track lost recover
          // first pair of frames we resolve relative transformation
          // ever - need to set 3D points always
          triangulateTracks(lastKeyframe_, curFrame, frame_tracks, map_.tracks);
        }

        Isometry3T& cur_pose = map_.tracking_states.at(curFrame).world_from_rig;

        keyframesForSBA_.push_back(curFrame);
        if (startResectioning_ && !predicted_scale) {
          // SBA on 3 frames with triangulated tracks visible in
          // all 3 frames
          sparseBAchain(4);
        }
        relativeTransform_ = last_kf_pose.inverse() * cur_pose;
        triangulateTracks(lastKeyframe_, curFrame, frame_tracks, map_.tracks);

        if (isHomography_) {
          TracePrint("\nHomography\n");
          sparseBAchain(5);
        }

        // as camera_[key2] was changed in sparse BA(), to be up to date
        relativeTransform_ = last_kf_pose.inverse() * cur_pose;
        curCamMat = cur_pose;
        startResectioning_ = true;
        lastKeyframe_ = curFrame;
        eradicatePast();

        if (!nodalCamera_) {
#if USE_ASYNC_SBA
          std::launch policy = std::launch::async;
#else
          std::launch policy = std::launch::deferred;
#endif
          assert(!bgFrameProcResult_.valid());
          copyWorldToSBA();

          auto bgProcCB = [curFrame, this]() -> FrameId {
            sparseBA(sba_map_, 5, true);
            sparseBA(sba_map_, 5, false);

            return curFrame;
          };
          assert(!bgFrameProcResult_.valid());
          bgFrameProcResult_ = kfThread_.addTask(policy, bgProcCB);
          // bgFrameProcResult_ = std::async(std::launch::async, bgProcCB);
          assert(bgFrameProcResult_.valid());
        }
      }
      if (predicted_pose != NULL) {
        last_kf_predicted_pose_ = *predicted_pose;
        last_kf_predicted_pose_valid_ = true;
      } else {
        last_kf_predicted_pose_valid_ = false;
      }
    }
#if !USE_ASYNC_SBA
    if (m_bgFrameProcResult.valid()) {
      FrameId bgResult = bgFrameProcResult_.get();
      updateWorldFromSBA();
      curCamMat = curCamMat;
    }
#endif
  }
  // non-key frame
  else {
    if (bgFrameProcResult_.valid()) {
      ahead_sba_observations_.emplace_back(frame_tracks);
    }
  }

  {
    const Isometry3T camera_from_world = curCamMat.inverse();
    getLastFewFramesTracks3DPosition(camera_from_world, out_tracks3d);
  }
  prevCamMat_ = curCamMat;

  // copy results
  if (out_tracks2d != nullptr) {
    const camera::ICameraModel& camera = *(rig_.intrinsics[0]);
    const size_t num_tracks = frame_tracks.size();
    out_tracks2d->reserve(num_tracks);
    out_tracks2d->clear();
    size_t k = 0;
    for (size_t i = 0; i < num_tracks; ++i) {
      Vector2T uv;
      if (camera.denormalizePoint(frame_tracks[i].xy, uv)) {
        const TrackId id = frame_tracks[i].id;
        out_tracks2d->push_back({0, id, uv});
        ++k;
      }
    }
    out_tracks2d->resize(k);
  }
  out_cameraExtrinsics = curCamMat;

  return ErrorCode::S_True;
}

}  // namespace cuvslam::pipelines

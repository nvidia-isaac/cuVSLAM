
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

#include "sba/mono_sba_solver.h"

#include "common/rotation_utils.h"
#include "epipolar/camera_projection.h"
#include "epipolar/camera_selection.h"
#include "math/robust_cost_function.h"

namespace {

const float LAMBDA_MAX = 1.0E+7f;

}  // namespace

namespace cuvslam::sba {

void MonoSBASolver::addFrame(FrameId id, bool constrained, const Isometry3T& a, const std::map<TrackId, bool>& filtered,
                             const std::vector<camera::Observation>& trackCoords) {
  assert(mapFrames_.find(id) == mapFrames_.end());

  size_t idxFrame = mapFrames_.size();

  frames_.emplace_back();

  mapFrames_.emplace(id, idxFrame);

  Frame& frame = frames_.back();

  frame.id = id;
  frame.constrained = constrained;
  frame.a = a.inverse();  // copy constructor called
  frame.da.setZero();
  frame.errA.setZero();

  if (constrained) {
    frame.aD = frame.a;
  }

  for (const auto& trackCoord : trackCoords) {
    if (filtered.at(trackCoord.id)) {
      continue;
    }
    // First time we're referring to this track?
    auto itTrack = mapTracks_.find(trackCoord.id);

    if (itTrack == mapTracks_.end()) {
      // must create it
      itTrack = mapTracks_.emplace(trackCoord.id, mapTracks_.size()).first;

      tracks_.emplace_back();
      // track will be kernels_initialized properly in addTrack
    }

    size_t idxTrack = itTrack->second;
    assert(idxTrack < tracks_.size());
    assert(idxTrack < mapTracks_.size());

    Track& track = tracks_[idxTrack];
    track.idxFrames.push_back(idxFrame);

    // Valid track ids are positive.
    // This assert should catch cases when we by accident try
    // to use default-kernels_initialized tracks.
    assert(track.id);

    if (idxTrack >= frame.coord.size()) {
      size_t newSize = mapTracks_.size();
      assert(idxTrack < newSize);

      frame.Y.resize(newSize);
      frame.W.resize(newSize);
      frame.WT.resize(newSize);
      frame.coord.resize(newSize);
      frame.err.resize(newSize);
      frame.weights.resize(newSize);
      frame.Adir.resize(newSize);
      frame.AdirT.resize(newSize);
      frame.Bdir.resize(newSize);
      frame.BdirT.resize(newSize);
    }

    frame.coord[idxTrack] = trackCoord.xy;
    frame.err[idxTrack].setZero();
  }

  std::fill(frame.weights.begin(), frame.weights.end(), 1.f);
}

void MonoSBASolver::addTrack(TrackId id, const Vector3T& b) {
  // First time we're referring to this track?
  auto itTrack = mapTracks_.find(id);

  if (itTrack == mapTracks_.end()) {
    // must create it
    itTrack = mapTracks_.emplace(id, mapTracks_.size()).first;

    // for some reason the tracks vector is too small?
    if (itTrack->second >= tracks_.size()) {
      tracks_.resize(itTrack->second + 1);
    }

    Track& track = tracks_[itTrack->second];
    track.idxFrames.clear();
  }

  assert(itTrack->second < mapTracks_.size());

  size_t idxTrack = itTrack->second;
  assert(idxTrack < tracks_.size());

  Track& track = tracks_[idxTrack];

  // Valid track ids are positive.
  // This assert should catch cases when we by accident try
  // to use default-kernels_initialized tracks.
  assert(id);

  track.id = id;
  track.b = b;
  track.db.setZero();
  track.skip = false;
  track.removed = false;
  track.infoMat.setIdentity();

  numTracks_++;
}

size_t MonoSBASolver::numFrames() const { return mapFrames_.size(); }

size_t MonoSBASolver::numTracks() const { return numTracks_; }

bool MonoSBASolver::removeTrack(TrackId id) {
  auto itTrack = mapTracks_.find(id);

  if (itTrack != mapTracks_.end()) {
    Track& track = tracks_[itTrack->second];
    track.skip = true;
    track.removed = true;
  }

  return true;
}

void MonoSBASolver::resetState() {
  for (Frame& frame : frames_) {
    frame.da.setZero();
    frame.errA.setZero();

    if (frame.constrained) {
      frame.aD = frame.a;
    }
  }

  for (Track& track : tracks_) {
    if (track.removed) {
      continue;
    }

    track.skip = false;
    track.db.setZero();
  }

  if (failed_) {
    residual_ = -1;
    failed_ = false;
  }

  lambda_ = 0.001f;
  iteration_ = -1;
}

void MonoSBASolver::reset() {
  frames_.clear();
  tracks_.clear();
  mapFrames_.clear();
  mapTracks_.clear();

  numTracks_ = 0;

  if (failed_) {
    residual_ = -1;
    failed_ = false;
  }

  lambda_ = 0.001f;
  iteration_ = -1;
}

int MonoSBASolver::solve() {
  int count = maxIterations_;

  filterHitherPoints();
  updateAdbD();

  // TODO (msmirnov): adjust the threshold dynamically
  // This value gives good results on our test dataset.
  huberThreshold_ = 0.5f;  // this values gives good results on our test data

  // populates Frame::err
  calculateResidual();

  // computes covariances using Frame::err
  calculateInformationMatrices();

  // computes weights as rho'(err^T infoMat err)
  computeWeights();

  // computes sum of rho(err^T infoMat err)
  residual_ = calculateResidual();

  // condition below can be true only if all points will be 'skipped' as bad ones
  // for SBA: for instance, all of them are closer than hither to camera center
  if (residual_ == std::numeric_limits<float>::max()) {
    covariance_ = Matrix6T::Identity();  // as we can not calculate it anyway
    return false;
  }

  residualInitial_ = residual_;

  bool newValues = true;
  float oldResidual = residual_;

  while (minResidual_ <= residual_ && count >= 0 && lambda_ <= LAMBDA_MAX) {
    if (newValues) {
      calculateDerivatives();
      calculateU();
      calculateV();
      calculateW();
    }

    stepUV();

    if (calculateAdjustments()) {
      updateAdbD();
      residual_ = calculateResidual();

      if (residual_ == std::numeric_limits<float>::max()) {
        return false;
      }
    } else {
      if (residual_ == std::numeric_limits<float>::max()) {
        return false;
      }

      // means, worst than it was SVD does not converge in BA
      residual_ *= 2.0f;
    }

    const float residualDelta = fabs(residual_ - oldResidual);

    if (residual_ > oldResidual) {
      lambda_ *= 10.0f;
      newValues = false;
    } else {
      if (lambda_ > 1e-25f) {
        lambda_ /= 10.0f;
      }

      oldResidual = residual_;
      newValues = true;
      update();
      filterHitherPoints();
      calculateInformationMatrices();

      // We have just accepted the step.
      // This means that we called calculateResidual which populated errors
      // with new values (corresponding to the new estimate).
      computeWeights();

      Eigen::CompleteOrthogonalDecomposition<Matrix6T> psInv(prepCovariance_);
      covariance_ = psInv.pseudoInverse();
    }

    if (minResidualSpeed_ * residual_ > residualDelta) {
      break;  // speed limit
    }

    count--;
  }

  residual_ = oldResidual;
  failed_ = lambda_ > LAMBDA_MAX || count < 0;
  // failed_ = residual_ >= residualInitial_;

  if (failed_) {
    // better something of the proper order than nothing
    Eigen::CompleteOrthogonalDecomposition<Matrix6T> psInv(prepCovariance_);
    covariance_ = psInv.pseudoInverse();
  }

  return iteration_ = maxIterations_ - count;
}

float MonoSBASolver::getResidual() const { return residual_; }

float MonoSBASolver::getResidualInitial() const { return residualInitial_; }

float MonoSBASolver::getLambda() const { return lambda_; }

const std::vector<Track>& MonoSBASolver::tracks() const { return tracks_; }

const std::vector<Frame>& MonoSBASolver::frames() const { return frames_; }

CameraMap MonoSBASolver::getCameras() const {
  CameraMap cameras;

  for (const Frame& frame : frames_) {
    cameras[frame.id] = frame.a.transform().inverse();
  }

  return cameras;
}

Tracks3DMap MonoSBASolver::getTracks3d() const {
  Tracks3DMap tracks3d;

  for (const Track& track : tracks_) {
    if (track.removed) {
      continue;
    }

    tracks3d[track.id] = track.b;
  }

  return tracks3d;
}

bool MonoSBASolver::getFailed() const { return failed_; }

Matrix6T MonoSBASolver::getCovariance() const { return covariance_; }

void MonoSBASolver::filterHitherPoints() {
  for (Track& track : tracks_) {
    if (track.skip) {
      continue;
    }

    for (size_t idxFrame : track.idxFrames) {
      const Vector3T loc3dInCamCoord = frames_[idxFrame].a.transform() * track.b;

      if (!cuvslam::epipolar::IsDepthInFront(loc3dInCamCoord.z())) {
        track.skip = true;
        break;
      }
    }
  }
}

void MonoSBASolver::calculateInformationMatrices() {
  covarMatDiag_.reserve(tracks_.size());
  covarMatDiag_.clear();
  covarMatXY_.reserve(tracks_.size());
  covarMatXY_.clear();
  covarMatX_.reserve(tracks_.size());
  covarMatX_.clear();
  covarMatY_.reserve(tracks_.size());
  covarMatY_.clear();

  size_t t = 0;

  for (const Track& track : tracks_) {
    if (!track.skip) {
      covarMatDiag_.emplace_back();
      covarMatXY_.emplace_back();
      covarMatX_.emplace_back();
      covarMatY_.emplace_back();

      for (size_t idxFrame : track.idxFrames) {
        const Frame& frame = frames_[idxFrame];

        covarMatDiag_.back()(frame.err[t]);
        covarMatXY_.back()(frame.err[t][0] * frame.err[t][1]);
        covarMatX_.back()(frame.err[t][0]);
        covarMatY_.back()(frame.err[t][1]);
      }
    }

    ++t;
  }

  const float thresh = 1e-5f;

  t = 0;
  Matrix2T covarMat;

  for (Track& track : tracks_) {
    if (track.skip) {
      continue;
    }

    const Vector2T variance = covarMatDiag_[t].variance();

    covarMat.setZero();
    covarMat(0, 0) = (variance.x() < thresh) ? thresh : variance.x();
    covarMat(1, 1) = (variance.y() < thresh) ? thresh : variance.y();

    // We need at least three observations to see 2D variance
    if (covarMatDiag_[t].count() > 3) {
      covarMat(0, 1) = covarMat(1, 0) = covarMatXY_[t].mean() - covarMatX_[t].mean() * covarMatY_[t].mean();

      // ensure that covariance is positive definite
      {
        auto ldlt = covarMat.ldlt();
        Vector2T d = ldlt.vectorD();
        d = d.cwiseMax(thresh);
        covarMat = ldlt.matrixL() * (d.asDiagonal() * ldlt.matrixL().transpose().toDenseMatrix());
      }
    }

    float determinant;
    bool invertible;
    covarMat.computeInverseAndDetWithCheck(track.infoMat, determinant, invertible, epsilon());

    if (!invertible) {
      covarMat(0, 1) = covarMat(1, 0) = 0;
      track.infoMat = covarMat.inverse();
    }

    ++t;
  }
}

// Computes robustifier weights based on the current values of residuals
void MonoSBASolver::computeWeights() {
  const int numTracks = static_cast<int>(tracks_.size());
  for (int t = 0; t < numTracks; ++t) {
    for (auto&& f : tracks_[t].idxFrames) {
      auto& frame = frames_[f];
      const auto s = frame.err[t].dot(tracks_[t].infoMat * frame.err[t]);
      frames_[f].weights[t] = math::ComputeDHuberLoss(s, huberThreshold_);
    }
  }
}

void MonoSBASolver::stepUV() {
  for (Frame& frame : frames_) {
    if (frame.constrained) {
      continue;
    }

    frame.Ua = frame.Uo;

    for (size_t i = 0; i < (size_t)frame.Ua.cols(); i++) {
      frame.Ua(i, i) *= 1.f + lambda_;
    }
  }

  size_t N = 0;

  for (Track& track : tracks_) {
    if (track.skip) {
      continue;
    }

    Matrix3T V = track.V;

    for (int i = 0; i < 3; i++) {
      V(i, i) *= 1.f + lambda_;
    }

    Matrix3T VInv;

    if (!invert3x3(V, VInv)) {
      // Non-invertible V always means point with unreasonable coordinates,
      // usually very far 3d location where algorithm will not be
      // stable anyway unless additionally to x,y,z we optimize w;
      // or very close to camera origin
      track.skip = true;
      N++;
    } else {
      track.invVa = VInv;
    }
  }

  // TraceMessageIf(N, "N 3D points to close to HITHER = %d", N);

  int t = 0;

  for (Track& track : tracks_) {
    if (!track.skip) {
      for (size_t idxFrame : track.idxFrames) {
        Frame& frame = frames_[idxFrame];

        if (frame.constrained) {
          continue;
        }

        frame.Y[t] = frame.W[t] * track.invVa;
      }
    }

    ++t;
  }

  // TraceMessageIf(N, "N 3D points to close to HITHER = %d", N);
}

void MonoSBASolver::calculateW() {
  size_t t = 0;

  for (Track& track : tracks_) {
    if (!track.skip) {
      for (size_t idxFrame : track.idxFrames) {
        Frame& frame = frames_[idxFrame];

        if (frame.constrained) {
          continue;
        }

        frame.W[t] = frame.AdirT[t] * (track.infoMat * frame.weights[t]) * frame.Bdir[t];
        frame.WT[t] = frame.W[t].transpose();
      }
    }

    ++t;
  }
}

void MonoSBASolver::calculateU() {
  for (Frame& frame : frames_) {
    if (frame.constrained) {
      continue;
    }

    frame.Uo.setZero();
    frame.errA.setZero();
  }

  size_t t = 0;

  for (const Track& track : tracks_) {
    if (!track.skip) {
      for (size_t idxFrame : track.idxFrames) {
        Frame& frame = frames_[idxFrame];

        if (frame.constrained) {
          continue;
        }

        math::TwistDerivativesT::CameraDerivsTransposed tmp = frame.AdirT[t] * (frame.weights[t] * track.infoMat);

        // HZ: U_ij += transp(A_ij) * inv(E_x_ij) * A_ij
        frame.Uo += tmp * frame.Adir[t];
        frame.errA += tmp * frame.err[t];
      }
    }

    ++t;
  }
}

void MonoSBASolver::calculateV() {
  size_t t = 0;

  for (Track& track : tracks_) {
    if (!track.skip) {
      track.V.setZero();
      track.errB.setZero();

      for (size_t idxFrame : track.idxFrames) {
        const Frame& frame = frames_[idxFrame];

        math::TwistDerivativesT::PointDerivsTransposed tmp = frame.BdirT[t] * (frame.weights[t] * track.infoMat);

        track.V += tmp * frame.Bdir[t];
        track.errB += tmp * frame.err[t];
      }
    }

    ++t;
  }
}

void MonoSBASolver::calculateDerivatives() {
  // for all frames calculate invCameraRot (to cache it)
  std::vector<Isometry3T> inv_cams;
  std::vector<Matrix3T> inv_cam_rots;
  {
    const size_t n_frames = frames_.size();
    inv_cams.resize(n_frames);
    inv_cam_rots.resize(n_frames);

    for (size_t i = 0; i < n_frames; ++i) {
      inv_cams[i] = frames_[i].a.transform();
      inv_cam_rots[i] = common::CalculateRotationFromSVD(inv_cams[i].matrix());
    }
  }
  size_t t = 0;

  for (Track& track : tracks_) {
    if (!track.skip) {
      for (size_t idxFrame : track.idxFrames) {
        Frame& frame = frames_[idxFrame];

        // camera (A) and points (B) derivatives
        math::TwistDerivativesT::PointDerivs bm;

        if (frame.constrained) {
          math::TwistDerivativesT::calcPointDerivs(track.b, inv_cams[idxFrame], inv_cam_rots[idxFrame], bm);
        } else {
          math::TwistDerivativesT::CameraDerivs am;
          math::TwistDerivativesT::calcCamPointDerivs(track.b, inv_cams[idxFrame], inv_cam_rots[idxFrame], am, bm);

          // -am as HZ minimize CV, i.e. inverse to cuVSLAM, camera parameters and in cuVSLAM case projections
          // of points and camera are moving in opposite directions
          frame.Adir[t] = -am;
          frame.AdirT[t] = frame.Adir[t].transpose();
        }

        frame.Bdir[t] = bm;
        frame.BdirT[t] = bm.transpose();
      }
    }

    ++t;
  }
}

float MonoSBASolver::calculateResidual() {
  size_t t = 0;
  int total = 0;
  float residual = 0;

  for (Track& track : tracks_) {
    if (!track.skip) {
      assert(!track.removed);

      for (size_t idxFrame : track.idxFrames) {
        Frame& frame = frames_[idxFrame];

        const Isometry3T& invCamMat = frame.aD.transform();

        Vector2T v3Proj;
        epipolar::Project3DPointInLocalCoordinates(invCamMat, track.bD, v3Proj);

        frame.err[t] = frame.coord[t] - v3Proj;

        const auto s = frame.err[t].dot(track.infoMat * frame.err[t]);
        residual += math::ComputeHuberLoss(s, huberThreshold_);

        total++;
      }
    }

    ++t;
  }

  return total != 0 ? residual / total : std::numeric_limits<float>::max();
}

void MonoSBASolver::updateAdbD() {
  for (Frame& frame : frames_) {
    if (frame.constrained) {
      continue;
    }

    frame.aD = frame.a + frame.da;
  }

  for (Track& track : tracks_) {
    if (track.skip) {
      continue;
    }

    track.bD = track.b + track.db;
  }
}

void MonoSBASolver::update() {
  for (Frame& frame : frames_) {
    if (frame.constrained) {
      continue;
    }

    frame.a = frame.aD;
  }

  for (Track& track : tracks_) {
    if (track.removed) {
      continue;
    }

    if (track.skip) {
      track.skip = false;
    } else {
      track.b = track.bD;
    }
  }
}

int MonoSBASolver::getIterations() const { return iteration_; }

void MonoSBASolver::calculateSMatrix(MatrixXT& S, VectorXT& b) {
  constexpr int INPUT_SIZE = 6;

  const Index ssize = b.size();
  assert(S.rows() == ssize && S.cols() == ssize);

  std::unordered_map<const Frame*, Index> mapFrameIdx;

  Index curFrameIdx = 0;

  for (const Frame& frame : frames_) {
    if (!frame.constrained) {
      b.segment<INPUT_SIZE>(INPUT_SIZE * curFrameIdx) = frame.epsilon;
      mapFrameIdx.emplace(&frame, curFrameIdx++);
    }
  }

  assert(INPUT_SIZE * curFrameIdx == ssize);

  size_t jj, kk;
  jj = kk = 0;

  for (const Frame& frame_j : frames_) {
    if (frame_j.constrained) {
      continue;
    }

    for (const Frame& frame_k : frames_) {
      if (frame_k.constrained) {
        continue;
      }

      if (&frame_j == &frame_k) {
        S.block<INPUT_SIZE, INPUT_SIZE>(jj, kk) = frame_j.Ua;
      } else {
        S.block<INPUT_SIZE, INPUT_SIZE>(jj, kk).setZero();
      }

      kk += INPUT_SIZE;
    }

    kk = 0;
    jj += INPUT_SIZE;
  }

  size_t t = 0;

  for (const Track& track : tracks_) {
    if (!track.skip) {
      for (size_t fj : track.idxFrames) {
        const Frame& frame_j = frames_[fj];

        if (frame_j.constrained) {
          continue;
        }

        auto itIdx_j = mapFrameIdx.find(&frame_j);
        assert(itIdx_j != mapFrameIdx.end());
        Index j = itIdx_j->second * INPUT_SIZE;

        for (size_t fk : track.idxFrames) {
          const Frame& frame_k = frames_[fk];

          if (frame_k.constrained) {
            continue;
          }

          auto itIdx_k = mapFrameIdx.find(&frame_k);
          assert(itIdx_k != mapFrameIdx.end());
          Index k = itIdx_k->second * INPUT_SIZE;

          S.block<INPUT_SIZE, INPUT_SIZE>(j, k) -= frame_j.Y[t] * frame_k.WT[t];
        }
      }
    }

    ++t;
  }

  jj = INPUT_SIZE * (ssize / INPUT_SIZE - 1);
  prepCovariance_ = S.block<INPUT_SIZE, INPUT_SIZE>(jj, jj);

  for (jj = 0; jj < INPUT_SIZE; jj++) {
    prepCovariance_(jj, jj) /= 1.0f + lambda_;
  }
}

void MonoSBASolver::calculateEps() {
  for (Frame& frame : frames_) {
    if (frame.constrained) {
      continue;
    }

    frame.epsilon = frame.errA;
  }

  size_t t = 0;

  for (const Track& track : tracks_) {
    if (!track.skip) {
      for (size_t idxFrame : track.idxFrames) {
        Frame& frame = frames_[idxFrame];

        if (frame.constrained) {
          continue;
        }

        frame.epsilon -= frame.Y[t] * track.errB;
      }
    }

    ++t;
  }
}

bool MonoSBASolver::calculateAdjustments() {
  calculateEps();

  size_t nonConstrainedFrameCount =
      count_if(frames_.begin(), frames_.end(), [](const Frame& frame) { return !frame.constrained; });

  const size_t nparam = 6;
  const size_t Ssize = nparam * nonConstrainedFrameCount;

  if (Ssize == 0) {
    for (Track& track : tracks_) {
      if (track.removed) {
        continue;
      }

      track.db.setZero();
    }

    return false;
  }

  // svdData * x = b
  MatrixXT svdData(Ssize, Ssize);
  VectorXT b(Ssize);
  calculateSMatrix(svdData, b);

  IterativeSolver<Eigen::JacobiSVD<MatrixXT>> solver(svdData, Eigen::ComputeThinU | Eigen::ComputeThinV);

  if ((size_t)solver.rank() < Ssize) {
    return false;
  }

  const VectorXT x = solver.solve(b);

  int j = 0;

  for (Frame& frame : frames_) {
    if (frame.constrained) {
      frame.da.setZero();
      continue;
    }

    for (size_t i = 0; i < 6; i++) {
      frame.da[i] = x[j++];
    }
  }

  size_t t = 0;

  for (Track& track : tracks_) {
    if (!track.skip) {
      Vector3T wda = Vector3T::Zero();

      for (size_t idxFrame : track.idxFrames) {
        const Frame& frame = frames_[idxFrame];

        if (frame.constrained) {
          continue;
        }

        wda += frame.WT[t] * frame.da;
      }

      Vector3T pt = track.errB - wda;
      track.db = track.invVa * pt;
    }

    ++t;
  }

  return true;
}

bool MonoSBASolver::invert3x3(const Matrix3T& i, Matrix3T& o) const {
  const float DET_THRESHOLD = 1e6f;
  float det = 0;
  bool invertable = false;
  const float threshDet = 1 / std::max(DET_THRESHOLD, lambda_ * lambda_ * lambda_);
  i.computeInverseAndDetWithCheck(o, det, invertable, threshDet);
  return invertable;
}

}  // namespace cuvslam::sba

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

#include "slam/localizer/localizer.h"

#include "slam/vpr/vpr_sof_image.h"

#include "common/log_types.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "profiler/profiler.h"

#include "slam/map/database/lmdb_slam_database.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"
#include "slam/view/map_to_view.h"

namespace cuvslam::slam {

void Localizer::LoopClosureStatus::SetLC(KeyFrameId _from_keyframe_id,
                                         const LocalizerAndMapper::LoopClosureStatus& _slam_loop_closure_status) {
  from_keyframe_id = _from_keyframe_id;
  success = _slam_loop_closure_status.success;
  result_pose = _slam_loop_closure_status.result_pose;
  result_pose_covariance = _slam_loop_closure_status.result_pose_covariance;
  good_landmarks_count = _slam_loop_closure_status.good_landmarks_count;
  tracked_landmarks_count = _slam_loop_closure_status.tracked_landmarks_count;
}

void Localizer::LoopClosureStatus::SetFrame(uint64_t _trial_id, float _dist_from_hint, const Isometry3T& _guess_pose) {
  trial_id = _trial_id;
  dist_from_hint = _dist_from_hint;
  guess_pose = _guess_pose;
}

float Localizer::LoopClosureStatus::Weight() const {
  if (tracked_landmarks_count == 0 || !success) {
    return 0;
  }
  float w = static_cast<float>(good_landmarks_count) / static_cast<float>(tracked_landmarks_count);
  float f = 2.f * static_cast<float>(atan(static_cast<float>(good_landmarks_count)) / 500.f) / cuvslam::PI;
  w *= f;
  return w;
}

void Localizer::Init(const camera::Rig& rig, const LocalizerOptions& options) {
  options_ = options;
  PrepareProbes();

  if (options_.horizontal_search_radius <= 0 || options_.vertical_search_radius <= 0) {
    TraceWarning("Localization search radius must be greater than zero.");
  }

  if (options_.horizontal_step <= 0 || options_.vertical_step <= 0 || options_.angle_step_rads <= 0) {
    TraceWarning("Localization step must be greater than zero.");
  }

  if (options_.angle_step_rads >= 2 * PI) {
    TraceWarning("Angular step must be less than 2 Pi");
  }

  rig_ = rig;
  slam_ = std::make_unique<LocalizerAndMapper>(rig_, FeatureDescriptorType::kShiTomasi2, options.use_gpu);
  slam_->SetReproduceMode(options.reproduce_mode);
  slam_->SetVprOptions(options.vpr_options);

  first_lcs_.reset(CreateLoopClosureSolver(LoopClosureSolverType::kSimplePoint, RansacType::kNone, true, rig));
  second_lcs_.reset(CreateLoopClosureSolver(LoopClosureSolverType::kTwoStepsEasy, RansacType::kPnP, true, rig));
}

bool Localizer::OpenDatabase(const std::string& path) {
  if (!slam_) {
    TraceError("Localizer not initialized. Call Init() before OpenDatabase().");
    return false;
  }
  if (!slam_->AttachToExistingReadOnlyDatabase(path)) {
    return false;
  }

  // The place recognition map was written into the same folder by SaveMap, and its entries are keyed
  // by the pose graph node ids this database just restored, so it loads with its ids intact. A map
  // saved without place recognition simply has no such file, which is not an error: localization
  // then needs the caller's guess pose, as it always did.
  if (slam_->IsVprEnabled()) {
    slam_->LoadVprMap(path, vpr::VprMap::LoadMode::kWithPoseGraph);
  }
  return true;
}

std::vector<Isometry3T> Localizer::RecognizePlaces(const sof::Images& images, size_t max_results) const {
  std::vector<Isometry3T> poses;
  if (!slam_ || !slam_->IsVprEnabled() || max_results == 0) {
    return poses;
  }

  const vpr::VprImage query = vpr::MakeVprImageFromImages(images);
  if (query.Empty()) {
    return poses;
  }

  const std::vector<vpr::VprPlace> places = slam_->RecognizePlaces(query, max_results);
  poses.reserve(places.size());
  for (const vpr::VprPlace& place : places) {
    poses.push_back(place.pose);
  }
  return poses;
}

void Localizer::UseRecognizedPlaceProbes() {
  options_.horizontal_search_radius = options_.horizontal_step;
  options_.vertical_search_radius = options_.vertical_step;
  options_.angle_search_range_rads = 4 * options_.angle_step_rads;
  PrepareProbes();
}

bool Localizer::Localize(const std::vector<Isometry3T>& guess_poses, const sof::Images& images,
                         LocalizationResult& result) {
  for (const Isometry3T& guess_pose : guess_poses) {
    if (Localize(guess_pose, images, result)) {
      return true;
    }
    // Localize() only moves slam_ out on success, so a rejected candidate leaves the map in place
    // and the next one can be tried against it.
  }
  return false;
}

// TODO: Why do we need current_pose here?
bool Localizer::Localize(const Isometry3T& guess_pose, const sof::Images& images, LocalizationResult& result) {
  guess_pose_ = guess_pose;

  // Each candidate is a fresh search: the trial counter and the best-so-far both bound the probe
  // walk (Step() stops as soon as a probe is farther from the hint than the best hit), so carrying
  // them over from a rejected candidate would cut the next one short.
  trial_ = 0;
  max_simple_loop_closure_status_ = LoopClosureStatus{};
  max_exact_loop_closure_status_ = LoopClosureStatus{};
  output_queue_ = {};

  VOFrameInfo fi;
  fi.images_ = images;
  fi.valid_ = true;

  // Cycle of localizations
  process_frame_info_ = fi;
  for (;;) {
    const bool to_be_continued = Step();
    if (!to_be_continued) {
      break;
    }
  }
  // Finish
  if (max_exact_loop_closure_status_.Weight() > 0) {
    // slam_localization_info
    SlamStdout("Localization successful (%s). ", sw_step_.Verbose().c_str());

    const auto& lc = max_exact_loop_closure_status_;

    result.slam_from = std::move(slam_);
    result.from_keyframe_id = lc.from_keyframe_id;
    result.pose_in_slam = lc.result_pose;
    result.pose_in_slam_covariance = lc.result_pose_covariance;
    return true;
  } else {
    // slam_localization_info
    SlamStdout("Localization failed (%s). ", sw_step_.Verbose().c_str());
  }

  return false;
}

bool Localizer::GenerateTrial(uint64_t trial, Isometry3T& isometry_current) const {
  if (trial >= shifts_.size()) {
    return false;
  }
  const Shift& shift = shifts_[trial];
  const Vector3T t(shift.x, shift.y, shift.z);
  const Eigen::Matrix3f r = Eigen::AngleAxis<float>(shift.angle, Vector3T::UnitY()).toRotationMatrix();
  Isometry3T isometry = Isometry3T::Identity();
  isometry.translate(t);
  isometry.rotate(r);
  isometry_current = guess_pose_ * isometry;

  return true;
}

void Localizer::TryAccurateLoopClosure(const LoopClosureStatus& simple_loop_closure_status) {
  LoopClosureStatus exact_loop_closure_status;

  if (!AccurateLoopClosure(simple_loop_closure_status, exact_loop_closure_status)) {
    return;
  }

  // exact solution is found so "simple_step" is good
  max_simple_loop_closure_status_ = simple_loop_closure_status;
  if (exact_loop_closure_status.Weight() > max_exact_loop_closure_status_.Weight()) {
    // the best is found
    SlamStdout("q");
    max_exact_loop_closure_status_ = exact_loop_closure_status;
  }
}

float Localizer::Shift::dist() const { return sqrtf(x * x + y * y + z * z); }

bool Localizer::Step() {
  SlamStdout(".");

  Isometry3T isometry_current;
  const uint64_t trial = trial_;
  if (!GenerateTrial(trial_, isometry_current)) {
    return false;
  }

  const float dist_from_hint = shifts_[trial].dist();

  if (dist_from_hint > max_exact_loop_closure_status_.dist_from_hint) {
    return false;
  }

  trial_++;

  const auto& frame_images = process_frame_info_.images_;

  LoopClosureStatus simple_loop_closure_status;
  simple_loop_closure_status.SetFrame(trial, dist_from_hint, isometry_current);
  FastLoopClosure(isometry_current, frame_images, simple_loop_closure_status);

  if (simple_loop_closure_status.success &&
      simple_loop_closure_status.Weight() > max_simple_loop_closure_status_.Weight()) {
    TryAccurateLoopClosure(simple_loop_closure_status);
  }

  return true;
}

bool Localizer::FastLoopClosure(const Isometry3T& isometry_current, const Images& frame_images,
                                LoopClosureStatus& simple_loop_closure_status) const {
  LocalizerAndMapper::LoopClosureStatus slam_loop_closure_status;
  const Isometry3T& guess = isometry_current;

  bool simple_lc_success = false;
  if (first_lcs_) {
    slam_->DetectLoopClosure(*first_lcs_, frame_images, guess, slam_loop_closure_status);
    simple_lc_success = slam_loop_closure_status.success;
  }

  const KeyFrameId best_kf = slam_->FindKeyframeWithMostLandmarks(slam_loop_closure_status.landmarks);
  simple_loop_closure_status.SetLC(best_kf, slam_loop_closure_status);

  return simple_lc_success;
}

bool Localizer::AccurateLoopClosure(const LoopClosureStatus& source, LoopClosureStatus& loop_closure_status) const {
  const auto& frame_images = process_frame_info_.images_;
  const Isometry3T& guess = source.result_pose;
  LocalizerAndMapper::LoopClosureStatus slam_loop_closure_status;

  if (!second_lcs_) {
    return false;
  }
  slam_->DetectLoopClosure(*second_lcs_, frame_images, guess, slam_loop_closure_status);
  if (!slam_loop_closure_status.success) {
    return false;
  }

  const KeyFrameId best_kf = slam_->FindKeyframeWithMostLandmarks(slam_loop_closure_status.landmarks);

  loop_closure_status.SetLC(best_kf, slam_loop_closure_status);
  loop_closure_status.SetFrame(source.trial_id, source.dist_from_hint, source.guess_pose);

  return true;
}

void Localizer::PrepareProbes() {
  const float& x_radius = options_.horizontal_search_radius;
  const float& y_radius = options_.vertical_search_radius;  // meters, y is vertical
  const float& z_radius = options_.horizontal_search_radius;

  const float& h_step = options_.horizontal_step;
  const float& v_step = options_.vertical_step;
  const float& a_step = options_.angle_step_rads;

  SlamStdout("horizontal_search_radius %f \n", options_.horizontal_search_radius);
  SlamStdout("vertical_search_radius %f \n", options_.vertical_search_radius);
  SlamStdout("horizontal_step %f \n", options_.horizontal_step);
  SlamStdout("vertical_step %f \n", options_.vertical_step);
  SlamStdout("angle_step_rads %f \n", options_.angle_step_rads);

  const size_t x_count = static_cast<size_t>((2 * x_radius) / h_step) + 1;
  const size_t y_count = static_cast<size_t>((2 * y_radius) / v_step) + 1;
  const size_t z_count = static_cast<size_t>((2 * z_radius) / h_step) + 1;
  const bool full_turn = options_.angle_search_range_rads >= 2 * PI;
  const float half_range = options_.angle_search_range_rads / 2;
  const size_t angle_count = full_turn ? static_cast<size_t>(2 * PI / a_step)
                                       : static_cast<size_t>(options_.angle_search_range_rads / a_step) + 1;
  shifts_.reserve(x_count * y_count * z_count * angle_count);
  shifts_.clear();
  for (float x = -x_radius; x <= x_radius; x += h_step) {
    for (float y = -y_radius; y <= y_radius; y += v_step) {
      for (float z = -z_radius; z <= z_radius; z += h_step) {
        if (full_turn) {
          for (float angle = 0; angle < 2 * PI; angle += a_step) {
            shifts_.push_back({x, y, z, angle});
          }
        } else {
          for (float angle = -half_range; angle <= half_range; angle += a_step) {
            shifts_.push_back({x, y, z, angle});
          }
        }
      }
    }
  }

  std::sort(shifts_.begin(), shifts_.end(), [](const Shift& lhs, const Shift& rhs) { return lhs.dist() < rhs.dist(); });

  SlamStdout("Prepared %zu probes for localization", shifts_.size());
}

}  // namespace cuvslam::slam

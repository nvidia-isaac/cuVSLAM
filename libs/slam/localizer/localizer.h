
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

#include <algorithm>
#include <queue>
#include <string>

#include "camera/rig.h"
#include "common/image.h"
#include "common/stopwatch.h"
#include "common/unaligned_types.h"

#include "slam/slam/slam.h"
#include "slam/view/view_landmarks.h"
#include "slam/view/view_manager.h"
#include "slam/vpr/vpr_types.h"
#include "sof/gradient_pyramid.h"

namespace cuvslam::slam {

using namespace cuvslam;

struct LocalizerOptions {
  bool reproduce_mode = false;  // allow to repeat results: ransac.seed(0), sync=true
  bool use_gpu = false;
  float horizontal_search_radius = 1.5f;  // meters
  float vertical_search_radius = 0.5;
  float horizontal_step = 0.5f;
  float vertical_step = 0.25f;
  float angle_step_rads = 2 * cuvslam::PI / 36;
  // Width of the yaw sweep around the guess. A full turn is right when the guess is a position with
  // no heading attached; a guess that came from place recognition already points the right way, and
  // sweeping a full turn around each of several candidates costs more probes than searching blind.
  float angle_search_range_rads = 2 * cuvslam::PI;
  // Place recognition backend of the map being opened. Must match the one the map was saved with,
  // and is what lets Localize run without a guess pose.
  vpr::VprOptions vpr_options;
};

struct LocalizationResult {
  // Ownership-transfer pattern: The caller takes ownership, configures the
  // LocalizerAndMapper as needed (parameters, sensors, callbacks), then swaps it with their
  // existing SLAM (receive->configure->swap) to replace the running system.
  std::unique_ptr<LocalizerAndMapper> slam_from;

  KeyFrameId from_keyframe_id;  //
  Isometry3T pose_in_slam;      // world_from_rig - global absolute pose
  Matrix6T pose_in_slam_covariance;
};

class Localizer {
public:
  void Init(const camera::Rig& rig, const LocalizerOptions& options);

  bool OpenDatabase(const std::string& path);

  // Places in the opened map that look like `images`, best first, as world poses to verify.
  // Empty when the map carries no place recognition map, or nothing matched.
  std::vector<Isometry3T> RecognizePlaces(const sof::Images& images, size_t max_results) const;

  // Narrow the probe grid to the neighbourhood of a recognized place. Call before localizing against
  // poses from RecognizePlaces(): those already point the right way and are only a frame spacing
  // away, so a full grid and a whole turn of yaw around each of several of them would cost more
  // probes than one blind search.
  void UseRecognizedPlaceProbes();

  bool Localize(const Isometry3T& guess_pose, const sof::Images& images, LocalizationResult& result);

  // Try each guess in turn and keep the first that verifies. Localizing without a pose guess means
  // handing this the poses RecognizePlaces() proposed, because a blind search of a building does
  // not terminate.
  bool Localize(const std::vector<Isometry3T>& guess_poses, const sof::Images& images, LocalizationResult& result);

private:
  struct LoopClosureStatus {
    bool success = false;
    uint64_t trial_id = 0;
    float dist_from_hint = std::numeric_limits<float>::infinity();
    KeyFrameId from_keyframe_id = InvalidKeyFrameId;
    Isometry3T guess_pose;
    Isometry3T result_pose;
    Matrix6T result_pose_covariance;
    uint32_t tracked_landmarks_count = 0;  // Count of Tracked Landmarks in LC
    uint32_t good_landmarks_count = 0;     // Count of good Landmarks in LC

    void SetLC(KeyFrameId from_keyframe_id, const LocalizerAndMapper::LoopClosureStatus& slam_loop_closure_status);
    void SetFrame(uint64_t trial_id, float _dist_from_hint, const Isometry3T& guess_pose);
    [[nodiscard]] float Weight() const;
  };

  struct VOFrameInfo {
    bool valid_ = false;
    sof::Images images_;

    FrameId frame_id() const {
      auto image = std::find_if(images_.begin(), images_.end(), [](const auto& image) { return image != nullptr; });
      return image == images_.end() ? FrameId{} : (*image)->get_image_meta().frame_id;
    }
    bool is_valid() const { return valid_; }
  };

  struct Shift {
    float x;
    float y;
    float z;
    float angle;  // 0-2Pi

    [[nodiscard]] float dist() const;
  };

  camera::Rig rig_;
  LocalizerOptions options_;
  std::unique_ptr<LocalizerAndMapper> slam_;
  std::unique_ptr<ILoopClosureSolver> first_lcs_;
  std::unique_ptr<ILoopClosureSolver> second_lcs_;
  Isometry3T guess_pose_;
  VOFrameInfo current_frame_info_;
  VOFrameInfo process_frame_info_;
  LoopClosureStatus max_simple_loop_closure_status_;
  LoopClosureStatus max_exact_loop_closure_status_;
  uint64_t trial_ = 0;
  std::queue<std::shared_ptr<LoopClosureStatus>> output_queue_;
  Stopwatch sw_step_;
  std::vector<Shift> shifts_;  // Hint-relative offsets (xyz, yaw) from PrepareProbes().

  // landmarks statistic. If localizer_observation_view_ is not null
  mutable std::unordered_map<LandmarkId, uint32_t> tracked_landmarks_;

  bool Step();

  bool GenerateTrial(uint64_t trial, Isometry3T& isometry_current) const;

  // Attempts accurate refinement of a fast loop closure result and updates best result if improved
  void TryAccurateLoopClosure(const LoopClosureStatus& simple_loop_closure_status);

  // Fast loop closure using SimplePoint solver without RANSAC for quick candidate screening
  bool FastLoopClosure(const Isometry3T& isometry_current, const Images& frame_image,
                       LoopClosureStatus& loop_closure_status) const;

  // Accurate loop closure using TwoStepsEasy solver with PnP RANSAC for precise pose refinement
  bool AccurateLoopClosure(const LoopClosureStatus& source, LoopClosureStatus& loop_closure_status) const;

  void PrepareProbes();
};

}  // namespace cuvslam::slam


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

#include <functional>
#include <list>

#include "common/types.h"
#include "params/params.h"

namespace cuvslam::pipelines {

struct StateMachineSettings {
  // Hard cap on inter-frame gap. Anything larger triggers a full reset (the IMU preintegration is
  // no longer trustworthy across that gap).
  int64_t max_integration_time_ns = 1e9;

  // Minimum number of keyframes that must accumulate before gravity init is attempted.
  size_t min_num_kf_for_gravity = 20;

  // Minimum time span covered by the keyframe history before gravity init is attempted.
  int64_t min_time_period_ns = 2e9;  // 2s

  // Sliding-window size for the keyframe history. Older keyframes are dropped.
  int64_t max_time_period_ns = 25e9;  // 25s

  // Period at which gravity is re-estimated once the SM is in the Ok state. Set to a very large
  // value (e.g. the default 5e12 = ~83 min) to effectively disable periodic re-estimation; gravity
  // is then estimated once at boot only.
  int64_t gravity_update_period_ns = 5e12;
};

// Three-state machine driving IMU initialization (gravity / bias / velocity estimation).
//
// Lifecycle:
//   Uninitialized -> Initializing : enough keyframes accumulated to attempt gravity init.
//   Initializing  -> Ok           : the gravity-init callback succeeded.
//   Initializing  -> Initializing : the callback failed; retry on the next keyframe (timeline
//                                   is preserved instead of being wiped).
//   Ok            -> Ok           : after `gravity_update_period_ns` elapsed since last successful
//                                   estimation, the callback fires again to refine gravity. On
//                                   failure, the timeline is preserved and we retry next keyframe.
//   any           -> Uninitialized: explicit Reset, or FrameDropped (gap > max_integration_time_ns).
//
// Per-frame visual-tracking failures (Event::FrameFailed) do NOT clear progress — the keyframe
// timeline only resets on dropped frames or explicit reset.
//
// Settings are passed per-call so they always reflect the latest values from
// ApplyPersistentInternalParameters without requiring a getter/setter on a live object.
class StateMachine {
public:
  enum class State { Uninitialized, Initializing, Ok };

  enum class Event {
    FrameOk,       // visual tracking succeeded and IMU samples covered the gap
    FrameFailed,   // visual tracking failed but IMU samples were intact (no reset)
    FrameDropped,  // IMU drop or large gap; trigger full reset
    Reset,         // explicit reset (e.g. odometry::reset())
  };

  // Returns true on success. A false return keeps the SM in Initializing so the caller retries on
  // the next keyframe rather than wiping progress.
  using GravityInitFn = std::function<bool(size_t num_kfs)>;

  StateMachine() = default;

  void set_gravity_init_fn(GravityInitFn fn);

  // Single dispatcher. Returns the post-event state.
  State on_event(Event e, int64_t ts_ns, bool is_keyframe, const StateMachineSettings& s);

  State state() const { return state_; }

  void reset();

private:
  bool ready_to_init(int64_t ts_ns, const StateMachineSettings& s) const;

  std::list<int64_t> keyframe_timeline_;
  int64_t last_successful_frame_time_ns_ = -1;
  int64_t last_gravity_estimation_time_ns_ = -1;

  GravityInitFn gravity_init_fn_ = nullptr;

  State state_ = State::Uninitialized;
};

}  // namespace cuvslam::pipelines

// Tunable fields of StateMachineSettings, addressed as `sm.<name>`. Only meaningful in Inertial
// mode, so the registry only exposes them when the tracker has a state machine.
CUVSLAM_PARAMS_BEGIN(cuvslam::pipelines::StateMachineSettings)
CUVSLAM_PARAM_BOUNDED(max_integration_time_ns, "Inter-frame gap above which IMU preintegration is reset, ns",
                      NonNegative())
CUVSLAM_PARAM(min_num_kf_for_gravity, "Keyframes required before gravity initialization is attempted")
CUVSLAM_PARAM_BOUNDED(min_time_period_ns, "Keyframe history span required before gravity initialization, ns",
                      NonNegative())
CUVSLAM_PARAM_BOUNDED(max_time_period_ns, "Sliding-window size of the keyframe history, ns", NonNegative())
CUVSLAM_PARAM_BOUNDED(gravity_update_period_ns, "Period between gravity re-estimations once initialized, ns",
                      NonNegative())
CUVSLAM_PARAMS_END()

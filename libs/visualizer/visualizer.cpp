
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

#include "visualizer.hpp"

#include <utility>

#include <rerun/archetypes/view_coordinates.hpp>

#include "common/log.h"

namespace cuvslam::visualizer {

// RerunVisualizer implementation
RerunVisualizer& RerunVisualizer::getInstance() {
  static RerunVisualizer instance;
  return instance;
}

RerunVisualizer::RerunVisualizer() {
  if (recording_.is_enabled()) {
    recording_.spawn().exit_on_failure();
  }
  // Static, so it applies on every timeline (frame_id/timestamp too), not just log_time.
  recording_.log_static(
      "world",
      rerun::archetypes::ViewCoordinates::RIGHT_HAND_Y_DOWN);  // OpenCV-style world (x right, y down, z forward)
}

RerunVisualizer::~RerunVisualizer() {
  // Call shutdown in destructor as backup, but explicit shutdown is preferred
  if (recording_.is_enabled()) {
    shutdown();
  }
}

void RerunVisualizer::setupTimeline(size_t frame_id, int64_t timestamp) {
  recording_.set_time_sequence("frame_id", frame_id);
  recording_.set_time_duration_nanos("timestamp", timestamp);
}

void RerunVisualizer::clearViewport(const std::string& viewport_name) { recording_.log(viewport_name, rerun::Clear()); }

void RerunVisualizer::shutdown() {
  static bool already_shutdown = false;
  if (already_shutdown) return;
  already_shutdown = true;

  try {
    std::ignore = recording_.flush_blocking();
  } catch (...) {
    // Ignore any exceptions during shutdown
  }
}

}  // namespace cuvslam::visualizer

// Global shutdown function for easy RERUN macro usage
namespace cuvslam {
void shutdownVisualizer() {
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  visualizer.shutdown();
}
}  // namespace cuvslam

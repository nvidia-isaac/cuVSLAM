
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

#ifdef USE_RERUN
#include <string>
#include <vector>

#include "common/frame_id.h"
#include "common/image.h"
#include "common/isometry.h"
#include "common/isometry_utils.h"
#include "common/log.h"
#include "common/rerun.h"
#include "common/vector_3t.h"

#include "visualizer/visualizer.hpp"

namespace cuvslam::launcher {
void setupFrameTimeline(const Metas& meta);

void logCameraImages(const Metas& meta, const Sources& images, const std::vector<CameraId>& cam_ids,
                     const std::vector<std::string>& viewport_names);

/**
 * Log depth image for visualization
 * @param depth_source The depth image source (F32 type expected, values in meters)
 * @param meta Image metadata containing shape information
 * @param viewport_name Rerun viewport name for the depth image
 * @param depth_meter Scale factor: how many depth units equal one meter (default 1.0 for metric depth)
 * @param min_depth Minimum expected depth value for colormap (default 0.0)
 * @param max_depth Maximum expected depth value for colormap (default 100.0 meters)
 */
void logDepthImage(const ImageSource& depth_source, const ImageMeta& meta, const std::string& viewport_name,
                   float depth_meter = 1.0f, float min_depth = 0.0f, float max_depth = 100.0f);

/**
 * Clear depth image visualization
 * @param viewport_name Rerun viewport name to clear
 */
void clearDepthImage(const std::string& viewport_name);

}  // namespace cuvslam::launcher
#endif

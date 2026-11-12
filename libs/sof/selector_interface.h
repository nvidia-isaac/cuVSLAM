
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

#include "sof/sof.h"

namespace cuvslam::sof {

struct Settings;

class ISelector {
public:
  virtual ~ISelector() = default;

  virtual void set_image_width(uint32_t width) = 0;
  virtual void reset_selector() = 0;
  // sof_settings: the current frame's settings (see Odometry::Track internals).
  virtual bool select(const TracksVector& cur_frame_tracks, const Settings& sof_settings) = 0;
  virtual void set_tracks(const TracksVector& cur_frame_tracks) = 0;
};

}  // namespace cuvslam::sof

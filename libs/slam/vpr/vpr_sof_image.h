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

#include "slam/common/slam_input_image.h"
#include "slam/vpr/vpr_types.h"

namespace cuvslam::slam::vpr {

/// Copy the pixels of one tracking image context into a VprImage: the level 0 8 bit CPU pyramid, or
/// a download of the GPU image, whichever one the context reports it has. Returns an empty image
/// when it holds neither, which is the case for a context that was reset and not refilled.
///
/// A recycled context already refilled with a newer frame is NOT detected here - it reports a valid
/// image, just the wrong one. The caller checks that the context still belongs to the frame it
/// wants; AsyncSlam compares frame ids before handing images to LocalizerAndMapper::AddKeyframe.
VprImage MakeVprImageFromContext(const ImageContextPtr& context);

/// Same, for the first camera of `images` that carries pixels. `images` is camera indexed and may
/// hold nullptr for cameras absent from this frame.
VprImage MakeVprImageFromImages(const Images& images);

}  // namespace cuvslam::slam::vpr

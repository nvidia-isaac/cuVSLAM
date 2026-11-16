
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

#include "common/types.h"

#include "sof/image_pyramid_float.h"

namespace cuvslam::utils {

enum class ImageTransformType { getSRGB, LinearToLog };

template <typename _PixelType, ImageTransformType _Type, size_t _Shift, size_t _Max, size_t _RANGE_MIN,
          size_t _RANGE_MAX>
struct _ImageTransform {
  static_assert(_Shift > 0, "_Shift should be 1 or greater");
  static_assert(_Max > 0, "_Max should be 1 or greater (usually 255)");
  static_assert(_RANGE_MAX > _RANGE_MIN, "_RANGE_MAX should be greater than _RANGE_MIN");

  static _PixelType sRGBtoLin(const _PixelType sRGB) {
    assert(0.f <= sRGB && sRGB <= 1.f);

    // see https://en.wikipedia.org/wiki/SRGB
    if (sRGB < 0.04045f) {
      return _PixelType(sRGB / 12.92f);
    } else {
      const float a = 0.055f;
      return _PixelType(std::pow((sRGB + a) / (1.f + a), 2.4f));
    }
  }

  static _PixelType LINtoSRGB(const _PixelType linearInput) {
    assert(0.f <= linearInput && linearInput <= 1.f);

    // see https://en.wikipedia.org/wiki/SRGB
    // clipping of output to (0.0 - 1.0) range omitted
    if (linearInput < 0.0031308f) {
      return _PixelType(linearInput * 12.92f);
    } else {
      const float powerFactor = 1.0f / 2.4f;
      const float a = 0.055f;
      return _PixelType(std::pow(linearInput, powerFactor) * (1.f + a) - a);
    }
  }

  static ImageMatrix<_PixelType> getSRGB(const ImageMatrix<_PixelType>& image) { return image; }

  static ImageMatrix<_PixelType> LinearToLog(const ImageMatrix<_PixelType>& image) {
#if 1
    ImageMatrix<_PixelType> linArray = (image.array() - _RANGE_MIN).cwiseMax(0) / _PixelType(_RANGE_MAX - _RANGE_MIN);
    const auto& logArray = linArray.unaryExpr(&LINtoSRGB).array() * _PixelType(255.0f);
    return logArray;
#else
    return ((image.array() - _RANGE_MIN).cwiseMax(0) / _PixelType(_RANGE_MAX - _RANGE_MIN) * _PixelType(_Max)).matrix();
#endif
  }

  static ImageMatrix<_PixelType> Apply(const ImageMatrix<_PixelType>& image) {
    return (_Type == ImageTransformType::getSRGB) ? getSRGB(image) : LinearToLog(image);
  }
};

template <typename _PixelType>
using ImageTransform = _ImageTransform<_PixelType, ImageTransformType::getSRGB, 2, 255, 0, 255>;

}  // namespace cuvslam::utils

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

#include "slam/vpr/vpr_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cuvslam::slam::vpr {

namespace {

inline uint8_t Rgb8ToGray(const uint8_t* p) {
  // BT.601 luma, the same weighting the tracker uses when it converts color input to grayscale.
  const int v = (77 * static_cast<int>(p[0]) + 150 * static_cast<int>(p[1]) + 29 * static_cast<int>(p[2])) >> 8;
  return static_cast<uint8_t>(std::min(255, v));
}

inline uint8_t FloatToGray(float v) {
  if (!(v > 0.f)) {
    return 0;
  }
  if (v >= 255.f) {
    return 255;
  }
  return static_cast<uint8_t>(v + 0.5f);
}

}  // namespace

VprImage MakeVprImage(const void* data, int width, int height, int pitch, VprPixelFormat format) {
  VprImage image;
  if (data == nullptr || width <= 0 || height <= 0) {
    return image;
  }

  int bytes_per_pixel = 1;
  if (format == VprPixelFormat::kRgb8) {
    bytes_per_pixel = 3;
  } else if (format == VprPixelFormat::kFloat) {
    bytes_per_pixel = static_cast<int>(sizeof(float));
  }
  const int row_bytes = width * bytes_per_pixel;
  if (pitch <= 0) {
    pitch = row_bytes;
  }
  if (pitch < row_bytes) {
    return image;
  }

  image.width = width;
  image.height = height;
  image.row.resize(static_cast<size_t>(width) * height);

  const auto* base = static_cast<const uint8_t*>(data);
  for (int y = 0; y < height; ++y) {
    const uint8_t* src = base + static_cast<size_t>(y) * pitch;
    uint8_t* dst = image.row.data() + static_cast<size_t>(y) * width;
    switch (format) {
      case VprPixelFormat::kMono8:
        std::memcpy(dst, src, static_cast<size_t>(width));
        break;
      case VprPixelFormat::kRgb8:
        for (int x = 0; x < width; ++x) {
          dst[x] = Rgb8ToGray(src + static_cast<size_t>(x) * 3);
        }
        break;
      case VprPixelFormat::kFloat: {
        const auto* fsrc = reinterpret_cast<const float*>(src);
        for (int x = 0; x < width; ++x) {
          dst[x] = FloatToGray(fsrc[x]);
        }
        break;
      }
    }
  }
  return image;
}

VprImage Downscale(const VprImage& image, int factor) {
  VprImage out;
  if (image.Empty()) {
    return out;
  }
  if (factor <= 1) {
    return image;
  }

  out.width = std::max(1, image.width / factor);
  out.height = std::max(1, image.height / factor);
  out.row.assign(static_cast<size_t>(out.width) * out.height, 0);

  // Average over the source block that maps to each destination pixel. Straight subsampling would
  // alias hard on the high frequency road and foliage texture that dominates driving datasets, and
  // the aliasing pattern depends on the exact sub-pixel alignment of the two passes, which is
  // exactly what has to stay stable for the same place to match itself.
  for (int y = 0; y < out.height; ++y) {
    const int y0 = y * image.height / out.height;
    const int y1 = std::max(y0 + 1, (y + 1) * image.height / out.height);
    for (int x = 0; x < out.width; ++x) {
      const int x0 = x * image.width / out.width;
      const int x1 = std::max(x0 + 1, (x + 1) * image.width / out.width);
      uint32_t sum = 0;
      uint32_t count = 0;
      for (int sy = y0; sy < y1; ++sy) {
        const uint8_t* src = image.row.data() + static_cast<size_t>(sy) * image.width;
        for (int sx = x0; sx < x1; ++sx) {
          sum += src[sx];
          ++count;
        }
      }
      out.row[static_cast<size_t>(y) * out.width + x] = static_cast<uint8_t>(sum / std::max(1u, count));
    }
  }
  return out;
}

VprImage Resize(const VprImage& image, int width, int height) {
  VprImage out;
  if (image.Empty() || width <= 0 || height <= 0) {
    return out;
  }
  if (width == image.width && height == image.height) {
    return image;
  }
  // Downscaling by more than a factor of two is done with the box filter first, so the bilinear
  // pass below never has to skip source pixels.
  const int box = std::min(image.width / std::max(1, width), image.height / std::max(1, height));
  if (box >= 2) {
    return Resize(Downscale(image, box), width, height);
  }

  out.width = width;
  out.height = height;
  out.row.assign(static_cast<size_t>(width) * height, 0);

  const float sx = static_cast<float>(image.width) / static_cast<float>(width);
  const float sy = static_cast<float>(image.height) / static_cast<float>(height);
  for (int y = 0; y < height; ++y) {
    const float fy = std::min(static_cast<float>(image.height - 1), (static_cast<float>(y) + 0.5f) * sy - 0.5f);
    const int y0 = std::max(0, static_cast<int>(fy));
    const int y1 = std::min(image.height - 1, y0 + 1);
    const float wy = std::max(0.f, fy - static_cast<float>(y0));
    const uint8_t* r0 = image.row.data() + static_cast<size_t>(y0) * image.width;
    const uint8_t* r1 = image.row.data() + static_cast<size_t>(y1) * image.width;
    for (int x = 0; x < width; ++x) {
      const float fx = std::min(static_cast<float>(image.width - 1), (static_cast<float>(x) + 0.5f) * sx - 0.5f);
      const int x0 = std::max(0, static_cast<int>(fx));
      const int x1 = std::min(image.width - 1, x0 + 1);
      const float wx = std::max(0.f, fx - static_cast<float>(x0));
      const float top = static_cast<float>(r0[x0]) * (1.f - wx) + static_cast<float>(r0[x1]) * wx;
      const float bottom = static_cast<float>(r1[x0]) * (1.f - wx) + static_cast<float>(r1[x1]) * wx;
      out.row[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(top * (1.f - wy) + bottom * wy + 0.5f);
    }
  }
  return out;
}

bool NormalizePixels(const VprImage& image, std::vector<float>& out) {
  out.assign(image.row.size(), 0.f);
  if (image.Empty()) {
    return false;
  }

  double sum = 0;
  for (const uint8_t v : image.row) {
    sum += v;
  }
  const double mean = sum / static_cast<double>(image.row.size());

  double variance = 0;
  for (const uint8_t v : image.row) {
    const double d = static_cast<double>(v) - mean;
    variance += d * d;
  }
  variance /= static_cast<double>(image.row.size());
  const double stddev = std::sqrt(variance);
  if (!(stddev > 1e-6)) {
    return false;
  }

  const float inv = static_cast<float>(1.0 / stddev);
  for (size_t i = 0; i < image.row.size(); ++i) {
    out[i] = (static_cast<float>(image.row[i]) - static_cast<float>(mean)) * inv;
  }
  return true;
}

}  // namespace cuvslam::slam::vpr

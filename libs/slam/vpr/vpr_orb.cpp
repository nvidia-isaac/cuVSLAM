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

#include "slam/vpr/vpr_orb.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

namespace cuvslam::slam::vpr {

namespace {

// 16 point Bresenham circle of radius 3, the ring FAST tests.
constexpr int kCircle[16][2] = {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0},  {3, 1},   {2, 2},   {1, 3},
                                {0, 3},  {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

// The orientation patch has radius 15 and a rotated BRIEF offset reaches 15*sqrt(2) ~ 22 px, so a
// keypoint closer than this to the edge would have its centroid biased by the samples that fall
// outside and its descriptor flattened by the edge clamping in the bilinear fetch.
constexpr int kBorder = 22;

/// 256 BRIEF bit pair offsets in [-15, 15], drawn once from a fixed seed so two runs, and two
/// machines, quantize the same image to the same words.
std::array<std::array<int8_t, 4>, 256> MakeBitPattern() {
  std::mt19937 rng(0x4b4b4b4bu);
  std::uniform_int_distribution<int> dist(-15, 15);
  std::array<std::array<int8_t, 4>, 256> pattern{};
  for (auto& pair : pattern) {
    for (auto& value : pair) {
      value = static_cast<int8_t>(dist(rng));
    }
  }
  return pattern;
}
const auto kBitPattern = MakeBitPattern();

struct PyramidLevel {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
  float scale = 1.f;  // relative to level 0
};

/// Separable 3x3 binomial blur, kernel [1 2 1] / 4, with the edge pixel weighted three times so the
/// borders do not darken.
void GaussianBlur3(const uint8_t* src, uint8_t* dst, int width, int height) {
  std::vector<int16_t> tmp(static_cast<size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * width;
    int16_t* out = tmp.data() + static_cast<size_t>(y) * width;
    out[0] = static_cast<int16_t>(row[0] * 3 + row[1]);
    for (int x = 1; x < width - 1; ++x) {
      out[x] = static_cast<int16_t>(row[x - 1] + row[x] * 2 + row[x + 1]);
    }
    out[width - 1] = static_cast<int16_t>(row[width - 2] + row[width - 1] * 3);
  }
  for (int x = 0; x < width; ++x) {
    dst[x] = static_cast<uint8_t>((tmp[x] * 3 + tmp[width + x]) >> 4);
    for (int y = 1; y < height - 1; ++y) {
      const size_t i = static_cast<size_t>(y) * width + x;
      dst[i] = static_cast<uint8_t>((tmp[i - width] + tmp[i] * 2 + tmp[i + width]) >> 4);
    }
    const size_t last = static_cast<size_t>(height - 1) * width + x;
    dst[last] = static_cast<uint8_t>((tmp[last - width] + tmp[last] * 3) >> 4);
  }
}

void Downsample(const uint8_t* src, int src_width, int src_height, uint8_t* dst, int width, int height) {
  for (int y = 0; y < height; ++y) {
    const uint8_t* src_row = src + static_cast<size_t>(y * src_height / height) * src_width;
    uint8_t* dst_row = dst + static_cast<size_t>(y) * width;
    for (int x = 0; x < width; ++x) {
      dst_row[x] = src_row[x * src_width / width];
    }
  }
}

/// FAST-9: nine or more consecutive ring pixels all brighter or all darker than the centre.
bool IsFast9(const uint8_t* image, int width, int cx, int cy, int threshold) {
  const int centre = image[static_cast<size_t>(cy) * width + cx];
  const int low = centre - threshold;
  const int high = centre + threshold;

  // Two of the four compass points have to agree before the full ring is worth reading.
  int bright = 0;
  int dark = 0;
  for (const int i : {0, 4, 8, 12}) {
    const int p = image[static_cast<size_t>(cy + kCircle[i][1]) * width + (cx + kCircle[i][0])];
    if (p > high) {
      ++bright;
    } else if (p < low) {
      ++dark;
    }
  }
  if (bright < 2 && dark < 2) {
    return false;
  }

  int ring[16];
  for (int i = 0; i < 16; ++i) {
    ring[i] = image[static_cast<size_t>(cy + kCircle[i][1]) * width + (cx + kCircle[i][0])];
  }
  for (int start = 0; start < 16; ++start) {
    bool all_bright = true;
    bool all_dark = true;
    for (int j = 0; j < 9; ++j) {
      const int p = ring[(start + j) % 16];
      if (p <= high) {
        all_bright = false;
      }
      if (p >= low) {
        all_dark = false;
      }
    }
    if (all_bright || all_dark) {
      return true;
    }
  }
  return false;
}

float HarrisScore(const uint8_t* image, int width, int cx, int cy) {
  float a = 0.f;
  float b = 0.f;
  float c = 0.f;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const size_t i = static_cast<size_t>(cy + dy) * width + (cx + dx);
      const float ix = static_cast<float>(image[i + 1]) - static_cast<float>(image[i - 1]);
      const float iy = static_cast<float>(image[i + width]) - static_cast<float>(image[i - width]);
      a += ix * ix;
      b += ix * iy;
      c += iy * iy;
    }
  }
  return (a * c - b * b) - 0.04f * (a + c) * (a + c);
}

/// Intensity centroid orientation over a circular patch of radius 15.
float ComputeOrientation(const uint8_t* image, int width, int height, int cx, int cy) {
  constexpr int kRadius = 15;
  int m10 = 0;
  int m01 = 0;
  for (int dy = -kRadius; dy <= kRadius; ++dy) {
    const int y = cy + dy;
    if (y < 0 || y >= height) {
      continue;
    }
    const int max_dx = static_cast<int>(std::sqrt(static_cast<float>(kRadius * kRadius - dy * dy)));
    for (int dx = -max_dx; dx <= max_dx; ++dx) {
      const int x = cx + dx;
      if (x < 0 || x >= width) {
        continue;
      }
      const int value = image[static_cast<size_t>(y) * width + x];
      m10 += dx * value;
      m01 += dy * value;
    }
  }
  return std::atan2(static_cast<float>(m01), static_cast<float>(m10));
}

uint8_t SampleBilinear(const uint8_t* image, int width, int height, float fx, float fy) {
  const int x0 = std::max(0, std::min(width - 2, static_cast<int>(fx)));
  const int y0 = std::max(0, std::min(height - 2, static_cast<int>(fy)));
  const float tx = fx - static_cast<float>(x0);
  const float ty = fy - static_cast<float>(y0);
  const size_t i = static_cast<size_t>(y0) * width + x0;
  const float top = (1 - tx) * image[i] + tx * image[i + 1];
  const float bottom = (1 - tx) * image[i + width] + tx * image[i + width + 1];
  return static_cast<uint8_t>((1 - ty) * top + ty * bottom);
}

void ComputeDescriptor(const uint8_t* image, int width, int height, int cx, int cy, float angle,
                       std::array<uint8_t, 32>& descriptor) {
  const float cos_a = std::cos(angle);
  const float sin_a = std::sin(angle);
  descriptor.fill(0);
  for (int bit = 0; bit < 256; ++bit) {
    const auto& offsets = kBitPattern[bit];
    const float x1 = offsets[0] * cos_a - offsets[1] * sin_a;
    const float y1 = offsets[0] * sin_a + offsets[1] * cos_a;
    const float x2 = offsets[2] * cos_a - offsets[3] * sin_a;
    const float y2 = offsets[2] * sin_a + offsets[3] * cos_a;
    const uint8_t p1 = SampleBilinear(image, width, height, static_cast<float>(cx) + x1, static_cast<float>(cy) + y1);
    const uint8_t p2 = SampleBilinear(image, width, height, static_cast<float>(cx) + x2, static_cast<float>(cy) + y2);
    if (p1 < p2) {
      descriptor[bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
    }
  }
}

}  // namespace

std::vector<OrbKeypoint> OrbExtractor::Extract(const VprImage& image) const {
  std::vector<OrbKeypoint> result;
  if (image.Empty() || image.width < 2 * kBorder || image.height < 2 * kBorder) {
    return result;
  }

  std::vector<PyramidLevel> pyramid(kNumLevels);
  pyramid[0].width = image.width;
  pyramid[0].height = image.height;
  pyramid[0].pixels = image.row;
  for (int level = 1; level < kNumLevels; ++level) {
    const PyramidLevel& previous = pyramid[level - 1];
    PyramidLevel& current = pyramid[level];
    current.scale = previous.scale / kScaleFactor;
    current.width = std::max(2 * kBorder, static_cast<int>(static_cast<float>(image.width) * current.scale));
    current.height = std::max(2 * kBorder, static_cast<int>(static_cast<float>(image.height) * current.scale));

    std::vector<uint8_t> blurred(previous.pixels.size());
    GaussianBlur3(previous.pixels.data(), blurred.data(), previous.width, previous.height);
    current.pixels.resize(static_cast<size_t>(current.width) * current.height);
    Downsample(blurred.data(), previous.width, previous.height, current.pixels.data(), current.width, current.height);
  }

  const int per_level = std::max(1, kMaxFeatures / kNumLevels);
  result.reserve(static_cast<size_t>(kMaxFeatures));

  struct Candidate {
    int x;
    int y;
    float score;
  };
  std::vector<Candidate> candidates;

  for (int level = 0; level < kNumLevels; ++level) {
    const PyramidLevel& current = pyramid[level];
    const uint8_t* pixels = current.pixels.data();
    const int width = current.width;
    const int height = current.height;

    candidates.clear();
    for (int y = kBorder; y < height - kBorder; ++y) {
      for (int x = kBorder; x < width - kBorder; ++x) {
        if (IsFast9(pixels, width, x, y, kFastThreshold)) {
          candidates.push_back({x, y, HarrisScore(pixels, width, x, y)});
        }
      }
    }

    if (static_cast<int>(candidates.size()) > per_level) {
      // Harris ranks the corners FAST accepted: FAST fires on edges too, and an edge point has no
      // repeatable position along the edge, so it would describe a different patch on every pass.
      std::partial_sort(candidates.begin(), candidates.begin() + per_level, candidates.end(),
                        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
      candidates.resize(static_cast<size_t>(per_level));
    }

    const float inv_scale = 1.f / current.scale;
    for (const Candidate& candidate : candidates) {
      OrbKeypoint keypoint;
      keypoint.angle_rad = ComputeOrientation(pixels, width, height, candidate.x, candidate.y);
      keypoint.x = static_cast<float>(candidate.x) * inv_scale;
      keypoint.y = static_cast<float>(candidate.y) * inv_scale;
      keypoint.level = level;
      ComputeDescriptor(pixels, width, height, candidate.x, candidate.y, keypoint.angle_rad, keypoint.descriptor);
      result.push_back(keypoint);
    }
  }

  return result;
}

}  // namespace cuvslam::slam::vpr

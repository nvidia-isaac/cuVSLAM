
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

#include "sof/selection.h"

#include "common/unaligned_types.h"

namespace {
using cuvslam::epsilon;
using cuvslam::ImageMatrixT;
using cuvslam::Vector2N;
using cuvslam::Vector2T;

// x1 = -1, x2 = 0, x3 = 1
float FindMinimum1D(float y1, float y2, float y3) {
  const float k = y1 - 2 * y2 + y3;

  const float m = std::abs(k) <= 10 * epsilon() ? 0 : 0.5f * (y1 - y3) / k;

  // clamp value in case of float err
  if (m >= 0.5f) {
    return 0.5f;
  }
  if (m <= -0.5f) {
    return -0.5f;
  }
  return m;
}

Vector2T refineFeaturePosition(const Vector2N& c, const ImageMatrixT& strength) {
  const int x = static_cast<int>(c.x());
  const int y = static_cast<int>(c.y());
  const int nCols = static_cast<int>(strength.cols());
  const int nRows = static_cast<int>(strength.rows());

  if (x == 0 || x == nCols - 1 || y == 0 || y == nRows - 1) {
    return {x, y};
  }

  const float gfttC = strength(y, x);
  const float resX = x + FindMinimum1D(strength(y, x - 1), gfttC, strength(y, x + 1));
  const float resY = y + FindMinimum1D(strength(y - 1, x), gfttC, strength(y + 1, x));
  assert(x - 0.5f <= resX && resX <= x + 0.5f);
  assert(y - 0.5f <= resY && resY <= y + 0.5f);
  return {resX, resY};
}

}  // namespace

namespace cuvslam::sof {

struct GoodFeaturesToTrackDetector::Bin {
  std::vector<int32_t> pixels;  // index
  float accGFTT;                // bin "goodness". Feature will be spread between bins according to this measure.
  Vector2N start;
  Vector2N size;

  bool isInside(const Vector2T& p) const {
    return start.x() <= p.x() && p.x() <= start.x() + size.x() && start.y() <= p.y() && p.y() <= start.y() + size.y();
  }
};

// GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector() = default;
GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector() : gftt_() {}
GoodFeaturesToTrackDetector::~GoodFeaturesToTrackDetector() = default;

void GoodFeaturesToTrackDetector::cutBinFromGFFT(const Vector2N& start, size_t binW, size_t binH,
                                                 GoodFeaturesToTrackDetector::Bin& bin, const ImageMatrixT& strength) {
  bin.start = start;
  bin.size = Vector2N(binW, binH);
  bin.pixels.resize(binW * binH);

  float acc = 0;

  int i = 0;

  // Heap should only contain local maximums,
  // so we are not going to add a pixel to the heap if its left
  // neighbor (if there is one) has higher strength.
  // Smallest possible strength is zero.
  float prev_strength = 0.f;

  for (size_t y = start.y(); y < binH + start.y(); ++y) {
    prev_strength = 0.f;
    for (size_t x = start.x(); x < binW + start.x(); ++x) {
      const float v = strength(y, x);
      bin.pixels[i] = static_cast<int32_t>(x + y * strength.cols());
      acc += v;

      if (v >= prev_strength) {
        ++i;
      }

      prev_strength = v;
    }
  }

  bin.accGFTT = acc;
  bin.pixels.resize(i);

  const float* strength_values = strength.data();
  auto comp = [strength_values](const int32_t& a, const int32_t& b) -> bool {
    return strength_values[a] < strength_values[b];
  };
  std::make_heap(bin.pixels.begin(), bin.pixels.end(), comp);
}

void GoodFeaturesToTrackDetector::splitGFFTToBins(size_t nRows, size_t nCols, const ImageMatrixT& strength) {
  const size_t binH = strength.rows() / nRows;
  const size_t binW = strength.cols() / nCols;
  const size_t nBins = nRows * nCols;  // cut up to 1.5% pixels worst case
  bins_.resize(nBins);

  size_t i = 0;

  for (size_t y = 0; y < nRows; ++y) {
    for (size_t x = 0; x < nCols; ++x) {
      cutBinFromGFFT(Vector2N(x * binW, y * binH), binW, binH, bins_[i], strength);
      ++i;
    }
  }

  assert(i == nBins);
}

float GoodFeaturesToTrackDetector::calculateSummGFTT() const noexcept {
  float summGFTT = 0;

  for (const Bin& bin : bins_) {
    summGFTT += bin.accGFTT;
  }

  return summGFTT;
}

void GoodFeaturesToTrackDetector::mask(size_t x, size_t y, size_t half) noexcept {
  const int startRow = std::max(0, (int)y - (int)half);
  const int startCol = std::max(0, (int)x - (int)half);
  const int endRow = std::min((int)mask_.rows() - 1, (int)y + (int)half);
  const int endCol = std::min((int)mask_.cols() - 1, (int)x + (int)half);
  mask_.block(startRow, startCol, endRow - startRow + 1, endCol - startCol + 1).setConstant(true);
}

void GoodFeaturesToTrackDetector::mask_border(int border_top, int border_bottom, int border_left,
                                              int border_right) noexcept {
  const int rows = mask_.rows();
  const int cols = mask_.cols();
  assert(rows > border_top && cols > border_left && rows - 1 > border_bottom && cols - 1 > border_right);

  mask_.block(0, 0, border_top, cols).setConstant(1);
  mask_.block(rows - 1 - border_bottom, 0, border_bottom, cols).setConstant(1);
  mask_.block(0, 0, rows, border_left).setConstant(1);
  mask_.block(0, cols - 1 - border_right, rows, border_right).setConstant(1);
}

static bool isMaximum3x3(const ImageMatrixT& smartGFFT, const Vector2N& c) noexcept {
  const int nRows = (int)smartGFFT.rows();
  const int nCols = (int)smartGFFT.cols();

  const int startRow = std::max(0, (int)c.y() - 1);
  const int startCol = std::max(0, (int)c.x() - 1);
  const int endRow = std::min(nRows - 1, (int)c.y() + 1);
  const int endCol = std::min(nCols - 1, (int)c.x() + 1);
  const float maxCoeff = smartGFFT.block(startRow, startCol, endRow - startRow + 1, endCol - startCol + 1).maxCoeff();
  // We don't know exactly if > or >= is required
  return maxCoeff == smartGFFT(c.y(), c.x());
}

void GoodFeaturesToTrackDetector::computeGFTTAndSelectFeatures(
    const GradientPyramidT& gradientPyramid, int border_top, int border_bottom, int border_left, int border_right,
    const ImageMatrix<uint8_t>* input_mask, const TracksVector& alreadySelectedFeatures,
    const size_t desiredNewFeaturesCount, std::vector<Vector2T>& newSelectedFeatures, size_t nBinX, size_t nBinY,
    size_t burnHalfSize, size_t alreadySelectedBurnHalfSize) {
  TRACE_EVENT ev = profiler_domain_.trace_event("computeGFTTAndSelectFeatures()", profiler_color_);

  const size_t level = 0;
  if (gradientPyramid.isGradientsLazyEvaluated(level)) {
    gradientPyramid.forceNonLazyEvaluation(level);
  }

  const ImageMatrixT& gradX = gradientPyramid.gradX()[level];
  const ImageMatrixT& gradY = gradientPyramid.gradY()[level];

  {
    TRACE_EVENT ev_gftt = profiler_domain_.trace_event("gftt_.compute()", profiler_color_);
    gftt_.compute(gradX, gradY);
  }

  {
    TRACE_EVENT ev_select = profiler_domain_.trace_event("selectFeatures()", profiler_color_);
    selectFeatures(gftt_.get(), input_mask, border_top, border_bottom, border_left, border_right,
                   alreadySelectedFeatures, desiredNewFeaturesCount, newSelectedFeatures, nBinX, nBinY, burnHalfSize,
                   alreadySelectedBurnHalfSize);
  }
}

void GoodFeaturesToTrackDetector::selectFeatures(const ImageMatrixT& imageGFTT, const ImageMatrix<uint8_t>* input_mask,
                                                 int border_top, int border_bottom, int border_left, int border_right,
                                                 const TracksVector& alreadySelectedFeatures,
                                                 const size_t desiredNewFeaturesCount,
                                                 std::vector<Vector2T>& newSelectedFeatures, size_t nBinX, size_t nBinY,
                                                 size_t burnHalfSize, size_t alreadySelectedBurnHalfSize) {
  const size_t n_rows = imageGFTT.rows();
  const size_t n_cols = imageGFTT.cols();
  mask_.resize(n_rows, n_cols);
  if (input_mask) {
    mask_ = *input_mask;
  } else {
    mask_.setConstant(0);
  }

  mask_border(border_top, border_bottom, border_left, border_right);

  for (size_t i = 0; i < alreadySelectedFeatures.size(); ++i) {
    const Track& track = alreadySelectedFeatures[i];

    if (!track.dead()) {
      const Vector2T& p = track.position();
      mask(static_cast<size_t>(p.x()), static_cast<size_t>(p.y()), alreadySelectedBurnHalfSize);
    }
  }

  splitGFFTToBins(nBinX, nBinY, imageGFTT);
  const float summGFTT = calculateSummGFTT();

  const size_t nExpectedTracks = alreadySelectedFeatures.get_num_alive() + desiredNewFeaturesCount;

  for (Bin& bin : bins_) {
    size_t nFeaturesFromPrevFrame = 0;

    for (size_t i = 0; i < alreadySelectedFeatures.size(); ++i) {
      const Track& track = alreadySelectedFeatures[i];

      if (!track.dead()) {
        if (bin.isInside(track.position())) {
          ++nFeaturesFromPrevFrame;
        }
      }
    }

    if (summGFTT == 0) {
      continue;
    }

    // we want to apportion points according to the total strength of each bin
    const auto nExpectedFeatures = static_cast<size_t>(std::round(nExpectedTracks * bin.accGFTT / summGFTT));
    size_t nFeatures = 0;

    if (nExpectedFeatures > nFeaturesFromPrevFrame) {
      nFeatures = nExpectedFeatures - nFeaturesFromPrevFrame;
    }

    size_t nAdded = 0;

    // get best features as bin.pixels sorted by gftt
    for (; !bin.pixels.empty() && nAdded < nFeatures;) {
      const float* gftt_values = imageGFTT.data();
      auto comp = [gftt_values](const int32_t& a, const int32_t& b) -> bool { return gftt_values[a] < gftt_values[b]; };

      std::pop_heap(bin.pixels.begin(), bin.pixels.end(), comp);
      const int32_t index = bin.pixels.back();
      bin.pixels.pop_back();

      const Vector2N c(index % imageGFTT.cols(), index / imageGFTT.cols());

      if (mask_(c.y(), c.x()) || !isMaximum3x3(imageGFTT, c)) {
        continue;
      }

      if (newSelectedFeatures.size() >= desiredNewFeaturesCount) {
        return;  // we never need to return more points than desired
      }

      newSelectedFeatures.push_back(refineFeaturePosition(c, imageGFTT));

      mask(c.x(), c.y(), burnHalfSize);
      ++nAdded;
    }
  }
}

}  // namespace cuvslam::sof

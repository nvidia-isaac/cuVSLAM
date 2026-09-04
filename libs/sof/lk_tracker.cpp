
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

#include "sof/lk_tracker.h"

#include "camera/observation.h"

#include "sof/gaussian_coefficients.h"

namespace cuvslam::sof {

const int MAX_POINT_TRACK_ITERATIONS = 10;
const float LK_CONVERGENCE_THRESHOLD = 0.08f;

void LKFeatureTracker::computeTGradient(LKFeatureTracker::FeaturePatch& tPatch,
                                        const LKFeatureTracker::FeaturePatch& img1,
                                        const LKFeatureTracker::FeaturePatch& img2) {
  const float delta = (img2.sum() - img1.sum()) / FeaturePatch::SizeAtCompileTime;

  // @TODO: skipped calc for contrast compensation. Add if proved to be needed after testing
  tPatch = img2 - img1 - FeaturePatch::Constant(delta);
}

float LKFeatureTracker::ncc(const LKFeatureTracker::FeaturePatch& patch1,
                            const LKFeatureTracker::FeaturePatch& patch2) {
  enum { NCC_DIM = 5 };

  assert(patch1.size() == patch2.size() && patch1.cols() == patch1.rows() && patch2.cols() == patch2.rows());
  assert(patch1.cols() >= NCC_DIM);

  const Vector2N start = Vector2N::Constant((patch1.rows() - NCC_DIM) / 2);

  const ImageMatrixPatch<float, NCC_DIM, NCC_DIM> img1 = patch1.block<NCC_DIM, NCC_DIM>(start.y(), start.x());
  const ImageMatrixPatch<float, NCC_DIM, NCC_DIM> img2 = patch2.block<NCC_DIM, NCC_DIM>(start.y(), start.x());

  const auto size = static_cast<float>(img1.size());

  const float a1 = img1.sum() / size;
  const float a2 = img2.sum() / size;

  const float d1 = img1.cwiseProduct(img1).sum() / size - a1 * a1;
  const float d2 = img2.cwiseProduct(img2).sum() / size - a2 * a2;
  const float d12 = img1.cwiseProduct(img2).sum() / size - a1 * a2;

  const float d1d2 = d1 * d2;
  return d1d2 < epsilon() ? 0 : d12 / std::sqrt(d1d2);
}

LKFeatureTracker::LKFeatureTracker()
    : gauss2DKernel_(
          (PatchColVector(GaussianWeightedFeatureCoeffs) * PatchRowVector(GaussianWeightedFeatureCoeffs)).eval()) {}

bool LKFeatureTracker::trackPoint(const GradientPyramidT& previous_image_gradient,
                                  const GradientPyramidT& /*current_image_gradient*/,
                                  const ImagePyramidT& previous_image, const ImagePyramidT& current_image,
                                  const Vector2T& previous_uv, Vector2T& current_uv, Matrix2T& current_info,
                                  float search_radius_px, float ncc_threshold) const {
  Vector2T offset = current_uv - previous_uv;
  current_uv = previous_uv;

  bool status = trackPoint(previous_image_gradient, previous_image, current_image, current_uv, offset, current_info,
                           search_radius_px, ncc_threshold);

  return status;
}

bool LKFeatureTracker::trackPoint(const GradientPyramidT& prevFrameGradPyramid,
                                  const ImagePyramidT& prevFrameImagePyramid,
                                  const ImagePyramidT& currentFrameImagePyramid, cuvslam::Vector2T& track,
                                  const cuvslam::Vector2T& offset, Matrix2T& info, float search_radius_px,
                                  float ncc_threshold) const {
  if (search_radius_px < 0.f) {
    return false;
  }

  info = camera::GetDefaultObservationInfoUV();

  const int levels = prevFrameGradPyramid.getLevelsCount();

  const int topLevelIndex = levels - 1;  // smallest image

  if (!prevFrameImagePyramid.isPointInImage(track, 0) ||
      !currentFrameImagePyramid.isPointInImage(Vector2T(track + offset), 0)) {
    // points outside image plane (ex: by offset hack)
    return false;
  }

  Vector2T xy1 = prevFrameImagePyramid.ScaleDownPoint(track, topLevelIndex);
  Vector2T xy2 = currentFrameImagePyramid.ScaleDownPoint(Vector2T(track + offset), topLevelIndex);

  bool isConverged = false;

  for (int i = 0; i < levels; ++i) {
    assert(topLevelIndex >= i);
    const int level = topLevelIndex - i;
    FeaturePatch gradPatchX, gradPatchY;

    if (!prevFrameGradPyramid.getGradXYPatches(gradPatchX, gradPatchY, xy1, level)) {
      ;
    } else {
      gradPatchX.array() -= gradPatchX.sum() / (float)gradPatchX.size();
      gradPatchY.array() -= gradPatchY.sum() / (float)gradPatchY.size();

      const float gxx = gauss2DKernel_(gradPatchX.cwiseProduct(gradPatchX)) + 1.f;
      const float gxy = gauss2DKernel_(gradPatchX.cwiseProduct(gradPatchY));
      const float gyy = gauss2DKernel_(gradPatchY.cwiseProduct(gradPatchY)) + 1.f;

      const float determinant = gxx * gyy - gxy * gxy;

      if (determinant < epsilon()) {
        return false;
      }

      FeaturePatch imagePatch1, imagePatch2;

      if (prevFrameImagePyramid.computePatch(imagePatch1, xy1, level)) {
        const Vector2T xy2Save = xy2;

        for (int iterations = 0; iterations < MAX_POINT_TRACK_ITERATIONS; iterations++) {
          if (!currentFrameImagePyramid.computePatch(imagePatch2, xy2, level)) {
            break;
          }

          FeaturePatch tPatch;
          computeTGradient(tPatch, imagePatch1, imagePatch2);

          const float ex = gauss2DKernel_(gradPatchX.cwiseProduct(tPatch));
          const float ey = gauss2DKernel_(gradPatchY.cwiseProduct(tPatch));
          const Vector2T v((gyy * ex - gxy * ey) / determinant, (gxx * ey - gxy * ex) / determinant);
          isConverged = (v.norm() < LK_CONVERGENCE_THRESHOLD);

          if (isConverged) {
            // TODO: investigate info matrix calculation,
            // now it's not correct and requires to use default info matrix
            // info << gxx, gxy, gxy, gyy;
            break;
          }

          xy2 += v;
        }

        if (isConverged) {
          const float level_radius = search_radius_px / static_cast<float>(1 << level);
          if ((xy2 - xy2Save).norm() > level_radius) {
            isConverged = false;
          }
        }

        const bool isNCC = isConverged && ncc(imagePatch1, imagePatch2) > ncc_threshold;

        if (!isNCC && level > 0) {
          xy2 = xy2Save;
        }

        isConverged = (level == 0) && isNCC;
      }  // can computePatch
    }    // can getGradXYPatches

    // go to next level
    if (i != levels - 1) {
      assert(level > 0);
      xy1 = prevFrameImagePyramid.ScaleUpPointToNextLevel(xy1);
      xy2 = currentFrameImagePyramid.ScaleUpPointToNextLevel(xy2);
    }
  }

  if (isConverged) {
    track = xy2;
  }

  return isConverged;
}

bool LKTrackerHorizontal::trackPoint(const GradientPyramidT& prevFrameGradPyramid,
                                     const ImagePyramidT& prevFrameImagePyramid,
                                     const ImagePyramidT& currentFrameImagePyramid, cuvslam::Vector2T& track,
                                     const cuvslam::Vector2T& offset, Matrix2T& info, float search_radius_px,
                                     float ncc_threshold) const {
  if (search_radius_px < 0.f) {
    return false;
  }

  info = camera::GetDefaultObservationInfoUV();

  const int levels = prevFrameGradPyramid.getLevelsCount();

  const int topLevelIndex = levels - 1;  // smallest image

  if (!prevFrameImagePyramid.isPointInImage(track, 0) ||
      !currentFrameImagePyramid.isPointInImage(Vector2T(track + offset), 0)) {
    // points outside image plane (ex: by offset hack)
    return false;
  }

  Vector2T xy1 = prevFrameImagePyramid.ScaleDownPoint(track, topLevelIndex);
  Vector2T xy2 = currentFrameImagePyramid.ScaleDownPoint(Vector2T(track + offset), topLevelIndex);

  bool isConverged = false;

  for (int i = 0; i < levels; ++i) {
    assert(topLevelIndex >= i);
    const int level = topLevelIndex - i;
    FeaturePatch gradPatchX;

    if (!prevFrameGradPyramid.getGradXPatch(gradPatchX, xy1, level)) {
      ;
    } else {
      gradPatchX.array() -= gradPatchX.sum() / (float)gradPatchX.size();

      const float gxx = gauss2DKernel_(gradPatchX.cwiseProduct(gradPatchX)) + 1.f;

      if (gxx < epsilon()) {
        return false;
      }

      FeaturePatch imagePatch1, imagePatch2;

      if (prevFrameImagePyramid.computePatch(imagePatch1, xy1, level)) {
        const Vector2T xy2Save = xy2;

        for (int iterations = 0; iterations < MAX_POINT_TRACK_ITERATIONS; iterations++) {
          if (!currentFrameImagePyramid.computePatch(imagePatch2, xy2, level)) {
            break;
          }

          FeaturePatch tPatch;
          computeTGradient(tPatch, imagePatch1, imagePatch2);

          const float ex = gauss2DKernel_(gradPatchX.cwiseProduct(tPatch));
          const Vector2T v(ex / gxx, 0.0f);
          isConverged = (v.norm() < LK_CONVERGENCE_THRESHOLD);

          if (isConverged) {
            // TODO: investigate info matrix calculation,
            // now it's not correct and requires to use default info matrix
            // info << gxx, 0.0f, 0.0f, gxx;
            break;
          }

          xy2 += v;
        }

        if (isConverged) {
          const float level_radius = search_radius_px / static_cast<float>(1 << level);
          if ((xy2 - xy2Save).norm() > level_radius) {
            isConverged = false;
          }
        }

        const bool isNCC = isConverged && ncc(imagePatch1, imagePatch2) > ncc_threshold;

        if (!isNCC && level > 0) {
          xy2 = xy2Save;
        }

        isConverged = (level == 0) && isNCC;
      }  // can computePatch
    }    // can getGradXPatch

    // go to next level
    if (i != levels - 1) {
      assert(level > 0);
      xy1 = prevFrameImagePyramid.ScaleUpPointToNextLevel(xy1);
      xy2 = currentFrameImagePyramid.ScaleUpPointToNextLevel(xy2);
    }
  }

  if (isConverged) {
    track = xy2;
  }

  return isConverged;
}

}  // namespace cuvslam::sof

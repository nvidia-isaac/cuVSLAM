
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

#include "sof/klt_tracker.h"

#include "camera/observation.h"
#include "common/unaligned_types.h"

#include "sof/gaussian_coefficients.h"

namespace cuvslam::sof {

void KLTTracker::compute_residual(KLTTracker::FeaturePatch& residual, const KLTTracker::FeaturePatch& img1,
                                  const KLTTracker::FeaturePatch& img2) {
  const float delta = (img2.sum() - img1.sum()) / (float)FeaturePatch::SizeAtCompileTime;
  residual = img2 - img1 - FeaturePatch::Constant(delta);
}

float KLTTracker::compute_ncc(const KLTTracker::FeaturePatch& patch1, const KLTTracker::FeaturePatch& patch2) {
  int NCC_DIM = 5;

  assert(patch1.size() == patch2.size() && patch1.cols() == patch1.rows() && patch2.cols() == patch2.rows());
  assert(patch1.cols() >= NCC_DIM);

  const Vector2N start = Vector2N::Constant((patch1.rows() - NCC_DIM) / 2);

  const auto a = patch1.block(start.y(), start.x(), NCC_DIM, NCC_DIM);
  const auto b = patch2.block(start.y(), start.x(), NCC_DIM, NCC_DIM);

  const auto size = static_cast<float>(a.size());

  const float mean_a = a.sum() / size;
  const float mean_b = b.sum() / size;

  const float var_a = a.cwiseProduct(a).sum() / size - mean_a * mean_a;
  const float var_b = b.cwiseProduct(b).sum() / size - mean_b * mean_b;
  const float cov = a.cwiseProduct(b).sum() / size - mean_a * mean_b;

  const float vavb = var_a * var_b;
  return vavb < epsilon() ? 0 : cov / std::sqrt(vavb);
}

KLTTracker::KLTTracker()
    : weights_(
          (Eigen::Matrix<float, FeaturePatch::RowsAtCompileTime, 1, Eigen::DontAlign>(GaussianWeightedFeatureCoeffs) *
           Eigen::Matrix<float, 1, FeaturePatch::ColsAtCompileTime>(GaussianWeightedFeatureCoeffs))
              .eval()) {}

bool KLTTracker::trackPoint(const GradientPyramidT&, const GradientPyramidT& current_image_gradients,
                            const ImagePyramidT& previous_image, const ImagePyramidT& current_image,
                            const Vector2T& previous_uv, Vector2T& current_uv, Matrix2T& current_info,
                            float search_radius_px, float ncc_threshold) const {
  const Vector2T offset = current_uv - previous_uv;
  current_uv = previous_uv;

  bool status = track_point(current_image_gradients, previous_image, current_image, current_uv, offset, current_info,
                            search_radius_px, ncc_threshold);
  return status;
}

bool KLTTracker::refine_position(const FeaturePatch& previous_patch, const GradientPyramidT& current_image_gradient,
                                 const ImagePyramidT& current_image, const int level, Vector2T& xy,
                                 Matrix2T& info) const {
  info = camera::GetDefaultObservationInfoUV();

  FeaturePatch gx, gy;
  FeaturePatch current_patch;

  if (!current_image_gradient.getGradXYPatches(gx, gy, xy, level)) {
    return false;
  }

  // Account for the fact that we subtract difference of means:
  // gx and gy are the entries of the jacobian matrix of the residual.
  gx.array() -= gx.sum() / (float)gx.size();
  gy.array() -= gy.sum() / (float)gy.size();

  float gxgx = weights_(gx.cwiseProduct(gx));
  float gxgy = weights_(gx.cwiseProduct(gy));
  float gygy = weights_(gy.cwiseProduct(gy));

  FeaturePatch residual;

  float current_cost;
  if (!compute_cost(&current_cost, current_patch, residual, current_image, previous_patch, xy, level)) {
    return false;
  }

  if (current_cost < epsilon()) {
    return true;
  }

  // Arbitrary, not tuned.
  float lambda = 0.001f;

  Vector2T scaling(gxgx, gygy);

  // In synthetic tests converges in about 20 iterations.
  constexpr int kMaxIterations = 10;
  for (int iterations = 0; iterations < kMaxIterations; ++iterations) {
    // hessian matrix of the cost function
    Matrix2T hessian;
    hessian << gxgx, gxgy, gxgy, gygy;

    // negative gradient of the cost function, not the image
    Vector2T negative_gradient;
    negative_gradient << weights_(gx.cwiseProduct(residual)), weights_(gy.cwiseProduct(residual));

    Matrix2T lhs = hessian;
    lhs.diagonal() += scaling * lambda;

    Eigen::JacobiSVD<Matrix2T, Eigen::NoQRPreconditioner> usv(lhs, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Vector2T step = usv.solve(negative_gradient);

    float cost;
    if (!compute_cost(&cost, current_patch, residual, current_image, previous_patch, xy + step, level)) {
      // We took a step too far for one of the two reasons:
      // - it is a valid step and the point moved outside of the image
      // - it is an invalid step
      // We can't tell between the two cases so we exit to err on the safe side.
      return false;
    }

    auto predicted_rel_reduction =
        step.dot(hessian * step) / current_cost + 2.f * lambda * step.dot(scaling.cwiseProduct(step)) / current_cost;

    // check for convergence
    if ((predicted_rel_reduction < sqrt_epsilon()) && (step.norm() < sqrt_epsilon())) {
      return true;
    }

    auto rho = (1.f - cost / current_cost) / predicted_rel_reduction;

    if (rho > 0.25f) {
      xy += step;
      current_cost = cost;
      // TODO: investigate info matrix calculation,
      // now it's not correct and requires to use default info matrix
      // info = hessian;

      if (current_cost < epsilon()) {
        return true;
      }

      if (rho > 0.75f) {
        lambda *= 0.125f;
      }

      // Update linear model.
      // TODO: we may not need to this often.
      current_image_gradient.getGradXYPatches(gx, gy, xy, level);

      gx.array() -= gx.sum() / (float)gx.size();
      gy.array() -= gy.sum() / (float)gy.size();

      gxgx = weights_(gx.cwiseProduct(gx));
      gxgy = weights_(gx.cwiseProduct(gy));
      gygy = weights_(gy.cwiseProduct(gy));

      scaling = scaling.cwiseMax(Vector2T(gxgx, gygy));
    } else {
      lambda *= 5.f;

      // restore patch and residual we overwritten while computing cost for xy + step
      if (!compute_cost(nullptr, current_patch, residual, current_image, previous_patch, xy, level)) {
        return false;
      }
    }
  }

  return true;
}

bool KLTTracker::compute_cost(float* cost, KLTTracker::FeaturePatch& current_patch, KLTTracker::FeaturePatch& residual,
                              const ImagePyramidT& current_image, const FeaturePatch& previous_patch,
                              const cuvslam::Vector2T& xy, int level) const {
  if (!current_image.computePatch(current_patch, xy, level)) {
    return false;
  }

  compute_residual(residual, previous_patch, current_patch);

  if (cost) {
    *cost = weights_(residual.cwiseProduct(residual));
  }

  return true;
}

bool KLTTracker::track_point(const GradientPyramidT& current_image_gradient, const ImagePyramidT& previous_image,
                             const ImagePyramidT& current_image, cuvslam::Vector2T& track,
                             const cuvslam::Vector2T& offset, Matrix2T& info, float search_radius_px,
                             float ncc_threshold) const {
  if (search_radius_px < 0.f) {
    return false;
  }

  const int num_levels = current_image_gradient.getLevelsCount();

  assert(num_levels > 0 && num_levels == previous_image.getLevelsCount() &&
         num_levels == current_image.getLevelsCount());

  // Taylor-series approximation is precise in a 1-pixel radius.
  // Each level of the pyramid "doubles" the radius of our approximation,
  // so we turn search radius into the number of levels by taking log
  // and truncating the value to the nearest integer.
  const int coarsest_level = std::min(num_levels - 1, static_cast<int>(std::log1p(search_radius_px)));

  assert(coarsest_level >= 0);
  assert(coarsest_level < num_levels);

  if (!previous_image.isPointInImage(track, 0)) {
    return false;
  }

  Vector2T current_uv = current_image.ScaleDownPoint(Vector2T(track + offset), coarsest_level);

  for (int level = coarsest_level; level >= 0; --level) {
    Vector2T previous_uv = previous_image.ScaleDownPoint(track, level);
    FeaturePatch previous_patch, current_patch;

    if (!current_image.isPointInImage(current_uv, level)) {
      return false;
    }

    if (!previous_image.computePatch(previous_patch, previous_uv, level)) {
      return false;
    }

    const Vector2T current_uv_to_save = current_uv;

    if (!refine_position(previous_patch, current_image_gradient, current_image, level, current_uv, info)) {
      if (level > 0) {
        current_uv = current_image.ScaleUpPointToNextLevel(current_uv_to_save);
        continue;
      } else {
        return false;
      }
    }

    if (!current_image.computePatch(current_patch, current_uv, level)) {
      return false;
    }

    const auto ncc_score = compute_ncc(previous_patch, current_patch);
    const bool is_match = ncc_score > ncc_threshold;

    if (!is_match) {
      if (level > 0) {
        current_uv = current_uv_to_save;
      } else {
        return false;
      }
    }

    if (level > 0) {
      current_uv = current_image.ScaleUpPointToNextLevel(current_uv);
    }
  }

  track = current_uv;

  return current_image.isPointInImage(track, 0);
}

bool KLTTrackerHorizontal::refine_position(const FeaturePatch& previous_patch,
                                           const GradientPyramidT& current_image_gradient,
                                           const ImagePyramidT& current_image, const int level, Vector2T& xy,
                                           Matrix2T& info) const {
  info.setZero();

  FeaturePatch gx;
  FeaturePatch current_patch;

  if (!current_image_gradient.getGradXPatch(gx, xy, level)) {
    return false;
  }

  // Account for the fact that we subtract difference of means:
  // gx and gy are the entries of the jacobian matrix of the residual.
  gx.array() -= gx.sum() / (float)gx.size();

  float gxgx = weights_(gx.cwiseProduct(gx));

  if (gxgx < epsilon()) {
    return false;
  }

  FeaturePatch residual;

  float current_cost;
  if (!compute_cost(&current_cost, current_patch, residual, current_image, previous_patch, xy, level)) {
    return false;
  }

  if (current_cost < epsilon()) {
    return true;
  }

  // Arbitrary, not tuned.
  float lambda = 0.001f;

  // In synthetic tests converges in about 20 iterations.
  constexpr int kMaxIterations = 10;
  for (int iterations = 0; iterations < kMaxIterations; ++iterations) {
    // negative gradient of the cost function, not the image
    float negative_gradient = weights_(gx.cwiseProduct(residual));

    float lhs = gxgx;
    lhs += gxgx * lambda;

    const Vector2T step(negative_gradient / lhs, 0);

    float cost;
    if (!compute_cost(&cost, current_patch, residual, current_image, previous_patch, xy + step, level)) {
      // We took a step too far for one of the two reasons:
      // - it is a valid step and the point moved outside of the image
      // - it is an invalid step
      // We can't tell between the two cases so we exit to err on the safe side.
      return false;
    }

    auto predicted_rel_reduction =
        step.dot(gxgx * step) / current_cost + 2.f * lambda * step.dot(gxgx * step) / current_cost;

    // check for convergence
    if ((predicted_rel_reduction < sqrt_epsilon()) && (step.norm() < sqrt_epsilon())) {
      return true;
    }

    auto rho = (1.f - cost / current_cost) / predicted_rel_reduction;

    if (rho > 0.25f) {
      xy += step;
      current_cost = cost;
      info(0, 0) = gxgx;
      info(1, 1) = gxgx;

      if (current_cost < epsilon()) {
        return true;
      }

      if (rho > 0.75f) {
        lambda *= 0.125f;
      }

      // Update linear model.
      // TODO: we may not need to this often.
      current_image_gradient.getGradXPatch(gx, xy, level);

      gx.array() -= gx.sum() / (float)gx.size();
    } else {
      lambda *= 5.f;

      // restore patch and residual we overwritten while computing cost for xy + step
      if (!compute_cost(nullptr, current_patch, residual, current_image, previous_patch, xy, level)) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace cuvslam::sof

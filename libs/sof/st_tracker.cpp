
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

#include "sof/st_tracker.h"

#include "camera/observation.h"
#include "common/log_types.h"
#include "common/types.h"
#include "common/unaligned_types.h"

#include "sof/gaussian_coefficients.h"
#include "sof/gradient_pyramid.h"
#include "sof/image_pyramid_float.h"
#include "sof/kernel_operator.h"

namespace cuvslam::sof {

namespace {

const float ST_CONVERGENCE_THRESHOLD = 0.05f;

constexpr auto epsilon = cuvslam::epsilon();
using FeaturePatch = cuvslam::ImageMatrixPatch<float, 9, 9>;
constexpr auto patch_cols = FeaturePatch::ColsAtCompileTime;
constexpr auto patch_rows = FeaturePatch::RowsAtCompileTime;

FeaturePatch make_x_patch() {
  FeaturePatch patch;
  const int m = (patch_cols - 1) / 2;
  for (int y = 0; y < patch_rows; ++y) {
    for (int x = 0; x < patch_cols; ++x) {
      patch(y, x) = static_cast<float>(x - m);
    }
  }
  return patch;
}

FeaturePatch make_y_patch() {
  FeaturePatch patch;
  const int m = (patch_rows - 1) / 2;
  for (int y = 0; y < patch_rows; ++y) {
    for (int x = 0; x < patch_cols; ++x) {
      patch(y, x) = static_cast<float>(y - m);
    }
  }
  return patch;
}

const FeaturePatch x_patch = make_x_patch();
const FeaturePatch y_patch = make_y_patch();
const FeaturePatch xx_patch = x_patch.cwiseProduct(x_patch);
const FeaturePatch yy_patch = y_patch.cwiseProduct(y_patch);
const FeaturePatch xy_patch = x_patch.cwiseProduct(y_patch);

// bilinear interpolation
float get_pixel(const ImagePyramidT::ImageType& image, const Vector2T& xy) {
  const float x = xy.x();
  const float y = xy.y();

  assert(x >= 0);
  assert(y >= 0);

  const int ix = static_cast<int>(x);  // floor
  const int iy = static_cast<int>(y);  // floor

  const float dx = x - static_cast<float>(ix);
  const float dy = y - static_cast<float>(iy);

  // interpolation coefficients
  assert(0 <= dx && dx < 1);
  assert(0 <= dy && dy < 1);

  const float c00 = (1.f - dx) * (1.f - dy);
  const float c01 = (1.f - dx) * dy;
  const float c10 = dx * (1.f - dy);
  const float c11 = dx * dy;

  return c00 * image(iy, ix) + c01 * image(iy + 1, ix) + c10 * image(iy, ix + 1) + c11 * image(iy + 1, ix + 1);
}

template <typename ImagePyramidT>
bool compute_patch(FeaturePatch& patch, const ImagePyramidT& image_pyramid, const Matrix2T& map, const Vector2T& xy,
                   int level) {
  // an optimization
  if (map == Matrix2T::Identity()) {
    return image_pyramid.computePatch(patch, xy, level);
  }

  const typename ImagePyramidT::ImageType& image = image_pyramid[level];

  // shift so center of the pixel has .0 coordinates
  // and shift so (x = 0, y = 0) has .0 coordinates
  const Vector2T xy00 = xy - Vector2T::Constant(0.5f) - map * Vector2T(patch_cols - 1, patch_rows - 1) / 2.f;
  const Vector2T xy01 = xy00 + map * Vector2T(0, patch_rows - 1);
  const Vector2T xy10 = xy00 + map * Vector2T(patch_cols - 1, 0);
  const Vector2T xy11 = xy00 + map * Vector2T(patch_cols - 1, patch_rows - 1);

  auto minmax_x = std::minmax<float>({xy00.x(), xy01.x(), xy10.x(), xy11.x()});
  auto minmax_y = std::minmax<float>({xy00.y(), xy01.y(), xy10.y(), xy11.y()});

  if (!(0 <= minmax_x.first && minmax_x.second < image.cols() - 1)) {
    return false;
  }
  if (!(0 <= minmax_y.first && minmax_y.second < image.rows() - 1)) {
    return false;
  }

  for (int y = 0; y < patch_rows; ++y) {
    for (int x = 0; x < patch_cols; ++x) {
      patch(y, x) = get_pixel(image, xy00 + map * Vector2T(x, y));
    }
  }

  return true;
}

bool compute_gradient_patches(FeaturePatch& gx_patch, FeaturePatch& gy_patch, const GradientPyramidT& gradient,
                              const Matrix2T& map, const Vector2T& xy, const int level) {
  assert(patch_cols == patch_rows);
  gradient.calcPatchGradients(xy, patch_cols, level);

  return compute_patch(gx_patch, gradient.gradX(), map, xy, level) &&
         compute_patch(gy_patch, gradient.gradY(), map, xy, level);
}

void compute_residual(FeaturePatch& residual, const FeaturePatch& patch1, const FeaturePatch& patch2) {
  const float delta = (patch1.sum() - patch2.sum()) / static_cast<float>(FeaturePatch::SizeAtCompileTime);

  // @TODO: skipped calc for contrast compensation. Add if proved to be needed after testing
  residual = patch1 - patch2 - FeaturePatch::Constant(delta);
}

//
// weights pixels within a patch
const cuvslam::sof::KernelOperator<FeaturePatch> gaussian_weights(
    (Eigen::Matrix<float, patch_rows, 1, Eigen::DontAlign>(cuvslam::sof::GaussianWeightedFeatureCoeffs) *
     Eigen::Matrix<float, 1, patch_cols>(cuvslam::sof::GaussianWeightedFeatureCoeffs))
        .eval());

float compute_ncc(const FeaturePatch& patch1, const FeaturePatch& patch2) {
#ifdef SOF_USE_SMALLER_NCC
  constexpr int NCC_DIM = 3;
#else
  constexpr int NCC_DIM = 5;
#endif

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
  return vavb < epsilon ? 0 : cov / std::sqrt(vavb);
}

// Tracks a single feature patch across an image pyramid. Deliberately not named Tracker: the public
// cuvslam::Tracker is visible here through cuvslam2.h, and a local Tracker would silently shadow it.
class PatchTracker {
public:
  PatchTracker(const GradientPyramidT& current_image_gradient, const ImagePyramidT& current_image,
               unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations, Matrix2T& info, float& ncc)
      : current_image_gradient(current_image_gradient),
        current_image(current_image),
        n_shift_only_iterations(n_shift_only_iterations),
        n_full_mapping_iterations(n_full_mapping_iterations),
        info(info),
        ncc(ncc),
        map(),
        xy() {
    assert(current_image_gradient.getLevelsCount() > 0);
    assert(current_image.getLevelsCount() > 0);
  }

  bool trackPoint(const ImagePyramidT& previous_image, Vector2T& track, const Vector2T& offset, float search_radius_px,
                  float ncc_threshold) {
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

    Vector2T previous_uv = current_image.ScaleDownPoint(Vector2T(track), coarsest_level);
    map = Matrix2T::Identity();
    xy = current_image.ScaleDownPoint(Vector2T(track + offset), coarsest_level);

    bool isConverged = false;

    for (int level = coarsest_level; level >= 0; --level) {
      if (!previous_image.isPointInImage(previous_uv, level)) {
        return false;
      }

      if (!current_image.isPointInImage(xy, level)) {
        return false;
      }

      FeaturePatch previous_patch;
      if (!previous_image.computePatch(previous_patch, previous_uv, level)) {
        if (level > 0) {
          xy = current_image.ScaleUpPointToNextLevel(xy);
          previous_uv = current_image.ScaleUpPointToNextLevel(previous_uv);
          continue;
        }
      }

      if (!refineMapping(previous_patch, level, isConverged)) {
        if (level > 0) {
          xy = current_image.ScaleUpPointToNextLevel(xy);
          previous_uv = current_image.ScaleUpPointToNextLevel(previous_uv);
          continue;
        } else {
          return false;
        }
      }

      FeaturePatch current_patch;
      if (!compute_patch(current_patch, current_image, map, xy, level)) {
        return false;
      }

      if (level == 0) {
        this->ncc = compute_ncc(previous_patch, current_patch);
        if (this->ncc <= ncc_threshold || !isConverged) {
          return false;
        }
      }

      if (level > 0) {
        xy = current_image.ScaleUpPointToNextLevel(xy);
        previous_uv = current_image.ScaleUpPointToNextLevel(previous_uv);
      }
    }

    track = xy;

    return current_image.isPointInImage(track, 0);
  }

  bool trackPoint(const uint32_t levels_mask, uint32_t previous_patches_size,
                  const cuvslam::sof::STFeaturePatch* previous_patches, Vector2T& track, const Vector2T& offset,
                  float search_radius_px, float ncc_threshold, const Matrix2T& initial_guess_map) {
    if (search_radius_px < 0.f) {
      return false;
    }

    assert(current_image_gradient.getLevelsCount() > 0);
    const int num_levels = std::min(current_image_gradient.getLevelsCount(), static_cast<int>(previous_patches_size));

    assert(num_levels > 0 && num_levels <= current_image.getLevelsCount());

    // Taylor-series approximation is precise in a 1-pixel radius.
    // Each level of the pyramid "doubles" the radius of our approximation,
    // so we turn search radius into the number of levels by taking log
    // and truncating the value to the nearest integer.
    const int coarsest_level = std::min(num_levels - 1, static_cast<int>(std::log1p(search_radius_px)));

    assert(coarsest_level >= 0);
    assert(coarsest_level < num_levels);

    Vector2T previous_uv = current_image.ScaleDownPoint(Vector2T(track), coarsest_level);
    map = initial_guess_map;
    xy = current_image.ScaleDownPoint(Vector2T(track + offset), coarsest_level);

    bool isConverged = false;

    for (int level = coarsest_level; level >= 0; --level) {
      if (!current_image.isPointInImage(xy, level)) {
        return false;
      }

      if ((levels_mask & (1 << level)) == 0) {
        if (level > 0) {
          xy = current_image.ScaleUpPointToNextLevel(xy);
          previous_uv = current_image.ScaleUpPointToNextLevel(previous_uv);
          continue;
        }
      }
      // here: copy from unaligned matrix to aligned matrix!
      const FeaturePatch previous_patch = previous_patches[level];
      if (!refineMapping(previous_patch, level, isConverged)) {
        if (level > 0) {
          xy = current_image.ScaleUpPointToNextLevel(xy);
          previous_uv = current_image.ScaleUpPointToNextLevel(previous_uv);
          continue;
        } else {
          return false;
        }
      }

      FeaturePatch current_patch;
      if (!compute_patch(current_patch, current_image, map, xy, level)) {
        return false;
      }

      if (level == 0) {
        this->ncc = compute_ncc(previous_patch, current_patch);
        if (this->ncc <= ncc_threshold || !isConverged) {
          return false;
        }
      }

      if (level > 0) {
        xy = current_image.ScaleUpPointToNextLevel(xy);
        previous_uv = current_image.ScaleUpPointToNextLevel(previous_uv);
      }
    }

    track = xy;

    return current_image.isPointInImage(track, 0);
  }

private:
  bool refineMapping(const FeaturePatch& previous_patch, const int level, bool& isConverged) {
    Matrix2T map_ = this->map;
    Vector2T xy_ = this->xy;

    info = cuvslam::camera::GetDefaultObservationInfoUV();

    const unsigned n_iterations = n_shift_only_iterations + n_full_mapping_iterations;
    for (unsigned i = 0; i < n_iterations; ++i) {
      const bool full_mapping = i >= n_shift_only_iterations;

      if (!current_image.isPointInImage(xy_, level)) {
        return false;
      }

      FeaturePatch current_patch;
      if (!compute_patch(current_patch, current_image, map_, xy_, level)) {
        return false;
      }

      FeaturePatch gx, gy;
      if (!compute_gradient_patches(gx, gy, current_image_gradient, map_, xy_, level)) {
        return false;
      }

      // Account for the fact that we subtract difference of means:
      // gx and gy are the entries of the jacobian matrix of the residual.
      gx.array() -= gx.sum() / (float)gx.size();
      gy.array() -= gy.sum() / (float)gy.size();

      const FeaturePatch gxgx = gx.cwiseProduct(gx);
      const FeaturePatch gxgy = gx.cwiseProduct(gy);
      const FeaturePatch gygy = gy.cwiseProduct(gy);

      FeaturePatch residual;
      compute_residual(residual, current_patch, previous_patch);

      const float w__gxgx = gaussian_weights(gxgx);
      const float w__gxgy = gaussian_weights(gxgy);
      const float w__gygy = gaussian_weights(gygy);

      const float wr_gx = gaussian_weights(residual.cwiseProduct(gx));
      const float wr_gy = gaussian_weights(residual.cwiseProduct(gy));

      {
        const float determinant = w__gxgx * w__gygy - w__gxgy * w__gxgy;
        const Vector2T v((w__gygy * wr_gx - w__gxgy * wr_gy) / determinant,
                         (w__gxgx * wr_gy - w__gxgy * wr_gx) / determinant);
        isConverged = (v.norm() < ST_CONVERGENCE_THRESHOLD);

        if (isConverged) {
          // TODO: investigate info matrix calculation,
          // now it's not correct and requires to use default info matrix
          // info <<
          //      w__gxgx, w__gxgy,
          //     w__gxgy, w__gygy;
          break;
        }
      }

      if (full_mapping) {
        const float wx_gxgx = gaussian_weights(x_patch.cwiseProduct(gxgx));
        const float wx_gxgy = gaussian_weights(x_patch.cwiseProduct(gxgy));
        const float wx_gygy = gaussian_weights(x_patch.cwiseProduct(gygy));

        const float wy_gxgx = gaussian_weights(y_patch.cwiseProduct(gxgx));
        const float wy_gxgy = gaussian_weights(y_patch.cwiseProduct(gxgy));
        const float wy_gygy = gaussian_weights(y_patch.cwiseProduct(gygy));

        const float wxxgxgx = gaussian_weights(xx_patch.cwiseProduct(gxgx));
        const float wxxgxgy = gaussian_weights(xx_patch.cwiseProduct(gxgy));
        const float wxxgygy = gaussian_weights(xx_patch.cwiseProduct(gygy));

        const float wxygxgx = gaussian_weights(xy_patch.cwiseProduct(gxgx));
        const float wxygxgy = gaussian_weights(xy_patch.cwiseProduct(gxgy));
        const float wxygygy = gaussian_weights(xy_patch.cwiseProduct(gygy));

        const float wyygxgx = gaussian_weights(yy_patch.cwiseProduct(gxgx));
        const float wyygxgy = gaussian_weights(yy_patch.cwiseProduct(gxgy));
        const float wyygygy = gaussian_weights(yy_patch.cwiseProduct(gygy));

        const float wrxgx = gaussian_weights(residual.cwiseProduct(x_patch.cwiseProduct(gx)));
        const float wrxgy = gaussian_weights(residual.cwiseProduct(x_patch.cwiseProduct(gy)));
        const float wrygx = gaussian_weights(residual.cwiseProduct(y_patch.cwiseProduct(gx)));
        const float wrygy = gaussian_weights(residual.cwiseProduct(y_patch.cwiseProduct(gy)));

        Matrix6T t;
        t << wxxgxgx, wxxgxgy, wxygxgx, wxygxgy, wx_gxgx, wx_gxgy, wxxgxgy, wxxgygy, wxygxgy, wxygygy, wx_gxgy, wx_gygy,
            wxygxgx, wxygxgy, wyygxgx, wyygxgy, wy_gxgx, wy_gxgy, wxygxgy, wxygygy, wyygxgy, wyygygy, wy_gxgy, wy_gygy,
            wx_gxgx, wx_gxgy, wy_gxgx, wy_gxgy, w__gxgx, w__gxgy, wx_gxgy, wx_gygy, wy_gxgy, wy_gygy, w__gxgy, w__gygy;

        Vector6T a;
        a << wrxgx, wrxgy, wrygx, wrygy, wr_gx, wr_gy;

        const Vector6T z = t.colPivHouseholderQr().solve(a);
        if (z.array().isNaN().any()) {
          return false;
        }

        const float& dxx = z(0);
        const float& dyx = z(1);
        const float& dxy = z(2);
        const float& dyy = z(3);
        const float& dx = z(4);
        const float& dy = z(5);

        Matrix2T z_map;
        z_map << dxx, dxy, dyx, dyy;

        const Vector2T z_xy(dx, dy);

        map_ += z_map;

        xy_ += z_xy;
      } else {
        Matrix2T t;
        t << w__gxgx, w__gxgy, w__gxgy, w__gygy;

        const Vector2T a{wr_gx, wr_gy};

        const Eigen::VectorXf z = t.colPivHouseholderQr().solve(a);
        assert(z.rows() == 2 && z.cols() == 1);
        if (z.array().isNaN().any()) {
          return false;
        }
        const Vector2T z_xy{z(0, 0), z(1, 0)};

        xy_ += z_xy;
      }
    }

    if (!current_image.isPointInImage(xy_, level)) {
      return false;
    }

    this->map = map_;
    this->xy = xy_;
    return true;
  }  // refineMapping

  const GradientPyramidT& current_image_gradient;
  const ImagePyramidT& current_image;
  const unsigned n_shift_only_iterations;
  const unsigned n_full_mapping_iterations;
  Matrix2T& info;
  float& ncc;

  Matrix2T map;
  Vector2T xy;
};  // class PatchTracker

}  // namespace

STTracker::STTracker(unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations)
    : n_shift_only_iterations(n_shift_only_iterations), n_full_mapping_iterations(n_full_mapping_iterations) {}

bool STTracker::trackPoint(const GradientPyramidT&, const GradientPyramidT& current_image_gradients,
                           const ImagePyramidT& previous_image, const ImagePyramidT& current_image,
                           const Vector2T& previous_uv, Vector2T& current_uv, Matrix2T& current_info,
                           float search_radius_px, float ncc_threshold) const {
  const Vector2T offset = current_uv - previous_uv;
  current_uv = previous_uv;

  float ncc;

  PatchTracker tracker(current_image_gradients, current_image, n_shift_only_iterations, n_full_mapping_iterations,
                       current_info, ncc);

  return tracker.trackPoint(previous_image, current_uv, offset, search_radius_px, ncc_threshold);
}

bool STTracker::BuildPointCache(const ImagePyramidT& previous_image, const Vector2T& track, uint32_t& levels_mask,
                                std::vector<STFeaturePatch>& image_patches) {
  int max_level = previous_image.getLevelsCount();
  if (max_level == 0) return false;

  int level = max_level - 1;
  Vector2T previous_uv = previous_image.ScaleDownPoint(track, level);

  levels_mask = 0;
  image_patches.resize(static_cast<size_t>(max_level));

  for (;; --level) {
    if (previous_image.isPointInImage(previous_uv, level)) {
      if (previous_image.computePatch(image_patches[level], previous_uv, level)) {
        levels_mask |= 1 << level;
      }
    }

    if (level > 0) {
      previous_uv = previous_image.ScaleUpPointToNextLevel(previous_uv);
    }
    if (level == 0) break;
  }
  return levels_mask != 0;
}

bool STTracker::TrackPointWithCache(const GradientPyramidT& current_image_gradients, const ImagePyramidT& current_image,
                                    const uint32_t levels_mask, uint32_t image_patches_size,
                                    const STFeaturePatch* image_patches, const float previous_uv[2],
                                    float current_uv[2], float& ncc, float current_info[4], float search_radius_px,
                                    float ncc_threshold, const Matrix2T& initial_guess_map) const {
  assert(current_image_gradients.getLevelsCount() > 0);
  assert(current_image.getLevelsCount() > 0);
  assert(previous_uv);
  assert(current_uv);

  Vector2T current = vec2(current_uv);
  Vector2T previous = vec2(previous_uv);
  Vector2T offset = current - previous;
  current = previous;

  Matrix2T info;

  PatchTracker tracker(current_image_gradients, current_image, n_shift_only_iterations, n_full_mapping_iterations, info,
                       ncc);

  bool status = tracker.trackPoint(levels_mask, image_patches_size, image_patches, current, offset, search_radius_px,
                                   ncc_threshold, initial_guess_map);

  if (status) {
    vec2(current_uv) = current;
  }

  if (current_info) {
    mat2(current_info) = info;
  }

  return status;
}

}  // namespace cuvslam::sof



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

#include <opencv2/opencv.hpp>

#include "camera/camera.h"
#include "common/include_gtest.h"
#include "common/unaligned_types.h"

namespace {
cv::Mat CreateCheckerboard(int squares_x, int squares_y, int square_size) {
  const int width = squares_x * square_size;
  const int height = squares_y * square_size;

  cv::Mat checkerboard(height, width, CV_8UC1, cv::Scalar(255));  // White background

  for (int y = 0; y < squares_y; ++y) {
    for (int x = 0; x < squares_x; ++x) {
      if ((x + y) % 2 == 0) {
        cv::rectangle(checkerboard, cv::Point(x * square_size, y * square_size),
                      cv::Point((x + 1) * square_size, (y + 1) * square_size), 0, cv::FILLED);
      }
    }
  }

  return checkerboard;
}

float random_a_b(float a, float b) { return a + (b - a) * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)); }

void UndistortWithModel(const cv::Mat& distorted, cv::Mat& undistort,
                        const cuvslam::camera::PinholeCameraModel& pinhole,
                        const cuvslam::camera::ICameraModel& model) {
  const cuvslam::Vector2T& resolution = model.getResolution();
  undistort = cv::Mat::zeros(distorted.size(), CV_8UC3);
  for (int y = 0; y < distorted.rows; ++y) {
    for (int x = 0; x < distorted.cols; ++x) {
      const cuvslam::Vector2T uv_of_undistorted(x, y);
      cuvslam::Vector2T xy;
      if (!pinhole.normalizePoint(uv_of_undistorted, xy)) {
        undistort.at<cv::Vec3b>(uv_of_undistorted.y(), uv_of_undistorted.x()) = cv::Vec3b(0, 255, 0);
        continue;
      }
      cuvslam::Vector2T uv_of_distorted;
      if (!model.denormalizePoint(xy, uv_of_distorted)) {
        undistort.at<cv::Vec3b>(uv_of_undistorted.y(), uv_of_undistorted.x()) = cv::Vec3b(0, 0, 255);
        continue;
      }
      if (0 <= uv_of_distorted.x() && uv_of_distorted.x() < resolution.x() && 0 <= uv_of_distorted.y() &&
          uv_of_distorted.y() < resolution.y()) {
        const uchar grayscale = distorted.at<uchar>(uv_of_distorted.y(), uv_of_distorted.x());
        undistort.at<cv::Vec3b>(uv_of_undistorted.y(), uv_of_undistorted.x()) =
            cv::Vec3b(grayscale, grayscale, grayscale);
      } else {
        undistort.at<cv::Vec3b>(uv_of_undistorted.y(), uv_of_undistorted.x()) = cv::Vec3b(255, 0, 0);
      }
    }
  }
}

void UndistortBrown5K(const cv::Mat& distorted, cv::Mat& undistort, const cv::Mat& camera_matrix,
                      const cv::Mat& dist_coeffs) {
  const cuvslam::Vector2T resolution(distorted.cols, distorted.rows);
  const cuvslam::Vector2T focal(camera_matrix.at<float>(0, 0), camera_matrix.at<float>(1, 1));
  const cuvslam::Vector2T principal(camera_matrix.at<float>(0, 2), camera_matrix.at<float>(1, 2));
  const cuvslam::camera::PinholeCameraModel pinhole(resolution, focal, principal);
  // dist_coeffs is in OpenCV order (k1, k2, p1, p2, k3), our model takes the radial ones first.
  const float K1 = dist_coeffs.at<float>(0);
  const float K2 = dist_coeffs.at<float>(1);
  const float P1 = dist_coeffs.at<float>(2);
  const float P2 = dist_coeffs.at<float>(3);
  const float K3 = dist_coeffs.at<float>(4);
  const cuvslam::camera::Brown5KCameraModel model(resolution, focal, principal, K1, K2, K3, P1, P2);
  UndistortWithModel(distorted, undistort, pinhole, model);
}

void UndistortPolynomial(const cv::Mat& distorted, cv::Mat& undistort, const cv::Mat& camera_matrix,
                         const cv::Mat& dist_coeffs) {
  const cuvslam::Vector2T resolution(distorted.cols, distorted.rows);
  const cuvslam::Vector2T focal(camera_matrix.at<float>(0, 0), camera_matrix.at<float>(1, 1));
  const cuvslam::Vector2T principal(camera_matrix.at<float>(0, 2), camera_matrix.at<float>(1, 2));
  const cuvslam::camera::PinholeCameraModel pinhole(resolution, focal, principal);
  // dist_coeffs is in OpenCV order (k1, k2, p1, p2, k3..k6), our model takes the radial ones first.
  const float K1 = dist_coeffs.at<float>(0);
  const float K2 = dist_coeffs.at<float>(1);
  const float P1 = dist_coeffs.at<float>(2);
  const float P2 = dist_coeffs.at<float>(3);
  const float K3 = dist_coeffs.at<float>(4);
  const float K4 = dist_coeffs.at<float>(5);
  const float K5 = dist_coeffs.at<float>(6);
  const float K6 = dist_coeffs.at<float>(7);
  const cuvslam::camera::PolynomialCameraModel model(resolution, focal, principal, K1, K2, K3, K4, K5, K6, P1, P2);
  UndistortWithModel(distorted, undistort, pinhole, model);
}

void RGBToMonoWithSkipColor(const cv::Mat& image_color, cv::Mat& image_mono, int skip_color = 255) {
  cv::cvtColor(image_color, image_mono, cv::COLOR_BGR2GRAY);

  const cv::Mat mask = image_mono != skip_color;
  image_mono.setTo(0, mask);
}
}  // namespace

namespace Test {
using namespace cuvslam;

TEST(TestCamera, OpenCVvsInternal) {
  constexpr int cell_size = 20;
  constexpr int n_cells = 16;
  constexpr int w = cell_size * n_cells;
  constexpr int h = cell_size * n_cells;
  const cv::Mat original_undistorted = CreateCheckerboard(n_cells, n_cells, cell_size);
  ASSERT_FALSE(original_undistorted.empty());

  cv::Mat cv_undistorted_mono;
  cv::Mat delta;
  cv::Mat cuvslam_undistorted_rgb;
  cv::Mat cuvslam_undistorted_mono;
  for (int i = 0; i < 500; ++i) {
    const float fx = random_a_b(1.f, w);
    const float fy = random_a_b(1.f, h);
    const float cx = random_a_b(10, w - 10);
    const float cy = random_a_b(10, h - 10);
    const cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    const float k1 = random_a_b(-3, 3);
    const float k2 = random_a_b(-3, 3);
    const float k3 = random_a_b(-3, 3);
    const float p1 = random_a_b(-0.5, 0.5);
    const float p2 = random_a_b(-0.5, 0.5);

    if (std::rand() % 2) {
      // brown
      const cv::Mat distortion_coefficients = (cv::Mat_<float>(1, 5) << k1, k2, p1, p2, k3);
      cv::undistort(original_undistorted, cv_undistorted_mono, camera_matrix, distortion_coefficients);
      UndistortBrown5K(original_undistorted, cuvslam_undistorted_rgb, camera_matrix, distortion_coefficients);
    } else {
      // polynomial
      const float k4 = random_a_b(-3, 3);
      const float k5 = random_a_b(-3, 3);
      const float k6 = random_a_b(-3, 3);
      const cv::Mat distortion_coefficients = (cv::Mat_<float>(1, 8) << k1, k2, p1, p2, k3, k4, k5, k6);
      cv::undistort(original_undistorted, cv_undistorted_mono, camera_matrix, distortion_coefficients);
      UndistortPolynomial(original_undistorted, cuvslam_undistorted_rgb, camera_matrix, distortion_coefficients);
    }
    RGBToMonoWithSkipColor(cuvslam_undistorted_rgb, cuvslam_undistorted_mono);

    cv::absdiff(cv_undistorted_mono, cuvslam_undistorted_mono, delta);
    /* Uncomment to display
    cv::imshow("Original", original_undistorted);
    cv::imshow("openCV", cv_undistorted_mono);
    cv::imshow("cuVSLAM", cuvslam_undistorted_mono);
    cv::imshow("delta", delta);
    cv::waitKey(1000);
    */
    const float percent_diff = 100 * static_cast<float>(countNonZero(delta)) / w / h;  // 0-100
    ASSERT_LE(percent_diff, 15);
  }
}
}  // namespace Test

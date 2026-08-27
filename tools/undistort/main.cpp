
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

#include <iostream>

#include "gflags/gflags.h"
#include "opencv2/opencv.hpp"

#include "camera/camera.h"
#include "edex/edex.h"

DEFINE_int32(camera, 0, "Camera number (only for input image)");

using namespace cuvslam;

template <typename Condition, typename Message>
void CheckCondition(const Condition& cond, const Message& message) {
  if (!cond) {
    std::cerr << message << std::endl;
    std::exit(1);
  }
}

std::unique_ptr<camera::ICameraModel> CreateCameraModel(const edex::Intrinsics& intrinsics) {
  return camera::CreateCameraModel(intrinsics.resolution, intrinsics.focal, intrinsics.principal,
                                   camera::StringToDistortionModel(intrinsics.distortion_model),
                                   intrinsics.distortion_params.data(), intrinsics.distortion_params.size());
}

void Undistort(const camera::ICameraModel& input_model, const camera::ICameraModel& output_model, const cv::Mat& input,
               cv::Mat& output) {
  cv::Mat map_x(output.size(), CV_32FC1);
  cv::Mat map_y(output.size(), CV_32FC1);

  for (int y = 0; y < output.rows; ++y) {
    for (int x = 0; x < output.cols; ++x) {
      Vector2T dst(x, y);
      Vector2T interim, src;
      if (!output_model.normalizePoint(dst, interim) || !input_model.denormalizePoint(interim, src)) {
        // Unmapped pixel — point the map outside the source so remap fills it from the border.
        map_x.at<float>(y, x) = -1.f;
        map_y.at<float>(y, x) = -1.f;
        continue;
      }
      map_x.at<float>(y, x) = src.x();
      map_y.at<float>(y, x) = src.y();
    }
  }

  cv::remap(input, output, map_x, map_y, cv::INTER_LINEAR);
}

int main(int argc, char** argv) {
  const std::string usage{
      "Usage: <input_image> <input_edex> <output_image> [<output_edex>]\n"
      "If output_edex is not set, input intrinsics and pinhole camera model are used for output.\n"
      "Otherwise camera 0 from output_edex is always used."};
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove flags = */ true);

  CheckCondition(4 <= argc && argc <= 5, "Wrong number of arguments.\n" + usage);
  std::string input_image = argv[1];
  std::string input_edex = argv[2];
  std::string output_image = argv[3];
  std::string output_edex = argc > 4 ? argv[4] : "";

  edex::Intrinsics input_intr;
  {
    edex::EdexFile edex;
    CheckCondition(edex.read(input_edex), "Can't read " + input_edex);
    CheckCondition(0 <= FLAGS_camera && FLAGS_camera < static_cast<intptr_t>(edex.cameras_.size()),
                   "No such camera in " + input_edex);
    input_intr = edex.cameras_[FLAGS_camera].intrinsics;
  }

  edex::Intrinsics output_intr;
  if (!output_edex.empty()) {
    edex::EdexFile edex;
    edex.read(output_edex);
    CheckCondition(0 < edex.cameras_.size(), "No camera in " + output_edex);
    output_intr = edex.cameras_[0].intrinsics;
  } else {
    output_intr = input_intr;
    output_intr.distortion_model = "pinhole";
    output_intr.distortion_params.clear();
  }

  std::unique_ptr<camera::ICameraModel> input_model;
  CheckCondition(input_model = CreateCameraModel(input_intr), "Cannot create input camera model");
  std::unique_ptr<camera::ICameraModel> output_model;
  CheckCondition(output_model = CreateCameraModel(output_intr), "Cannot create output camera model");

  try {
    cv::Mat input = cv::imread(input_image, cv::IMREAD_UNCHANGED);
    CheckCondition(!input.empty(), "Cannot load input image: " + input_image);

    cv::Mat output(output_intr.resolution.y(), output_intr.resolution.x(), input.type());
    Undistort(*input_model, *output_model, input, output);

    CheckCondition(cv::imwrite(output_image, output), "Cannot save output image: " + output_image);
  } catch (const std::exception& e) {
    CheckCondition(false, e.what());
  }

  return 0;
}

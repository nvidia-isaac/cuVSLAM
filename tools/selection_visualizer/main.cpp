
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

#include <filesystem>
#include <iostream>

#include "Eigen/Dense"
#include "gflags/gflags.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

#include "camera_rig_edex/camera_rig_edex.h"
#include "common/environment.h"
#include "edex/file_name_tools.h"
#include "odometry/svo_config.h"
#include "sof/image_context.h"
#include "sof/image_manager.h"
#include "sof/sof_config_gflags.h"
#include "sof/sof_create.h"
#include "utils/image_transform.h"

using namespace cuvslam;

static cv::Mat CreateImage(const ImageSource& image, const ImageShape& shape) {
  ImageMatrixT mat;

  if (image.type == ImageSource::U8) {
    mat = image.as<uint8_t>(shape).cast<float>();
  } else {
    mat = image.as<float>(shape);
  }

  mat = utils::ImageTransform<float>::Apply(mat);

  cv::Mat cvmat;
  cv::eigen2cv(mat, cvmat);
  if (cvmat.channels() == 1) {
    cv::cvtColor(cvmat, cvmat, cv::COLOR_GRAY2RGB);
  }
  return cvmat;
}

static void DoMonoTrack(const std::string& edexFile, std::string outputFolder, sof::Settings& sof_settings) {
  outputFolder += "mono/";

  std::error_code ec;
  if (!std::filesystem::create_directories(outputFolder, ec) && ec) {
    std::cout << "Can't create output folder: " << outputFolder << ", error: " << ec.message() << std::endl;
    return;
  }

  camera_rig_edex::CameraRigEdex rig(edexFile);

  ErrorCode err;
  if (!(err = rig.start())) {
    std::cout << "Failed to start camera rig: " << ErrorCode::GetString(err) << std::endl;
    return;
  }

  CameraId left_cam = 0;
  const auto& intrinsicsL = rig.getIntrinsic(0);

  auto selector = std::make_unique<sof::SelectorStereo>(sof_settings.feature_selection_settings);
  std::unique_ptr<sof::IMonoSOF> sof =
      CreateMonoSOF(sof::Implementation::kCPU, left_cam, intrinsicsL, std::move(selector), nullptr, sof_settings);
  sof::ImageManager image_manager;

  struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
  };

  std::map<TrackId, Color> trackColors;
  FrameId frameId;

  std::vector<camera::Observation> tracks;

  Sources curr_sources;
  Sources masks_sources;
  DepthSources depth_sources;
  Metas curr_meta;
  sof::Images curr_image_ptrs = {};
  sof::Images prev_image_ptrs = {};

  while (err != ErrorCode::E_Bounds) {
    err = rig.getFrame(curr_sources, curr_meta, masks_sources, depth_sources);
    if (err != ErrorCode::S_True) {
      if (err != ErrorCode::E_Bounds) {
        std::cout << "Can't get image for tracking" << ErrorCode::GetString(err) << std::endl;
      }

      return;
    }

    constexpr CameraId cam_id = 0;
    if (curr_sources.empty() || cam_id >= curr_meta.size() || curr_sources[cam_id].data == nullptr) {
      continue;
    }

    if (!image_manager.is_initialized()) {
      image_manager.init(curr_meta[cam_id].shape, curr_sources.size() * 4, false);
    }

    if (prev_image_ptrs.size() != curr_sources.size()) {
      prev_image_ptrs.assign(curr_sources.size(), nullptr);
    }
    if (cam_id < curr_image_ptrs.size() && curr_image_ptrs[cam_id] != nullptr) {
      prev_image_ptrs[cam_id] = curr_image_ptrs[cam_id];
    }
    curr_image_ptrs.assign(curr_sources.size(), nullptr);
    for (CameraId id = 0; id < curr_sources.size(); ++id) {
      if (curr_sources[id].data == nullptr) {
        continue;
      }
      sof::ImageContextPtr ptr = image_manager.acquire();
      if (ptr == nullptr) {
        TraceError("ImageManager::acquire returned nullptr");
        return;
      }
      ptr->set_image_meta(curr_meta[id]);
      curr_image_ptrs[id] = ptr;
    }

    if (curr_image_ptrs.empty() || curr_image_ptrs[cam_id] == nullptr) {
      continue;
    }

    sof::ImageContextPtr prev_image;
    if (cam_id < prev_image_ptrs.size()) {
      prev_image = prev_image_ptrs[cam_id];
    }

    const ImageSource* mask_source =
        (cam_id < masks_sources.size() && masks_sources[cam_id].data != nullptr) ? &masks_sources[cam_id] : nullptr;
    odom::KeyFrameSettings kf_settings;
    const sof::MonoSOFFrameSettings sof_frame_settings{sof_settings, kf_settings};
    sof->track({curr_sources[cam_id], curr_image_ptrs[cam_id]}, prev_image, Isometry3T::Identity(), sof_frame_settings,
               mask_source);
    sof::FrameState state;
    const auto& tracks_vector = sof->finish(state, sof_frame_settings);
    if (state != sof::FrameState::Key) {
      continue;
    }

    tracks_vector.export_to_observations_vector(intrinsicsL, tracks);

    frameId = curr_meta[0].frame_id;
    std::cout << "Track frame " << frameId << std::endl;

    auto imgDrawL = CreateImage(curr_sources[0], curr_meta[0].shape);

    for (const auto& track : tracks) {
      if (trackColors.count(track.id) == 0) {
        trackColors[track.id] = {(unsigned char)(std::rand() % 255), (unsigned char)(std::rand() % 255),
                                 (unsigned char)(std::rand() % 255)};
      }

      const Color& c = trackColors.at(track.id);

      Vector2T uvL;
      if (!intrinsicsL.denormalizePoint(track.xy, uvL)) {
        continue;
      }
      cv::circle(imgDrawL, cv::Point(uvL.x(), uvL.y()), 2, cv::Scalar(c.b, c.g, c.r), cv::FILLED);
    }

    cv::imshow("Frame: " + std::to_string(frameId) + " left", imgDrawL);
    cv::waitKey(1);

    std::ostringstream fileNameL;
    std::ostringstream fileNameR;
    fileNameL << outputFolder << "left." << std::setw(5) << std::setfill('0') << frameId << ".png";
    fileNameR << outputFolder << "right." << std::setw(5) << std::setfill('0') << frameId << ".png";

    cv::imwrite(fileNameR.str(), imgDrawL);
  }
}

int main(int argC, char** ppArgV) {
  gflags::ParseCommandLineFlags(&argC, &ppArgV, /*remove flags = */ true);
  sof::Settings sof_settings;
  sof::ParseSettings(sof_settings);

  try {
    // check environment
    std::string sequenceFolder = Environment::GetVar(Environment::CUVSLAM_DATASETS);
    std::string outputFolder = Environment::GetVar(Environment::CUVSLAM_OUTPUT);

    if (sequenceFolder.size() == 0) {
      std::cout << "Can't understand where is edexes data folder" << std::endl;
      return -1;
    }

    if (outputFolder.size() == 0) {
      std::cout << "Can't understand where is output folder" << std::endl;
      return -1;
    }

    std::error_code ec;
    if (!std::filesystem::create_directories(outputFolder, ec) && ec) {
      std::cout << "Can't create output folder: " << outputFolder << ", error: " << ec.message() << std::endl;
      return -1;
    }

    // parse command line
    std::string edexFile;
    std::string resultFolder;
    std::string outputPdfName;
    std::string seqSubPath;

    if (argC != 3 && argC != 4) {
      std::cout << "Wrong command line.\n"
                   "Use: <edex file name> <relative path to sequence>\n"
                   "Example: -lr_tracker=lk test_stereo.edex kitti/12\n";
      return -1;
    }

    if (argC > 1) {
      edexFile = ppArgV[1];
    }

    if (argC > 2) {
      seqSubPath = ppArgV[2];

      if (!IsPathEndWithSlash(seqSubPath)) {
        seqSubPath += '/';
      }
    }

    if (argC > 3) {
      resultFolder = outputFolder + ppArgV[3];

      if (!IsPathEndWithSlash(resultFolder)) {
        resultFolder += '/';
      }
    } else {
      const std::string edexFileWithoutExt = edex::filepath::StripExt(edexFile);
      const std::string outputEdexFolder = outputFolder + edexFileWithoutExt + "/";

      std::error_code ec;
      if (!std::filesystem::create_directories(outputEdexFolder, ec) && ec) {
        std::cout << "Can't create output edex folder: " << outputEdexFolder << ", error: " << ec.message()
                  << std::endl;
        return -1;
      }

      resultFolder = outputEdexFolder + std::to_string(std::time(nullptr)) + "/";
    }

    if (!std::filesystem::create_directories(resultFolder, ec) && ec) {
      std::cout << "Can't create result folder: " << resultFolder << ", error: " << ec.message() << std::endl;
      return -1;
    }

    std::cout << "Output folder - " + resultFolder << std::endl;

    const std::string fileListFullName = sequenceFolder + seqSubPath + edexFile;
    DoMonoTrack(fileListFullName, resultFolder, sof_settings);

    std::cout << "Tracking done." << std::endl;

    return 0;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return -1;  // non zero is error
  }
}


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

#include <algorithm>
#include <filesystem>
#include <iostream>

#include "Eigen/Dense"
#include "gflags/gflags.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

#include "camera_rig_edex/camera_rig_edex.h"
#include "common/environment.h"
#include "common/image_dropper.h"
#include "edex/file_name_tools.h"
#include "odometry/svo_config.h"
#include "odometry/svo_config_gflags.h"
#include "sof/image_context.h"
#include "sof/image_manager.h"
#include "sof/sof_config_gflags.h"
#include "sof/sof_create.h"
#include "utils/image_transform.h"

DEFINE_bool(use_gpu, true, "Use GPU");
DEFINE_bool(store_raw_images, false, "Store PNG images");
DEFINE_string(output, "", "Output folder");
DEFINE_bool(show_cov, true, "Show covariances of observations");

DEFINE_double(image_drop_rate, 0.0, "Image drop rate");
DEFINE_string(image_drop_type, "steady", "Image dropping type: steady, normal, sticky");

// Local copy of the FIG-shaping flag. The launcher binaries declare their own copy in
// libs/launcher/multi_camera_launcher_base.cpp; the visualizer doesn't link the launcher TU
// and is a self-contained top-level binary, so we define it here directly.
DEFINE_bool(allow_stereo_track_for_depth, false,
            "Allow stereo 2D tracking between depth-aligned cameras and other cameras.");

using namespace cuvslam;

cv::Mat CreateImage(const ImageSource& image, const ImageShape& shape) {
  ImageMatrix<uint8_t> mat;

  if (image.type == ImageSource::U8) {
    mat = image.as<uint8_t>(shape);
  } else {
    mat = image.as<float>(shape).cast<uint8_t>();
  }

  cv::Mat cvmat;
  cv::eigen2cv(mat, cvmat);
  cv::cvtColor(cvmat, cvmat, cv::COLOR_GRAY2RGB);
  return cvmat;
}

void DrawCovariance(cv::Mat& dest, const camera::ICameraModel& intrinsics, const camera::Observation& track,
                    cv::Scalar color) {
  Vector2T uv;
  Matrix2T uv_cov;
  if (!intrinsics.denormalizePoint(track.xy, uv)) {
    return;
  }

  const Matrix2T focal = intrinsics.getFocal().asDiagonal();

  uv_cov = focal * track.xy_info.inverse() * focal.transpose();

  Eigen::SelfAdjointEigenSolver<decltype(uv_cov)> es(uv_cov);

  cv::Size axes(2 * sqrt(es.eigenvalues()(0)), 2 * sqrt(es.eigenvalues()(1)));  // 2 sigma
  double angle = atan2(es.eigenvectors()(0, 1), es.eigenvectors()(0, 0)) * 180.0 / CV_PI;

  cv::ellipse(dest, {(int)uv.x(), (int)uv.y()}, axes, angle, 0.0, 360.0, color);
}

void PutText(cv::InputOutputArray img, const cv::String& text, cv::Point origin) {
  cv::putText(img, text, origin, cv::FONT_HERSHEY_SIMPLEX, 1., {0, 0, 0}, 3);        // black outline
  cv::putText(img, text, origin, cv::FONT_HERSHEY_SIMPLEX, 1., {255, 255, 255}, 1);  // white text
}

void LineIfExists(cv::InputOutputArray img, TrackId track_id, const Vector2T& origin,
                  const std::vector<camera::Observation>& target_points, const camera::ICameraModel& target_cam,
                  bool target_primary, cv::Scalar color) {
  auto target = std::find_if(target_points.begin(), target_points.end(),
                             [track_id](const camera::Observation& obs) { return obs.id == track_id; });

  if (target != target_points.end()) {
    Vector2T uvTo;
    if (!target_cam.denormalizePoint(target->xy, uvTo)) {
      return;
    }
    cv::line(img, {(int)origin.x(), (int)origin.y()}, {(int)uvTo.x(), (int)uvTo.y()}, color);
    if (target_primary) {
      cv::circle(img, {(int)uvTo.x(), (int)uvTo.y()}, 2, color, cv::FILLED);
    }
  }
}

static void DoStereoTrack(const std::string& edexFile, std::string outputFolder, sof::Settings& sof_settings,
                          odom::KeyFrameSettings& kf_settings) {
  outputFolder += "stereo/";

  std::error_code ec;
  if (!std::filesystem::create_directories(outputFolder) && ec) {
    std::cout << "Can't create output folder: " + outputFolder << ", error: " << ec.message() << std::endl;
    return;
  }

  camera_rig_edex::CameraRigEdex rig(edexFile);

  ErrorCode err;
  if (!(err = rig.start())) {
    std::cout << "Failed to start camera rig: " << ErrorCode::GetString(err) << std::endl;
    return;
  }

  camera::Rig cam_rig;
  cam_rig.num_cameras = rig.getCamerasNum();
  for (int32_t cam = 0; cam < cam_rig.num_cameras; ++cam) {
    cam_rig.intrinsics[cam] = &rig.getIntrinsic(cam);
    cam_rig.camera_from_rig[cam] = rig.getExtrinsic(cam);
  }

  // The visualizer only consumes legacy stereo edex datasets where camera 0 is the convention-
  // ally anchored reference camera, so we tag {0} as the depth-id set unconditionally. This
  // matches the pre-FigSettings refactor behavior and is intentional for this debug tool — if
  // a future use case needs a configurable depth camera here, expose a `-cfg_depth_camera`
  // flag and feed it into depth_ids below.
  camera::FigSettings fig_settings{
      /*.mode=*/sof_settings.multicam_mode,
      /*.depth_ids=*/{0},
      /*.allow_stereo_track_for_depth=*/FLAGS_allow_stereo_track_for_depth,
      /*.manual_setup=*/sof_settings.multicam_setup,
  };
  camera::FrustumIntersectionGraph fig(cam_rig, fig_settings);

  const auto& primary_cams = fig.primary_cameras();
  std::vector<bool> is_primary(static_cast<size_t>(cam_rig.num_cameras), false);

  std::cout << "primary cam ids: ";
  for (CameraId cam_id : primary_cams) {
    is_primary[cam_id] = true;
    std::cout << (int)cam_id << " ";
  }
  std::cout << std::endl;

  sof::Implementation implementation = sof::Implementation::kCPU;
#ifdef USE_CUDA
  if (FLAGS_use_gpu) {
    implementation = sof::Implementation::kGPU;
  }
#endif
  std::unique_ptr<sof::IMultiSOF> stereo_sof =
      CreateMultiSOF(implementation, cam_rig, fig, nullptr, sof_settings, kf_settings);
  sof::ImageManager image_manager;

  struct Color {
    float r;
    float g;
    float b;
  };

  std::map<TrackId, Color> trackColors;

  FrameId frameId;

  cuvslam::sof::FrameState frameState;

  std::vector<std::vector<camera::Observation>> tracks(cam_rig.num_cameras);
  std::vector<std::vector<camera::Observation>> prev_tracks(cam_rig.num_cameras);

  Sources curr_sources;
  Sources masks_sources;
  Metas curr_meta;
  DepthSources depth_sources;
  sof::Images curr_image_ptrs = {};
  sof::Images prev_image_ptrs = {};

  const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');  // Codec
  const double fps = 30.0;
  std::vector<cv::VideoWriter> writers;
  for (int32_t cam = 0; cam < cam_rig.num_cameras; ++cam) {
    const auto res = cam_rig.intrinsics[cam]->getResolution();
    writers.emplace_back(outputFolder + "cam" + std::to_string(cam) + ".mp4", fourcc, fps,
                         cv::Size{(int)res.x(), (int)res.y()});
  }

  std::random_device dev;
  auto drop = CreateImageDropper(ParseImageDropperType(FLAGS_image_drop_type), std::mt19937{dev()});

  while (err != ErrorCode::E_Bounds) {
    err = rig.getFrame(curr_sources, curr_meta, masks_sources, depth_sources);
    if (err != ErrorCode::S_True) {
      if (err != ErrorCode::E_Bounds) {
        std::cout << "Can't get image for tracking" << ErrorCode::GetString(err) << std::endl;
      }
      return;
    }

    auto dropped_cams = drop->GetDroppedImages(FLAGS_image_drop_rate, rig.getCamerasNum());
    for (const auto& cam_id : dropped_cams) {
      curr_sources[cam_id] = {};
    }
    const bool has_sources = std::any_of(curr_sources.begin(), curr_sources.end(),
                                         [](const ImageSource& source) { return source.data != nullptr; });
    if (!has_sources) {
      continue;
    }

    if (!image_manager.is_initialized()) {
      image_manager.init(curr_meta[0].shape, curr_sources.size() * 4, FLAGS_use_gpu);
    }

    if (prev_image_ptrs.size() != curr_image_ptrs.size()) {
      prev_image_ptrs.assign(curr_image_ptrs.size(), nullptr);
    }
    for (CameraId cam_id = 0; cam_id < curr_image_ptrs.size(); ++cam_id) {
      if (curr_image_ptrs[cam_id] != nullptr) {
        prev_image_ptrs[cam_id] = curr_image_ptrs[cam_id];
      }
    }
    curr_image_ptrs.assign(curr_sources.size(), nullptr);
    for (CameraId cam_id = 0; cam_id < curr_sources.size(); ++cam_id) {
      if (curr_sources[cam_id].data == nullptr) {
        continue;
      }
      sof::ImageContextPtr ptr = image_manager.acquire();
      if (ptr == nullptr) {
        TraceError("ImageManager::acquire returned nullptr");
        return;
      }
      ptr->set_image_meta(curr_meta[cam_id]);
      curr_image_ptrs[cam_id] = ptr;
    }

    frameId = curr_meta[0].frame_id;
    std::cout << "Track frame " << frameId << std::endl;
    odom::TrackPerFrameSettings per_frame_settings{};
    per_frame_settings.sof = sof_settings;
    per_frame_settings.kf = kf_settings;
    stereo_sof->trackNextFrame(curr_sources, curr_image_ptrs, prev_image_ptrs, masks_sources, Isometry3T::Identity(),
                               tracks, frameState, per_frame_settings);

    std::unordered_map<CameraId, cv::Mat> drawings;
    for (CameraId cam = 0; cam < curr_sources.size(); ++cam) {
      if (curr_sources[cam].data != nullptr) {
        drawings.emplace(cam, CreateImage(curr_sources[cam], curr_meta[cam].shape));
      }
    }

    if (frameState == sof::FrameState::Key) {
      std::cout << "Keyframe!" << std::endl;
    }

    for (CameraId cam = 0; cam < curr_sources.size(); ++cam) {
      if (curr_sources[cam].data == nullptr) {
        continue;
      }
      std::cout << "num_tracks in cam " << cam << " = " << tracks[cam].size() << std::endl;
      for (const auto& track : tracks[cam]) {
        if (trackColors.count(track.id) == 0) {
          trackColors[track.id] = {(float)(std::rand() % 255), (float)(std::rand() % 255), (float)(std::rand() % 255)};
        }

        const Color& c = trackColors.at(track.id);
        const auto rgb = cv::Scalar{c.r, c.g, c.b};
        auto& intrinsics = *cam_rig.intrinsics[cam];
        if (FLAGS_show_cov) {
          DrawCovariance(drawings[cam], intrinsics, track, rgb);
        }
        Vector2T uv;
        if (!intrinsics.denormalizePoint(track.xy, uv)) {
          continue;
        }
        if (frameState == sof::FrameState::Key) {
          if (is_primary[cam]) {
            cv::circle(drawings[cam], {(int)uv.x(), (int)uv.y()}, 2, rgb, cv::FILLED);
            for (auto sec_id : fig.secondary_cameras(cam)) {
              LineIfExists(drawings[cam], track.id, uv, tracks[sec_id], *cam_rig.intrinsics[sec_id], is_primary[sec_id],
                           rgb);
            }
          } else {  // is primary cam
            for (auto prim_id : primary_cams) {
              auto sec_ids = fig.secondary_cameras(prim_id);
              if (std::none_of(sec_ids.begin(), sec_ids.end(), [cam](auto sec_id) { return cam == sec_id; })) {
                continue;
              }
              LineIfExists(drawings[cam], track.id, uv, tracks[prim_id], *cam_rig.intrinsics[prim_id],
                           true,  // target_primary
                           rgb);
            }
          }
        } else {
          LineIfExists(drawings[cam], track.id, uv, prev_tracks[cam], *cam_rig.intrinsics[cam],
                       false,  // target_primary
                       rgb);
        }
      }
      PutText(drawings[cam], std::to_string(frameId), {0, 24});  // left-top corner shifted down by font height
      cv::imshow("Camera " + std::to_string(cam), drawings[cam]);
      writers[cam].write(drawings[cam]);
      if (FLAGS_store_raw_images) {
        std::ostringstream fileNameStr;
        fileNameStr << "cam" << std::to_string(cam) << "." << std::setw(5) << std::setfill('0') << frameId << ".png";
        namespace fs = std::filesystem;
        const fs::path fileName = fileNameStr.str();
        const fs::path fullPath = outputFolder / fileName;
        cv::imwrite(fullPath, drawings[cam]);
      }
    }
    prev_tracks = tracks;
    cv::waitKey(1);
  }
}

int main(int argC, char** ppArgV) {
  gflags::ParseCommandLineFlags(&argC, &ppArgV, /*remove flags = */ true);
  odom::Settings svo_settings;
  odom::ParseSettings(svo_settings.kf_settings);
  sof::ParseSettings(svo_settings.sof_settings);

  try {
    // check environment
    std::string sequenceFolder = Environment::GetVar(Environment::CUVSLAM_DATASETS);
    std::string outputFolder = Environment::GetVar(Environment::CUVSLAM_OUTPUT);

    if (!FLAGS_output.empty()) {
      outputFolder = FLAGS_output;
    }

    if (outputFolder.size() == 0) {
      std::cout << "Can't understand where is output folder" << std::endl;
      return 0;
    }

    std::error_code ec;
    if (!std::filesystem::create_directories(outputFolder) && ec) {
      std::cout << "Can't create output folder: " + outputFolder << ", error: " << ec.message() << std::endl;
      return 0;
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
      return 0;
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

    if (!FLAGS_output.empty()) {
      resultFolder = outputFolder;
      if (!IsPathEndWithSlash(resultFolder)) {
        resultFolder += '/';
      }
    } else if (argC > 3) {
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
        return 0;
      }

      resultFolder = outputEdexFolder + std::to_string(std::time(nullptr)) + "/";
    }

    if (!std::filesystem::create_directories(resultFolder, ec) && ec) {
      std::cout << "Can't create result folder: " << resultFolder << ", error: " << ec.message() << std::endl;
      return 0;
    }

    std::cout << "Output folder - " + resultFolder << std::endl;

    const std::string fileListFullName = sequenceFolder + seqSubPath + edexFile;
    DoStereoTrack(fileListFullName, resultFolder, svo_settings.sof_settings, svo_settings.kf_settings);

    std::cout << "Tracking done." << std::endl;

    return 0;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;  // non zero is error
  }
}


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

#include "cuvslam/debug_dump.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <thread>

#include "cnpy.h"
#include "common/imu_measurement.h"
#include "common/include_json.h"
#include "common/isometry.h"
#include "common/tga.h"
#include "common/unaligned_types.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "sof/image_manager.h"

#include "cuvslam/internal.h"
#include "version.h"

namespace cuvslam {

namespace {

std::string DistortionModelToString(Distortion::Model model) {
  switch (model) {
    case Distortion::Model::Polynomial:
      return "polynomial";
    case Distortion::Model::Brown:
      return "brown5k";
    case Distortion::Model::Pinhole:
      return "pinhole";
    case Distortion::Model::Fisheye:
      return "fisheye4";
  }
  throw std::invalid_argument{"Incorrect distortion model: " +
                              std::to_string(static_cast<std::underlying_type_t<Distortion::Model>>(model))};
}
void ConvertRgbToGray(uint8_t* dst, size_t dpitch, const uint8_t* src, size_t spitch, size_t width, size_t height) {
  assert(dst && src && dpitch >= width && spitch >= width * 3);
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      uint8_t r = src[i * spitch + j * 3];
      uint8_t g = src[i * spitch + j * 3 + 1];
      uint8_t b = src[i * spitch + j * 3 + 2];
      // 0.299 * red + 0.587 * green + 0.114 * blue but in integers
      dst[i * dpitch + j] = (306 * r + 601 * g + 117 * b) / 1024;
    }
  }
}
}  // namespace

void DumpConfiguration(const std::string& input_dump_root_dir, const Rig& rig, const Odometry::Config& cfg) {
  if (input_dump_root_dir.empty()) {
    return;
  }

  const auto dir = std::filesystem::path(input_dump_root_dir);
  std::error_code ec;
  if (std::filesystem::exists(dir)) {
    std::filesystem::remove_all(dir, ec);
    if (ec) {
      std::cout << "Failed to delete " << dir.string() << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::filesystem::create_directory(dir, ec);
  if (ec) {
    std::cout << "Failed to create " << dir.string() << std::endl;
  }
  std::filesystem::create_directory(dir / "images", ec);
  if (ec) {
    std::cout << "Failed to create " << (dir / "images").string() << std::endl;
  }
  if (cfg.odometry_mode == Odometry::OdometryMode::RGBD) {
    std::filesystem::create_directory(dir / "depths", ec);
    if (ec) {
      std::cout << "Failed to create " << (dir / "depths").string() << std::endl;
    }
  }

  Json::Value root;
  root[0]["frame_start"] = 0;
  root[0]["frame_end"] = 1;
  root[0]["version"] = "0.9";

  root[0]["cuvslam_version_major"] = CUVSLAM_API_VERSION_MAJOR;
  root[0]["cuvslam_version_minor"] = CUVSLAM_API_VERSION_MINOR;

  for (Json::ArrayIndex c = 0; c < rig.cameras.size(); c++) {
    auto& src = rig.cameras[c];
    Json::Value& dst = root[0]["cameras"][c];

    // intrinsics
    dst["intrinsics"]["size"][0] = src.size[0];
    dst["intrinsics"]["size"][1] = src.size[1];
    dst["intrinsics"]["principal"][0] = src.principal[0];
    dst["intrinsics"]["principal"][1] = src.principal[1];
    dst["intrinsics"]["focal"][0] = src.focal[0];
    dst["intrinsics"]["focal"][1] = src.focal[1];

    dst["intrinsics"]["distortion_model"] = DistortionModelToString(src.distortion.model);
    for (Json::ArrayIndex i = 0; i < src.distortion.parameters.size(); i++) {
      dst["intrinsics"]["distortion_params"][i] = src.distortion.parameters[i];
    }

    std::string distortion_key = "distortion_";
    distortion_key += DistortionModelToString(src.distortion.model);
    for (Json::ArrayIndex p = 0; p < src.distortion.parameters.size(); p++) {
      dst["intrinsics"][distortion_key][p] = src.distortion.parameters[p];
    }

    // transform
    Isometry3T rig_from_camera = ConvertPoseToIsometry(src.rig_from_camera);
    for (int y = 0; y < 3; y++) {
      for (int x = 0; x < 4; x++) {
        dst["transform"][y][x] = rig_from_camera(y, x);
      }
    }

    if (cfg.odometry_mode == Odometry::OdometryMode::RGBD && (uint32_t)cfg.rgbd_settings.depth_camera_id == c) {
      dst["depth_id"] = 0;
    }
  }
  // imu
  if (cfg.odometry_mode == Odometry::OdometryMode::Inertial) {
    Json::Value& dst_cfg = root[0]["imu"];
    dst_cfg["measurements"] = "IMU.jsonl";
    {
      // rig_from_imu
      Isometry3T rig_from_imu = ConvertPoseToIsometry(rig.imus[0].rig_from_imu);
      for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 4; x++) {
          dst_cfg["transform"][y][x] = rig_from_imu(y, x);
        }
      }
    }
  }

  // configuration
  {
    Json::Value& dst_cfg = root[0]["configuration"];

    dst_cfg["use_motion_model"] = cfg.use_motion_model;
    dst_cfg["use_denoising"] = cfg.use_denoising;
    dst_cfg["use_gpu"] = cfg.use_gpu;
    dst_cfg["async_sba"] = cfg.async_sba;
    dst_cfg["rectified_stereo_camera"] = cfg.rectified_stereo_camera;
    dst_cfg["enable_observations_export"] = cfg.enable_observations_export;
    dst_cfg["enable_landmarks_export"] = cfg.enable_landmarks_export;
    dst_cfg["enable_final_landmarks_export"] = cfg.enable_final_landmarks_export;
    dst_cfg["debug_imu_mode"] = cfg.debug_imu_mode;
    dst_cfg["max_frame_delta_s"] = cfg.max_frame_delta_s;
    dst_cfg["multicam_mode"] = ToUnderlying(cfg.multicam_mode);
    dst_cfg["odometry_mode"] = ToUnderlying(cfg.odometry_mode);
    dst_cfg["min_depth"] = cfg.min_depth;
    dst_cfg["max_depth"] = cfg.max_depth;
    dst_cfg["rgbd_settings"]["depth_scale_factor"] = cfg.rgbd_settings.depth_scale_factor;
    dst_cfg["rgbd_settings"]["depth_camera_id"] = cfg.rgbd_settings.depth_camera_id;
    dst_cfg["rgbd_settings"]["enable_depth_stereo_tracking"] = cfg.rgbd_settings.enable_depth_stereo_tracking;
    if ((cfg.odometry_mode == Odometry::OdometryMode::Inertial ||
         cfg.odometry_mode == Odometry::OdometryMode::Multisensor) &&
        !rig.imus.empty()) {
      dst_cfg["imu"]["gyroscope_noise_density"] = rig.imus[0].gyroscope_noise_density;
      dst_cfg["imu"]["gyroscope_random_walk"] = rig.imus[0].gyroscope_random_walk;
      dst_cfg["imu"]["accelerometer_noise_density"] = rig.imus[0].accelerometer_noise_density;
      dst_cfg["imu"]["accelerometer_random_walk"] = rig.imus[0].accelerometer_random_walk;
      dst_cfg["imu"]["frequency"] = rig.imus[0].frequency;
    }
    if (cfg.odometry_mode == Odometry::OdometryMode::Multisensor) {
      Json::Value depth_cams(Json::arrayValue);
      for (int32_t id : cfg.multisensor_settings.depth_camera_ids) depth_cams.append(id);
      dst_cfg["multisensor_settings"]["depth_camera_ids"] = depth_cams;
      dst_cfg["multisensor_settings"]["depth_scale_factor"] = cfg.multisensor_settings.depth_scale_factor;
      dst_cfg["multisensor_settings"]["enable_depth_stereo_tracking"] =
          cfg.multisensor_settings.enable_depth_stereo_tracking;
    }
  }
  {
    root[1]["frame_metadata"] = "frame_metadata.jsonl";
    for (Json::ArrayIndex c = 0; c < rig.cameras.size(); c++) {
      root[1]["sequence"][c] = "images/cam" + std::to_string(c) + ".00000.tga";
    }
    if (cfg.odometry_mode == Odometry::OdometryMode::RGBD) {
      auto depth_cam_id = cfg.rgbd_settings.depth_camera_id;
      root[1]["depth_sequence"][depth_cam_id] = "depths/cam" + std::to_string(depth_cam_id) + ".00000.npy";
    }
  }

  Json::StreamWriterBuilder builder;
  std::string jsonStr = Json::writeString(builder, root);

  std::ofstream file;
  std::filesystem::path edex = dir / "stereo.edex";
  file.open(edex.string());
  if (file.is_open()) {
    file << jsonStr;
    file.close();
    std::cout << "Save " << edex.string() << std::endl;
  }
}

void DumpTrackCall(const std::string& input_dump_root_dir, size_t frame_id, const std::vector<Image>& images,
                   const std::vector<Image>& masks, const std::vector<Image>& depths) {
  if (input_dump_root_dir.empty()) {
    return;
  }

  ////////////////////////////////
  // TODO: HORRIBLE. REDO.
  ////////////////////////////////
  const auto dir = std::filesystem::path(input_dump_root_dir);
  try {
    std::vector<std::string> filenames(images.size());
    std::vector<std::string> depth_filenames(depths.size());

    for (size_t i = 0; i < images.size(); i++) {
      const auto& image = images[i];
      auto cam_id = image.camera_index;

      std::ostringstream fileName;
      fileName << "images/cam" << cam_id << "." << std::setw(5) << std::setfill('0') << frame_id << ".tga";

      if (image.pixels == nullptr) {
        std::cout << "  Skip empty image for " << fileName.str() << std::endl;
        continue;
      }
      assert(image.data_type == Image::DataType::UINT8);
      const uint8_t* pixels = static_cast<const uint8_t*>(image.pixels);
      size_t bpp = image.encoding == Image::Encoding::RGB ? 3 : 1;
      thread_local std::vector<uint8_t, cuda::HostAllocator<uint8_t>> cpu_image;
      thread_local std::vector<uint8_t, cuda::HostAllocator<uint8_t>> cpu_input_mask;
      cuda::Stream s;
      cuda::GPUImage8 gpu_mask_original;
      cuda::GPUImage8 gpu_mask_resized;
      ImageMatrix<uint8_t> cpu_mask_resized;
      if (image.is_gpu_mem) {
        cpu_image.resize(image.height * image.width * bpp);
        cudaMemcpy2D((void*)cpu_image.data(), image.width * bpp, image.pixels, image.pitch, image.width * bpp,
                     image.height, cudaMemcpyDeviceToHost);
        pixels = cpu_image.data();
      }

      thread_local std::vector<uint8_t> gray_image;
      if (image.encoding == Image::Encoding::RGB) {
        gray_image.resize(image.height * image.width);
        // `pixels` is either `cpu_image` or cpu buffer in `image.pixels`,
        // both have `image.width * bpp` pitch (stride)
        ConvertRgbToGray(gray_image.data(), image.width, pixels, image.width * bpp, image.width, image.height);
        pixels = gray_image.data();
      }
      auto map = Eigen::Map<const ImageMatrix<uint8_t>>(pixels, image.height, image.width);

      std::filesystem::path path_img = dir / fileName.str();
      SaveTga(map, path_img.string());
      std::cout << "  Save " << path_img.string() << std::endl;
      filenames[i] = fileName.str();

      auto mask_it =
          std::find_if(masks.begin(), masks.end(), [&cam_id](const auto& mask) { return mask.camera_index == cam_id; });
      if (mask_it != masks.end()) {
        auto& mask = *mask_it;
        assert(mask.pixels != nullptr && mask.data_type == Image::DataType::UINT8 &&
               mask.encoding == Image::Encoding::MONO);
        const uint8_t* input_mask = static_cast<const uint8_t*>(mask.pixels);
        if (mask.is_gpu_mem) {
          cpu_input_mask.resize(image.height * image.width);
          if (mask.height == image.height && mask.width == image.width) {
            cudaMemcpy2D((void*)cpu_input_mask.data(), mask.width, input_mask, mask.pitch, mask.width, mask.height,
                         cudaMemcpyDeviceToHost);
          } else {
            gpu_mask_resized.init(image.width, image.height);
            uint2 src_size = {(unsigned)mask.width, (unsigned)mask.height};
            uint2 dst_size = {(unsigned)image.width, (unsigned)image.height};
            CUDA_CHECK(cuda::resize_mask(input_mask, src_size, mask.pitch, gpu_mask_resized.ptr(), dst_size,
                                         gpu_mask_resized.pitch(), s.get_stream()));
            gpu_mask_resized.copy(cuda::GPUCopyDirection::ToCPU, cpu_input_mask.data(), s.get_stream());
          }
          cudaStreamSynchronize(s.get_stream());
          input_mask = cpu_input_mask.data();
        } else {
          if (mask.height != image.height && mask.width != image.width) {
            auto cpu_mask_map =
                Eigen::Map<ImageMatrix<uint8_t>>(const_cast<uint8_t*>(input_mask), mask.height, mask.width);
            cpu_mask_resized.resize(image.height, image.width);
            float scale_x = static_cast<float>(mask.width) / static_cast<float>(image.width);
            float scale_y = static_cast<float>(mask.height) / static_cast<float>(image.height);
            for (int y = 0; y < image.height; ++y) {
              for (int x = 0; x < image.width; ++x) {
                int orig_x = static_cast<int>(static_cast<float>(x) * scale_x);
                int orig_y = static_cast<int>(static_cast<float>(y) * scale_y);

                orig_x = std::min(orig_x, mask.width - 1);
                orig_y = std::min(orig_y, mask.height - 1);

                cpu_mask_resized(y, x) = cpu_mask_map(orig_y, orig_x);
              }
            }
            input_mask = cpu_mask_resized.data();
          }
        }
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask_map =
            Eigen::Map<const ImageMatrix<uint8_t>>(input_mask, image.height, image.width);

        bool mask_is_binary =
            (mask_map.array() == 0).all() || ((mask_map.array() == 1).any() && (mask_map.array() <= 1).all());
        if (mask_is_binary) {
          mask_map *= 255;
        }
        std::ostringstream fileName_mask;
        fileName_mask << "images/cam" << cam_id << "." << std::setw(5) << std::setfill('0') << frame_id << "_mask.tga";
        std::filesystem::path path_img_mask = dir / fileName_mask.str();
        SaveTga(mask_map, path_img_mask.string());
      }
    }

    // Handle depth data if available
    for (size_t i = 0; i < depths.size(); i++) {
      const auto& depth = depths[i];
      auto cam_id = depth.camera_index;
      assert(depth.pixels != nullptr && depth.data_type == Image::DataType::UINT16);
      const uint16_t* depth_data = static_cast<const uint16_t*>(depth.pixels);

      if (depth.is_gpu_mem) {
        thread_local std::vector<uint16_t, cuda::HostAllocator<uint16_t>> cpu_depth;
        cpu_depth.resize(depth.height * depth.width);
        cudaMemcpy2D((void*)cpu_depth.data(), depth.width * sizeof(uint16_t), depth.pixels, depth.pitch,
                     depth.width * sizeof(uint16_t), depth.height, cudaMemcpyDeviceToHost);
        depth_data = cpu_depth.data();
      }

      std::ostringstream depthFileName;
      depthFileName << "depths/cam" << cam_id << "." << std::setw(5) << std::setfill('0') << frame_id << ".npy";
      std::filesystem::path path_depth = dir / depthFileName.str();
      std::vector<size_t> shape = {(size_t)depth.height, (size_t)depth.width};
      cnpy::npy_save(path_depth.string(), depth_data, shape);
      std::cout << "  Save " << path_depth.string() << std::endl;
      depth_filenames[i] = depthFileName.str();
    }

    std::ofstream file_events;
    std::filesystem::path path_events = dir / "frame_metadata.jsonl";
    file_events.open(path_events.string(), std::ofstream::out | std::ofstream::app);
    if (file_events.is_open()) {
      file_events << "{\"frame_id\": " << frame_id;
      file_events << ",\"cams\": [";
      for (size_t i = 0; i < images.size(); i++) {
        const auto& image = images[i];
        int cam_id = image.camera_index;
        if (i > 0) file_events << ", ";
        file_events << "{\"id\": " << cam_id << ", \"filename\": \"" << (filenames[i]) << "\""
                    << ", \"timestamp\": " << (image.timestamp_ns) << "}";
      }
      file_events << "], \"depths\": [";
      for (size_t i = 0; i < depths.size(); i++) {
        const auto& depth = depths[i];
        int cam_id = depth.camera_index;
        if (i > 0) file_events << ", ";
        file_events << "{\"id\": " << cam_id << ", \"filename\": \"" << (depth_filenames[i]) << "\""
                    << ", \"timestamp\": " << (depth.timestamp_ns) << "}";
      }
      file_events << "]}" << std::endl;
      file_events.close();
    }
  } catch (...) {
  }
  frame_id++;
}

void DumpRegisterImuMeasurementCall(const std::string& input_dump_root_dir, const ImuMeasurement& data) {
  if (input_dump_root_dir.empty()) {
    return;
  }

  imu::ImuMeasurement m = {
      data.timestamp_ns,
      {data.linear_accelerations[0], data.linear_accelerations[1], data.linear_accelerations[2]},
      {data.angular_velocities[0], data.angular_velocities[1], data.angular_velocities[2]},
  };

  const auto dir = std::filesystem::path(input_dump_root_dir);
  try {
    std::ofstream file_events;
    std::filesystem::path path_events = dir / "IMU.jsonl";
    file_events.open(path_events.string(), std::ofstream::out | std::ofstream::app);
    if (file_events.is_open()) {
      file_events << "{\"type\":\"imu_data\"";
      file_events << ", \"timestamp\": " << (m.time_ns);
      file_events << ", \"AngularVelocityX\": " << m.angular_velocity[0];
      file_events << ", \"AngularVelocityY\": " << m.angular_velocity[1];
      file_events << ", \"AngularVelocityZ\": " << m.angular_velocity[2];
      file_events << ", \"LinearAccelerationX\": " << m.linear_acceleration[0];
      file_events << ", \"LinearAccelerationY\": " << m.linear_acceleration[1];
      file_events << ", \"LinearAccelerationZ\": " << m.linear_acceleration[2];
      file_events << "}" << std::endl;
      file_events.close();
    }
  } catch (...) {
  }
}

}  // namespace cuvslam

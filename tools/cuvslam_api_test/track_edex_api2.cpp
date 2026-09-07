
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

#include "track_edex_api2.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "camera_rig_edex/camera_rig_edex.h"
#include "common/coordinate_system.h"
#include "common/isometry.h"
#include "common/log.h"
#include "common/stream.h"
#include "common/time.h"
#ifdef USE_CUDA
#include "cuda_modules/cuda_helper.h"
#endif
#include "cuvslam/cuvslam2.h"

#define VERIFY_TRACE(condition, ...) \
  if (!(condition)) {                \
    TraceError(__VA_ARGS__);         \
    return false;                    \
  }

namespace cuvslam {

namespace {

Distortion::Model ToDistortionModel(const std::string& str) {
  if (str == "pinhole") {
    return Distortion::Model::Pinhole;
  } else if (str == "fisheye4" || str == "fisheye") {
    return Distortion::Model::Fisheye;
  } else if (str == "brown5k") {
    return Distortion::Model::Brown;
  } else if (str == "polynomial") {
    return Distortion::Model::Polynomial;
  } else {
    throw std::invalid_argument("Invalid distortion model " + str);
  }
}

Rig EdexRigToApiRig(const edex::EdexFile& edex_file, const camera_rig_edex::CameraRigEdex* edex_rig) {
  Rig rig;
  rig.cameras.reserve(edex_rig->getCamerasNum());
  for (size_t i = 0; i < edex_rig->getCamerasNum(); i++) {
    const edex::Camera& edex_cam = edex_file.cameras_[edex_rig->getCameraIds()[i]];

    Camera cam{};
    cam.size[0] = edex_cam.intrinsics.resolution.x();
    cam.size[1] = edex_cam.intrinsics.resolution.y();
    cam.principal = {edex_cam.intrinsics.principal.x(), edex_cam.intrinsics.principal.y()};
    cam.focal = {edex_cam.intrinsics.focal.x(), edex_cam.intrinsics.focal.y()};
    const Isometry3T cam_pose = LegacyEdexIsometryToOpenCV(edex_cam.transform);
    vec<3>(cam.rig_from_camera.translation) = cam_pose.translation();
    Eigen::Quaternionf quat(cam_pose.linear());
    cam.rig_from_camera.rotation = {quat.x(), quat.y(), quat.z(), quat.w()};

    cam.distortion.model = ToDistortionModel(edex_cam.intrinsics.distortion_model);
    cam.distortion.parameters = edex_cam.intrinsics.distortion_params;
    rig.cameras.push_back(std::move(cam));
  }

  // TODO(vikuznetsov): This should either be provided by the edex rig class, or we should get rid of it altogether.
  // Also, is there a more convenient way to check if imu calibration is available?
  if (!edex_file.imu_.imu_log_path_.empty()) {
    ImuCalibration imu;
    const Isometry3T imu_pose = LegacyEdexImuExtrinsicToOpenCV(edex_file.imu_.transform);
    vec<3>(imu.rig_from_imu.translation) = imu_pose.translation();
    Eigen::Quaternionf quat(imu_pose.linear());
    imu.rig_from_imu.rotation = {quat.x(), quat.y(), quat.z(), quat.w()};
    imu.gyroscope_noise_density = edex_file.imu_.gyroscope_noise_density;
    imu.gyroscope_random_walk = edex_file.imu_.gyroscope_random_walk;
    imu.accelerometer_noise_density = edex_file.imu_.accelerometer_noise_density;
    imu.accelerometer_random_walk = edex_file.imu_.accelerometer_random_walk;
    imu.frequency = edex_file.imu_.frequency;
    rig.imus.push_back(imu);
  }

  return rig;
}

#ifdef USE_CUDA
std::vector<cuda::GPUOnlyArray<uint8_t>> CreateGpuImages(edex::EdexFile& edex_file, bool use_gpu_mem) {
  std::vector<cuda::GPUOnlyArray<uint8_t>> gpu_images;
  if (use_gpu_mem) {
    gpu_images.reserve(edex_file.cameras_.size());
    for (auto&& cam : edex_file.cameras_) {
      // we don't know yet if incoming images will be mono or rgb. SAD
      gpu_images.emplace_back(cam.intrinsics.resolution[0] * cam.intrinsics.resolution[1] * 3);
    }
  }
  return gpu_images;
}
#endif

std::ostream& operator<<(std::ostream& stream, const Pose& pose) {
  Isometry3T iso_pose;
  iso_pose.setIdentity();
  Eigen::Quaternionf quat{pose.rotation.data()};
  iso_pose.linear() = quat.toRotationMatrix();
  iso_pose.translation() = vec<3>(pose.translation);
  return cuvslam::operator<<(stream, iso_pose);
}

void PrintPose(std::ostream& stream, bool is_valid, int64_t timestamp_ns, const Pose& pose, bool print_nan_on_failure) {
  constexpr const float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr const Pose kNanPose = {{kNaN, kNaN, kNaN, kNaN}, {kNaN, kNaN, kNaN}};
  if (stream) {
    const Timestamp timestamp{timestamp_ns};
    if (is_valid) {
      stream << timestamp << " " << pose << std::endl;
    } else if (print_nan_on_failure) {
      stream << timestamp << " " << kNanPose << std::endl;
    }
  }
}

std::string GetEdexPath(const std::string& data_folder, const std::string& edex_name = "stereo.edex") {
  return std::filesystem::path{data_folder} / edex_name;
}

}  // namespace

bool DoesEdexExist(const std::string& data_folder) {
  const std::string edex_name{GetEdexPath(data_folder)};
  edex::EdexFile edex_file;
  return edex_file.read(edex_name);
}

bool DoesEdexHaveImu(const std::string& data_folder) {
  const std::string edex_name{GetEdexPath(data_folder)};
  edex::EdexFile edex_file;
  return edex_file.read(edex_name) && !edex_file.imu_.imu_log_path_.empty();
}

bool TrackEdexApi2(const TestingSettings& settings, const cuvslam::Odometry::Config& cfg) {
  FrameId frame = 0;
  try {
    WarmUpGPU();

    const std::string edex_name{GetEdexPath(settings.data_folder)};
    auto edex_rig = std::make_unique<camera_rig_edex::CameraRigEdex>(edex_name, settings.camera_ids);

    edex::EdexFile edex_file;
    VERIFY_TRACE(edex_file.read(edex_name), "Failed to read %s\n", edex_name.c_str());

    const ErrorCode ret = edex_rig->start();
    VERIFY_TRACE(ret == ErrorCode::S_True, "Failed to start camera rig %s\n", ret.str());

    Rig rig = EdexRigToApiRig(edex_file, edex_rig.get());
    Odometry tracker{rig, cfg};

    Slam::Config slam_cfg;
    slam_cfg.sync_mode = !cfg.async_sba;
    // arbitraryly enable reading internals in some tests
    slam_cfg.enable_reading_internals = cfg.enable_final_landmarks_export;
    Slam slam{rig, tracker.GetPrimaryCameras(), slam_cfg};

    edex_rig->registerIMUCallback([&](const imu::ImuMeasurement& measurement) {
      // Multisensor mode also needs IMU when the rig has one.
      if (cfg.odometry_mode != Odometry::OdometryMode::Inertial &&
          cfg.odometry_mode != Odometry::OdometryMode::Multisensor) {
        return;
      }
      cuvslam::ImuMeasurement imu_measurement;
      imu_measurement.timestamp_ns = measurement.time_ns;
      vec<3>(imu_measurement.linear_accelerations) = measurement.linear_acceleration;
      vec<3>(imu_measurement.angular_velocities) = measurement.angular_velocity;
      tracker.RegisterImuMeasurement(0, imu_measurement);
    });

    std::ofstream out_odom_poses(settings.odom_poses_file);
    std::ofstream out_slam_poses(settings.slam_poses_file);
    // Set precision for printing.
    out_odom_poses << std::fixed << std::setprecision(9);
    out_slam_poses << std::fixed << std::setprecision(9);

    if (settings.start_frame > 0) {
      frame = static_cast<FrameId>(settings.start_frame);
      edex_rig->setCurrentFrame(frame);
    }

#ifdef USE_CUDA
    // alloc in advance to reuse on each frame
    std::vector<cuda::GPUOnlyArray<uint8_t>> gpu_images = CreateGpuImages(edex_file, settings.use_gpu_mem);
    std::vector<cuda::GPUOnlyArray<uint8_t>> gpu_masks = CreateGpuImages(edex_file, settings.use_gpu_mem);
#else
    VERIFY_TRACE(!settings.use_gpu_mem, "use_gpu_mem requires a build with CUDA support (USE_CUDA=ON)");
#endif
    while (true) {
      ScopedThrottler throttle(settings.max_fps);
      Sources cur_sources;
      Sources masks_sources;
      DepthSources depth_sources;
      Metas cur_meta;
      const ErrorCode ret = edex_rig->getFrame(cur_sources, cur_meta, masks_sources, depth_sources);
      if (ret != ErrorCode::S_True) {
        VERIFY_TRACE(ret == ErrorCode::E_Bounds, "getFrame fails with error %s at frame %zu", ErrorCode::GetString(ret),
                     frame);
        // if not another error, we successfully reached last frame
        break;
      }

      cuvslam::Odometry::ImageSet images;
      cuvslam::Odometry::ImageSet masks;
      for (uint32_t i = 0; i < cur_sources.size(); i++) {
        auto&& src = cur_sources[i];
        if (src.data == nullptr) {
          continue;
        }
        auto&& mask_src = masks_sources[i];
        auto&& meta = cur_meta[i];
        images.emplace_back(Image{
            {src.data, meta.shape.width, meta.shape.height, src.pitch, static_cast<Image::Encoding>(src.image_encoding),
             ImageData::DataType::UINT8, false /* is_gpu_mem */},
            meta.timestamp,
            i});
        if (mask_src.data != nullptr) {
          masks.emplace_back(Image{{mask_src.data, meta.mask_shape.width, meta.mask_shape.height, mask_src.pitch,
                                    static_cast<Image::Encoding>(mask_src.image_encoding), ImageData::DataType::UINT8,
                                    false /* is_gpu_mem */},
                                   meta.timestamp,
                                   i});
        }

        if (settings.use_gpu_mem) {
#ifdef USE_CUDA
          int bpp = src.image_encoding == ImageEncoding::RGB8 ? 3 : 1;
          CUDA_CHECK(cudaMemcpy(gpu_images[i].ptr(), src.data, meta.shape.height * meta.shape.width * bpp,
                                cudaMemcpyHostToDevice));
          images.back().pixels = gpu_images[i].ptr();
          images.back().pitch = meta.shape.width * bpp;
          images.back().is_gpu_mem = true;

          if (mask_src.data != nullptr) {
            CUDA_CHECK(cudaMemcpy(gpu_masks[i].ptr(), mask_src.data, meta.mask_shape.height * meta.mask_shape.width,
                                  cudaMemcpyHostToDevice));
            masks.back().pixels = gpu_masks[i].ptr();
            masks.back().pitch = meta.mask_shape.width;
            masks.back().is_gpu_mem = true;
          }
#endif
        }
      }

      // Build depth ImageSet for modes that consume depth (RGBD, Multisensor).
      cuvslam::Odometry::ImageSet depths;
      for (CameraId cid = 0; cid < depth_sources.size(); ++cid) {
        const auto& dsrc = depth_sources[cid];
        if (dsrc.data == nullptr) continue;
        const auto& meta = cur_meta[cid];
        const auto dtype =
            (dsrc.type == ImageSource::Type::F32) ? ImageData::DataType::FLOAT32 : ImageData::DataType::UINT16;
        depths.emplace_back(Image{{dsrc.data, meta.shape.width, meta.shape.height, dsrc.pitch, Image::Encoding::MONO,
                                   dtype, false /* is_gpu_mem */},
                                  meta.timestamp,
                                  static_cast<uint32_t>(cid)});
      }

      auto pose_estimate = tracker.Track(images, masks, depths);
      if (!pose_estimate.world_from_rig.has_value()) {
        TraceWarning("CUVSLAM_Track(): Tracking lost at frame %zu.", frame);
        continue;
      }
      auto odom_pose = pose_estimate.world_from_rig.value().pose;
      PrintPose(out_odom_poses, true, pose_estimate.timestamp_ns, odom_pose, settings.print_nan_on_failure);

      if (settings.enable_slam) {
        Odometry::State state;
        tracker.GetState(state);
        slam.Track(state);
        const auto slam_pose = slam.GetPose();
        PrintPose(out_slam_poses, true, pose_estimate.timestamp_ns, slam_pose, settings.print_nan_on_failure);
      }

      frame++;
    }
  } catch (const std::exception& ex) {
    TraceError("Frame %zu, error: %s", frame, ex.what());
    return false;
  }
  return true;
}

}  // namespace cuvslam

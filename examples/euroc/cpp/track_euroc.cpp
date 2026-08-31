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
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#include <climits>
#endif

#include "cuvslam/cuvslam2.h"
#include "png.h"
#include "yaml-cpp/yaml.h"

#ifdef USE_RERUN
#include "rerun.hpp"
#include "rerun/archetypes/arrows3d.hpp"
#include "rerun/archetypes/image.hpp"
#include "rerun/archetypes/line_strips3d.hpp"
#include "rerun/archetypes/points2d.hpp"
#include "rerun/archetypes/scalars.hpp"
#include "rerun/archetypes/transform3d.hpp"

// Generate pseudo-random color from integer identifier for visualization
std::array<uint8_t, 3> colorFromId(int32_t id) {
  return {static_cast<uint8_t>((id * 17) % 256), static_cast<uint8_t>((id * 31) % 256),
          static_cast<uint8_t>((id * 47) % 256)};
}

#endif

struct ImageBuffer {
  std::vector<uint8_t> data;
  int32_t width;
  int32_t height;
};

// Load PNG image (grayscale)
ImageBuffer loadPNG(const std::string &filename) {
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    throw std::runtime_error("Failed to open image: '" + filename + "'");
  }

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) {
    fclose(fp);
    throw std::runtime_error("Failed to create PNG read struct");
  }

  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_read_struct(&png, nullptr, nullptr);
    fclose(fp);
    throw std::runtime_error("Failed to create PNG info struct");
  }

  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    throw std::runtime_error("Error during PNG reading");
  }

  png_init_io(png, fp);
  png_read_info(png, info);

  ImageBuffer img;
  img.width = png_get_image_width(png, info);
  img.height = png_get_image_height(png, info);
  png_byte color_type = png_get_color_type(png, info);
  png_byte bit_depth = png_get_bit_depth(png, info);

  // Convert to 8-bit grayscale
  if (bit_depth == 16) png_set_strip_16(png);
  if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
  if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
  if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGBA || color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_rgb_to_gray(png, 1, 0.299, 0.587);
  if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA || color_type == PNG_COLOR_TYPE_RGBA) png_set_strip_alpha(png);

  png_read_update_info(png, info);

  img.data.resize(img.height * img.width);
  std::vector<png_bytep> row_pointers(img.height);
  for (int y = 0; y < img.height; y++) {
    row_pointers[y] = img.data.data() + y * img.width;
  }

  png_read_image(png, row_pointers.data());
  png_destroy_read_struct(&png, &info, nullptr);
  fclose(fp);

  return img;
}

// Parse transformation matrix from YAML config
std::vector<double> parseTransformMatrix(const YAML::Node &config, const std::string &key) {
  auto T = config[key]["data"];
  std::vector<double> matrix;
  if (T.IsSequence()) {
    for (const auto &val : T) {
      matrix.push_back(val.as<double>());
    }
  }
  if (matrix.size() != 16) {
    throw std::runtime_error(key + " transform matrix must have 16 elements");
  }
  return matrix;
}

// Convert 4x4 transformation matrix to cuvslam::Pose
cuvslam::Pose transformToPose(const std::vector<double> &T) {
  assert(T.size() == 16);
  // T is row-major 4x4 matrix
  // Extract rotation matrix (top-left 3x3)
  double R[9] = {T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]};

  // Convert rotation matrix to quaternion
  double trace = R[0] + R[4] + R[8];
  double qw, qx, qy, qz;

  if (trace > 0.0) {
    double s = 0.5 / std::sqrt(trace + 1.0);
    qw = 0.25 / s;
    qx = (R[7] - R[5]) * s;
    qy = (R[2] - R[6]) * s;
    qz = (R[3] - R[1]) * s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    double s = 2.0 * std::sqrt(1.0 + R[0] - R[4] - R[8]);
    qw = (R[7] - R[5]) / s;
    qx = 0.25 * s;
    qy = (R[1] + R[3]) / s;
    qz = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    double s = 2.0 * std::sqrt(1.0 + R[4] - R[0] - R[8]);
    qw = (R[2] - R[6]) / s;
    qx = (R[1] + R[3]) / s;
    qy = 0.25 * s;
    qz = (R[5] + R[7]) / s;
  } else {
    double s = 2.0 * std::sqrt(1.0 + R[8] - R[0] - R[4]);
    qw = (R[3] - R[1]) / s;
    qx = (R[2] + R[6]) / s;
    qy = (R[5] + R[7]) / s;
    qz = 0.25 * s;
  }

  cuvslam::Pose pose;
  pose.rotation = {static_cast<float>(qx), static_cast<float>(qy), static_cast<float>(qz), static_cast<float>(qw)};
  pose.translation = {static_cast<float>(T[3]), static_cast<float>(T[7]), static_cast<float>(T[11])};

  return pose;
}

// Invert a 4x4 homogeneous transformation matrix (rotation + translation)
// For a rigid transform T = [R | t], inv(T) = [R^T | -R^T * t]
std::vector<double> invertTransform(const std::vector<double> &T) {
  assert(T.size() == 16);
  std::vector<double> T_inv(16);

  // Extract rotation matrix R (top-left 3x3) and transpose it
  T_inv[0] = T[0];
  T_inv[1] = T[4];
  T_inv[2] = T[8];
  T_inv[3] = 0.0;
  T_inv[4] = T[1];
  T_inv[5] = T[5];
  T_inv[6] = T[9];
  T_inv[7] = 0.0;
  T_inv[8] = T[2];
  T_inv[9] = T[6];
  T_inv[10] = T[10];
  T_inv[11] = 0.0;

  // Compute -R^T * t
  T_inv[3] = -(T_inv[0] * T[3] + T_inv[1] * T[7] + T_inv[2] * T[11]);
  T_inv[7] = -(T_inv[4] * T[3] + T_inv[5] * T[7] + T_inv[6] * T[11]);
  T_inv[11] = -(T_inv[8] * T[3] + T_inv[9] * T[7] + T_inv[10] * T[11]);

  // Bottom row
  T_inv[12] = 0.0;
  T_inv[13] = 0.0;
  T_inv[14] = 0.0;
  T_inv[15] = 1.0;

  return T_inv;
}

// Multiply two 4x4 transformation matrices (row-major order)
std::vector<double> multiplyTransforms(const std::vector<double> &A, const std::vector<double> &B) {
  assert(A.size() == 16 && B.size() == 16);
  std::vector<double> C(16, 0.0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        C[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
      }
    }
  }

  return C;
}

// Transform sensor pose to be relative to cam0 (cam0 becomes identity)
std::vector<double> transformToCam0Reference(const std::vector<double> &cam0_transform,
                                             const std::vector<double> &sensor_transform) {
  assert(cam0_transform.size() == 16 && sensor_transform.size() == 16);
  auto cam0_body = invertTransform(cam0_transform);
  return multiplyTransforms(cam0_body, sensor_transform);
}

// Load camera configuration from YAML
cuvslam::Camera loadCameraFromYAML(const std::string &yaml_path) {
  YAML::Node config = YAML::LoadFile(yaml_path);

  cuvslam::Camera cam;

  auto resolution = config["resolution"].as<std::vector<int>>();
  cam.size = {resolution[0], resolution[1]};

  auto intrinsics = config["intrinsics"].as<std::vector<double>>();
  cam.focal = {static_cast<float>(intrinsics[0]), static_cast<float>(intrinsics[1])};
  cam.principal = {static_cast<float>(intrinsics[2]), static_cast<float>(intrinsics[3])};

  std::string distortion_model = config["distortion_model"].as<std::string>();
  auto distortion_coeffs = config["distortion_coefficients"].as<std::vector<double>>();

  if (distortion_model == "radial-tangential") {
    cam.distortion.model = cuvslam::Distortion::Model::Brown;
    cam.distortion.parameters.resize(5, 0.0f);
    // only k1, k2, p1, p2 are provided in the default calibration
    for (size_t i = 0; i < std::min(distortion_coeffs.size(), 4ul); i++) {
      // so no need to set k3
      cam.distortion.parameters[i >= 2 ? i + 1 : i] = static_cast<float>(distortion_coeffs[i]);
    }
  } else if (distortion_model == "fisheye") {
    cam.distortion.model = cuvslam::Distortion::Model::Fisheye;
    cam.distortion.parameters.resize(4, 0.0f);
    for (size_t i = 0; i < std::min(distortion_coeffs.size(), 4ul); i++) {
      cam.distortion.parameters[i] = static_cast<float>(distortion_coeffs[i]);
    }
  } else {
    throw std::runtime_error("Unsupported distortion model: " + distortion_model);
  }

  return cam;
}

// Load IMU calibration from YAML
cuvslam::ImuCalibration loadImuFromYAML(const std::string &yaml_path) {
  YAML::Node config = YAML::LoadFile(yaml_path);

  cuvslam::ImuCalibration imu;

  imu.gyroscope_noise_density = config["gyroscope_noise_density"].as<float>();
  imu.gyroscope_random_walk = config["gyroscope_random_walk"].as<float>();
  imu.accelerometer_noise_density = config["accelerometer_noise_density"].as<float>();
  imu.accelerometer_random_walk = config["accelerometer_random_walk"].as<float>();
  imu.frequency = config["rate_hz"].as<float>();

  return imu;
}

// CSV data structures
struct CameraData {
  int64_t timestamp;
  std::string filename;
};

// Read camera CSV file
std::vector<CameraData> readCameraCSV(const std::string &csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open CSV: " + csv_path);
  }

  std::vector<CameraData> data;
  std::string line;

  // Skip header
  std::getline(file, line);

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string timestamp_str, filename;

    if (std::getline(iss, timestamp_str, ',') && std::getline(iss, filename)) {
      if (filename.empty()) {
        throw std::runtime_error("Empty filename in CSV: " + line);
      }
      if (filename.back() == '\r') {
        filename.pop_back();
      }
      CameraData cam_data;
      cam_data.timestamp = std::stoll(timestamp_str);
      cam_data.filename = filename;
      data.push_back(cam_data);
    }
  }

  return data;
}

// Read IMU CSV file
std::vector<cuvslam::ImuMeasurement> readImuCSV(const std::string &csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open CSV: " + csv_path);
  }

  std::vector<cuvslam::ImuMeasurement> data;
  std::string line;

  // Skip header
  std::getline(file, line);

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string timestamp_str;
    std::string gyro_x, gyro_y, gyro_z;
    std::string accel_x, accel_y, accel_z;

    if (std::getline(iss, timestamp_str, ',') && std::getline(iss, gyro_x, ',') && std::getline(iss, gyro_y, ',') &&
        std::getline(iss, gyro_z, ',') && std::getline(iss, accel_x, ',') && std::getline(iss, accel_y, ',') &&
        std::getline(iss, accel_z, ',')) {
      cuvslam::ImuMeasurement imu_measurement;
      imu_measurement.timestamp_ns = std::stoll(timestamp_str);
      imu_measurement.angular_velocities = {static_cast<float>(std::stod(gyro_x)),
                                            static_cast<float>(std::stod(gyro_y)),
                                            static_cast<float>(std::stod(gyro_z))};
      imu_measurement.linear_accelerations = {static_cast<float>(std::stod(accel_x)),
                                              static_cast<float>(std::stod(accel_y)),
                                              static_cast<float>(std::stod(accel_z))};
      data.push_back(imu_measurement);
    }
  }

  return data;
}

// Get Rig object from EuRoC dataset path with transformations relative to cam0
cuvslam::Rig getRig(const std::string &euroc_path) {
  // Determine which calibration files to use (recalibrated or default)
  std::string cam0_yaml = euroc_path + "/cam0/sensor_recalibrated.yaml";
  std::string cam1_yaml = euroc_path + "/cam1/sensor_recalibrated.yaml";
  std::string imu_yaml = euroc_path + "/imu0/sensor_recalibrated.yaml";
  bool is_default = false;

  std::ifstream test_file(cam0_yaml);
  if (!test_file.good()) {
    cam0_yaml = euroc_path + "/cam0/sensor.yaml";
    cam1_yaml = euroc_path + "/cam1/sensor.yaml";
    imu_yaml = euroc_path + "/imu0/sensor.yaml";
    is_default = true;
  }
  test_file.close();

  std::cout << "Using " << (is_default ? "default" : "recalibrated") << " calibration" << std::endl;

  // Load camera configurations
  auto cam0 = loadCameraFromYAML(cam0_yaml);
  auto cam1 = loadCameraFromYAML(cam1_yaml);

  // Load transformation matrices
  YAML::Node cam0_config = YAML::LoadFile(cam0_yaml);
  YAML::Node cam1_config = YAML::LoadFile(cam1_yaml);
  YAML::Node imu_config = YAML::LoadFile(imu_yaml);

  auto cam0_T_body = parseTransformMatrix(cam0_config, "T_BS");
  auto cam1_T_body = parseTransformMatrix(cam1_config, "T_BS");
  auto imu_T_body = parseTransformMatrix(imu_config, "T_BS");

  // Set cam0 as identity (rig frame)
  cam0.rig_from_camera = cuvslam::Pose();
  cam0.rig_from_camera.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
  cam0.rig_from_camera.translation = {0.0f, 0.0f, 0.0f};

  // Compute cam1 and IMU relative to cam0
  std::vector<double> cam1_T_cam0;
  std::vector<double> imu_T_cam0;

  if (is_default) {
    // Transform to cam0 reference frame
    cam1_T_cam0 = transformToCam0Reference(cam0_T_body, cam1_T_body);
    imu_T_cam0 = transformToCam0Reference(cam0_T_body, imu_T_body);
  } else {
    // For recalibrated data, transforms are already in cam0 frame
    cam1_T_cam0 = cam1_T_body;
    imu_T_cam0 = imu_T_body;
  }

  cam1.rig_from_camera = transformToPose(cam1_T_cam0);

  // Load IMU calibration
  auto imu = loadImuFromYAML(imu_yaml);
  imu.rig_from_imu = transformToPose(imu_T_cam0);

  // Create rig
  cuvslam::Rig rig;
  rig.cameras = {cam0, cam1};
  rig.imus = {imu};

  return rig;
}

// Initialize cuvslam::Image from ImageBuffer and metadata
void initializeImage(cuvslam::Image &image, const ImageBuffer &img_buffer, int64_t timestamp, uint32_t camera_index) {
  image.pixels = img_buffer.data.data();
  image.width = img_buffer.width;
  image.height = img_buffer.height;
  image.pitch = img_buffer.width;
  image.encoding = cuvslam::ImageData::Encoding::MONO;
  image.data_type = cuvslam::ImageData::DataType::UINT8;
  image.is_gpu_mem = false;
  image.timestamp_ns = timestamp;
  image.camera_index = camera_index;
}

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
      return 1;
    }
    std::string dataset_path = argv[1];
    std::cout << "Dataset path: " << dataset_path << std::endl;

    // Get version
    int32_t major, minor, patch;
    auto version_str = cuvslam::GetVersion(&major, &minor, &patch);
    std::cout << "cuVSLAM version: " << version_str << std::endl;

    // Set verbosity
    cuvslam::SetVerbosity(1);

#ifdef USE_RERUN
    // Initialize Rerun
    const auto rec = rerun::RecordingStream("cuVSLAM EuRoC Tracker");
    rec.spawn().exit_on_failure();

    // Set coordinate system (right-hand, Y-down, Z-forward)
    rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN);

    // Load blueprint from the same directory as the executable
    std::string exe_dir{"."};
#ifdef __linux__
    char exe_buf[PATH_MAX];
    ssize_t exe_len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    if (exe_len > 0) {
      exe_buf[exe_len] = '\0';
      exe_dir = std::string(exe_buf, exe_len);
      exe_dir = exe_dir.substr(0, exe_dir.rfind('/'));
    }
#endif
    std::string blueprint_path = exe_dir + "/euroc_blueprint.rbl";
    std::ifstream blueprint_file(blueprint_path);
    if (blueprint_file.good()) {
      rec.log_file_from_path(blueprint_path);
      std::cout << "Loaded blueprint from " << blueprint_path << std::endl;
    } else {
      std::cout << "Blueprint file not found. Using default layout." << std::endl;
    }
    blueprint_file.close();

    std::cout << "Rerun visualization enabled" << std::endl;
#endif

    // Get camera rig
    cuvslam::Rig rig = getRig(dataset_path);

    // Configure the tracker for visual-inertial odometry with SLAM
    cuvslam::Odometry::Config odometry_config;
    odometry_config.odometry_mode = cuvslam::Odometry::OdometryMode::Inertial;
    odometry_config.async_sba = false;
    odometry_config.rectified_stereo_camera = false;
    odometry_config.enable_observations_export = true;
    odometry_config.enable_landmarks_export = true;
    odometry_config.enable_final_landmarks_export = true;

    cuvslam::Slam::Config slam_config;
    slam_config.sync_mode = true;
    slam_config.enable_reading_internals = true;

    std::cout << "Initializing tracker (odometry + SLAM)..." << std::endl;
    cuvslam::Tracker tracker(rig, odometry_config, &slam_config);

    // Reading SLAM data layers is not mirrored on Tracker, so reach the SLAM instance directly.
    // It is non-null here because the config above enables SLAM.
    cuvslam::Slam &slam = tracker.GetSlam();

    // Enable SLAM data layers for visualization
    slam.EnableReadingData(cuvslam::Slam::DataLayer::Map, 100000);         // Map landmarks
    slam.EnableReadingData(cuvslam::Slam::DataLayer::LoopClosure, 10000);  // Loop closures
    // slam.EnableReadingData(cuvslam::Slam::DataLayer::PoseGraph, 10000);   // Future: Pose graph

    // Load camera timestamps
    auto cam0_data = readCameraCSV(dataset_path + "/cam0/data.csv");
    auto cam1_data = readCameraCSV(dataset_path + "/cam1/data.csv");

    if (cam0_data.size() != cam1_data.size()) {
      throw std::runtime_error("Camera 0 and 1 have different number of frames");
    }

    // Verify timestamps match
    for (size_t i = 0; i < cam0_data.size(); i++) {
      if (cam0_data[i].timestamp != cam1_data[i].timestamp) {
        throw std::runtime_error("Timestamp mismatch at frame " + std::to_string(i));
      }
    }

    std::cout << "Found " << cam0_data.size() << " camera frames" << std::endl;

    // Load IMU data
    auto imu_data = readImuCSV(dataset_path + "/imu0/data.csv");
    std::cout << "Found " << imu_data.size() << " IMU measurements" << std::endl;

    std::cout << "\nFrame |   Timestamp (ns)    |    Translation [x, y, z]    |"
                 "    Rotation [qx, qy, qz, qw]     | IMU count"
              << std::endl;
    std::cout << "------+---------------------+-----------------------------+"
                 "----------------------------------+----------"
              << std::endl;

    // Track each frame
    int tracked_frames = 0;
    size_t imu_index = 0;
    int64_t last_camera_timestamp = -1;
#ifdef USE_RERUN
    std::vector<rerun::Vec3D> trajectory_odom;
    std::vector<rerun::Vec3D> trajectory_slam;
    std::vector<rerun::Vec3D> loop_closure_poses;
#endif
    std::set<int64_t> reported_loop_closures;

    for (size_t frame = 0; frame < cam0_data.size(); frame++) {
      int64_t current_timestamp = cam0_data[frame].timestamp;

      // Register IMU measurements between camera frames
      int imu_count = 0;
      while (imu_index < imu_data.size() && imu_data[imu_index].timestamp_ns < current_timestamp) {
        // Only register IMU after first camera frame
        if (last_camera_timestamp >= 0) {
          tracker.RegisterImuMeasurement(0, imu_data[imu_index]);
          imu_count++;
        }
        imu_index++;
      }

      // Warn if no IMU measurements between frames
      if (last_camera_timestamp >= 0 && imu_count == 0) {
        std::cout << "Warning: No IMU measurements between timestamps " << last_camera_timestamp << " and "
                  << current_timestamp << std::endl;
      }

      last_camera_timestamp = current_timestamp;

      // Load images for left and right cameras
      std::string img0_path = dataset_path + "/cam0/data/" + cam0_data[frame].filename;
      std::string img1_path = dataset_path + "/cam1/data/" + cam1_data[frame].filename;

      auto img_left = loadPNG(img0_path);
      auto img_right = loadPNG(img1_path);

      // Create Image structures
      cuvslam::Odometry::ImageSet images(2);

      initializeImage(images[0], img_left, current_timestamp, 0);
      initializeImage(images[1], img_right, current_timestamp, 1);

      // Track: runs odometry and, since SLAM is enabled, feeds it the odometry state
      auto result = tracker.Track(images);

      if (result.odometry.world_from_rig.has_value()) {
        const auto &pose = result.odometry.world_from_rig.value().pose;
        const auto &t = pose.translation;
        const auto &q = pose.rotation;

        std::cout << std::setw(5) << frame << " | " << std::setw(14) << current_timestamp << " | " << std::fixed
                  << std::setprecision(3) << "[" << std::setw(7) << t[0] << ", " << std::setw(7) << t[1] << ", "
                  << std::setw(7) << t[2] << "] | "
                  << "[" << std::setw(6) << q[0] << ", " << std::setw(6) << q[1] << ", " << std::setw(6) << q[2] << ", "
                  << std::setw(6) << q[3] << "] | " << std::setw(9) << imu_count << std::endl;

        tracked_frames++;

        // Get loop closure poses
        std::vector<cuvslam::PoseStamped> current_lc_poses;
        slam.GetLoopClosurePoses(current_lc_poses);
        for (const auto &lc : current_lc_poses) {
          if (reported_loop_closures.find(lc.timestamp_ns) == reported_loop_closures.end()) {
            reported_loop_closures.insert(lc.timestamp_ns);
#ifdef USE_RERUN
            loop_closure_poses.push_back({lc.pose.translation[0], lc.pose.translation[1], lc.pose.translation[2]});
#endif
            std::cout << "  Loop closure detected at timestamp " << lc.timestamp_ns << std::endl;
          }
        }

#ifdef USE_RERUN
        rec.set_time_sequence("frame", static_cast<int64_t>(frame));

        // Log odometry trajectory
        trajectory_odom.push_back({t[0], t[1], t[2]});
        rec.log_static("world/trajectory_odom", rerun::LineStrips3D(rerun::components::LineStrip3D(trajectory_odom))
                                                    .with_colors({rerun::Color(0, 64, 255)}));

        // Log SLAM trajectory. Track() already returned the SLAM pose for this frame.
        const auto &st = result.slam->translation;
        trajectory_slam.push_back({st[0], st[1], st[2]});
        rec.log_static("world/trajectory_slam", rerun::LineStrips3D(rerun::components::LineStrip3D(trajectory_slam))
                                                    .with_colors({rerun::Color(0, 224, 0)}));

        // Log loop closure poses
        if (!loop_closure_poses.empty()) {
          rec.log_static(
              "world/loop_closure_poses",
              rerun::Points3D(loop_closure_poses).with_radii({0.05f}).with_colors({rerun::Color(255, 0, 0)}));
        }

        // Read and visualize SLAM data layers

        // Map landmarks
        auto map_landmarks = slam.ReadLandmarks(cuvslam::Slam::DataLayer::Map);
        if (map_landmarks && !map_landmarks->landmarks.empty()) {
          std::vector<rerun::Vec3D> map_positions;
          map_positions.reserve(map_landmarks->landmarks.size());
          for (const auto &lm : map_landmarks->landmarks) {
            map_positions.push_back({lm.coords[0], lm.coords[1], lm.coords[2]});
          }
          rec.log_static("world/slam_map",
                         rerun::Points3D(map_positions).with_radii({0.02f}).with_colors({rerun::Color(128, 128, 128)}));
        }

        // Loop closures landmarks
        auto lc_landmarks = slam.ReadLandmarks(cuvslam::Slam::DataLayer::LoopClosure);
        if (lc_landmarks && !lc_landmarks->landmarks.empty()) {
          std::vector<rerun::Vec3D> lc_positions;
          lc_positions.reserve(lc_landmarks->landmarks.size());
          for (const auto &lm : lc_landmarks->landmarks) {
            lc_positions.push_back({lm.coords[0], lm.coords[1], lm.coords[2]});
          }
          rec.log_static("world/loop_closure_landmarks",
                         rerun::Points3D(lc_positions).with_radii({0.025f}).with_colors({rerun::Color(255, 0, 0)}));
        }

        // Future extension example - Pose Graph
        // auto pose_graph = slam.ReadPoseGraph();
        // if (pose_graph && !pose_graph->nodes.empty()) {
        //   // Visualize pose graph nodes and edges
        // }

        // Log camera pose
        rec.log("world/camera_0",
                rerun::Transform3D::from_translation_rotation(rerun::Vec3D{t[0], t[1], t[2]},
                                                              rerun::Quaternion::from_xyzw(q[0], q[1], q[2], q[3])));

        // Log coordinate axes
        std::vector<rerun::Vec3D> axes_vectors = {
            {0.2f, 0.0f, 0.0f},  // X-axis (red)
            {0.0f, 0.2f, 0.0f},  // Y-axis (green)
            {0.0f, 0.0f, 0.2f}   // Z-axis (blue)
        };
        std::vector<rerun::Color> axes_colors = {rerun::Color(255, 0, 0), rerun::Color(0, 255, 0),
                                                 rerun::Color(0, 0, 255)};
        rec.log("world/camera_0/axes", rerun::Arrows3D::from_vectors(axes_vectors).with_colors(axes_colors));

        // Log observations
        auto observations = tracker.GetOdometry().GetLastObservations(0);
        if (!observations.empty()) {
          std::vector<rerun::Vec2D> points;
          std::vector<rerun::Color> colors;

          for (const auto &obs : observations) {
            points.push_back({obs.u, obs.v});
            auto color = colorFromId(obs.id);
            colors.push_back(rerun::Color(color[0], color[1], color[2]));
          }

          rec.log("world/camera_0/observations", rerun::Points2D(points).with_colors(colors).with_radii({5.0f}));
        }

        // Log camera image
        rec.log("world/camera_0/image",
                rerun::Image::from_grayscale8(
                    img_left.data, {static_cast<uint32_t>(img_left.width), static_cast<uint32_t>(img_left.height)}));

        rec.log("world/camera_1/image",
                rerun::Image::from_grayscale8(
                    img_right.data, {static_cast<uint32_t>(img_right.width), static_cast<uint32_t>(img_right.height)}));

        // Log gravity vector
        auto gravity = tracker.GetOdometry().GetLastGravity();
        if (gravity.has_value()) {
          const auto &g = gravity.value();
          rec.log("world/camera_0/gravity",
                  rerun::Arrows3D::from_vectors({{g[0], g[1], g[2]}}).with_colors({rerun::Color(255, 0, 0)}));
        }

        // Log IMU data (last measurement)
        if (imu_index > 0) {
          const auto &imu_last = imu_data[imu_index - 1];
          rec.log("world/imu/accel/x", rerun::Scalars(imu_last.linear_accelerations[0]));
          rec.log("world/imu/accel/y", rerun::Scalars(imu_last.linear_accelerations[1]));
          rec.log("world/imu/accel/z", rerun::Scalars(imu_last.linear_accelerations[2]));
          rec.log("world/imu/gyro/x", rerun::Scalars(imu_last.angular_velocities[0]));
          rec.log("world/imu/gyro/y", rerun::Scalars(imu_last.angular_velocities[1]));
          rec.log("world/imu/gyro/z", rerun::Scalars(imu_last.angular_velocities[2]));
        }
#endif
      } else {
        std::cout << std::setw(5) << frame << " | " << std::setw(14) << current_timestamp << " | "
                  << "TRACKING FAILED" << std::endl;
      }
    }

    std::cout << "\nTracked " << tracked_frames << " frames out of " << cam0_data.size() << " with "
              << reported_loop_closures.size() << " loop closures" << std::endl;

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}


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
#include <chrono>
#include <experimental/iterator>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "json/json.h"

#include "camera_rig_edex/camera_rig_edex.h"
#include "camera_rig_edex/repeated_camera_rig_edex.h"
#include "camera_rig_edex/shuttle_camera_rig_edex.h"
#include "common/coordinate_system.h"
#include "common/json_to_eigen.h"
#include "common/log.h"
#include "common/stream.h"
#include "common/time.h"
#include "common/types.h"
#include "common/utils.h"
#include "cuvslam/cuvslam2.h"
#include "cuvslam/cuvslam_params.h"
#include "cuvslam/internal.h"
#include "params/cli.h"
#include "params/registry.h"
#include "utils/image_loader.h"

using namespace cuvslam;

// Get default configurations to use for flag defaults
namespace {
const Odometry::Config kDefaultOdomCfg = Odometry::GetDefaultConfig();
const Slam::Config kDefaultSlamCfg = Slam::GetDefaultConfig();
}  // namespace

DEFINE_int32(verbosity, 2, "Verbosity level");
DEFINE_string(params, "",
              "Path to a parameter file: one 'key: value' or 'key = value' per line, # starts a comment. "
              "Covers configuration (odometry.*, slam.*) and solver parameters (sof.*, sba.*, vo_pnp.*, ...). "
              "Repeatable -Pkey=value overrides single entries and wins over the file; --list_params prints "
              "every name with its type, default and description.");
DEFINE_bool(list_params, false, "Print every parameter with its type, default and description, then exit");
// replay settings
DEFINE_string(edex, "", "Path to the .edex file to run on (required)");
DEFINE_string(cameras, "", "Comma-separated list of cameras");
DEFINE_int32(start_frame, 0, "Frame to start with");
DEFINE_int32(repeat, 1, "Number of repetitions of the dataset");
DEFINE_bool(shuttle, false, "Shuttle replay forward and backward");
DEFINE_double(max_fps, 0., "Throttle main loop to not exceed max_fps. 0 (default) to disable");
DEFINE_bool(ignore_tracking_errors, false, "Don't stop on tracking errors");
// output settings
DEFINE_string(print_odom_poses, "", "Path to save odometry poses");
DEFINE_string(print_slam_poses, "", "Path to save SLAM poses");
DEFINE_string(print_loc_poses, "", "Path to save localization poses");
DEFINE_string(print_format, "matrix", "Format of output poses (matrix, tum)");
DEFINE_bool(print_nan_on_failure, false, "Print a NaN pose when the localization failed");
DEFINE_bool(ros_frame_conversion, false, "Convert input/output poses from/to ROS frame");
DEFINE_string(output_map, "", "Path to output map (cuVSLAM will try to save it at the end)");
DEFINE_string(print_map_keyframes, "", "Path to save map keyframes");
DEFINE_bool(print_stats, true, "Print an end-of-run performance and observation summary to stdout");
// hinted localization settings
DEFINE_string(loc_input_map, "", "Path to input map (cuVSLAM will try to localize in it)");
DEFINE_string(loc_input_hints, "", "Path to hint data for localization (format: `timestamp x y z`)");
DEFINE_string(loc_hint_ts_format, "detect", "Localization hint timestamp format (detect, int, float)");
DEFINE_int32(loc_start_frame, 0, "Start frame for localization");
DEFINE_int32(loc_skip_frames, 0, "Number of frames to skip after previous localization");
DEFINE_int32(loc_retries, 0, "Number of retries if localization fails");
DEFINE_double(loc_hint_noise, 0., "Introduce noise to hints (uniform, max deviation in meters)");
DEFINE_bool(loc_random_rot, false, "Randomize hint rotation");
DEFINE_bool(localize_wait, false, "Wait for localization to finish (not the same as reproduce_mode in slam)");
DEFINE_bool(localize_forever, false, "Run localization continuously (each time previous call finished)");
// cuvslam configuration
DEFINE_string(debug_dump, "", "Path to debug dump");
DEFINE_int32(cfg_odom_mode, static_cast<int>(kDefaultOdomCfg.odometry_mode),
             "Odometry mode: Multicamera (0), Inertial (1), RGBD (2), Mono (3)");
DEFINE_int32(cfg_multicam_mode, static_cast<int>(kDefaultOdomCfg.multicam_mode),
             "Multicamera mode: performance (0), precision (1), or moderate (2)");
DEFINE_bool(cfg_async_sba, false, "Enable asynchronous sparse bundle adjustment");
DEFINE_bool(cfg_denoising, kDefaultOdomCfg.use_denoising, "Enable image denoising");
DEFINE_bool(cfg_horizontal, kDefaultOdomCfg.rectified_stereo_camera,
            "Enable tracking for rectified cameras with principal points on the horizontal line");
DEFINE_bool(cfg_planar, kDefaultSlamCfg.planar_constraints,
            "Slam poses are so that the camera moves on a horizontal plane");
DEFINE_bool(cfg_enable_slam, false, "Enable localization and mapping");
DEFINE_bool(cfg_sync_slam, kDefaultSlamCfg.sync_mode,
            "Run localization and mapping in the same thread with visual odometry");
DEFINE_int32(cfg_slam_max_map_size, kDefaultSlamCfg.max_map_size,
             "Maximum numbers of poses in SLAM pose graph, 0 means unlimited pose-graph");
DEFINE_double(cfg_max_frame_delta_s, kDefaultOdomCfg.max_frame_delta_s,
              "Set maximum camera frame time in seconds to warn users");
DEFINE_bool(cfg_enable_export, false, "Enable export of observations & landmarks");
// Tracker-compatible SLAM flags (load map + localize on a chosen edex frame)
DEFINE_int32(slam_simulate_slow_map_load, 0, "Delay in ms invoked from localize start_cb (after map load begins).");
DEFINE_string(slam_input_database, "", "Folder with SLAM map DB to localize in (LocalizeInMap).");
DEFINE_string(slam_localize_image, "", "Optional image path for localization cam0; if empty, use current frame.");
DEFINE_string(slam_localize_guess_translation, "",
              "Guess translation only, format: \"[float, float, float]\" (identity rotation).");
DEFINE_double(slam_load_and_localize_timestamp, -1,
              "Timestamp in seconds for LocalizeInMap; if <=0 use current frame image timestamp.");
DEFINE_int32(slam_load_and_localize_on_frame, -1, "If >=0, run LocalizeInMap when edex frame_number matches.");
DEFINE_bool(slam_reproduce_mode, true, "If set: SLAM sync_mode (synced SLAM thread / reproduce semantics).");
DEFINE_int32(max_pose_graph_nodes, 300, "SLAM pose graph node limit (maps to Slam::Config.max_map_size).");
// image crop settings
DEFINE_int32(border_top, 0, "top border to ignore in pixels (0 to use full frame)");
DEFINE_int32(border_bottom, 0, "bottom border to ignore in pixels (0 to use full frame)");
DEFINE_int32(border_left, 0, "left border to ignore in pixels (0 to use full frame)");
DEFINE_int32(border_right, 0, "right border to ignore in pixels (0 to use full frame)");
// rgbd settings
// set default to 0, so users don't have to specify it explicitly for the most common case,
// while library defaults to -1 to avoid errors
DEFINE_int32(cfg_depth_camera, 0, "Depth camera index");
DEFINE_double(cfg_depth_scale_factor, kDefaultOdomCfg.rgbd_settings.depth_scale_factor, "Depth scale factor");
DEFINE_bool(cfg_enable_depth_stereo_tracking, kDefaultOdomCfg.rgbd_settings.enable_depth_stereo_tracking,
            "Enable depth stereo tracking");
/// One entry of a parameter file, with the line it came from so errors can point at it.
struct ParamEntry {
  std::string key;
  std::string value;
  size_t line = 0;  ///< 0 for a -P override, which has no line
};

#define VERIFY_TRACE(condition, ...) \
  if (!(condition)) {                \
    TraceError(__VA_ARGS__);         \
    return false;                    \
  }

struct Hint {
  int64_t timestamp;
  std::array<float, 3> hint;
};
namespace {

void setStreamFormat(std::ofstream& out_poses) {
  out_poses << std::fixed << std::setprecision(9)
            << PoseIOManip(FLAGS_print_format == "tum" ? PoseFormat::TUM : PoseFormat::MATRIX,
                           FLAGS_ros_frame_conversion);
}

// Accumulates per-frame timing and visible-observation counts over a tracking run and prints an
// end-of-run summary. Only receives numbers from the tracking loop; no cuVSLAM state of its own.
struct RunStats {
  size_t num_frames = 0;
  double total_track_seconds = 0.0;
  double max_track_seconds = 0.0;
  FrameId max_track_frame = 0;
  bool has_observations = false;
  size_t total_observations = 0;
  size_t min_observations = std::numeric_limits<size_t>::max();
  FrameId min_observations_frame = 0;

  // One successfully tracked frame. num_observations is empty when observation export is disabled.
  void AddFrame(double track_seconds, std::optional<size_t> num_observations, FrameId frame) {
    ++num_frames;
    total_track_seconds += track_seconds;
    if (track_seconds > max_track_seconds) {
      max_track_seconds = track_seconds;
      max_track_frame = frame;
    }
    if (num_observations.has_value()) {
      has_observations = true;
      total_observations += num_observations.value();
      if (num_observations.value() < min_observations) {
        min_observations = num_observations.value();
        min_observations_frame = frame;
      }
    }
  }
};

// total_wall_seconds is measured by the caller around the tracking loop (fps basis).
void printRunStats(std::ostream& stream, const RunStats& stats, double total_wall_seconds) {
  if (stats.num_frames == 0) {
    stream << "No frames tracked." << std::endl;
    return;
  }
  const double fps = total_wall_seconds > 0 ? static_cast<double>(stats.num_frames) / total_wall_seconds : 0.0;
  const double avg_track_seconds = stats.total_track_seconds / static_cast<double>(stats.num_frames);
  stream << "frames = " << stats.num_frames << ", duration = " << total_wall_seconds << " s, fps = " << fps << "\n";
  stream << "average track() duration = " << avg_track_seconds
         << " s, max track() duration = " << stats.max_track_seconds << " s (in frame #" << stats.max_track_frame
         << ")\n";
  if (stats.has_observations) {
    const double avg_observations =
        static_cast<double>(stats.total_observations) / static_cast<double>(stats.num_frames);
    stream << "average visible observations = " << avg_observations
           << ", min visible observations = " << stats.min_observations << " (in frame #"
           << stats.min_observations_frame << ")\n";
  } else {
    stream << "observation stats unavailable: rerun with --cfg_enable_export\n";
  }
  stream << std::flush;
}

template <typename T>
bool loadValues(const std::string& file_name, std::vector<std::vector<T>>& values) {
  std::ifstream file(file_name);
  VERIFY_TRACE(file.is_open(), "Unable to open file %s", file_name.c_str());

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::vector<T> vec;
    T val;
    while (ss >> val) {
      vec.push_back(val);
    }
    values.emplace_back(std::move(vec));
  }

  return true;
}

double getTimestampMultiplier(const std::vector<std::vector<double>>& values) {
  if (FLAGS_loc_hint_ts_format == "detect") {
    // if all timestamps are integer
    if (std::all_of(values.begin(), values.end(),
                    [](const std::vector<double>& vec) { return vec[0] == static_cast<int64_t>(vec[0]); })) {
      return 1;
    } else {
      return 1e9;
    }
  } else if (FLAGS_loc_hint_ts_format == "float") {
    return 1e9;
  } else if (FLAGS_loc_hint_ts_format == "int") {
    return 1;
  }
  TraceError("Unknown hint timestamp format %s", FLAGS_loc_hint_ts_format.c_str());
  return 1;
}

bool loadHints(const std::string& file_name, std::vector<Hint>& hints) {
  std::vector<std::vector<double>> values;
  if (!loadValues(file_name, values)) {
    return false;
  }
  const double mul = getTimestampMultiplier(values);
  for (auto&& vec : values) {
    VERIFY_TRACE(vec.size() == 4 || vec.size() == 8, "4/8 values per line expected (ts, x, y, z [, quaternion])");
    auto ts = static_cast<int64_t>(vec[0] * mul);
    VERIFY_TRACE(hints.empty() || ts > hints.back().timestamp, "Hints must be sorted by timestamp (%zu > %zu)", ts,
                 hints.back().timestamp);
    hints.push_back({ts, {(float)vec[1], (float)vec[2], (float)vec[3]}});
  }
  return !hints.empty();
}

Pose getHintPose(std::array<float, 3> hint, float noise, bool random_rot) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dis(-noise, noise);
  for (float& v : hint) {
    v += dis(gen);
  }
  Isometry3T transform(Translation3T{hint[0], hint[1], hint[2]});
  if (random_rot) {
    transform.rotate(Matrix3T::Random());
  }
  if (FLAGS_ros_frame_conversion) {
    transform = kCuvslamFromRos * transform * kRosFromCuvslam;
  }
  Pose pose;
  vec<3>(pose.translation) = transform.translation();
  Eigen::Quaternionf quat(transform.linear());
  pose.rotation = {quat.x(), quat.y(), quat.z(), quat.w()};
  return pose;
}

// get the latest hint, but no later than a given frame
bool getLatestHint(const std::vector<Hint>& hints, int64_t timestamp, Hint& out) {
  auto it = std::upper_bound(hints.begin(), hints.end(), timestamp,
                             [](int64_t ts, const Hint& hint) { return ts < hint.timestamp; });
  if (it != hints.begin()) {
    out = *(--it);
    return true;
  }
  return false;
}

std::ostream& operator<<(std::ostream& stream, const Pose& pose) {
  Isometry3T iso_pose;
  iso_pose.setIdentity();
  Eigen::Quaternionf quat{pose.rotation.data()};
  iso_pose.linear() = quat.toRotationMatrix();
  iso_pose.translation() = vec<3>(pose.translation);
  return ::cuvslam::operator<<(stream, iso_pose);
}

void printTsPose(std::ostream& stream, bool is_valid, int64_t ts, const Pose& pose) {
  constexpr const float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr const Pose kNanPose = {{kNaN, kNaN, kNaN, kNaN}, {kNaN, kNaN, kNaN}};
  if (stream) {
    const auto timestamp = Timestamp(ts);
    if (is_valid) {
      stream << timestamp << " " << pose << std::endl;
    } else if (FLAGS_print_nan_on_failure) {
      stream << timestamp << " " << kNanPose << std::endl;
    }
  }
}

// Asynchronous response for CUVSLAM_LocalizeInExistDb()
enum class LocalizeInMapStatus { NOT_LOCALIZED, IN_PROGRESS, LOCALIZED };

struct LocalizeInMapContext {
  LocalizeInMapStatus status;
  int64_t timestamp;
  std::ofstream out_poses;
};

void localize_in_exist_db_response(const Result<Pose>& result, LocalizeInMapContext* ctx) {
  TraceDebug("LocalizeInMap sent status: %s", result.data.has_value() ? "SUCCESS" : result.error_message.data());
  printTsPose(ctx->out_poses, result.data.has_value(), ctx->timestamp,
              result.data.has_value() ? result.data.value() : Pose{});
  ctx->status = (result.data.has_value() ? LocalizeInMapStatus::LOCALIZED : LocalizeInMapStatus::NOT_LOCALIZED);
}

// Asynchronous response for SaveMap()
enum class SaveMapStatus { NOT_SAVED, IN_PROGRESS, SAVED };

void save_to_slam_db_response(bool success, SaveMapStatus* context) {
  TraceDebug("SaveMap sent status: %s", success ? "SUCCESS" : "FAILED");
  *context = (success ? SaveMapStatus::SAVED : SaveMapStatus::NOT_SAVED);
}

Rig createRig(const edex::EdexFile& edex_file, const camera_rig_edex::ICameraRigReplay* edex_rig) {
  Rig rig;
  rig.cameras.reserve(edex_rig->getCamerasNum());
  for (size_t i = 0; i < edex_rig->getCamerasNum(); i++) {
    const edex::Camera& cam = edex_file.cameras_[edex_rig->getCameraIds()[i]];

    Camera camera{};
    camera.size[0] = cam.intrinsics.resolution.x();
    camera.size[1] = cam.intrinsics.resolution.y();
    camera.principal = {cam.intrinsics.principal.x(), cam.intrinsics.principal.y()};
    camera.focal = {cam.intrinsics.focal.x(), cam.intrinsics.focal.y()};

    // Legacy EDEX files store rig extrinsics in pre-OpenCV cuVSLAM frame.
    camera.rig_from_camera = ConvertIsometryToPose(LegacyEdexIsometryToOpenCV(cam.transform));

    camera.distortion.model = camera::StringToDistortionModel(cam.intrinsics.distortion_model);
    camera.distortion.parameters = cam.intrinsics.distortion_params;
    camera.border_top = FLAGS_border_top;
    camera.border_bottom = FLAGS_border_bottom;
    camera.border_left = FLAGS_border_left;
    camera.border_right = FLAGS_border_right;

    rig.cameras.push_back(std::move(camera));
  }

  // Add IMU calibration if available
  if (!edex_file.imu_.imu_log_path_.empty()) {
    ImuCalibration imu_calib;
    imu_calib.rig_from_imu = ConvertIsometryToPose(LegacyEdexImuExtrinsicToOpenCV(edex_file.imu_.transform));
    imu_calib.gyroscope_noise_density = edex_file.imu_.gyroscope_noise_density;
    imu_calib.gyroscope_random_walk = edex_file.imu_.gyroscope_random_walk;
    imu_calib.accelerometer_noise_density = edex_file.imu_.accelerometer_noise_density;
    imu_calib.accelerometer_random_walk = edex_file.imu_.accelerometer_random_walk;
    imu_calib.frequency = edex_file.imu_.frequency;
    rig.imus.push_back(imu_calib);
  }

  return rig;
}

// Tracker-style load + LocalizeInMap on a specific edex frame (gflags select the frame and paths).
bool RunSlamLoadAndLocalizeOnMatchingFrame(Slam& slam, const Slam::ImageSet& images) {
  constexpr int64_t kNsPerSecond = 1000000000LL;
  const int64_t timestamp_ns =
      FLAGS_slam_load_and_localize_timestamp > 0
          ? static_cast<int64_t>(FLAGS_slam_load_and_localize_timestamp * static_cast<double>(kNsPerSecond))
          : images[0].timestamp_ns;

  Vector3T translation = Vector3T::Zero();
  if (!FLAGS_slam_localize_guess_translation.empty() &&
      !ReadEigenFromString(FLAGS_slam_localize_guess_translation, translation)) {
    TraceError("Invalid slam_localize_guess_translation");
    return false;
  }
  Pose guess_pose;
  guess_pose.translation = {translation.x(), translation.y(), translation.z()};

  Slam::ImageSet localize_images;
  std::vector<uint8_t> loaded_pixels;
  if (!FLAGS_slam_localize_image.empty()) {
    utils::ImageLoaderT loader;
    VERIFY_TRACE(loader.load(FLAGS_slam_localize_image), "Failed to load slam_localize_image %s",
                 FLAGS_slam_localize_image.c_str());
    const ImageMatrix<uint8_t>& mat = loader.getImage();
    loaded_pixels.resize(static_cast<size_t>(mat.size()));
    std::copy_n(mat.data(), mat.size(), loaded_pixels.begin());
    localize_images.push_back(
        Image{{loaded_pixels.data(), static_cast<int32_t>(mat.cols()), static_cast<int32_t>(mat.rows()), 0,
               Image::Encoding::MONO, ImageData::DataType::UINT8, false},
              timestamp_ns,
              0u});
  } else {
    localize_images = images;
  }

  Slam::LocalizationSettings loc_settings{};
  loc_settings.horizontal_search_radius = 1.5f;
  loc_settings.vertical_search_radius = 0.5f;
  loc_settings.horizontal_step = 0.5f;
  loc_settings.vertical_step = 0.25f;
  loc_settings.angular_step_rads = 2 * PI / 36;

  const Slam::LocalizeStartCB start_cb = [] {
    if (FLAGS_slam_simulate_slow_map_load > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(FLAGS_slam_simulate_slow_map_load));
    }
  };

  slam.LocalizeInMap(FLAGS_slam_input_database, timestamp_ns, guess_pose, localize_images, loc_settings, start_cb,
                     nullptr);
  return true;
}

bool trackEdexDataSet(const std::string& edex_name, const Odometry::Config& odom_cfg, const Slam::Config& slam_cfg,
                      const std::vector<ParamEntry>& solver_params, const std::string& input_map_name,
                      const std::string& output_map_name) {
  std::vector<CameraId> camera_ids{StringToIntVector<CameraId>(FLAGS_cameras, ',')};
  std::unique_ptr<camera_rig_edex::ICameraRigReplay> edex_rig;
  if (FLAGS_shuttle) {
    edex_rig = std::make_unique<camera_rig_edex::ShuttleCameraRigEdex>(
        std::make_unique<camera_rig_edex::CameraRigEdex>(edex_name, camera_ids), FLAGS_repeat);
  } else {
    edex_rig = std::make_unique<camera_rig_edex::RepeatedCameraRigEdex>(
        std::make_unique<camera_rig_edex::CameraRigEdex>(edex_name, camera_ids), FLAGS_repeat);
  }

  edex::EdexFile edex_file;
  VERIFY_TRACE(edex_file.read(edex_name), "Failed to read %s\n", edex_name.c_str());

  const ErrorCode ret = edex_rig->start();
  VERIFY_TRACE(ret == ErrorCode::S_True, "Failed to start camera rig %s\n", ret.str());

  WarmUpGPU();
  Rig rig = createRig(edex_file, edex_rig.get());
  std::unique_ptr<Odometry> odom = std::make_unique<Odometry>(rig, odom_cfg);
  // Solver parameters are only reachable once the tracker exists, since which of them apply
  // depends on the odometry mode it was built with.
  for (const ParamEntry& entry : solver_params) {
    odom->SetParameter(entry.key, entry.value);
  }
  TraceMessage("Odometry tracker created");

  std::unique_ptr<Slam> slam;
  if (FLAGS_cfg_enable_slam) {
    slam = std::make_unique<Slam>(rig, odom->GetPrimaryCameras(), slam_cfg);
    TraceMessage("SLAM created");
  }

  edex_rig->registerIMUCallback([&](const imu::ImuMeasurement& measurement) {
    if (odom_cfg.odometry_mode != Odometry::OdometryMode::Inertial) {
      return;
    }
    ImuMeasurement imu_measurement;
    imu_measurement.timestamp_ns = measurement.time_ns;
    imu_measurement.linear_accelerations = {measurement.linear_acceleration.x(), measurement.linear_acceleration.y(),
                                            measurement.linear_acceleration.z()};
    imu_measurement.angular_velocities = {measurement.angular_velocity.x(), measurement.angular_velocity.y(),
                                          measurement.angular_velocity.z()};
    odom->RegisterImuMeasurement(0, imu_measurement);
  });

  VERIFY_TRACE(FLAGS_loc_input_map.empty() == FLAGS_loc_input_hints.empty(),
               "Both `loc_input_map` and `loc_input_hints` must be provided for localization");
  std::vector<Hint> hints;
  if (!FLAGS_loc_input_hints.empty()) {
    VERIFY_TRACE(loadHints(FLAGS_loc_input_hints, hints), "Failed to load localization hints from %s",
                 FLAGS_loc_input_hints.c_str());
  }

  std::ofstream out_odom_poses(FLAGS_print_odom_poses);
  std::ofstream out_slam_poses(FLAGS_print_slam_poses);

  LocalizeInMapContext loc_context{LocalizeInMapStatus::NOT_LOCALIZED, 0, std::ofstream(FLAGS_print_loc_poses)};

  setStreamFormat(out_odom_poses);
  setStreamFormat(out_slam_poses);
  setStreamFormat(loc_context.out_poses);

  if (FLAGS_localize_forever) {
    FLAGS_loc_retries = std::numeric_limits<int32_t>::max();
  }

  FrameId frame = static_cast<FrameId>(FLAGS_start_frame);
  edex_rig->setCurrentFrame(frame);
  FrameId next_loc_frame = FLAGS_loc_start_frame;
  RunStats run_stats;
  const auto run_start = std::chrono::steady_clock::now();
  bool success = true;
  while (true) {
    ScopedThrottler throttle(FLAGS_max_fps);
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

    std::vector<Image> images;
    std::vector<Image> masks;
    std::vector<Image> depths;
    std::vector<CameraId> frame_camera_ids;
    frame_camera_ids.reserve(cur_sources.size());
    for (CameraId cam_id = 0; cam_id < cur_sources.size(); ++cam_id) {
      if (cur_sources[cam_id].data != nullptr) {
        frame_camera_ids.push_back(cam_id);
      }
    }
    std::sort(frame_camera_ids.begin(), frame_camera_ids.end());

    for (CameraId cam_id : frame_camera_ids) {
      const auto& src = cur_sources[cam_id];
      const auto& meta = cur_meta[cam_id];
      images.emplace_back(Image{{src.data, meta.shape.width, meta.shape.height, src.pitch,
                                 static_cast<Image::Encoding>(src.image_encoding), ImageData::DataType::UINT8, false},
                                meta.timestamp,
                                static_cast<uint32_t>(cam_id)});
      if (masks_sources[cam_id].data != nullptr) {
        const auto& mask_src = masks_sources[cam_id];
        masks.emplace_back(
            Image{{mask_src.data, meta.mask_shape.width, meta.mask_shape.height, mask_src.pitch,
                   static_cast<Image::Encoding>(mask_src.image_encoding), ImageData::DataType::UINT8, false},
                  meta.timestamp,
                  static_cast<uint32_t>(cam_id)});
      }
    }
    if (odom_cfg.odometry_mode == Odometry::OdometryMode::RGBD) {
      for (CameraId cam_id = 0; cam_id < depth_sources.size(); ++cam_id) {
        const auto& depth_src = depth_sources[cam_id];
        if (depth_src.data != nullptr) {
          const auto& meta = cur_meta[cam_id];
          const auto depth_data_type =
              depth_src.type == ImageSource::F32 ? ImageData::DataType::FLOAT32 : ImageData::DataType::UINT16;
          depths.emplace_back(Image{{depth_src.data, meta.shape.width, meta.shape.height, depth_src.pitch,
                                     Image::Encoding::MONO, depth_data_type, false},
                                    meta.timestamp,
                                    static_cast<uint32_t>(cam_id)});
        }
      }
    }

    const auto track_start = std::chrono::steady_clock::now();
    PoseEstimate pose_estimate = odom->Track(images, masks, depths);
    const double track_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - track_start).count();
    if (!pose_estimate.world_from_rig.has_value()) {
      TraceWarning("Track(): Tracking lost at frame %zu.", frame);
      if (FLAGS_ignore_tracking_errors) {
        continue;
      } else {
        success = false;
        break;
      }
    }
    auto odom_pose = pose_estimate.world_from_rig.value().pose;
    printTsPose(out_odom_poses, true, pose_estimate.timestamp_ns, odom_pose);

    if (slam) {
      Odometry::State state;
      odom->GetState(state);
      slam->Track(state);
      const Pose slam_pose = slam->GetPose();
      printTsPose(out_slam_poses, true, pose_estimate.timestamp_ns, slam_pose);

      const auto frame_source = std::find_if(cur_sources.begin(), cur_sources.end(),
                                             [](const ImageSource& source) { return source.data != nullptr; });
      const size_t frame_source_index = static_cast<size_t>(std::distance(cur_sources.begin(), frame_source));
      if (FLAGS_slam_load_and_localize_on_frame >= 0 && !FLAGS_slam_input_database.empty() && !cur_meta.empty() &&
          frame_source != cur_sources.end() && frame_source_index < cur_meta.size()) {
        const auto& frame_meta = cur_meta[frame_source_index];
        if (frame_meta.frame_number == FLAGS_slam_load_and_localize_on_frame) {
          if (!RunSlamLoadAndLocalizeOnMatchingFrame(*slam, images)) {
            success = false;
            break;
          }
        }
      }
    }

    std::optional<size_t> num_observations;
    if (odom_cfg.enable_observations_export) {
      // Observations for the reference camera (camera 0).
      num_observations = odom->GetLastObservations(0).size();
      TraceDebug("Exported %zu observations at frame %zu", num_observations.value(), frame);
    }
    run_stats.AddFrame(track_seconds, num_observations, frame);

    if (odom_cfg.enable_landmarks_export) {
      std::vector<Landmark> landmarks = odom->GetLastLandmarks();
      TraceDebug("Exported %zu landmarks at frame %zu", landmarks.size(), frame);
    }

    if (!input_map_name.empty() && !hints.empty() && slam && images.size() > 0) {
      if (loc_context.status == LocalizeInMapStatus::NOT_LOCALIZED && frame >= next_loc_frame) {
        Hint hint;
        const int64_t timestamp_ns = images[0].timestamp_ns;
        if (getLatestHint(hints, timestamp_ns, hint) && FLAGS_loc_retries-- >= 0) {
          Pose guess_pose = getHintPose(hint.hint, FLAGS_loc_hint_noise, FLAGS_loc_random_rot);
          loc_context.status = LocalizeInMapStatus::IN_PROGRESS;
          loc_context.timestamp = timestamp_ns;

          // Default localization settings from SlamLocalizerOptions
          Slam::LocalizationSettings settings;
          settings.horizontal_search_radius = 1.5f;
          settings.vertical_search_radius = 0.5f;
          settings.horizontal_step = 0.5f;
          settings.vertical_step = 0.25f;
          settings.angular_step_rads = 2 * PI / 36;

          slam->LocalizeInMap(
              input_map_name, timestamp_ns, guess_pose, images, settings, nullptr,
              [&loc_context](const Result<Pose>& result) { localize_in_exist_db_response(result, &loc_context); });

          while (FLAGS_localize_wait && loc_context.status == LocalizeInMapStatus::IN_PROGRESS) {
            // Wait for localization to complete
            // Make dummy tracking calls to allow async localization to progress
            std::vector<Image> dummy_images = images;
            for (auto&& im : dummy_images) {
              im.timestamp_ns += 1000;
            }
            odom->Track(dummy_images, /*masks=*/{}, /*depths=*/{});
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          next_loc_frame = frame + FLAGS_loc_skip_frames + 1;
        }
      }

      if (FLAGS_localize_forever && loc_context.status == LocalizeInMapStatus::LOCALIZED) {
        // Slam can end up in a bad state, so we need to recreate tracker and slam for next localization attempt
        odom = std::make_unique<Odometry>(rig, odom_cfg);
        slam = std::make_unique<Slam>(rig, odom->GetPrimaryCameras(), slam_cfg);
        loc_context.status = LocalizeInMapStatus::NOT_LOCALIZED;
        next_loc_frame = frame + FLAGS_loc_skip_frames + 1;
      }
    }

    frame++;
  }

  if (FLAGS_print_stats) {
    const double total_wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
    printRunStats(std::cout, run_stats, total_wall_seconds);
  }

  if (!success) {
    return false;
  }

  while (loc_context.status == LocalizeInMapStatus::IN_PROGRESS) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  if (!output_map_name.empty() && slam) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto save_status{SaveMapStatus::IN_PROGRESS};
    TraceDebug("Calling SaveMap with %s", output_map_name.c_str());
    slam->SaveMap(output_map_name, [&save_status](bool success) { save_to_slam_db_response(success, &save_status); });
    while (save_status == SaveMapStatus::IN_PROGRESS) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::vector<PoseStamped> kf_poses;
    slam->GetAllSlamPoses(kf_poses, slam_cfg.max_map_size);
    std::ofstream out_kfs(FLAGS_print_map_keyframes);
    setStreamFormat(out_kfs);
    for (const auto& kf : kf_poses) {
      printTsPose(out_kfs, true, kf.timestamp_ns, kf.pose);
    }
  }

  return true;
}

/**
 * @brief Reads a parameter file: `key: value` or `key = value` per line, `#` starts a comment.
 *
 * Entries are returned rather than applied, because they are split across two phases -- the
 * configuration must be final before the tracker is built, the solver parameters only exist after.
 *
 * @throws std::runtime_error if the file cannot be read or a line has no separator.
 */
std::vector<ParamEntry> ReadParamFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open parameter file '" + path + "'");
  }

  const auto trim = [](std::string_view text) {
    const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!text.empty() && is_space(text.front())) {
      text.remove_prefix(1);
    }
    while (!text.empty() && is_space(text.back())) {
      text.remove_suffix(1);
    }
    return text;
  };

  std::vector<ParamEntry> entries;
  std::string line;
  for (size_t line_number = 1; std::getline(file, line); ++line_number) {
    std::string_view content = line;
    const size_t comment = content.find('#');
    if (comment != std::string_view::npos) {
      content = content.substr(0, comment);
    }
    content = trim(content);
    if (content.empty()) {
      continue;
    }
    const size_t separator = content.find_first_of(":=");
    if (separator == std::string_view::npos) {
      throw std::runtime_error(path + ":" + std::to_string(line_number) + ": expected 'key: value', got '" +
                               std::string(content) + "'");
    }
    entries.push_back(ParamEntry{std::string(trim(content.substr(0, separator))),
                                 std::string(trim(content.substr(separator + 1))), line_number});
  }
  return entries;
}

/**
 * @brief Prints every parameter with its type, default and description.
 *
 * Built from the descriptors, so it cannot fall behind the parameters that actually exist. Mode
 * matters: the solver groups listed here are the ones an inertial tracker exposes, which is the
 * widest set.
 */
void PrintParameterReference() {
  Odometry::Config odom_cfg = Odometry::GetDefaultConfig();
  Slam::Config slam_cfg = Slam::GetDefaultConfig();
  odom_cfg.odometry_mode = Odometry::OdometryMode::Inertial;

  params::Registry registry;
  registry.Add("odometry", odom_cfg);
  registry.Add("odometry.rgbd", odom_cfg.rgbd_settings);
  registry.Add("odometry.multisensor", odom_cfg.multisensor_settings);
  registry.Add("slam", slam_cfg);

  size_t width = 0;
  std::vector<params::ParamInfo> infos = registry.List();
  for (const params::ParamInfo& info : infos) {
    width = std::max(width, info.key.size());
  }
  std::cout << "Configuration, applied before the tracker is built:\n";
  for (const params::ParamInfo& info : infos) {
    std::cout << "  " << std::left << std::setw(static_cast<int>(width)) << info.key << "  " << info.type << " = "
              << info.default_value << "\n      " << info.doc << "\n";
  }
  std::cout << "\nSolver parameters, applied to the tracker once it exists. Availability depends on the\n"
               "odometry mode; sm.*, imu_pnp.* and inertial_stereo_pnp.* need an IMU:\n"
            << "  run with --verbosity=2 and no --params to see the set a given mode exposes.\n";
}

}  // end of anonymous namespace

int main(int arg_c, char** arg_v) {
  const std::vector<std::string> param_overrides = params::ExtractOverrides(arg_c, arg_v);
  gflags::ParseCommandLineFlags(&arg_c, &arg_v, /*remove flags = */ true);
  if (arg_c != 1) {
    std::cout << "This tools doesn't expect command line arguments, only flags listed with --help" << std::endl
              << "If you see this message, please check your command line." << std::endl;
    return EXIT_FAILURE;
  }

  // Before the --edex check: listing the parameters does not need a dataset.
  if (FLAGS_list_params) {
    PrintParameterReference();
    return EXIT_SUCCESS;
  }

  if (FLAGS_edex.empty()) {
    std::cout << "Error: -edex is required (path to the .edex file to run on)." << std::endl;
    return EXIT_FAILURE;
  }

  Trace::SetVerbosity(Trace::ToVerbosity(FLAGS_verbosity));
  SetVerbosity(FLAGS_verbosity);

  // Start with default configurations
  Odometry::Config odom_cfg = Odometry::GetDefaultConfig();
  Slam::Config slam_cfg = Slam::GetDefaultConfig();

  // Parameters arrive from a file and from repeatable -P overrides, and split across two phases:
  // Odometry::Config and Slam::Config must be final before the tracker is built, while the solver
  // parameters only exist once it is. Each entry goes to whichever registry knows its name.
  params::Registry config_registry;
  config_registry.Add("odometry", odom_cfg);
  config_registry.Add("odometry.rgbd", odom_cfg.rgbd_settings);
  config_registry.Add("odometry.multisensor", odom_cfg.multisensor_settings);
  config_registry.Add("slam", slam_cfg);

  std::vector<ParamEntry> entries;
  try {
    if (!FLAGS_params.empty()) {
      entries = ReadParamFile(FLAGS_params);
    }
  } catch (const std::exception& e) {
    TraceError("%s\n", e.what());
    return EXIT_FAILURE;
  }
  // -P wins over the file, so it goes last.
  for (const std::string& override_arg : param_overrides) {
    const size_t eq = override_arg.find('=');
    if (eq == std::string::npos) {
      TraceError("-P expects key=value, got '%s'\n", override_arg.c_str());
      return EXIT_FAILURE;
    }
    entries.push_back(ParamEntry{override_arg.substr(0, eq), override_arg.substr(eq + 1), 0});
  }

  std::vector<ParamEntry> solver_params;
  for (const ParamEntry& entry : entries) {
    const std::string where = entry.line != 0 ? FLAGS_params + ":" + std::to_string(entry.line) : std::string("-P");
    try {
      if (config_registry.Knows(entry.key)) {
        config_registry.Set(entry.key, entry.value, params::Source::File);
      } else {
        // Deferred; an unknown name is reported once the tracker knows which parameters it has.
        solver_params.push_back(entry);
      }
    } catch (const std::exception& e) {
      TraceError("%s: %s\n", where.c_str(), e.what());
      return EXIT_FAILURE;
    }
  }

  // Apply command-line flag overrides. Only apply when the user explicitly set the flag
  // (not at default), so YAML config values are not silently overwritten by flag defaults.
  auto flag_is_set = [](const char* name) {
    gflags::CommandLineFlagInfo info;
    return gflags::GetCommandLineFlagInfo(name, &info) && !info.is_default;
  };

  if (!FLAGS_debug_dump.empty()) {
    odom_cfg.debug_dump_directory = FLAGS_debug_dump;
  }
  if (flag_is_set("cfg_async_sba")) odom_cfg.async_sba = FLAGS_cfg_async_sba;
  if (flag_is_set("cfg_denoising")) odom_cfg.use_denoising = FLAGS_cfg_denoising;
  if (flag_is_set("cfg_horizontal")) odom_cfg.rectified_stereo_camera = FLAGS_cfg_horizontal;
  if (flag_is_set("cfg_max_frame_delta_s"))
    odom_cfg.max_frame_delta_s = static_cast<float>(FLAGS_cfg_max_frame_delta_s);
  if (flag_is_set("cfg_enable_export")) {
    odom_cfg.enable_observations_export = FLAGS_cfg_enable_export;
    odom_cfg.enable_landmarks_export = FLAGS_cfg_enable_export;
  }

  // Set multicamera mode (only when explicitly set)
  if (flag_is_set("cfg_multicam_mode")) {
    if (FLAGS_cfg_multicam_mode == 0) {
      odom_cfg.multicam_mode = Odometry::MulticameraMode::Performance;
    } else if (FLAGS_cfg_multicam_mode == 1) {
      odom_cfg.multicam_mode = Odometry::MulticameraMode::Precision;
    } else if (FLAGS_cfg_multicam_mode == 2) {
      odom_cfg.multicam_mode = Odometry::MulticameraMode::Moderate;
    } else {
      TraceError("Invalid multicamera mode");
      return EXIT_FAILURE;
    }
  }

  // Set odometry mode (only when explicitly set)
  if (flag_is_set("cfg_odom_mode")) {
    if (FLAGS_cfg_odom_mode == 0) {
      odom_cfg.odometry_mode = Odometry::OdometryMode::Multicamera;
    } else if (FLAGS_cfg_odom_mode == 1) {
      odom_cfg.odometry_mode = Odometry::OdometryMode::Inertial;
    } else if (FLAGS_cfg_odom_mode == 2) {
      odom_cfg.odometry_mode = Odometry::OdometryMode::RGBD;
    } else if (FLAGS_cfg_odom_mode == 3) {
      odom_cfg.odometry_mode = Odometry::OdometryMode::Mono;
    } else {
      TraceError("Unsupported odometry mode");
      return EXIT_FAILURE;
    }
  }

  // Apply RGBD-specific flags whenever the final mode is RGBD, regardless of
  // whether the mode came from a flag or from YAML config.
  if (odom_cfg.odometry_mode == Odometry::OdometryMode::RGBD) {
    if (flag_is_set("cfg_depth_camera")) odom_cfg.rgbd_settings.depth_camera_id = FLAGS_cfg_depth_camera;
    if (flag_is_set("cfg_depth_scale_factor"))
      odom_cfg.rgbd_settings.depth_scale_factor = static_cast<float>(FLAGS_cfg_depth_scale_factor);
    if (flag_is_set("cfg_enable_depth_stereo_tracking"))
      odom_cfg.rgbd_settings.enable_depth_stereo_tracking = FLAGS_cfg_enable_depth_stereo_tracking;
  }

  // Apply SLAM config from flags (only when explicitly set)
  if (flag_is_set("cfg_sync_slam")) slam_cfg.sync_mode = FLAGS_cfg_sync_slam;
  if (flag_is_set("cfg_slam_max_map_size")) slam_cfg.max_map_size = FLAGS_cfg_slam_max_map_size;
  if (flag_is_set("max_pose_graph_nodes")) slam_cfg.max_map_size = static_cast<uint32_t>(FLAGS_max_pose_graph_nodes);
  if (flag_is_set("slam_reproduce_mode")) slam_cfg.sync_mode = FLAGS_slam_reproduce_mode;
  if (flag_is_set("cfg_planar")) slam_cfg.planar_constraints = FLAGS_cfg_planar;

  return trackEdexDataSet(FLAGS_edex, odom_cfg, slam_cfg, solver_params, FLAGS_loc_input_map, FLAGS_output_map) ? 0 : 1;
}

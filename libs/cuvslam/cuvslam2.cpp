
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

#include "cuvslam/cuvslam2.h"
#include "cuvslam/cuvslam2_internal.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/error.h"
#include "common/isometry.h"
#include "common/parse_utils.h"
#include "math/twist_to_angle.h"
#include "odometry/increment_pose.h"
#include "odometry/mono_visual_odometry.h"
#include "odometry/multi_visual_odometry.h"
#include "odometry/rgbd_odometry.h"
#include "odometry/stereo_inertial_odometry.h"
#include "odometry/svo_config.h"
#include "slam/async_slam/async_slam.h"
#ifdef USE_CUNLS
#include "odometry/multisensor_odometry.h"
#endif

#include "cuvslam/debug_dump.h"
#include "cuvslam/internal.h"

namespace cuvslam {

namespace {

std::string_view TrackOptionName(std::string_view expression) {
  constexpr std::string_view internals_prefix = "internals.";
  if (expression.substr(0, internals_prefix.size()) == internals_prefix) {
    return expression.substr(internals_prefix.size());
  }
  return expression;
}

int32_t RequireNonNegative(int32_t value, std::string_view expression) {
  if (value < 0) {
    const std::string_view name = TrackOptionName(expression);
    throw std::invalid_argument("Internals::" + std::string(name) + " must be non-negative");
  }
  return value;
}

#define REQUIRE_NON_NEGATIVE(x) RequireNonNegative((x), #x)

// Builds a TrackPerFrameSettings from per-frame options. Internals always carries concrete
// values; pass Internals{} to use all defaults. Add new per-frame categories to
// TrackPerFrameSettings rather than adding parameters here or to track().
odom::TrackPerFrameSettings BuildTrackFrameSettings(const cuvslam::internal::Internals& internals) {
  odom::TrackPerFrameSettings result;
  result.sof.num_desired_tracks = internals.num_desired_tracks;
  result.sof.border_top = internals.border_top;
  result.sof.border_bottom = internals.border_bottom;
  result.sof.border_left = internals.border_left;
  result.sof.border_right = internals.border_right;
  result.sof.box3_prefilter = internals.box3_prefilter;
  result.sof.ransac_filter = internals.ransac_filter;
  result.kf.survivor_from_last = internals.kf_survivor_from_last;
  result.kf.max_timedelta_between_kfs_s = internals.kf_max_timedelta_between_kfs_s;
  result.kf.override_frame_selection = internals.kf_override_frame_selection;
  result.vo_pnp.lambda = internals.vo_pnp_lambda;
  result.vo_pnp.huber = internals.vo_pnp_huber;
  result.vo_pnp.max_iteration = REQUIRE_NON_NEGATIVE(internals.vo_pnp_max_iteration);
  result.vo_pnp.recalculate_cov = internals.vo_pnp_recalculate_cov;
  result.vo_pnp.filter_new_observations = internals.vo_pnp_filter_new_observations;
  result.vo_pnp.max_obs_per_camera = REQUIRE_NON_NEGATIVE(internals.vo_pnp_max_obs_per_camera);
  result.vo_pnp.point_z_thresh = internals.vo_pnp_point_z_thresh;
  result.vo_pnp.min_observations = REQUIRE_NON_NEGATIVE(internals.vo_pnp_min_observations);
  result.vo_pnp.cost_thresh = internals.vo_pnp_cost_thresh;
  result.inertial_stereo_pnp.lambda = internals.inertial_stereo_pnp_lambda;
  result.inertial_stereo_pnp.huber = internals.inertial_stereo_pnp_huber;
  result.inertial_stereo_pnp.max_iteration = REQUIRE_NON_NEGATIVE(internals.inertial_stereo_pnp_max_iteration);
  result.inertial_stereo_pnp.recalculate_cov = internals.inertial_stereo_pnp_recalculate_cov;
  result.inertial_stereo_pnp.filter_new_observations = internals.inertial_stereo_pnp_filter_new_observations;
  result.inertial_stereo_pnp.max_obs_per_camera =
      REQUIRE_NON_NEGATIVE(internals.inertial_stereo_pnp_max_obs_per_camera);
  result.inertial_stereo_pnp.point_z_thresh = internals.inertial_stereo_pnp_point_z_thresh;
  result.inertial_stereo_pnp.min_observations = REQUIRE_NON_NEGATIVE(internals.inertial_stereo_pnp_min_observations);
  result.inertial_stereo_pnp.cost_thresh = internals.inertial_stereo_pnp_cost_thresh;
  result.imu_pnp.robustifier_scale = internals.imu_pnp_robustifier_scale;
  result.imu_pnp.max_iteration = REQUIRE_NON_NEGATIVE(internals.imu_pnp_max_iteration);
  result.imu_pnp.min_observations = REQUIRE_NON_NEGATIVE(internals.imu_pnp_min_observations);
  result.icp.lambda = internals.icp_lambda;
  result.icp.huber_vis = internals.icp_huber_vis;
  result.icp.huber_depth = internals.icp_huber_depth;
  result.icp.max_iteration = REQUIRE_NON_NEGATIVE(internals.icp_max_iteration);
  result.icp.cost_thresh = internals.icp_cost_thresh;
  result.icp.min_scale_level = REQUIRE_NON_NEGATIVE(internals.icp_min_scale_level);
  result.icp.max_scale_level = REQUIRE_NON_NEGATIVE(internals.icp_max_scale_level);
  result.icp.num_iters_per_scale = REQUIRE_NON_NEGATIVE(internals.icp_num_iters_per_scale);
  result.icp.blending_alpha = internals.icp_blending_alpha;
  return result;
}

#undef REQUIRE_NON_NEGATIVE

// TODO(vikuznetsov): Remove camera::MulticameraMode & reuse cuvslam enum? What about Manual mode hidden from
// cuvslam API?
camera::MulticameraMode ToMulticamMode(Odometry::MulticameraMode mode) {
  switch (mode) {
    case Odometry::MulticameraMode::Performance:
      return camera::MulticameraMode::Performance;
    case Odometry::MulticameraMode::Precision:
      return camera::MulticameraMode::Precision;
    case Odometry::MulticameraMode::Moderate:
      return camera::MulticameraMode::Moderate;
  }
  throw std::invalid_argument{"Incorrect multicamera mode: " +
                              std::to_string(ToUnderlying<Odometry::MulticameraMode>(mode))};
}

ImageEncoding ToImageEncoding(Image::Encoding mode) {
  switch (mode) {
    case Image::Encoding::MONO:
      return ImageEncoding::MONO8;
    case Image::Encoding::RGB:
      return ImageEncoding::RGB8;
  }
  throw std::invalid_argument{"Incorrect image encoding: " + std::to_string(ToUnderlying<Image::Encoding>(mode))};
}

void CreateCameraModel(const Camera& camera, std::unique_ptr<camera::ICameraModel>& camera_model) {
  Vector2T resolution{camera.size[0], camera.size[1]};
  Vector2T focal{camera.focal[0], camera.focal[1]};
  Vector2T principal{camera.principal[0], camera.principal[1]};

  THROW_INVALID_ARG_IF(principal[0] < 0.0 || principal[1] < 0.0, "Principal point coords must be >= 0.0");
  THROW_INVALID_ARG_IF(focal[0] <= 0.0 || focal[1] <= 0.0, "Focal length must be > 0.0");
  THROW_INVALID_ARG_IF(resolution[0] <= 0 || resolution[1] <= 0, "Image width/height must be > 0");

  // everything built from the extrinsics inherits a NaN: rays, epipolar curves, frustum overlap
  const auto is_finite = [](float v) { return std::isfinite(v); };
  const auto& pose = camera.rig_from_camera;
  THROW_INVALID_ARG_IF(!std::all_of(pose.rotation.begin(), pose.rotation.end(), is_finite) ||
                           !std::all_of(pose.translation.begin(), pose.translation.end(), is_finite),
                       "Camera rig_from_camera must be finite");

  camera_model = camera::CreateCameraModel(resolution, focal, principal, camera.distortion.model,
                                           camera.distortion.parameters.data(), camera.distortion.parameters.size());
  THROW_INVALID_ARG_IF(!camera_model, "Failed to create camera model with " +
                                          std::to_string(camera.distortion.parameters.size()) + " parameters");
}

void SetTrackerRigAndIntrinsics(std::vector<std::unique_ptr<camera::ICameraModel>>& cameras_models,
                                camera::Rig& internal_rig, const std::vector<Camera>& cameras) {
  internal_rig.num_cameras = cameras.size();
  cameras_models.resize(cameras.size());

  for (size_t k = 0; k < cameras.size(); ++k) {
    CreateCameraModel(cameras[k], cameras_models[k]);
    internal_rig.intrinsics[k] = cameras_models[k].get();
    {
      const Isometry3T rig_from_camera = ConvertPoseToIsometry(cameras[k].rig_from_camera);
      internal_rig.camera_from_rig[k] = rig_from_camera.inverse();
    }
  }
}

void CheckCameras(const cuvslam::Rig& rig) {
  THROW_INVALID_ARG_IF(rig.cameras.empty(), "No cameras in a rig");
  THROW_INVALID_ARG_IF(rig.cameras.size() > camera::Rig::kMaxCameras, "Number of cameras limit exceeded");
  for (const auto& cam : rig.cameras) {
    THROW_INVALID_ARG_IF(cam.size[0] != rig.cameras[0].size[0] || cam.size[1] != rig.cameras[0].size[1],
                         "All cameras resolutions must be the same");
  }
  // Check that no two cameras have identical poses
  for (size_t i = 0; i < rig.cameras.size(); i++) {
    for (size_t j = i + 1; j < rig.cameras.size(); j++) {
      const auto& p_i = rig.cameras[i].rig_from_camera;
      const auto& p_j = rig.cameras[j].rig_from_camera;
      bool same_r = std::equal(p_i.rotation.begin(), p_i.rotation.end(), p_j.rotation.begin());
      bool same_t = std::equal(p_i.translation.begin(), p_i.translation.end(), p_j.translation.begin());
      THROW_INVALID_ARG_IF(same_r && same_t,
                           "Cameras " + std::to_string(i) + " and " + std::to_string(j) + " have identical poses");
    }
  }
}

void CheckRectifiedStereoCamera(const cuvslam::Rig& rig) {
  THROW_INVALID_ARG_IF(rig.cameras.size() % 2 != 0,
                       "Rectified stereo camera mode only works with 1+ stereo cameras. "
                       "Number of cameras must be even.");
  for (size_t i = 0; i < rig.cameras.size(); ++i) {
    const auto& dist = rig.cameras[i].distortion;
    THROW_INVALID_ARG_IF(
        dist.model != Distortion::Model::Pinhole ||
            std::any_of(dist.parameters.begin(), dist.parameters.end(), [](const float& p) { return p != 0.0f; }),
        "Rectified stereo camera mode requires pinhole distortion model for camera " + std::to_string(i));
  }
  for (size_t i = 0; i < rig.cameras.size(); i += 2) {
    const auto& rot_0 = rig.cameras[i].rig_from_camera.rotation;
    const auto& rot_1 = rig.cameras[i + 1].rig_from_camera.rotation;
    THROW_INVALID_ARG_IF(
        !std::equal(rot_0.begin(), rot_0.end(), rot_1.begin(),
                    [](float a, float b) { return std::abs(a - b) < 1e-6f; }),
        "Rectified stereo camera mode requires rectified stereo cameras. Rotation matrices of cameras " +
            std::to_string(i) + " and " + std::to_string(i + 1) + " differ");
  }
}

void CheckImuCalibration(const ImuCalibration& imu_calibration) {
  Eigen::Quaternionf quat{imu_calibration.rig_from_imu.rotation.data()};
  // The user-supplied quaternion may be only near-unit (numerical drift in
  // upstream calibrations is common, ~1e-4 is normal).  Normalise before
  // building the rotation matrix so the orthonormality check below tests
  // geometric validity, not float precision.  We only reject when the
  // norm itself is degenerate (zero or NaN).
  THROW_INVALID_ARG_IF(!std::isfinite(quat.norm()) || quat.norm() < 1e-6f,
                       "IMU Calibration: rig from IMU rotation quaternion is degenerate");
  quat.normalize();
  Matrix3T rot = quat.toRotationMatrix();

  THROW_INVALID_ARG_IF(rot.array().isNaN().any() ||
                           vec<3>(imu_calibration.rig_from_imu.translation).array().isNaN().any() ||
                           !rot.inverse().isApprox(rot.transpose()) || std::abs(rot.determinant() - 1.f) > 1e-6,
                       "IMU Calibration: rig from IMU transform is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.gyroscope_noise_density),
                       "IMU Calibration: gyroscope_noise_density is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.gyroscope_random_walk),
                       "IMU Calibration: gyroscope_random_walk is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.accelerometer_noise_density),
                       "IMU Calibration: accelerometer_noise_density is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.accelerometer_random_walk),
                       "IMU Calibration: accelerometer_random_walk is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.frequency), "IMU Calibration: IMU data frequency is not valid");
}

void CheckImages(const Odometry::ImageSet& images, int64_t frame_sync_threshold_ns,
                 const std::vector<std::unique_ptr<camera::ICameraModel>>& cameras_models) {
  THROW_INVALID_ARG_IF(images.empty(), "No images provided");
  for (size_t i = 0; i < images.size(); ++i) {
    THROW_INVALID_ARG_IF(images[i].pixels == nullptr, "No buffer provided for image " + std::to_string(i));
#ifdef USE_CUDA
    THROW_INVALID_ARG_IF(images[i].is_gpu_mem != cuda::IsGpuPointer(images[i].pixels),
                         "is_gpu_mem flag mismatch for image " + std::to_string(i));
#endif
    THROW_INVALID_ARG_IF(images[i].data_type != Image::DataType::UINT8,
                         "Image data type must be UINT8 for image " + std::to_string(i));
    THROW_INVALID_ARG_IF(images[i].camera_index >= cameras_models.size(),
                         "camera_index >= number of cameras for image " + std::to_string(i));
    const auto& resolution = cameras_models[images[i].camera_index].get()->getResolution();
    THROW_INVALID_ARG_IF(images[i].width != resolution[0] || images[i].height != resolution[1],
                         "Image dimensions (" + std::to_string(images[i].width) + "x" +
                             std::to_string(images[i].height) + ") do not correspond to camera resolution (" +
                             std::to_string(resolution[0]) + "x" + std::to_string(resolution[1]) + ") for image " +
                             std::to_string(i));

    for (size_t j = 0; j < i; ++j) {
      THROW_INVALID_ARG_IF(std::abs(images[i].timestamp_ns - images[j].timestamp_ns) >= frame_sync_threshold_ns,
                           "Timestamps differ by more than " + std::to_string(frame_sync_threshold_ns / 1e6) + " ms" +
                               " for images " + std::to_string(j) + ", " + std::to_string(i));
      THROW_INVALID_ARG_IF(images[j].pixels == images[i].pixels,
                           "The same image buffer for images " + std::to_string(j) + ", " + std::to_string(i));
      THROW_INVALID_ARG_IF(images[j].camera_index == images[i].camera_index,
                           "The same camera index for images " + std::to_string(j) + ", " + std::to_string(i));
    }
  }
}

void FillImageSourceAndShape(const Image& image, ImageSource& source, ImageShape& shape) {
  source.data = const_cast<void*>(image.pixels);
  switch (image.data_type) {
    case Image::DataType::UINT8:
      source.type = ImageSource::U8;
      break;
    case Image::DataType::UINT16:
      source.type = ImageSource::U16;
      break;
    case Image::DataType::FLOAT32:
      source.type = ImageSource::F32;
      break;
    default:
      throw std::invalid_argument{"Unsupported image data type"};
  }
  source.memory_type = image.is_gpu_mem ? ImageSource::Device : ImageSource::Host;
  source.pitch = image.pitch;
  source.image_encoding = ToImageEncoding(image.encoding);

  shape.width = image.width;
  shape.height = image.height;
}

// Validates depth images for the RGBD path: exactly one depth image is required.
void CheckDepths(const Odometry::ImageSet& depths,
                 const std::vector<std::unique_ptr<camera::ICameraModel>>& cameras_models) {
  THROW_INVALID_ARG_IF(depths.empty(), "Depth images are required for RGBD odometry");
  THROW_INVALID_ARG_IF(depths.size() > 1, "Only one depth image is supported by RGBD odometry");
  for (size_t i = 0; i < depths.size(); ++i) {
    THROW_INVALID_ARG_IF(depths[i].pixels == nullptr, "No buffer provided for depth image " + std::to_string(i));
#ifdef USE_CUDA
    THROW_INVALID_ARG_IF(depths[i].is_gpu_mem != cuda::IsGpuPointer(depths[i].pixels),
                         "is_gpu_mem flag mismatch for depth image " + std::to_string(i));
#endif
    THROW_INVALID_ARG_IF(
        depths[i].data_type != Image::DataType::UINT16 && depths[i].data_type != Image::DataType::FLOAT32,
        "Depth data type must be UINT16 or FLOAT32");
    THROW_INVALID_ARG_IF(depths[i].camera_index >= cameras_models.size(),
                         "camera_index >= number of cameras for depth image " + std::to_string(i));
    const auto& resolution = cameras_models[depths[i].camera_index].get()->getResolution();
    THROW_INVALID_ARG_IF(depths[i].width != resolution[0] || depths[i].height != resolution[1],
                         "Depth image dimensions (" + std::to_string(depths[i].width) + "x" +
                             std::to_string(depths[i].height) + ") do not correspond to camera resolution (" +
                             std::to_string(resolution[0]) + "x" + std::to_string(resolution[1]) + ")");
  }
}

// Validates depth images for the Multisensor path: zero or more depth images, each tied to a
// distinct camera id from `expected_depth_cam_ids` (the configured depth-camera set). Permitting
// fewer than configured covers frame drops on individual depth streams.
void CheckMultisensorDepths(const Odometry::ImageSet& depths, const std::vector<int32_t>& expected_depth_cam_ids,
                            const std::vector<std::unique_ptr<camera::ICameraModel>>& cameras_models) {
  for (size_t i = 0; i < depths.size(); ++i) {
    THROW_INVALID_ARG_IF(depths[i].pixels == nullptr,
                         "No buffer provided for multisensor depth image " + std::to_string(i));
#ifdef USE_CUDA
    THROW_INVALID_ARG_IF(depths[i].is_gpu_mem != cuda::IsGpuPointer(depths[i].pixels),
                         "is_gpu_mem flag mismatch for multisensor depth image " + std::to_string(i));
#endif
    THROW_INVALID_ARG_IF(
        depths[i].data_type != Image::DataType::UINT16 && depths[i].data_type != Image::DataType::FLOAT32,
        "Multisensor depth data type must be UINT16 or FLOAT32");
    THROW_INVALID_ARG_IF(depths[i].camera_index >= cameras_models.size(),
                         "camera_index >= number of cameras for multisensor depth image " + std::to_string(i));
    const auto& resolution = cameras_models[depths[i].camera_index]->getResolution();
    THROW_INVALID_ARG_IF(depths[i].width != resolution[0] || depths[i].height != resolution[1],
                         "Multisensor depth image dimensions do not match camera resolution");
    const int32_t cam_id = static_cast<int32_t>(depths[i].camera_index);
    THROW_INVALID_ARG_IF(
        std::find(expected_depth_cam_ids.begin(), expected_depth_cam_ids.end(), cam_id) == expected_depth_cam_ids.end(),
        "Multisensor depth image for camera " + std::to_string(cam_id) +
            " was not configured in MultisensorSettings::depth_camera_ids");
    for (size_t j = 0; j < i; ++j) {
      THROW_INVALID_ARG_IF(depths[j].camera_index == depths[i].camera_index,
                           "Duplicate depth image for camera " + std::to_string(cam_id));
    }
  }
}

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SetVerbosity(int verbosity) {
  constexpr const Trace::Verbosity max_allowed =
#if defined(NDEBUG)
      Trace::Verbosity::Message;
#else
      Trace::Verbosity::Debug;
#endif
  Trace::SetVerbosity(Trace::ToVerbosity(verbosity, max_allowed));
}

void WarmUpGPU() { WarmUpGpuImpl(); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Odometry class implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// using ImageContexts = std::unordered_map<CameraId, std::shared_ptr<Odometry::State::Context>>;

class Odometry::Impl {
public:
  // objects
  camera::Rig rig;  // rig and cameras_models
  camera::FrustumIntersectionGraph fig;
  std::vector<std::unique_ptr<camera::ICameraModel>> cameras_models;
  std::unique_ptr<sof::ImageManager> image_manager;
  std::unique_ptr<odom::IVisualOdometry> visual_odometry;
  // state
  FrameId frame_id{0};
  sof::Images prev_image_ptrs;
  Sources image_sources;
  Sources masks_sources;
  DepthSources depth_sources;
  Metas image_metas;
  sof::Images curr_image_ptrs;
  Isometry3T prev_abs_pose{Isometry3T::Identity()};
  int64_t last_timestamp_ns{std::numeric_limits<int64_t>::min()};
  int64_t last_frame_timestamp_ns;
  Isometry3T last_delta{Isometry3T::Identity()};
  State::ContextMap image_contexts;
  // debug
  std::string debug_dump_directory;
  int64_t max_frame_delta_ns;
  int64_t frame_sync_threshold_ns{1'000'000};  // 1 ms
  // settings
  odom::Settings svo_settings;  // construction-time settings passed to odometry components
  bool imu_fusion_enabled;
  RGBDSettings rgbd_settings;
  MultisensorSettings multisensor_settings;
  Odometry::OdometryMode odometry_mode;
  // stats
  bool enable_final_landmarks_export{false};
  std::unordered_map<uint64_t, Vector3f> final_landmarks;

  // data helpers

  // if camera_index == std::numeric_limits<uint32_t>::max() will return observations for all cameras
  void GetLastObservations(uint32_t camera_index, std::vector<Observation>& observations) const {
    const auto& stat = visual_odometry->get_last_stat();
    THROW_INVALID_ARG_IF(stat == nullptr, "Set enable_observations_export to true to get last observations");
    observations.clear();
    if (camera_index == std::numeric_limits<uint32_t>::max()) {
      observations.reserve(stat->tracks2d.size());
      for (const auto& t : stat->tracks2d) {
        observations.emplace_back(Observation{t.track_id, t.uv.x(), t.uv.y(), t.cam_id});
      }
    } else {
      observations.reserve(stat->tracks2d.size() / fig.primary_cameras().size());
      for (const auto& t : stat->tracks2d) {
        if (t.cam_id == camera_index) {
          observations.emplace_back(Observation{t.track_id, t.uv.x(), t.uv.y(), t.cam_id});
        }
      }
    }
  }

  void GetLastLandmarks(std::vector<Landmark>& landmarks) const {
    const auto& stat = visual_odometry->get_last_stat();
    THROW_INVALID_ARG_IF(stat == nullptr, "Set enable_landmarks_export to true to get last landmarks");
    landmarks.clear();
    landmarks.reserve(stat->tracks3d.size());
    for (const auto& t : stat->tracks3d) {
      landmarks.emplace_back(Landmark{t.first, {{t.second.x(), t.second.y(), t.second.z()}}});
    }
  }
};

Odometry::Odometry(Odometry&&) noexcept = default;

Odometry::~Odometry() = default;

Odometry::Odometry(const Rig& rig, const Config& cfg) {
  std::string message;
  TracePrintIf(!CheckCudaCompatibility(message), "[WARNING] %s\n", message.c_str());
#ifdef ENFORCE_GPU
  THROW_INVALID_ARG_IF(!cfg.use_gpu, "cfg.use_gpu must be enabled");
#endif

  CheckCameras(rig);

  THROW_INVALID_ARG_IF(cfg.odometry_mode == OdometryMode::Inertial && rig.imus.empty(),
                       "IMU fusion is enabled, but IMU calibration is not provided");
  THROW_INVALID_ARG_IF(rig.imus.size() > 1, "Only one IMU sensor is supported");

  // For Multisensor mode, IMU presence is auto-detected from the rig (same convention as Inertial).
  const bool multisensor_with_imu = cfg.odometry_mode == OdometryMode::Multisensor && !rig.imus.empty();

  odom::Settings svo_settings;
  svo_settings.verbose = Trace::GetVerbosity() > Trace::Verbosity::None;

  svo_settings.sba_settings.async = cfg.async_sba;
  // Auto-pick SBA mode: inertial-CPU when IMU fusion is on (Inertial mode, or Multisensor with IMU),
  // otherwise the standard GPU bundler.
  svo_settings.sba_settings.mode = (cfg.odometry_mode == OdometryMode::Inertial || multisensor_with_imu)
                                       ? sba::Mode::InertialCPU
                                       : sba::Mode::OriginalGPU;

  // Use only first camera border settings. Other camera border settings are ignored now.
  svo_settings.sof_settings.multicam_mode = ToMulticamMode(cfg.multicam_mode);
  svo_settings.sof_settings.border_top = rig.cameras[0].border_top;
  svo_settings.sof_settings.border_bottom = rig.cameras[0].border_bottom;
  svo_settings.sof_settings.border_left = rig.cameras[0].border_left;
  svo_settings.sof_settings.border_right = rig.cameras[0].border_right;
  svo_settings.sof_settings.box3_prefilter = cfg.use_denoising;
  svo_settings.sof_settings.min_depth = cfg.min_depth;
  svo_settings.sof_settings.max_depth = cfg.max_depth;
  if (cfg.rectified_stereo_camera) {
    CheckRectifiedStereoCamera(rig);
    svo_settings.sof_settings.lr_tracker = sof::TrackerType::LKHorizontal;
  }

  auto tracker{std::make_unique<Odometry::Impl>()};

  SetTrackerRigAndIntrinsics(tracker->cameras_models, tracker->rig, rig.cameras);

  // Build the frustum-intersection graph. Each odometry mode contributes a different depth-camera
  // list and stereo-depth-tracking choice; collect them up front for clarity.
  std::vector<CameraId> depth_ids;
  bool enable_depth_stereo_tracking_fig = false;
  if (cfg.odometry_mode == OdometryMode::RGBD) {
    depth_ids.push_back(static_cast<CameraId>(cfg.rgbd_settings.depth_camera_id));
    enable_depth_stereo_tracking_fig = cfg.rgbd_settings.enable_depth_stereo_tracking;
  } else if (cfg.odometry_mode == OdometryMode::Multisensor) {
    depth_ids.reserve(cfg.multisensor_settings.depth_camera_ids.size());
    for (int32_t id : cfg.multisensor_settings.depth_camera_ids) {
      THROW_INVALID_ARG_IF(id < 0 || static_cast<size_t>(id) >= rig.cameras.size(),
                           "MultisensorSettings::depth_camera_ids contains out-of-range camera id");
      THROW_INVALID_ARG_IF(std::find(depth_ids.begin(), depth_ids.end(), static_cast<CameraId>(id)) != depth_ids.end(),
                           "MultisensorSettings::depth_camera_ids contains duplicate camera id");
      depth_ids.push_back(static_cast<CameraId>(id));
    }
    enable_depth_stereo_tracking_fig = cfg.multisensor_settings.enable_depth_stereo_tracking;
  } else {
    depth_ids.push_back(static_cast<CameraId>(0));
  }
  camera::FigSettings fig_settings{
      /*.mode=*/svo_settings.sof_settings.multicam_mode,
      /*.depth_ids=*/depth_ids,
      /*.allow_stereo_track_for_depth=*/enable_depth_stereo_tracking_fig,
      /*.manual_setup=*/svo_settings.sof_settings.multicam_setup,
  };
  tracker->fig = camera::FrustumIntersectionGraph(tracker->rig, fig_settings);
  if (!tracker->fig.is_valid()) {
    if (cfg.odometry_mode == OdometryMode::Multisensor) {
      throw std::invalid_argument{
          "Bad calibration. Multisensor mode needs at least one RGB-D camera configured in "
          "MultisensorSettings::depth_camera_ids or one camera pair with overlapping frustums."};
    }
    throw std::invalid_argument{
        "Bad calibration. cuVSLAM needs at least one stereo pair for Multicamera or Inertial mode."};
  }

  switch (cfg.odometry_mode) {
    case OdometryMode::Multicamera: {
      svo_settings.use_prediction = cfg.use_motion_model;
      tracker->visual_odometry =
          std::make_unique<odom::MultiVisualOdometry>(tracker->rig, tracker->fig, svo_settings, cfg.use_gpu);
      break;
    }
    case OdometryMode::Inertial: {
      const auto& imu_calibration = rig.imus[0];
      CheckImuCalibration(imu_calibration);
      const Isometry3T rig_from_imu = ConvertPoseToIsometry(imu_calibration.rig_from_imu);
      svo_settings.imu_calibration =
          imu::ImuCalibration(rig_from_imu, imu_calibration.gyroscope_noise_density,
                              imu_calibration.gyroscope_random_walk, imu_calibration.accelerometer_noise_density,
                              imu_calibration.accelerometer_random_walk, imu_calibration.frequency);
      svo_settings.use_prediction = cfg.use_motion_model;
      tracker->visual_odometry = std::make_unique<odom::StereoInertialOdometry>(
          tracker->rig, tracker->fig, svo_settings, cfg.use_gpu, cfg.debug_imu_mode,
          /*disable_fusion_except_gravity=*/true);
      break;
    }
    case OdometryMode::Mono: {
      tracker->visual_odometry = std::make_unique<odom::MonoVisualOdometry>(tracker->rig, svo_settings, cfg.use_gpu);
      break;
    }
    case OdometryMode::RGBD: {
      tracker->rgbd_settings = cfg.rgbd_settings;
      tracker->visual_odometry =
          std::make_unique<odom::RGBDOdometry>(tracker->rig, tracker->fig, svo_settings, cfg.use_gpu);
      break;
    }
    case OdometryMode::Multisensor: {
#ifdef USE_CUNLS
      tracker->multisensor_settings = cfg.multisensor_settings;
      svo_settings.multisensor_settings.with_imu = multisensor_with_imu;
      svo_settings.multisensor_settings.depth_camera_ids = cfg.multisensor_settings.depth_camera_ids;
      svo_settings.multisensor_settings.depth_scale_factor = cfg.multisensor_settings.depth_scale_factor;
      svo_settings.multisensor_settings.enable_depth_stereo_tracking =
          cfg.multisensor_settings.enable_depth_stereo_tracking;
      if (multisensor_with_imu) {
        const auto& imu_calibration = rig.imus[0];
        CheckImuCalibration(imu_calibration);
        const Isometry3T rig_from_imu = ConvertPoseToIsometry(imu_calibration.rig_from_imu);
        svo_settings.imu_calibration =
            imu::ImuCalibration(rig_from_imu, imu_calibration.gyroscope_noise_density,
                                imu_calibration.gyroscope_random_walk, imu_calibration.accelerometer_noise_density,
                                imu_calibration.accelerometer_random_walk, imu_calibration.frequency);
      }
      svo_settings.use_prediction = cfg.use_motion_model;
      tracker->visual_odometry =
          std::make_unique<odom::MultisensorOdometry>(tracker->rig, tracker->fig, svo_settings, cfg.use_gpu);
      break;
#else
      throw std::invalid_argument{"OdometryMode::Multisensor requires a build with cuNLS support (USE_CUNLS=ON)"};
#endif
    }
    default:
      throw std::invalid_argument{"Unsupported odometry mode " +
                                  std::to_string(ToUnderlying<Odometry::OdometryMode>(cfg.odometry_mode))};
  }
  tracker->visual_odometry->enable_stat(cfg.enable_observations_export || cfg.enable_landmarks_export ||
                                        cfg.enable_final_landmarks_export);
  tracker->enable_final_landmarks_export = cfg.enable_final_landmarks_export;

  tracker->svo_settings = svo_settings;
  tracker->imu_fusion_enabled = cfg.odometry_mode == OdometryMode::Inertial || multisensor_with_imu;
  tracker->debug_dump_directory = cfg.debug_dump_directory;
  tracker->max_frame_delta_ns = static_cast<int64_t>(cfg.max_frame_delta_s * 1e9);
  tracker->odometry_mode = cfg.odometry_mode;

  // Each depth-providing camera draws from the depth-capable pool, every other camera from the
  // no-depth pool. Both pools must hold cache_size contexts per camera so the pipeline can keep
  // the previous frame plus async pipelining slots alive while the current frame is being tracked.
  const size_t cache_size = 4;
  size_t num_depth_cams = 0;
  if (cfg.odometry_mode == OdometryMode::RGBD) {
    num_depth_cams = 1;
  } else if (cfg.odometry_mode == OdometryMode::Multisensor) {
    num_depth_cams = cfg.multisensor_settings.depth_camera_ids.size();
  }
  const size_t num_no_depth_cams = rig.cameras.size() - num_depth_cams;
  tracker->image_manager = std::make_unique<sof::ImageManager>();
  ImageShape shape{rig.cameras[0].size[0], rig.cameras[0].size[1]};
  tracker->image_manager->init(shape, num_no_depth_cams * cache_size, cfg.use_gpu, num_depth_cams * cache_size);

  impl = std::move(tracker);

  DumpConfiguration(impl->debug_dump_directory, rig, cfg);
}

void Odometry::RegisterImuMeasurement(uint32_t sensor_index, const ImuMeasurement& imu) {
  THROW_INVALID_ARG_IF(!impl->imu_fusion_enabled, "IMU fusion is not enabled");
  THROW_INVALID_ARG_IF(sensor_index != 0, "Only one IMU sensor is supported");
  THROW_INVALID_ARG_IF(imu.timestamp_ns < impl->last_timestamp_ns, "Timestamps are non-monotonic");
  impl->last_timestamp_ns = imu.timestamp_ns;

  DumpRegisterImuMeasurementCall(impl->debug_dump_directory, imu);

  Vector3T acc(imu.linear_accelerations[0], imu.linear_accelerations[1], imu.linear_accelerations[2]);
  Vector3T gyro(imu.angular_velocities[0], imu.angular_velocities[1], imu.angular_velocities[2]);

  const imu::ImuMeasurement m = {imu.timestamp_ns, acc, gyro};
  if (impl->odometry_mode == OdometryMode::Inertial) {
    static_cast<odom::StereoInertialOdometry*>(impl->visual_odometry.get())->add_imu_measurement(m);
  } else {
#ifdef USE_CUNLS
    static_cast<odom::MultisensorOdometry*>(impl->visual_odometry.get())->add_imu_measurement(m);
#else
    throw std::invalid_argument{"Multisensor IMU fusion requires a build with USE_CUNLS"};
#endif
  }
}

PoseEstimate Odometry::Track(const ImageSet& images, const ImageSet& masks, const ImageSet& depths,
                             const cuvslam::internal::Internals* internals) {
  odom::TrackPerFrameSettings per_frame_setting =
      BuildTrackFrameSettings(internals ? *internals : cuvslam::internal::Internals{});
  per_frame_setting.sba = impl->svo_settings.sba_settings;
  per_frame_setting.sm = impl->svo_settings.sm_settings;

  CheckImages(images, impl->frame_sync_threshold_ns, impl->cameras_models);
  if (impl->odometry_mode == OdometryMode::RGBD) {
    CheckDepths(depths, impl->cameras_models);
  } else if (impl->odometry_mode == OdometryMode::Multisensor) {
    CheckMultisensorDepths(depths, impl->multisensor_settings.depth_camera_ids, impl->cameras_models);
  } else {
    THROW_INVALID_ARG_IF(!depths.empty(), "Depth images are only accepted for RGBD or Multisensor odometry");
  }
  DumpTrackCall(impl->debug_dump_directory, impl->frame_id, images, masks, depths);

  Sources& image_sources = impl->image_sources;
  Sources& masks_sources = impl->masks_sources;
  DepthSources& depth_sources = impl->depth_sources;
  Metas& image_metas = impl->image_metas;
  sof::Images& cuvslam_images_ptrs = impl->curr_image_ptrs;

  const size_t num_cameras = static_cast<size_t>(impl->rig.num_cameras);
  image_sources.assign(num_cameras, {});
  masks_sources.assign(num_cameras, {});
  depth_sources.assign(num_cameras, {});
  image_metas.assign(num_cameras, {});
  cuvslam_images_ptrs.assign(num_cameras, nullptr);
  if (impl->prev_image_ptrs.size() != num_cameras) {
    impl->prev_image_ptrs.assign(num_cameras, nullptr);
  }
  impl->image_contexts.clear();

  const FrameId frame_id = impl->frame_id++;
  const int64_t current_time_ns = images[0].timestamp_ns;
  THROW_INVALID_ARG_IF(current_time_ns < impl->last_timestamp_ns, "Timestamps are non-monotonic");

  const bool has_prev_images = std::any_of(impl->prev_image_ptrs.begin(), impl->prev_image_ptrs.end(),
                                           [](const auto& image) { return image != nullptr; });
  if (has_prev_images) {
    THROW_INVALID_ARG_IF(current_time_ns <= impl->last_frame_timestamp_ns,
                         "Frame timestamps must be strictly increasing");
    auto current_frame_delta_ns = current_time_ns - impl->last_frame_timestamp_ns;
    TraceWarningIf(current_frame_delta_ns > impl->max_frame_delta_ns,
                   "Delta between frames at frame %d is %.0f ms that is longer than desired %.0f ms. Check camera fps "
                   "and sync settings.",
                   frame_id, current_frame_delta_ns / 1e6, impl->max_frame_delta_ns / 1e6);
  }
  impl->last_timestamp_ns = current_time_ns;
  impl->last_frame_timestamp_ns = current_time_ns;

  for (const auto& image : images) {
    bool is_rgbd = false;
    auto cam_id = image.camera_index;
    ImageSource& source = image_sources[cam_id];
    ImageSource& mask_source = masks_sources[cam_id];  // a mask for each image is required even if it is empty
    ImageMeta& meta = image_metas[cam_id];

    meta.frame_id = frame_id;
    meta.camera_index = cam_id;
    meta.timestamp = current_time_ns;

    FillImageSourceAndShape(image, source, meta.shape);

    auto mask_it =
        std::find_if(masks.begin(), masks.end(), [&cam_id](const auto& mask) { return mask.camera_index == cam_id; });
    if (mask_it != masks.end()) {
      FillImageSourceAndShape(*mask_it, mask_source, meta.mask_shape);
    }

    // RGBD: exactly one depth image (validated in CheckDepths) → match it to its camera id.
    if (impl->odometry_mode == OdometryMode::RGBD && depths[0].camera_index == cam_id) {
      auto& depth_source = depth_sources[cam_id];
      FillImageSourceAndShape(depths[0], depth_source, meta.shape);
      meta.pixel_scale_factor = impl->rgbd_settings.depth_scale_factor;
      is_rgbd = true;
    }
    // Multisensor: any of the supplied depth images can match this camera id (multi-RGBD rigs).
    if (impl->odometry_mode == OdometryMode::Multisensor) {
      for (const auto& depth : depths) {
        if (depth.camera_index == cam_id) {
          auto& depth_source = depth_sources[cam_id];
          FillImageSourceAndShape(depth, depth_source, meta.shape);
          meta.pixel_scale_factor = impl->multisensor_settings.depth_scale_factor;
          is_rgbd = true;
          break;
        }
      }
    }

    sof::ImageContextPtr ptr = is_rgbd ? impl->image_manager->acquire_with_depth() : impl->image_manager->acquire();
    THROW_RUNTIME_ERROR_IF(ptr == nullptr, "Failed to acquire image context from image_manager");
    ptr->set_image_meta(meta);
    cuvslam_images_ptrs[cam_id] = ptr;
    impl->image_contexts.insert({cam_id, std::static_pointer_cast<Odometry::State::Context>(ptr)});
  }

  Matrix6T static_info_exp = Matrix6T::Identity();
  bool result = impl->visual_odometry->track(image_sources, depth_sources, cuvslam_images_ptrs, impl->prev_image_ptrs,
                                             masks_sources, impl->last_delta, static_info_exp, per_frame_setting);

  for (CameraId cam_id = 0; cam_id < cuvslam_images_ptrs.size(); ++cam_id) {
    if (cuvslam_images_ptrs[cam_id] != nullptr) {
      impl->prev_image_ptrs[cam_id] = cuvslam_images_ptrs[cam_id];
    }
  }

  if (result) {
    impl->prev_abs_pose = odom::increment_pose(impl->prev_abs_pose, impl->last_delta);
  } else {
    return PoseEstimate{};
  }
  const Isometry3T internal_pose = impl->prev_abs_pose;

  // static pose covariance in exponential mapping form in WCS
  const Matrix6T static_pose_covariance_exp = static_info_exp.ldlt().solve(Matrix6T::Identity());
  // static pose covariance in WCS, ordered as [x, y, z, roll, pitch, yaw]
  const Matrix6T static_pose_covariance_xyz_rpy =
      math::PoseCovToXYZRollPitchYawCov(static_pose_covariance_exp, internal_pose);

  PoseEstimate pose_estimate;
  pose_estimate.timestamp_ns = current_time_ns;
  PoseWithCovariance pose_with_covariance;
  pose_with_covariance.pose = ConvertIsometryToPose(internal_pose);
  mat<6>(pose_with_covariance.covariance_xyz_rpy) = static_pose_covariance_xyz_rpy;
  pose_estimate.world_from_rig = pose_with_covariance;

  if (impl->enable_final_landmarks_export) {
    const auto& stat = impl->visual_odometry->get_last_stat();
    assert(stat != nullptr);  // Final landmarks need to be saved, but VO stats are not enabled; this should not happen

    for (const auto& t : stat->tracks3d) {
      const TrackId id = t.first;
      const Vector3T internal_lm = internal_pose * t.second;
      impl->final_landmarks[id] = {{internal_lm.x(), internal_lm.y(), internal_lm.z()}};
    }
  }

  return pose_estimate;
}

std::vector<Observation> Odometry::GetLastObservations(uint32_t camera_index) const {
  std::vector<Observation> observations;
  THROW_INVALID_ARG_IF(camera_index > static_cast<uint32_t>(impl->rig.num_cameras), "Camera index out of range");
  impl->GetLastObservations(camera_index, observations);
  return observations;
}

std::vector<Landmark> Odometry::GetLastLandmarks() const {
  std::vector<Landmark> landmarks;
  impl->GetLastLandmarks(landmarks);
  return landmarks;
}

std::optional<Odometry::Gravity> Odometry::GetLastGravity() const {
  THROW_INVALID_ARG_IF(!impl->imu_fusion_enabled, "IMU fusion is disabled");

  std::optional<Vector3T> gravity_estimate;
  if (impl->odometry_mode == OdometryMode::Inertial) {
    gravity_estimate = static_cast<odom::StereoInertialOdometry*>(impl->visual_odometry.get())->get_gravity();
  } else {
#ifdef USE_CUNLS
    gravity_estimate = static_cast<odom::MultisensorOdometry*>(impl->visual_odometry.get())->get_gravity();
#endif
  }
  if (!gravity_estimate.has_value()) {
    return std::nullopt;  // Gravity is not available yet
  }

  const auto& internal_g = gravity_estimate.value();
  return Gravity{internal_g.x(), internal_g.y(), internal_g.z()};
}

std::optional<Odometry::ImuState> Odometry::GetImuState() const {
  THROW_INVALID_ARG_IF(!impl->imu_fusion_enabled, "IMU fusion is disabled");

  if (impl->odometry_mode == OdometryMode::Inertial) {
    const auto s = static_cast<odom::StereoInertialOdometry*>(impl->visual_odometry.get())->GetImuState();
    if (!s.has_value()) return std::nullopt;
    return ImuState{
        {s->velocity.x(), s->velocity.y(), s->velocity.z()},
        {s->gyro_bias.x(), s->gyro_bias.y(), s->gyro_bias.z()},
        {s->acc_bias.x(), s->acc_bias.y(), s->acc_bias.z()},
    };
  }
#ifdef USE_CUNLS
  const auto s = static_cast<odom::MultisensorOdometry*>(impl->visual_odometry.get())->GetImuState();
  if (!s.has_value()) return std::nullopt;
  return ImuState{
      {s->velocity.x(), s->velocity.y(), s->velocity.z()},
      {s->gyro_bias.x(), s->gyro_bias.y(), s->gyro_bias.z()},
      {s->acc_bias.x(), s->acc_bias.y(), s->acc_bias.z()},
  };
#else
  return std::nullopt;
#endif
}

void Odometry::GetState(Odometry::State& state) const {
  const auto& stat = impl->visual_odometry->get_last_stat();
  THROW_INVALID_ARG_IF(stat == nullptr, "Enable export of observations and/or landmarks to get state");
  state.frame_id = impl->frame_id - 1;  // frame_id is incremented after tracking, so we need to subtract 1
  state.timestamp_ns = impl->last_frame_timestamp_ns;
  state.delta = ConvertIsometryToPose(impl->last_delta);
  state.keyframe = stat->keyframe;
  state.warming_up = stat->heating;
  state.gravity = impl->imu_fusion_enabled ? GetLastGravity() : std::nullopt;
  impl->GetLastObservations(std::numeric_limits<uint32_t>::max(), state.observations);
  impl->GetLastLandmarks(state.landmarks);
  state.context = impl->image_contexts;
}

std::unordered_map<uint64_t, Vector3f> Odometry::GetFinalLandmarks() const {
  const auto& stat = impl->visual_odometry->get_last_stat();
  THROW_INVALID_ARG_IF(stat == nullptr, "Set enable_final_landmarks_export to true to get final landmarks");
  // final_landmarks are already in OpenCV coordinate system
  return impl->final_landmarks;
}

const std::vector<uint8_t>& Odometry::GetPrimaryCameras() const { return impl->fig.primary_cameras(); }

void Odometry::ApplyPersistentInternalParameters(const std::vector<cuvslam::internal::InternalParameter>& parameters) {
  static std::once_flag warned;
  std::call_once(warned, [] {
    TraceWarning(
        "ApplyPersistentInternalParameters: modifying internal parameters (InternalParameter) may cause instability or "
        "degraded tracking performance. "
        "SBA settings are applied on the next keyframe trigger; only the latest value takes effect — "
        "any intermediate values set while SBA is running are overwritten.\n");
  });

  const bool has_sm = (impl->odometry_mode == OdometryMode::Inertial);

  sba::Settings& sba = impl->svo_settings.sba_settings;
  pipelines::StateMachineSettings& sm = impl->svo_settings.sm_settings;

  for (const cuvslam::internal::InternalParameter& parameter : parameters) {
    const std::string_view key = parameter.key;
    const std::string_view value = parameter.value;
    // SBA settings (all modes)
    try {
      if (key == "sba.num_sba_frames") {
        sba.num_sba_frames = common::ParseInt32(value);
      } else if (key == "sba.num_inertial_sba_frames") {
        sba.num_inertial_sba_frames = common::ParseInt32(value);
      } else if (key == "sba.num_fixed_sba_frames") {
        sba.num_fixed_sba_frames = common::ParseInt32(value);
      } else if (key == "sba.num_sba_iterations") {
        sba.num_sba_iterations = common::ParseInt32(value);
      } else if (key == "sba.robustifier_scale") {
        sba.robustifier_scale = common::ParseFloat(value);
      } else if (key == "sba.use_sba_winsorizer") {
        sba.use_sba_winsorizer = common::ParseBool(value);
        // StateMachine settings (Inertial mode only)
      } else if (key.substr(0, 3) == "sm.") {
        if (!has_sm) {
          TraceWarning("ApplyPersistentInternalParameters: key \"%.*s\" requires Inertial mode (ignored)\n",
                       static_cast<int>(key.size()), key.data());
          continue;
        }
        if (key == "sm.gravity_update_period_ns") {
          sm.gravity_update_period_ns = common::ParseInt64(value);
        } else if (key == "sm.max_integration_time_ns") {
          sm.max_integration_time_ns = common::ParseInt64(value);
        } else if (key == "sm.min_num_kf_for_gravity") {
          const int32_t tmp = common::ParseInt32(value);
          if (tmp < 0) {
            throw std::runtime_error("expected non-negative int32, got: " + std::string(value));
          }
          sm.min_num_kf_for_gravity = static_cast<size_t>(tmp);
        } else if (key == "sm.min_time_period_ns") {
          sm.min_time_period_ns = common::ParseInt64(value);
        } else if (key == "sm.max_time_period_ns") {
          sm.max_time_period_ns = common::ParseInt64(value);
        } else {
          TraceWarning("ApplyPersistentInternalParameters: unknown key \"%.*s\" (ignored)\n",
                       static_cast<int>(key.size()), key.data());
        }
      } else {
        TraceWarning("ApplyPersistentInternalParameters: unknown key \"%.*s\" (ignored)\n",
                     static_cast<int>(key.size()), key.data());
      }
    } catch (const std::exception& e) {
      TraceWarning("ApplyPersistentInternalParameters: failed parsing key \"%.*s\": %s (ignored)\n",
                   static_cast<int>(key.size()), key.data(), e.what());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Slam class implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Slam::Impl {
public:
  camera::Rig rig_;
  std::vector<std::unique_ptr<camera::ICameraModel>> cameras_models_;
  std::vector<uint8_t> primary_cameras_;
  bool use_gpu_ = true;
  bool gt_align_mode_ = false;
  std::unique_ptr<slam::AsyncSlam> async_slam_;
  std::atomic<uint64_t> frame_id_ = 0;

  // views
  bool enable_reading_internals_ = false;
  static constexpr auto kMaxDataLayer = ToUnderlying(DataLayer::Max);
  std::shared_ptr<slam::ViewManager<slam::ViewLandmarks>> landmarks_views_[kMaxDataLayer];
  std::shared_ptr<Landmarks> landmarks_[kMaxDataLayer];
  std::shared_ptr<slam::ViewManager<slam::ViewPoseGraph>> pose_graph_view_;
  std::shared_ptr<PoseGraph> pose_graph_;  // return type is different, so cannot just return a pointer to a view

  Impl(const Rig& rig, const std::vector<uint8_t>& primary_cameras, const Config& config)
      : primary_cameras_(primary_cameras) {
    std::string message;
    TracePrintIf(!CheckCudaCompatibility(message), "[WARNING] %s\n", message.c_str());

    SetTrackerRigAndIntrinsics(cameras_models_, rig_, rig.cameras);

    slam::AsyncSlamOptions options;
    options.map_cache_path = config.map_cache_path;
    options.use_gpu = config.use_gpu;
    options.reproduce_mode = config.sync_mode;
    options.pose_for_frame_required = true;  // required for localization
    options.spatial_index_options.cell_size = config.map_cell_size;
    options.max_landmarks_distance = config.max_landmarks_distance;
    options.planar_constraints = config.planar_constraints;
    options.throttling_time_ms = config.throttling_time_ms;
    options.retention_time_ms = config.retention_time_ms;
    options.delay_warning_queue_size = config.delay_warning_queue_size;
    use_gpu_ = config.use_gpu;
    gt_align_mode_ = config.gt_align_mode;
    if (config.gt_align_mode) {
      THROW_INVALID_ARG_IF(!config.sync_mode, "sync_mode should be enabled for gt_align_mode.");
      THROW_INVALID_ARG_IF(config.planar_constraints, "planar_constraints should be disabled for gt_align_mode.");
      options.loop_closure_solver_type = slam::LoopClosureSolverType::kDummy;
      options.pgo_options.type = slam::PoseGraphOptimizerType::Dummy;
    } else {
      options.max_pose_graph_nodes = config.max_map_size;
    }
    async_slam_ = std::make_unique<slam::AsyncSlam>(rig_, primary_cameras_, options);

    enable_reading_internals_ = config.enable_reading_internals;
    if (enable_reading_internals_) {
      pose_graph_view_ = std::make_shared<slam::ViewManager<slam::ViewPoseGraph>>();
      pose_graph_ = std::make_shared<PoseGraph>();

      for (DataLayer layer : {DataLayer::Map, DataLayer::LoopClosure, DataLayer::Landmarks}) {
        landmarks_views_[ToUnderlying(layer)] = std::make_shared<slam::ViewManager<slam::ViewLandmarks>>();
        landmarks_[ToUnderlying(layer)] = std::make_shared<Landmarks>();
      }

      async_slam_->SetLandmarksView(landmarks_views_[ToUnderlying(DataLayer::Map)]);
      async_slam_->SetLoopClosureView(landmarks_views_[ToUnderlying(DataLayer::LoopClosure)]);
      async_slam_->SetPoseGraphView(pose_graph_view_);
    }
  }

  ~Impl() = default;

  void Track(FrameId frame_id, int64_t timestamp_ns, const odom::IVisualOdometry::VOFrameStat& stat,
             const Isometry3T& delta, const sof::Images& images) {
    sof::Images slam_images(rig_.num_cameras, nullptr);
    // only pass images for primary cameras; TODO: custom primary cameras
    for (CameraId cam_id : primary_cameras_) {
      if (cam_id < images.size() && images[cam_id] != nullptr) {
        slam_images[cam_id] = images[cam_id];
      }
    }
    async_slam_->TrackResult(frame_id, timestamp_ns, stat, slam_images, delta);
    // for synchronous execution:
    async_slam_->ProcessInputSynchronously();
  }
};

Slam::Slam(const Rig& rig, const std::vector<uint8_t>& primary_cameras, const Config& config)
    : impl(std::make_unique<Impl>(rig, primary_cameras, config)) {}

Slam::Slam(Slam&& other) noexcept = default;

Slam::~Slam() = default;

void Slam::Track(const Odometry::State& state, const Pose* gt_pose) {
  // Convert from public API types to internal types
  Isometry3T internal_delta = ConvertPoseToIsometry(state.delta);
  if (impl->gt_align_mode_) {
    THROW_INVALID_ARG_IF(gt_pose == nullptr, "gt_pose should be provided if gt_align_mode is enabled.");
    const Isometry3T prev_pose = impl->async_slam_->GetSlamPose();
    internal_delta = prev_pose.inverse() * ConvertPoseToIsometry(*gt_pose);
    internal_delta.linear() = common::CalculateRotationFromSVD(internal_delta.matrix());
  } else {
    THROW_INVALID_ARG_IF(gt_pose != nullptr, "gt_pose should be nullptr if gt_align_mode is disabled.");
  }

  odom::IVisualOdometry::VOFrameStat stat;
  stat.keyframe = state.keyframe;
  stat.heating = state.warming_up;

  // Convert observations to tracks2d
  stat.tracks2d.reserve(state.observations.size());
  for (const auto& obs : state.observations) {
    Track2D track;
    track.track_id = obs.id;
    track.uv = Vector2T(obs.u, obs.v);
    track.cam_id = obs.camera_index;
    stat.tracks2d.push_back(track);
  }

  // Convert landmarks to tracks3d
  for (const auto& lm : state.landmarks) {
    Vector3T internal_lm{lm.coords[0], lm.coords[1], lm.coords[2]};
    stat.tracks3d[lm.id] = internal_lm;
  }

  // Convert image contexts to sof::Images
  sof::Images slam_images(impl->rig_.num_cameras, nullptr);
  for (CameraId cam_id : impl->primary_cameras_) {
    if (cam_id >= slam_images.size()) {
      continue;
    }
    auto it = state.context.find(cam_id);
    if (it != state.context.end()) {
      slam_images[cam_id] = std::static_pointer_cast<sof::ImageContext>(it->second);
    }
  }

  impl->Track(state.frame_id, state.timestamp_ns, stat, internal_delta, slam_images);
  impl->frame_id_ = state.frame_id;

  // copy odometry landmarks to observations view
  auto view_manager = impl->landmarks_views_[ToUnderlying(DataLayer::Landmarks)];
  auto landmarks_view = view_manager ? view_manager->acquire_earliest() : nullptr;
  if (landmarks_view) {
    const Isometry3T slam_pose = impl->async_slam_->GetSlamPose();
    for (const auto& [id, v] : stat.tracks3d) {
      if (landmarks_view->landmarks.size() >= landmarks_view->landmarks.capacity()) {
        break;
      }
      landmarks_view->landmarks.push_back({static_cast<uint64_t>(id), 1, ToArray<float, 3>(slam_pose * v)});
    }
    landmarks_view->timestamp_ns = state.timestamp_ns;
  }
}

Pose Slam::GetPose() const { return ConvertIsometryToPose(impl->async_slam_->GetSlamPose()); }

void Slam::GetAllSlamPoses(std::vector<PoseStamped>& poses, uint32_t max_poses_count) const {
  std::map<uint64_t, storage::Isometry3<float>> frames;
  if (!impl->async_slam_->GetPosesForAllFrames(frames)) {
    throw std::runtime_error("Failed to get SLAM poses");
  }

  // Resize output vector to match the number of frames, but respect max_poses_count if specified
  size_t num_poses =
      max_poses_count > 0 ? std::min(frames.size(), static_cast<size_t>(max_poses_count)) : frames.size();
  poses.resize(num_poses);

  // Convert each pose to the public API format
  size_t i = 0;
  for (const auto& [timestamp_ns, pose] : frames) {
    if (i >= num_poses) {
      break;
    }
    poses[i].timestamp_ns = static_cast<int64_t>(timestamp_ns);
    poses[i].pose = ConvertIsometryToPose(pose);
    i++;
  }
}

void Slam::SaveMap(const std::string_view& folder_name, std::function<void(bool success)> callback) const {
  impl->async_slam_->CopyToDatabase(std::string{folder_name}, callback);
}

#define CALLBACK_AND_RETURN_IF(condition, callback, type, message) \
  if (condition) {                                                 \
    callback(Result<type>::Error(message));                        \
    return;                                                        \
  }

// timestamp_ns - localization timestamp
void Slam::LocalizeInMap(const std::string_view& folder_name, int64_t timestamp_ns, const Pose& guess_pose,
                         const ImageSet& images, const LocalizationSettings& settings, LocalizeStartCB start_cb,
                         LocalizeFinishCB finish_cb) {
  slam::LocalizerOptions localizer_options;
  localizer_options.use_gpu = impl->use_gpu_;

  localizer_options.horizontal_search_radius = settings.horizontal_search_radius;
  localizer_options.vertical_search_radius = settings.vertical_search_radius;
  localizer_options.horizontal_step = settings.horizontal_step;
  localizer_options.vertical_step = settings.vertical_step;
  localizer_options.angle_step_rads = settings.angular_step_rads;

  THROW_INVALID_ARG_IF(images.empty(), "No images provided");

  const Isometry3T isometry_guess_pose = ConvertPoseToIsometry(guess_pose);

  // copy user images
  Sources image_sources(impl->rig_.num_cameras);
  Metas image_metas(impl->rig_.num_cameras);
  sof::Images images_ptrs(impl->rig_.num_cameras, nullptr);
  // Get image dimensions from the first camera (assuming all cameras have same resolution)
  const ImageShape shape{images[0].width, images[0].height};
  constexpr size_t cache_size = 4;
  const auto image_manager = std::make_unique<sof::ImageManager>();
  const auto& primary_cameras = impl->primary_cameras_;
  const bool use_gpu = impl->use_gpu_;
  image_manager->init(shape, primary_cameras.size() * cache_size, use_gpu, 0);

#ifdef USE_CUDA
  cuda::Stream s{true};
#endif
  for (const auto& image : images) {
    auto cam_id = image.camera_index;
    if (std::none_of(primary_cameras.begin(), primary_cameras.end(), [cam_id](auto&& id) { return id == cam_id; })) {
      continue;
    }

    ImageSource& source = image_sources[cam_id];
    ImageMeta& meta = image_metas[cam_id];

    meta.frame_id = impl->frame_id_;
    meta.camera_index = cam_id;
    meta.timestamp = images[0].timestamp_ns;  // image timestamp could be different from localization timestamp

    FillImageSourceAndShape(image, source, meta.shape);

    sof::ImageContextPtr ptr = image_manager->acquire();
    THROW_INVALID_ARG_IF(ptr == nullptr, "Failed to acquire image context from image_manager");
    ptr->set_image_meta(meta);
    if (use_gpu) {
#ifdef USE_CUDA
      ptr->build_gpu_image_pyramid(source, false, s.get_stream());
      ptr->build_gpu_gradient_pyramid(false, s.get_stream());
#endif
    } else {
      ptr->build_cpu_image_pyramid(source, false);
      ptr->build_cpu_gradient_pyramid(false);
    }
    images_ptrs[cam_id] = ptr;
  }
#ifdef USE_CUDA
  if (use_gpu) {
    CUDA_CHECK(cudaStreamSynchronize(s.get_stream()));  // Synchronize once after all pyramids are built
  }
#endif

  const bool has_images =
      std::any_of(images_ptrs.begin(), images_ptrs.end(), [](const auto& image) { return image != nullptr; });
  THROW_INVALID_ARG_IF(!has_images, "No valid images to localize");

  impl->async_slam_->LocalizeInMap(folder_name, timestamp_ns, isometry_guess_pose, images_ptrs, settings, start_cb,
                                   finish_cb);
}

void Slam::GetSlamMetrics(Metrics& metrics) const {
  slam::AsyncSlamLCTelemetry telemetry;

  if (!impl->async_slam_->GetLastTelemetry(telemetry)) {
    throw std::runtime_error("Failed to get SLAM metrics");
  }

  // Convert metrics
  metrics.timestamp_ns = telemetry.timestamp_ns;
  metrics.lc_status = telemetry.lc_status;
  metrics.pgo_status = telemetry.pgo_status;
  metrics.lc_selected_landmarks_count = telemetry.lc_selected_landmarks_count;
  metrics.lc_tracked_landmarks_count = telemetry.lc_tracked_landmarks_count;
  metrics.lc_pnp_landmarks_count = telemetry.lc_pnp_landmarks_count;
  metrics.lc_good_landmarks_count = telemetry.lc_good_landmarks_count;
}

void Slam::GetLoopClosurePoses(std::vector<PoseStamped>& poses) const {
  // Get loop closure poses directly from AsyncSlam
  const std::list<slam::LoopClosureStamped>& last_loop_closures_stamped =
      impl->async_slam_->GetLastLoopClosuresStamped();

  // Resize output vector to match the number of loop closures
  poses.resize(last_loop_closures_stamped.size());

  // Convert each loop closure pose to the public API format
  size_t i = 0;
  for (const auto& lc_stamped : last_loop_closures_stamped) {
    poses[i].timestamp_ns = lc_stamped.timestamp_ns;
    poses[i].pose = ConvertIsometryToPose(lc_stamped.pose);
    i++;
  }
}

void Slam::EnableReadingData(DataLayer layer, uint32_t max_items_count) {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  switch (layer) {
    case DataLayer::Map:
    case DataLayer::LoopClosure:
    case DataLayer::Landmarks:
      assert(impl->landmarks_views_[ToUnderlying(layer)]);
      impl->landmarks_views_[ToUnderlying(layer)]->init(2, max_items_count);
      break;
    case DataLayer::PoseGraph:
      assert(impl->pose_graph_view_);
      impl->pose_graph_view_->init(2, max_items_count);
      break;
    default:
      throw std::runtime_error("Invalid data layer: " + std::to_string(ToUnderlying(layer)));
  }
}

void Slam::DisableReadingData(DataLayer layer) {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  switch (layer) {
    case DataLayer::Map:
    case DataLayer::LoopClosure:
    case DataLayer::Landmarks:
      assert(impl->landmarks_views_[ToUnderlying(layer)]);
      impl->landmarks_views_[ToUnderlying(layer)]->reset();
      break;
    case DataLayer::PoseGraph:
      assert(impl->pose_graph_view_);
      impl->pose_graph_view_->reset();
      break;
    default:
      throw std::runtime_error("Invalid data layer: " + std::to_string(ToUnderlying(layer)));
  }
}

std::shared_ptr<const Slam::Landmarks> Slam::ReadLandmarks(DataLayer layer) {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  const auto layer_index = ToUnderlying(layer);
  if (layer_index >= Slam::Impl::kMaxDataLayer) {
    throw std::runtime_error("Invalid data layer: " + std::to_string(ToUnderlying(layer)));
  } else if (layer == DataLayer::PoseGraph) {
    throw std::runtime_error("For DataLayer::PoseGraph use ReadPoseGraph() instead");
  }
  assert(impl->landmarks_views_[layer_index]);
  auto latest = impl->landmarks_views_[layer_index]->acquire_latest();
  if (latest) {
    impl->landmarks_[layer_index]->timestamp_ns = latest->timestamp_ns;
    impl->landmarks_[layer_index]->landmarks.clear();
    impl->landmarks_[layer_index]->landmarks.reserve(latest->landmarks.size());
    for (const auto& lm : latest->landmarks) {
      impl->landmarks_[layer_index]->landmarks.push_back({lm.id, lm.weight, lm.coords});
    }
  }
  return impl->landmarks_[layer_index];
}

std::shared_ptr<const Slam::PoseGraph> Slam::ReadPoseGraph() {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  assert(impl->pose_graph_view_);
  auto latest = impl->pose_graph_view_->acquire_latest();
  if (latest) {
    impl->pose_graph_->timestamp_ns = latest->timestamp_ns;
    impl->pose_graph_->nodes.clear();
    impl->pose_graph_->edges.clear();
    impl->pose_graph_->nodes.reserve(latest->nodes.size());
    impl->pose_graph_->edges.reserve(latest->edges.size());
    for (const auto& node : latest->nodes) {
      impl->pose_graph_->nodes.push_back({node.id, ConvertIsometryToPose(node.node_pose)});
    }
    for (const auto& edge : latest->edges) {
      PoseCovariance covariance;
      mat<6>(covariance) = edge.covariance;
      impl->pose_graph_->edges.push_back(
          {edge.node_from, edge.node_to, ConvertIsometryToPose(edge.transform), covariance});
    }
  }
  return impl->pose_graph_;
}

}  // namespace cuvslam

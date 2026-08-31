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

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

/// @cond Doxygen_Suppress
#ifdef _WIN32
#ifdef CUVSLAM_EXPORT
#define CUVSLAM_API __declspec(dllexport)
#else
#define CUVSLAM_API __declspec(dllimport)
#endif
#else
#define CUVSLAM_API __attribute__((visibility("default")))
#endif
/// @endcond

namespace cuvslam {
namespace internal {
struct Internals;
struct InternalParameter;
}  // namespace internal

/**
 * @brief Get the version of the library.
 * Any one of the pointers could be null.
 * @param[out] major   - major version
 * @param[out] minor   - minor version
 * @param[out] patch   - patch version
 * @return semantic version string view
 */
CUVSLAM_API
std::string_view GetVersion(int32_t* major, int32_t* minor, int32_t* patch);

/**
 * Set verbosity. The higher the value, the more output from the library. 0 (default) for no output.
 * @param[in] verbosity new verbosity value
 */
CUVSLAM_API
void SetVerbosity(int verbosity);

/**
 * Warms up GPU, creates CUDA runtime context.
 * This function is not mandatory to call, but helps to save some time in tracker initialization.
 * It can also be used to quickly diagnose issues with CUDA or CUDA libraries.
 * @throws std::runtime_error if CUDA, cusolver or cublas initialization fails.
 */
CUVSLAM_API
void WarmUpGPU();

/**
 * Static-size array of 32-bit floats
 */
template <std::size_t N>
using Array = std::array<float, N>;

/**
 * 3D vector of floats
 */
using Vector3f = Array<3>;

/**
 * Static-size array of 32-bit integers
 */
template <std::size_t N>
using IntArray = std::array<int32_t, N>;

/**
 * Transformation from one frame to another.
 * cuVSLAM uses OpenCV coordinate system convention: x is right, y is down, z is forward.
 */
struct Pose {
  Array<4> rotation = {0, 0, 0, 1};  ///< rotation quaternion in (x, y, z, w) order
  Array<3> translation = {0, 0, 0};  ///< translation vector
};

/**
 * 6x6 covariance matrix
 */
using PoseCovariance = Array<6 * 6>;

/**
 * @brief Distortion model with parameters
 *
 * Terminology:
 * - principal point \f$(c_x, c_y)\f$
 * - focal length \f$(f_x, f_y)\f$
 * - 2x2 diagonal matrix \f$\mathrm{diag}(f_x, f_y) = \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}\f$
 *
 * Supported distortion models:
 *
 * - Pinhole (0 parameters)
 *   - No distortion; equivalent to Brown with \f$k_0=k_1=k_2=p_0=p_1=0\f$.
 *
 * - Fisheye (4 parameters)
 *   - Also known as equidistant model for pinhole cameras.
 *   - Coefficients \f$k_1, k_2, k_3, k_4\f$ are compatible with ethz-asl/kalibr (pinhole-equi) and OpenCV::fisheye.
 *   - Limitation: this (pinhole + undistort) approach works only for FOV < 180°. TUMVI has ~190°.
 *     EuRoC and ORB_SLAM3 use a different approach (direct project/unproject without pinhole) and support > 180°;
 *     their coefficients are incompatible with this model.
 *   - Parameters:
 *     - 0..3: \f$(k_1, k_2, k_3, k_4)\f$
 *   - Projection:
 *     - \f$(u, v) = (c_x, c_y) + \mathrm{diag}(f_x, f_y) \cdot \frac{\mathrm{radial}(r) \cdot (x_n, y_n)}{r}\f$
 *     - where:
 *       - \f$\mathrm{radial}(r) = \arctan(r) \cdot \left(1 + k_1 \arctan^2(r) + k_2 \arctan^4(r) + k_3 \arctan^6(r)
 * + k_4 \arctan^8(r)\right)\f$
 *       - \f$x_n = x/z\f$
 *       - \f$y_n = y/z\f$
 *       - \f$r = \sqrt{x_n^2 + y_n^2}\f$
 *
 * - Brown (5 parameters)
 *   - Equivalent to Polynomial model with \f$k_4=k_5=k_6=0\f$; \b note a different order of parameters.
 *   - Parameters:
 *     - 0..2: radial \f$(k_1, k_2, k_3)\f$
 *     - 3..4: tangential \f$(p_1, p_2)\f$
 *   - Projection:
 *     - \f$(u, v) = (c_x, c_y) + \mathrm{diag}(f_x, f_y) \cdot \left( \mathrm{radial} \cdot (x_n, y_n) + (t_x, t_y)
 * \right)\f$
 *     - where:
 *       - \f$\mathrm{radial} = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6\f$
 *       - \f$t_x = 2 p_1 x_n y_n + p_2 (r^2 + 2 x_n^2)\f$
 *       - \f$t_y = p_1 (r^2 + 2 y_n^2) + 2 p_2 x_n y_n\f$
 *       - \f$x_n = x/z\f$
 *       - \f$y_n = y/z\f$
 *       - \f$r = \sqrt{x_n^2 + y_n^2}\f$
 *
 * - Polynomial (8 parameters)
 *   - Coefficients are compatible with the first 8 coefficients of the OpenCV distortion model.
 *   - Parameters:
 *     - 0..1: radial \f$(k_1, k_2)\f$
 *     - 2..3: tangential \f$(p_1, p_2)\f$
 *     - 4..7: radial \f$(k_3, k_4, k_5, k_6)\f$
 *   - Projection:
 *     - \f$(u, v) = (c_x, c_y) + \mathrm{diag}(f_x, f_y) \cdot \left( \mathrm{radial} \cdot (x_n, y_n) + (t_x, t_y)
 * \right)\f$
 *     - where:
 *       - \f$\mathrm{radial} = \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}\f$
 *       - \f$t_x = 2 p_1 x_n y_n + p_2 (r^2 + 2 x_n^2)\f$
 *       - \f$t_y = p_1 (r^2 + 2 y_n^2) + 2 p_2 x_n y_n\f$
 *       - \f$x_n = x/z\f$
 *       - \f$y_n = y/z\f$
 *       - \f$r = \sqrt{x_n^2 + y_n^2}\f$
 */
struct Distortion {
  /**
   * Distortion model type
   */
  enum class Model : uint8_t {
    Pinhole,
    Fisheye,
    Brown,
    Polynomial,
  };

  Model model = Model::Pinhole;   ///< distortion model @see Model
  std::vector<float> parameters;  ///< array of distortion parameters depending on model
};

/**
 * @brief Camera parameters
 *
 * Describes intrinsic and extrinsic parameters of a camera and per-camera settings.
 *
 * For camera coordinate system top left pixel has (0, 0) coordinate (y is down, x is right).
 * It's compatible with ROS CameraInfo/OpenCV.
 */
struct Camera {
  IntArray<2> size;          ///< image size in pixels (width, height)
  Array<2> principal;        ///< principal point in pixels \f$(c_x, c_y)\f$
  Array<2> focal;            ///< focal length in pixels \f$(f_x, f_y)\f$
  Pose rig_from_camera;      ///< transformation from coordinate frame of the camera to frame of the rig
  Distortion distortion;     ///< distortion parameters
  int32_t border_top{0};     ///< offset from the top border where visual features will be ignored (default: 0)
  int32_t border_bottom{0};  ///< offset from the bottom border where visual features will be ignored (default: 0)
  int32_t border_left{0};    ///< offset from the left border where visual features will be ignored (default: 0)
  int32_t border_right{0};   ///< offset from the right border where visual features will be ignored (default: 0)
};

/**
 * @brief IMU Calibration parameters
 *
 * Describes intrinsic and extrinsic (noise and random walk) parameters of an IMU sensor.
 * See [IMU Noise Model](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)
 */
struct ImuCalibration {
  Pose rig_from_imu;                  /**< Rig from imu transformation.
                                           vRig = rig_from_imu * vImu
                                           - vImu - vector in imu coordinate system
                                           - vRig - vector in rig coordinate system */
  float gyroscope_noise_density;      ///< \f$rad / (s * \sqrt{hz})\f$
  float gyroscope_random_walk;        ///< \f$rad / (s^2 * \sqrt{hz})\f$
  float accelerometer_noise_density;  ///< \f$m / (s^2 * \sqrt{hz})\f$
  float accelerometer_random_walk;    ///< \f$m / (s^3 * \sqrt{hz})\f$
  float frequency;                    ///< \f$hz\f$
};

/**
 * @brief Rig consisting of cameras and IMU sensors
 *
 * @note 1 to 32 cameras are supported now.
 * @note 0 or 1 IMU sensor is supported now.
 * @note An IMU sensor can be fused in Odometry::OdometryMode::Inertial (stereo + IMU) and in
 * Odometry::OdometryMode::Multisensor (one or more cameras, at least one RGB-D camera or overlapping pair, and an
 * optional single IMU).
 */
struct Rig {
  std::vector<Camera> cameras;       ///< Cameras; 1 to 32 cameras are supported now
  std::vector<ImuCalibration> imus;  ///< IMU sensors; 0 or 1 sensor is supported now
};

/**
 * @brief Image data structure
 *
 * @note Image pixels must be stored row-wise (right to left, top to bottom)
 * @note Image width and height must match Camera::size
 */
struct ImageData {
  /// @brief Image encoding
  enum class Encoding : uint8_t {
    MONO,  ///< grayscale or other single-channel data
    RGB,   ///< RGB
  };
  /// @brief Image data type
  enum class DataType : uint8_t {
    UINT8,    ///< 8-bit unsigned integer
    UINT16,   ///< 16-bit unsigned integer
    FLOAT32,  ///< 32-bit floating point
  };

  const void* pixels;  ///< Pixels must be stored row-wise (right to left, top to bottom)
  int32_t width;       ///< image width must match Camera::size
  int32_t height;      ///< image height must match Camera::size
  int32_t pitch;       ///< bytes per image row including padding for GPU memory images, ignored for CPU images
  Encoding encoding;   ///< grayscale and RGB are supported now
  DataType data_type;  ///< image data type
  bool is_gpu_mem;     ///< is pixels pointer points to GPU or CPU memory buffer
};

/**
 * @brief Image with timestamp and camera index
 */
struct Image : ImageData {
  int64_t timestamp_ns;   ///< Image timestamp in nanoseconds
  uint32_t camera_index;  ///< index of the camera in the rig
};

/**
 * @brief IMU measurement
 */
struct ImuMeasurement {
  int64_t timestamp_ns;           ///< IMU measurement timestamp in nanoseconds
  Array<3> linear_accelerations;  ///<  \f$m / s^2\f$
  Array<3> angular_velocities;    ///< \f$rad / s\f$
};

/**
 * @brief Pose with timestamp
 */
struct PoseStamped {
  int64_t timestamp_ns;  ///< Pose timestamp in nanoseconds
  Pose pose;             ///< Pose (transformation between two coordinate frames)
};

/**
 * @brief Pose with covariance
 *
 * Pose covariance is exposed over the public pose variables:
 * (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis).
 * Translation is in meters. Rotation uses a fixed-axis representation in radians.
 */
struct PoseWithCovariance {
  Pose pose;                         ///< Pose (transformation between two coordinate frames)
  PoseCovariance covariance_xyz_rpy; /**< Row-major representation of the 6x6 covariance matrix.
                                      The orientation parameters use a fixed-axis representation.
                                      In order, the parameters are:
                                      (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis).
                                      Translation in meters, rotation in radians.*/
};

/**
 * @brief Rig pose estimate from the tracker
 *
 * The rig coordinate frame is user-defined and depends on the extrinsic parameters of the cameras.
 * The cameras' coordinate frames may not match the rig coordinate frame - depending on camers extrinsics.
 * The world coordinate frame is an arbitrary 3D coordinate frame. It coincides with the rig coordinate frame at the
 * first frame.
 */
struct PoseEstimate {
  int64_t timestamp_ns;                              ///< Pose timestamp (in nanoseconds) will match image timestamp
  std::optional<PoseWithCovariance> world_from_rig;  ///< Transform from rig coordinate frame to world coordinate frame
};

/**
 * @brief Observation
 *
 * 2D point with coordinates in image.
 * (0, 0) is the top-left corner.
 */
struct Observation {
  uint64_t id;            ///< observation id
  float u;                ///< 0 <= u < image width; (0, 0) is the top-left corner
  float v;                ///< 0 <= v < image height; (0, 0) is the top-left corner
  uint32_t camera_index;  ///< camera index
};

/**
 * @brief Landmark
 *
 * 3D point with coordinates in meters in world frame
 */
struct Landmark {
  uint64_t id;      ///< landmark id
  Vector3f coords;  ///< x, y, z in meters in world frame
};

/**
 * @brief Estimates rig motion from camera, depth, and IMU input
 *
 * Processes synchronized sensor data to estimate the rig pose. Use directly for odometry-only
 * workflows, or through Tracker when coordinating with SLAM.
 */
class CUVSLAM_API Odometry {
public:
  /// Image set
  using ImageSet = std::vector<Image>;
  /// Gravity acceleration vector (magnitude ~9.81 m/s²) in the rig / VO frame; +Y is down (OpenCV), so it is
  /// approximately (0, +g, 0) when the rig is upright.
  using Gravity = Vector3f;

  /// IMU state: velocity, gyro bias, accelerometer bias
  struct ImuState {
    Vector3f velocity;
    Vector3f gyro_bias;
    Vector3f acc_bias;
  };

  /**
   * @brief Multicamera mode
   *
   * Multicamera mode defines which cameras will be used for mono SOF (primary cameras)
   */
  enum class MulticameraMode : uint8_t {
    /// primary cameras auto selection, each secondary camera must be connected to only one primary camera
    Performance,
    /// all cameras are primary cameras
    Precision,
    /// primary cameras auto selection, secondary cameras may be connected to more than one primary camera
    Moderate,
  };

  /**
   * @brief Odometry mode
   *
   * Odometry mode defines what data is expected and how it will be used by odometry tracker
   */
  enum class OdometryMode : uint8_t {
    Multicamera,  ///< Uses multiple synchronized stereo cameras, all cameras need to have frustum overlap with at least
                  ///< one another camera.
    Inertial,     ///< Uses stereo camera and IMU measurements. A single stereo-camera with a single IMU sensor is
                  ///< supported.
    RGBD,  ///< Uses RGB-D camera for tracking. A single RGB-D camera is supported. RGB & Depth images must be aligned.
    Mono,  ///< Uses a single camera, tracking is accurate up to scale.

    /// @warning Experimental: tracking may be inaccurate or fail for some sensor configurations and scenes.
    ///
    /// Unified multi-sensor mode (cuNLS-based). Supports any mix of plain RGB cameras,
    /// RGB-D cameras (any subset of the rig), with or without a single IMU. IMU fusion
    /// is enabled automatically when the rig contains an IMU; sba_mode is forced to the
    /// inertial bundler in that case. Per-camera depth presence is configured through
    /// MultisensorSettings::depth_camera_ids.
    ///
    /// @note Requirements (construction throws `std::invalid_argument` otherwise):
    ///  - Build must have cuNLS enabled (`-DUSE_CUNLS=ON`).
    ///  - The rig must provide at least one RGB-D camera through
    ///    MultisensorSettings::depth_camera_ids, or at least one camera pair with
    ///    overlapping frustums. A single RGB-D camera is valid, with or without an IMU.
    ///  - The current cuNLS solver supports pinhole cameras. Other camera models emit a
    ///    warning and are not supported in Multisensor mode.
    ///  - Depth images passed to `Track()` must use `Encoding::MONO` with
    ///    `DataType::UINT16` or `DataType::FLOAT32`; see `Track()` for the per-frame
    ///    matching rules against `MultisensorSettings::depth_camera_ids`.
    Multisensor,
  };

  /**
   * @brief Multisensor odometry settings
   *
   * @warning Experimental: tracking may be inaccurate or fail for some sensor configurations and scenes.
   *
   * Used only when Config::odometry_mode == OdometryMode::Multisensor.
   *
   * Multisensor mode supports any mix of:
   *   - one or more plain RGB cameras
   *   - one or more RGB-D cameras (any subset of the rig)
   *   - a single optional IMU (configured through Rig::imus)
   *
   * The rig's IMU presence drives IMU fusion automatically — there is no separate
   * `with_imu` flag. The only fields here describe depth handling.
   */
  struct MultisensorSettings {
    /// @brief Camera ids (matching Rig::cameras indices) that supply depth images at Track() time.
    ///
    /// Empty means: no depth (pure multi-camera-RGB). Cameras not listed here are treated as
    /// RGB-only. Mixed rigs (some RGB, some RGB-D) are supported by listing only the depth-capable
    /// cameras.
    std::vector<int32_t> depth_camera_ids;

    /// @brief Scale factor for depth measurements (denominator: raw depth divided by this is meters).
    /// Applied uniformly across all depth cameras. Default: 1.f.
    float depth_scale_factor = 1.f;

    /// @brief Allow stereo 2D tracking between depth-aligned cameras and other cameras.
    ///
    /// Default: true. Multisensor mode benefits from cross-camera 2D tracks (depth-aligned
    /// cameras are typically informative anchors), so we enable this by default. Set to false
    /// to opt out.
    bool enable_depth_stereo_tracking = true;
  };

  /**
   * @brief RGBD odometry settings
   */
  struct RGBDSettings {
    /// @brief Scale of provided depth measurements. Default: 1.f
    ///
    /// The scale factor is the denominator used to convert raw depth values from the input depth image to actual
    /// distance measurements in meters. For example, in [TUM
    /// RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect)
    /// a factor of 5000 is used for 16-bit PNG images, meaning each pixel value should be divided by 5000 to get the
    /// depth in meters, while a factor of 1 is used for 32-bit float images, where the depth values are already in
    /// meters.
    float depth_scale_factor = 1.f;

    /// @brief Depth camera id.
    ///
    /// Depth image is supposed to be pixel-to-pixel aligned with some RGB camera image.
    /// This field specifies camera id, that depth is aligned with. Default: -1
    int32_t depth_camera_id = -1;

    /// Allows stereo 2D tracking between depth-aligned camera and any other camera. Default: false
    bool enable_depth_stereo_tracking = false;
  };

  /**
   * @brief Configuration parameters of the VIO tracker
   */
  struct Config {
    /// Multicamera mode. Default: MulticameraMode::Precision
    MulticameraMode multicam_mode = MulticameraMode::Precision;
    /// Odometry mode. Default: OdometryMode::Multicamera
    OdometryMode odometry_mode = OdometryMode::Multicamera;
    /// Enable tracking using GPU. Default: true.
    bool use_gpu = true;
    /// Enable SBA asynchronous mode. Default: true.
    bool async_sba = true;
    /**
     * @brief Enable internal pose prediction mechanism. Default: true
     *
     * If frame rate is high enough it improves tracking performance and stability.
     * As a general rule it is better to use a pose prediction mechanism
     * tailored to a specific application. If you have an IMU, consider using
     * it to provide pose predictions to cuVSLAM.
     */
    bool use_motion_model = true;
    /// Enable image denoising. Disable if the input images have already passed through a denoising filter.
    /// Default: false
    bool use_denoising = false;
    /// Enable fast and robust tracking between rectified cameras with principal points on the horizontal line.
    /// Default: false
    bool rectified_stereo_camera = false;
    /// Enable GetLastObservations(). Warning: export flags slow down execution and result in additional memory usage.
    /// Default: false
    bool enable_observations_export = false;
    /// Enable GetLastLandmarks(). Warning: export flags slow down execution and result in additional memory usage.
    /// Default: false
    bool enable_landmarks_export = false;
    /// Enable GetFinalLandmarks(). Warning: export flags slow down execution and result in additional memory usage.
    /// This flag also sets enable_landmarks_export and enable_observations_export.
    /// Default: false
    bool enable_final_landmarks_export = false;
    /// Maximum frame delta in seconds. Odometry will warn if time delta between frames is higher than the threshold.
    /// Default: 1.f
    float max_frame_delta_s = 1.f;
    /// Directory where input data will be dumped in edex format.
    std::string_view debug_dump_directory;
    /// Enable IMU debug mode. Default: false
    bool debug_imu_mode = false;
    /// RGBD odometry settings.
    RGBDSettings rgbd_settings;
    /// Multisensor odometry settings. Used only when odometry_mode == OdometryMode::Multisensor.
    MultisensorSettings multisensor_settings;
    /// Minimum scene depth (meters) sampled along the epipolar curve when generating LK initial
    /// guesses for left-to-right (L2R) tracking. Used in every odometry mode that tracks between
    /// overlapping camera pairs — Multicamera, Inertial, RGBD and Multisensor. Ignored in
    /// OdometryMode::Mono, which has no L2R stage, and in any rig without an overlapping pair.
    /// Any negative value (e.g. -1) auto-detects from the pair baseline: small stereo (~7 cm)
    /// → 0.1 m, KITTI-scale (~0.5 m) → 7 m. Default: -1.f (auto).
    float min_depth = -1.f;
    /// Maximum scene depth (meters) sampled along the epipolar curve when generating LK initial
    /// guesses for left-to-right (L2R) tracking. Used in the same modes as min_depth.
    /// Any negative value (e.g. -1) auto-detects from the pair baseline: small stereo (~7 cm)
    /// → 20 m, KITTI-scale (~0.5 m) → 1000 m. Default: -1.f (auto).
    /// This only places the farthest initial guess; nothing is clamped. Points beyond max_depth
    /// still track and triangulate normally — the leftover disparity there (fx*B/max_depth, about
    /// 0.4 px for KITTI at 1000 m) is well inside LK's basin. Infinity is therefore unnecessary,
    /// and non-finite values are rejected.
    float max_depth = -1.f;
  };

  // TODO(vikuznetsov): remove when https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165 is fixed
  /// @brief Get default configuration.
  ///
  /// @see Config for default values.
  /// @return default configuration
  static Config GetDefaultConfig() { return Config{}; }

  /**
   * @brief State of the odometry tracker
   *
   * Only available if data export is enabled in Config.
   */
  struct State {
    struct Context;
    using ContextMap = std::unordered_map<uint8_t, std::shared_ptr<Context>>;

    uint64_t frame_id;               ///< Internal frame id
    int64_t timestamp_ns;            ///< Timestamp in nanoseconds
    Pose delta;                      ///< Pose change since last keyframe
    bool keyframe;                   ///< Is this frame a keyframe?
    bool warming_up;                 ///< Is the tracker in warming up phase?
    std::optional<Gravity> gravity;  ///< Optional gravity. Available in Inertial or Multisensor mode with an IMU.
    std::vector<Observation> observations;  ///< Observations for this frame
    std::vector<Landmark> landmarks;        ///< Landmarks for this frame
    ContextMap context;                     ///< Opaque context information for this frame (used internally by Slam)
  };

  /**
   * @brief Construct a tracker
   *
   * @param[in] rig  rig setup
   * @param[in] cfg  tracker configuration
   * @throws std::runtime_error if tracker fails to initialize
   * @throws std::invalid_argument if rig or config is invalid
   */
  explicit Odometry(const Rig& rig, const Config& cfg = GetDefaultConfig());

  /**
   * @brief Move constructor
   *
   * @param[in] other other tracker
   */
  Odometry(Odometry&& other) noexcept;

  /// @brief Destructor
  ~Odometry();

  /**
   * @brief Track a rig pose using current frame
   *
   * Track current frame synchronously: the function blocks until the tracker has computed a pose.
   * By default, this function uses visual odometry to compute a pose.
   * If visual odometry tracker fails to compute a pose, in Inertial mode or Multisensor mode with an IMU the function
   * returns the position calculated from user-provided IMU data.
   * If after several calls of Track() visual odometry is not able to recover,
   * then invalid pose will be returned.
   *
   * The track will output poses in the same coordinate system until a loss of tracking.
   *
   * Image timestamps have to match. cuVSLAM will use timestamp from the camera 0 image.
   * If a camera rig provides "almost synchronized" frames, the timestamps should be within 1 millisecond.
   * The number of images for tracker should not exceed rig->num_cameras.
   *
   * @param[in]  images  an array of synchronized images no more than rig->num_cameras.
   * Must use ImageData::DataType::UINT8. Partial ImageSet is supported, for example due to frame drops. Corresponding
   * cameras are identified by Image::camera_index.
   * @param[in]  masks  (Optional) an array of corresponding masks no more than rig->num_cameras.
   * Must use ImageData::DataType::UINT8. Partial ImageSet is supported, for example if mask is calculated on for some
   * cameras. Corresponding cameras are identified by Image::camera_index.
   * @param[in]  depths  (Optional) an array of depth images. In OdometryMode::RGBD exactly one depth
   * image must be provided. In OdometryMode::Multisensor pass one depth image per depth-providing
   * camera; each entry is matched to its rig camera by Image::camera_index and every camera_index
   * must appear in MultisensorSettings::depth_camera_ids. Other modes must pass an empty array.
   * Must use ImageData::Encoding::MONO and ImageData::DataType::UINT16 or ImageData::DataType::FLOAT32.
   * @param[in]  internals (Optional) pointer to internal per-frame development parameters; pass nullptr (default) to
   * use built-in defaults. Not intended for production use.
   *
   * @return On success `PoseEstimate` contains estimated rig pose, on failure `PoseEstimate::world_from_rig` will be
   * `nullopt`.
   * @throws std::invalid_argument if image parameters are invalid
   * @throws std::runtime_error in case of unexpected errors
   */
  PoseEstimate Track(const ImageSet& images, const ImageSet& masks = {}, const ImageSet& depths = {},
                     const cuvslam::internal::Internals* internals = nullptr);

  /**
   * @brief Register IMU measurement
   *
   * If visual odometry loses camera position, it briefly continues execution
   * using user-provided IMU measurements while trying to recover the position.
   * You should call these functions in monotonic timestamp order however many IMU measurements you have
   * between image acquisitions:
   *
   * - tracker.Track
   * - tracker.RegisterImuMeasurement
   * - ...
   * - tracker.RegisterImuMeasurement
   * - tracker.Track
   *
   * IMU measurements and frame images both have timestamps, so calls to Track() and RegisterImuMeasurement()
   * on the same Odometry instance must be made in non-decreasing timestamp order and externally serialized.
   * Do not call them concurrently from different threads.
   * If IMU samples are captured on a separate thread, buffer them and submit them in timestamp order from
   * the same sequence that calls Track(), or protect all calls with caller-owned synchronization.
   *
   * @param[in] sensor_index Sensor index; must be 0, as only one sensor is supported now
   * @param[in] imu IMU measurements
   * @throws std::invalid_argument if IMU fusion is disabled or if called out of the order of timestamps
   * @see Track
   */
  void RegisterImuMeasurement(uint32_t sensor_index, const ImuMeasurement& imu);

  /**
   * @brief Get Last Observations
   *
   * Get an array of observations from the last VO frame for a specific camera
   *
   * @param[in] camera_index Index of the camera to get observations for
   * @return Array of observations
   * @throws std::invalid_argument if stats export is disabled
   * @see Observation
   */
  std::vector<Observation> GetLastObservations(uint32_t camera_index) const;

  /**
   * @brief Get Last Landmarks
   *
   * Get an array of landmarks from the last VO frame;
   * Landmarks are 3D points in the last camera frame.
   * @return Array of landmarks
   * @throws std::invalid_argument if stats export is disabled
   * @see Landmark
   */
  std::vector<Landmark> GetLastLandmarks() const;

  /**
   * @brief Get Last Gravity
   *
   * Get gravity acceleration vector in the last VO / rig frame (+Y down, OpenCV convention).
   * @return Optional gravity vector. Empty if gravity is not yet available.
   * @throws std::invalid_argument if IMU fusion is disabled
   * @see Gravity
   */
  std::optional<Gravity> GetLastGravity() const;

  /**
   * @brief Get current IMU state (velocity, gyro bias, acc bias).
   * @return IMU state or nullopt before the first successful track.
   */
  std::optional<ImuState> GetImuState() const;

  /**
   * @brief Get tracker state
   *
   * Only available if data export is enabled in Config.
   *
   * @param[out] state Odometry state to be filled
   * @throws std::invalid_argument if stats export is disabled
   * @see State
   */
  void GetState(State& state) const;

  /**
   * @brief Get all final landmarks from all frames
   *
   * Landmarks are 3D points in the odometry start frame.
   * @return std::unordered_map<uint64_t, Vector3f>
   * @throws std::invalid_argument if stats export is disabled
   * @see Landmark
   */
  std::unordered_map<uint64_t, Vector3f> GetFinalLandmarks() const;

  /**
   * @brief Get primary camera indices used for tracking
   *
   * @return Vector of primary camera indices
   */
  const std::vector<uint8_t>& GetPrimaryCameras() const;

  /**
   * @brief Apply internal parameters by string key/value pairs.
   *
   * Allows setting internal runtime settings by name. Unknown keys log a warning and are ignored.
   * Invalid values or keys not applicable to the current odometry mode throw std::invalid_argument.
   *
   * For internal use only.
   *
   * Supported keys (grouped by prefix):
   *
   * SBA (all modes):
   *   `sba.num_sba_frames`, `sba.num_inertial_sba_frames`, `sba.num_fixed_sba_frames`,
   *   `sba.num_sba_iterations`, `sba.robustifier_scale`, `sba.use_sba_winsorizer`
   *
   * Note: `sba.async` and `sba.mode` are construction-time settings that determine whether the
   * SBA background thread is spawned and which bundler is used. They cannot be changed after the
   * tracker is created. Set `Odometry::Config::async_sba` and `Odometry::Config::odometry_mode` before
   * constructing the Odometry object instead.
   *
   * StateMachine / IMU gravity estimation (Inertial mode only):
   *   `sm.gravity_update_period_ns`, `sm.max_integration_time_ns`, `sm.min_num_kf_for_gravity`,
   *   `sm.min_time_period_ns`, `sm.max_time_period_ns`
   *
   * @param[in] parameters Key/value pairs to apply.
   */
  void ApplyPersistentInternalParameters(const std::vector<cuvslam::internal::InternalParameter>& parameters);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/**
 * @brief Result type that can hold either success data or error information.
 *
 * For use in callbacks. Result::error_message should not outlive the callback scope.
 */
// TODO(C++23): replace with std::expected
template <typename T>
struct Result {
  std::optional<T> data;           ///< data
  std::string_view error_message;  ///< error message

  /// Create a success result
  /// @param[in] value data
  /// @return Result
  static Result Success(T&& value) { return Result{std::move(value), ""}; }

  /// Create an error result
  /// @param[in] message error message
  /// @return Result
  static Result Error(std::string_view message) { return Result{std::nullopt, message}; }
};

/**
 * @brief Builds and optimizes a reusable map from odometry results
 *
 * Consumes Odometry::State to maintain a pose graph, detect loop closures, save and load maps, and
 * relocalize in an existing map.
 *
 * Thread safety: all methods must be called from a single thread, except LocalizeInMap()
 * and SaveMap() which may be called concurrently with any other method from another thread.
 */
class CUVSLAM_API Slam {
public:
  /// @brief Image set
  using ImageSet = std::vector<Image>;

  /**
   * @brief SLAM configuration parameters
   */
  struct Config {
    /// If empty, map is kept in memory only. Else, map is synced to disk (LMDB) at this path, allowing large-scale
    /// maps; if the path already exists it will be overwritten. To load an existing map, use LocalizeInMap(). To save
    /// map, use SaveMap().
    std::string_view map_cache_path = "";
    /// Enable GPU use for SLAM
    bool use_gpu = true;
    /// Synchronous mode (does not run a separate work thread if true)
    bool sync_mode = false;
    /// Enable reading internal data from SLAM (Pose Graph, Loop Closures, Landmarks, etc.).
    /// Additionally separate data layers are enabled by `EnableReadingData`.
    bool enable_reading_internals = false;
    /// Planar constraints. SLAM poses will be modified so that the camera moves on a horizontal plane.
    bool planar_constraints = false;
    /// Special SLAM mode for visual map building in case ground truth is present.
    /// Not realtime, no loop closure, no map global optimization, SBA must be in main thread.
    bool gt_align_mode = false;
    /// Size of map cell. Default is 0 (the size will be calculated from the camera baseline).
    float map_cell_size = 0.0f;
    /// Maximum distance from camera to landmark for inclusion in map. Default is 100 meters.
    float max_landmarks_distance = 100.f;
    /// Maximum number of poses in SLAM pose graph. 300 is suitable for real-time mapping.
    /// The special value 0 means unlimited pose-graph.
    uint32_t max_map_size = 300;
    /// Minimum time interval between loop closure events in milliseconds.
    /// 1000 is suitable for real-time mapping.
    uint32_t throttling_time_ms = 0;
    /// How long the past is preserved. Maximum time to keep odometries delta history to be able to process
    /// LocalizeInMap within timestamps from past.
    uint32_t retention_time_ms = 5000;
    /// Length of the SLAM input queue at which cuVSLAM warns that SLAM is falling behind odometry. Diagnostic only:
    /// exceeding it does not change tracking behavior, it only prints a warning. SLAM runs in a background thread and
    /// is fed a queue of commands (keyframes, map localization, map saving); if it cannot keep up, the queue grows and
    /// the poses and loop closures SLAM reports refer to an increasingly old point of the trajectory. The warning
    /// requires verbosity Warning or higher (see SetVerbosity). Default: 10 queued commands.
    uint32_t delay_warning_queue_size = 10;
  };

  // TODO(vikuznetsov): remove when https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165 is fixed
  /// Get default configuration
  /// @return default configuration
  static Config GetDefaultConfig() { return Config{}; }

  /**
   * @brief Localization settings for use in LocalizeInMap
   */
  struct LocalizationSettings {
    float horizontal_search_radius;  ///< horizontal search radius in meters
    float vertical_search_radius;    ///< vertical search radius in meters
    float horizontal_step;           ///< horizontal step in meters
    float vertical_step;             ///< vertical step in meters
    float angular_step_rads;         ///< angular step around vertical axis in radians
  };

  /**
   * @brief Metrics
   */
  struct Metrics {
    int64_t timestamp_ns;                  ///< timestamp of these measurements (in nanoseconds)
    bool lc_status;                        ///< loop closure status
    bool pgo_status;                       ///< pose graph optimization status
    uint32_t lc_selected_landmarks_count;  ///< Count of Landmarks Selected
    uint32_t lc_tracked_landmarks_count;   ///< Count of Landmarks Tracked
    uint32_t lc_pnp_landmarks_count;       ///< Count of Landmarks in PNP
    uint32_t lc_good_landmarks_count;      ///< Count of Landmarks in LC
  };

  /**
   * @brief Data layer for SLAM
   */
  enum class DataLayer : uint8_t {
    Landmarks,    ///< Landmarks that are visible in the current frame
    Map,          ///< Landmarks of the map
    LoopClosure,  ///< Map's landmarks that are visible in the last loop closure event
    PoseGraph,    ///< Pose Graph
    Max,
  };

  /**
   * @brief Pose graph node
   */
  struct PoseGraphNode {
    uint64_t id;     ///< node identifier
    Pose node_pose;  ///< node pose
  };

  /**
   * @brief Pose graph edge
   */
  struct PoseGraphEdge {
    uint64_t node_from;         ///< node id
    uint64_t node_to;           ///< node id
    Pose transform;             ///< transform
    PoseCovariance covariance;  ///< covariance
  };

  /**
   * @brief Pose graph
   */
  struct PoseGraph {
    int64_t timestamp_ns;              ///< timestamp of the pose graph in nanoseconds
    std::vector<PoseGraphNode> nodes;  ///< nodes list
    std::vector<PoseGraphEdge> edges;  ///< edges list
  };

  /**
   * @brief Landmark with additional information
   */
  struct Landmark {
    uint64_t id;      ///< identifier
    float weight;     ///< weight (ignored now)
    Vector3f coords;  ///< x, y, z in meters in world frame
  };

  /**
   * Landmarks array
   */
  struct Landmarks {
    int64_t timestamp_ns;  ///< timestamp of landmarks in nanoseconds; corresponds to the timestamp of the frame where
                           ///< the landmarks were observed
    std::vector<Landmark> landmarks;  ///< landmarks list
  };

  /**
   * Construct a SLAM instance with rig and primary cameras
   * @param[in] rig Camera rig configuration
   * @param[in] primary_cameras Vector of primary camera indices
   * @param[in] config SLAM configuration
   * @throws std::runtime_error if SLAM initialization fails
   */
  Slam(const Rig& rig, const std::vector<uint8_t>& primary_cameras, const Config& config = GetDefaultConfig());

  /**
   * Move constructor
   * @param[in] other other SLAM instance
   */
  Slam(Slam&& other) noexcept;

  /// Destructor
  ~Slam();

  /**
   * Process tracking results from `Odometry::Track`. This should be called after each successful tracking.
   * @param[in] state Odometry state containing all tracking data
   * @param[in] gt_pose Optional ground truth pose. Should be provided if `gt_align_mode` is enabled, otherwise
   * should be nullptr.
   * @see `Odometry::Track`
   * @throws std::invalid_argument if `gt_pose` is passed incorrectly
   */
  void Track(const Odometry::State& state, const Pose* gt_pose = nullptr);

  /**
   * Get the current SLAM rig pose in the world frame.
   *
   * Returns the most recent pose computed by SLAM. Before the first keyframe is processed the pose
   * is the identity transform.
   * @return Current rig pose in world frame
   */
  Pose GetPose() const;

  /**
   * Get all SLAM poses for each frame.
   * @param[in] max_poses_count maximum number of poses to return
   * @param[out] poses Vector of poses with timestamps
   * This call could be blocked by slam thread.
   */
  void GetAllSlamPoses(std::vector<PoseStamped>& poses, uint32_t max_poses_count = 0) const;

  /**
   * Save SLAM database (map) to folder asynchronously.
   * This folder will be created, if it does not exist.
   * Contents of the folder will be overwritten.
   * @param[in] folder_name Folder name, where SLAM database (map) will be saved
   * @param[in] callback Callback function to be called when save is complete, may be called in a separate thread
   */
  void SaveMap(const std::string_view& folder_name, std::function<void(bool success)> callback) const;

  /// Callback invoked when localization starts, may be called in a separate thread
  using LocalizeStartCB = std::function<void()>;
  /// Callback invoked when localization finishes, may be called in a separate thread
  using LocalizeFinishCB = std::function<void(const Result<Pose>& result)>;

  /**
   * Localize in the existing database (map).
   * If `Config.sync_mode` is false, the request is queued for the background slam thread; the callback runs when
   * localization finishes (possibly on another thread than the caller). If `Config.sync_mode` is true, localization
   * runs immediately before this call returns.
   * Finds the rig pose in the saved map. If successful, replace current map with saved one.
   * @param[in] folder_name Folder containing the saved SLAM map (database)
   * @param[in] timestamp_ns Time in nanoseconds for the localized pose. If all images timestamps are equal, it makes
   *                         sense to use images[0].timestamp_ns.
   * @param[in] guess_pose Initial guess for rig pose at the images' timestamp
   * @param[in] images Observed images from multicamera (1 - mono, 2 - stereo, etc.)
   * @param[in] settings Localization settings
   * @param[in] start_cb Called when localization starts, may be called in a separate thread
   * @param[in] finish_cb Called when localization completes, may be called in a separate thread
   */
  void LocalizeInMap(const std::string_view& folder_name, int64_t timestamp_ns, const Pose& guess_pose,
                     const ImageSet& images, const LocalizationSettings& settings, LocalizeStartCB start_cb,
                     LocalizeFinishCB finish_cb);

  /**
   * Get SLAM metrics.
   * @param[out] metrics SLAM metrics
   */
  void GetSlamMetrics(Metrics& metrics) const;

  /**
   * Get list of last 10 loop closure poses with timestamps.
   * @param[out] poses Vector of poses with timestamps
   */
  void GetLoopClosurePoses(std::vector<PoseStamped>& poses) const;

  /**
   * Enable reading data layer.
   * @param[in] layer Data layer to enable/disable
   * @param[in] max_items_count Maximum number of items to allocate in the layer
   */
  void EnableReadingData(DataLayer layer, uint32_t max_items_count);

  /**
   * Disable reading data layer.
   * @param[in] layer Data layer to disable
   */
  void DisableReadingData(DataLayer layer);

  /**
   * Read landmarks from a given data layer. Enabled by `EnableReadingData`.
   * @param[in] layer Data layer to read
   * @return Landmarks
   */
  std::shared_ptr<const Landmarks> ReadLandmarks(DataLayer layer);

  /**
   * Read pose graph. Enabled by `EnableReadingData(DataLayer::PoseGraph)`.
   * @return Pose graph
   */
  std::shared_ptr<const PoseGraph> ReadPoseGraph();

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/**
 * @brief Visual odometry with optional SLAM, combined behind a single interface
 *
 * Tracker owns an Odometry instance and, when constructed with a non-null SLAM configuration,
 * a Slam instance. It runs the standard per-frame sequence for you: Odometry::Track(), then
 * Odometry::GetState() and Slam::Track() when odometry produced a pose, then Slam::GetPose().
 *
 * Everything Tracker does can be done by driving Odometry and Slam directly; use those when you
 * need full control over the two components. Tracker is the recommended entry point otherwise.
 *
 * The underlying components stay reachable through GetOdometry() and GetSlam(). Tracker only
 * coordinates inputs that advance the odometry/SLAM pipeline; queries and module-specific
 * operations are performed on the component that owns them.
 *
 * Thread safety: same as the underlying components. Track() and RegisterImuMeasurement() must be
 * called from a single thread in non-decreasing timestamp order. Operations performed through
 * GetOdometry() and GetSlam() follow the thread-safety contracts of those classes.
 */
class CUVSLAM_API Tracker {
public:
  /// Image set
  using ImageSet = Odometry::ImageSet;

  /**
   * @brief Result of a single Track() call
   */
  struct TrackResult {
    /// Odometry pose estimate. On failure `odometry.world_from_rig` is `nullopt`.
    PoseEstimate odometry;
    /// SLAM pose in the world frame. Empty when SLAM is disabled or when odometry failed for this
    /// frame.
    std::optional<Pose> slam;
  };

  /**
   * @brief Construct a tracker
   *
   * When `slam_config` is non-null, Tracker enables observation and landmark export on its own copy
   * of `odometry_config`, because SLAM needs both. The configs passed by the caller are not modified.
   *
   * @param[in] rig               rig setup
   * @param[in] odometry_config   odometry configuration
   * @param[in] slam_config       optional SLAM configuration; pass nullptr (the default) to disable
   * SLAM. `Slam::Config::gt_align_mode` is not supported by Tracker; use standalone Odometry and
   * Slam instances for ground-truth-aligned map building.
   * @throws std::runtime_error if odometry or SLAM fails to initialize
   * @throws std::invalid_argument if rig or configuration is invalid
   */
  explicit Tracker(const Rig& rig, const Odometry::Config& odometry_config = Odometry::GetDefaultConfig(),
                   const Slam::Config* slam_config = nullptr);

  /**
   * @brief Move constructor
   *
   * @param[in] other other tracker
   */
  Tracker(Tracker&& other) noexcept = default;

  /// @brief Destructor
  ~Tracker() = default;

  /**
   * @brief Track a rig pose using current frame, and update SLAM
   *
   * Runs Odometry::Track() and, when SLAM is enabled and odometry produced a pose, feeds the
   * odometry state to Slam::Track() and reads back the SLAM pose.
   *
   * Odometry poses stay in the same coordinate frame until a loss of tracking. SLAM poses jump when
   * a loop closure is detected and the pose graph is optimized; they are never adjusted
   * retroactively, so use GetSlam().GetAllSlamPoses() to get a smooth trajectory up to the latest
   * frame. In asynchronous mode loop closure runs on a separate thread to keep Track() fast, so
   * SLAM poses are not updated immediately.
   *
   * @param[in]  images     synchronized images, no more than the number of cameras in the rig
   * @param[in]  masks      (Optional) corresponding masks
   * @param[in]  depths     (Optional) depth images, see Odometry::Track()
   *
   * @return odometry pose estimate and, when available, the SLAM pose
   * @throws std::invalid_argument if image parameters are invalid
   * @throws std::runtime_error in case of unexpected errors
   * @see Odometry::Track, Slam::Track
   */
  TrackResult Track(const ImageSet& images, const ImageSet& masks = {}, const ImageSet& depths = {});

  /**
   * @brief Register IMU measurement
   *
   * @param[in] sensor_index Sensor index; must be 0, as only one sensor is supported now
   * @param[in] imu IMU measurements
   * @throws std::invalid_argument if IMU fusion is disabled or if called out of the order of
   * timestamps
   * @see Odometry::RegisterImuMeasurement
   */
  void RegisterImuMeasurement(uint32_t sensor_index, const ImuMeasurement& imu);

  /**
   * @brief Is SLAM enabled
   *
   * @return true when the tracker was constructed with a SLAM configuration
   */
  bool IsSlamEnabled() const;

  /**
   * @brief Get the underlying odometry
   *
   * The returned object is for queries only. Do not call Odometry::Track() on an odometry obtained
   * from Tracker; use Tracker::Track() so SLAM receives every successful odometry state.
   *
   * @return read-only odometry owned by this tracker
   */
  const Odometry& GetOdometry() const;

  /**
   * @brief Get the underlying SLAM
   *
   * Do not call Slam::Track() on a SLAM instance obtained from Tracker; use Tracker::Track() so
   * odometry and SLAM remain synchronized.
   *
   * @return SLAM owned by this tracker
   * @throws std::logic_error if SLAM is disabled
   */
  Slam& GetSlam();

  /// @copydoc Tracker::GetSlam()
  const Slam& GetSlam() const;

private:
  Odometry odometry_;
  std::unique_ptr<Slam> slam_;  ///< null when SLAM is disabled
};

}  // namespace cuvslam

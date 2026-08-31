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

#include <chrono>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/array.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/unordered_map.h"
#include "nanobind/stl/vector.h"

#include "cuvslam/cuvslam2.h"
#include "cuvslam/cuvslam2_internal.h"

#define THROW_INVALID_ARG_IF(condition, message) \
  do {                                           \
    if (condition) {                             \
      throw std::invalid_argument(message);      \
    }                                            \
  } while (0)

namespace nb = nanobind;
// using namespace nb::literals;
NB_MAKE_OPAQUE(std::shared_ptr<cuvslam::Odometry::State::Context>);

namespace cuvslam {

namespace {

template <typename T, typename ElementType, size_t N>
auto bind_array_accessors(nb::class_<T>& cls, const char* name, std::array<ElementType, N> T::*member,
                          const char* doc = "") {
  return cls.def_prop_rw(
      name,
      // Getter returns numpy array referencing the underlying data
      [member](T& self) { return nb::ndarray<ElementType, nb::numpy, nb::shape<N>>((self.*member).data()); },
      // Setter accepts an ndarray (e.g. np.ndarray) or a sequence and copies it to the underlying data
      [member](T& self, const nb::object& arr) {
        if (nb::isinstance<nb::ndarray<ElementType, nb::shape<N>, nb::device::cpu>>(arr)) {
          auto a = nb::cast<nb::ndarray<ElementType, nb::shape<N>, nb::device::cpu>>(arr);
          std::copy_n(a.data(), N, (self.*member).data());
        } else if (nb::isinstance<nb::sequence>(arr)) {
          nb::sequence a = nb::cast<nb::sequence>(arr);
          // any better way to check the size of nb::sequence?
          if (PySequence_Size(a.ptr()) != N) {
            throw nb::value_error(("Sequence must have exactly " + std::to_string(N) + " elements").c_str());
          }
          for (size_t i = 0; i < N; i++) {
            (self.*member)[i] = nb::cast<ElementType>(a[i]);
          }
        } else {
          throw nb::type_error(("Data must be an np::array or sequence of size " + std::to_string(N) +
                                ". Array must be 1D with element type " + std::string(typeid(ElementType).name()))
                                   .c_str());
        }
      },
      doc);
}

// PyCuVSLAM prioritises convenience over the C++ defaults: observation and landmark export are on
// unless the caller turns them off. Both the Odometry.Config constructor and Tracker's default
// odometry config use these, so the two cannot drift apart.
constexpr bool kPyDefaultEnableObservationsExport = true;
constexpr bool kPyDefaultEnableLandmarksExport = true;

Odometry::Config PyDefaultOdometryConfig() {
  Odometry::Config cfg;
  cfg.enable_observations_export = kPyDefaultEnableObservationsExport;
  cfg.enable_landmarks_export = kPyDefaultEnableLandmarksExport;
  return cfg;
}

enum class ArrayType {
  Image,
  Mask,
  Depth,
};

constexpr const char* GetArrayErrorMsg(ArrayType type) {
  switch (type) {
    case ArrayType::Image:
      return "Image must be 2D or 3D uint8 ndarray of shape (height, width) or (height, width, channels), "
             "with 1 or 3 channels.";
    case ArrayType::Mask:
      return "Mask must be 2D uint8 ndarray of shape (height, width).";
    case ArrayType::Depth:
      return "Depth must be 2D uint16 ndarray of shape (height, width).";
    default:
      assert(false);
      return "Unknown array type";
  }
}

cuvslam::Image ImageFromNDArray(const nb::ndarray<nb::ro>& tensor, int64_t timestamp, uint32_t camera_index,
                                ArrayType array_type) {
  assert(tensor.data() && tensor.size() != 0);
  auto error_msg = GetArrayErrorMsg(array_type);
  switch (array_type) {
    case ArrayType::Image:
      THROW_INVALID_ARG_IF(tensor.ndim() < 2 || 3 < tensor.ndim(), error_msg);
      THROW_INVALID_ARG_IF(tensor.dtype() != nb::dtype<uint8_t>(), error_msg);
      break;
    case ArrayType::Mask:
      THROW_INVALID_ARG_IF(tensor.ndim() != 2, error_msg);
      THROW_INVALID_ARG_IF(tensor.dtype() != nb::dtype<uint8_t>(), error_msg);
      break;
    case ArrayType::Depth:
      THROW_INVALID_ARG_IF(tensor.ndim() != 2, error_msg);
      THROW_INVALID_ARG_IF(tensor.dtype() != nb::dtype<uint16_t>(), error_msg);
      break;
  }

  cuvslam::Image img;
  img.timestamp_ns = timestamp;
  img.camera_index = camera_index;

  img.data_type = array_type == ArrayType::Depth ? cuvslam::Image::DataType::UINT16 : cuvslam::Image::DataType::UINT8;
  img.height = tensor.shape(0);
  img.width = tensor.shape(1);
  img.pitch = tensor.stride(0) * tensor.itemsize();
  if (tensor.ndim() == 2) {
    THROW_INVALID_ARG_IF(tensor.stride(1) != 1, "For 2D ndarrays, the width must be contiguous (stride(1) == 1)");
    img.encoding = cuvslam::Image::Encoding::MONO;
  } else if (tensor.ndim() == 3 && (tensor.shape(2) == 1 || tensor.shape(2) == 3)) {
    THROW_INVALID_ARG_IF(tensor.stride(2) != 1 || tensor.stride(1) != (int64_t)tensor.shape(2),
                         "For 3D ndarrays, the width must be contiguous and the number of channels must be 1 or 3 "
                         "(stride(1) == shape(2), stride(2) == 1)");
    img.encoding = tensor.shape(2) == 1 ? cuvslam::Image::Encoding::MONO : cuvslam::Image::Encoding::RGB;
  } else {
    THROW_INVALID_ARG_IF(true, error_msg);
  }
  if (tensor.device_type() == nb::device::cpu::value || tensor.device_type() == nb::device::cuda_host::value) {
    THROW_INVALID_ARG_IF(tensor.stride(0) * tensor.shape(0) != tensor.size(),
                         "For CPU and CUDA Host ndarrays, the data must be contiguous");
    img.is_gpu_mem = false;
  } else if (tensor.device_type() == nb::device::cuda::value ||
             tensor.device_type() == nb::device::cuda_managed::value) {
    img.is_gpu_mem = true;
  } else {
    THROW_INVALID_ARG_IF(true, "ndarray device type must be CPU, CUDA, CUDA Host or CUDA Managed");
  }
  img.pixels = tensor.data();

  return img;
}

// Converts a per-camera list of arrays into an ImageSet. Cameras are identified by their position in
// the list; empty entries are skipped, which is how callers signal "no data for this camera".
Odometry::ImageSet ImageSetFromNDArrays(const std::vector<nb::ndarray<nb::ro>>& images, int64_t timestamp,
                                        ArrayType array_type) {
  Odometry::ImageSet images_set;
  images_set.reserve(images.size());
  uint32_t cam_id = 0;
  for (auto& image : images) {
    if (image.data() && image.size() != 0) {
      images_set.push_back(ImageFromNDArray(image, timestamp, cam_id, array_type));
    }
    ++cam_id;
  }
  return images_set;
}

}  // namespace

NB_MODULE(pycuvslam, m) {
  // Free functions

  m.def(
      "get_version",
      []() -> std::tuple<std::string_view, int32_t, int32_t, int32_t> {
        int32_t major_val, minor_val, patch_val;
        auto ret = GetVersion(&major_val, &minor_val, &patch_val);
        return std::make_tuple(ret, major_val, minor_val, patch_val);
      },
      "Get the version of cuVSLAM library.\n\n"
      "Returns a tuple with the semantic version string and major, minor and patch values.");

  m.def(
      "set_verbosity", [](int verbosity) { SetVerbosity(verbosity); },
      "Set the verbosity level of the library.\n\n"
      "Available values: 0 (default) for no output, 1 for error messages, 2 for warnings, 3 for info messages.");

  m.def(
      "warm_up_gpu", []() { WarmUpGPU(); },
      "Warm up the GPU (CUDA runtime).\n\n"
      "It is not necessary to call it before the first call to cuvslam, "
      "but it can help to reduce the first call latency."
      "Throws an exception if CUDA, cusolver or cublas initialization fails.");

  // Data structures

  auto pose_cls =
      nb::class_<Pose>(m, "Pose",
                       "Transformation from one frame to another.\n\n"
                       "Consists of rotation (quaternion in x, y, z, w order) and translation (3-vector).\n"
                       "cuVSLAM uses OpenCV coordinate system convention: x is right, y is down, z is forward.")
          .def(nb::init<>())
          .def(nb::init<Array<4>, Array<3>>(), nb::arg("rotation"), nb::arg("translation"))
          .def("__str__",
               [](const Pose& pose) {
                 return nb::str("(rotation={}, translation={})").format(pose.rotation, pose.translation);
               })
          .def("__repr__", [](const Pose& pose) { return nb::str("cuvslam.Pose{}").format(pose); });
  bind_array_accessors(pose_cls, "rotation", &Pose::rotation, "Rotation (quaternion in x, y, z, w order)");
  bind_array_accessors(pose_cls, "translation", &Pose::translation, "Translation (3-vector)");

  auto distortion_cls =
      nb::class_<Distortion>(
          m, "Distortion",
          "Camera distortion model with parameters.\n\n"
          "Supports Pinhole (no distortion), Brown (radial and tangential), "
          "Fisheye (equidistant), and Polynomial distortion models.\n\n"
          "Common definitions used below:\n\n"
          "* principal point :math:`(c_x, c_y)`\n"
          "* focal length :math:`(f_x, f_y)`\n\n"
          "Supported values of distortion_model:\n\n"
          "**Pinhole (0 parameters):**\n"
          "No distortion\n\n"
          "**Fisheye (4 parameters):**\n"
          "Also known as equidistant distortion model for pinhole cameras.\n"
          "Coefficients k1, k2, k3, k4 are 100% compatible with ethz-asl/kalibr/pinhole-equi and OpenCV::fisheye.\n"
          "See: 'A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses' "
          "by Juho Kannala and Sami S. Brandt for further information.\n"
          "Please note, this approach (pinhole+undistort) has a limitation and works only field-of-view below 180°\n"
          "For the TUMVI dataset FOV is ~190°.\n"
          "EuRoC and ORB_SLAM3 use a different approach (direct project/unproject without pinhole) and support >180°,"
          " so their coefficient is incompatible with this model.\n\n"
          "* 0-3: fisheye distortion coefficients :math:`(k_1, k_2, k_3, k_4)`\n\n"
          "Each 3D point :math:`(x, y, z)` is projected in the following way:\n"
          ":math:`(u, v) = (c_x, c_y) + diag(f_x, f_y) \\cdot (distortedR(r) \\cdot (x_n, y_n) / r)`\n\n"
          "where:\n\n"
          ":math:`distortedR(r) = \\arctan(r) \\cdot (1 + k_1 \\cdot \\arctan^2(r) + k_2 \\cdot \\arctan^4(r) + "
          "k_3 \\cdot \\arctan^6(r) + k_4 \\cdot \\arctan^8(r))`\n"
          ":math:`x_n = x/z`\n"
          ":math:`y_n = y/z`\n"
          ":math:`r = \\sqrt{x_n^2 + y_n^2}`\n\n"
          "**Brown (5 parameters):**\n\n"
          "* 0-2: radial distortion coefficients :math:`(k_1, k_2, k_3)`\n"
          "* 3-4: tangential distortion coefficients :math:`(p_1, p_2)`\n\n"
          "Each 3D point :math:`(x, y, z)` is projected in the following way:\n"
          ":math:`(u, v) = (c_x, c_y) + diag(f_x, f_y) \\cdot (radial \\cdot (x_n, y_n) + tangential)`\n\n"
          "where:\n\n"
          ":math:`radial = (1 + k_1 \\cdot r^2 + k_2 \\cdot r^4 + k_3 \\cdot r^6)`\n"
          ":math:`tangential.x = 2 \\cdot p_1 \\cdot x_n \\cdot y_n + p_2 \\cdot (r^2 + 2 \\cdot x_n^2)`\n"
          ":math:`tangential.y = p_1 \\cdot (r^2 + 2 \\cdot y_n^2) + 2 \\cdot p_2 \\cdot x_n \\cdot y_n`\n"
          ":math:`x_n = x/z`\n"
          ":math:`y_n = y/z`\n"
          ":math:`r = \\sqrt{x_n^2 + y_n^2}`\n\n"
          "**Polynomial (8 parameters):**\n"
          "Coefficients are compatible with first 8 coefficients of OpenCV distortion model.\n\n"
          "* 0-1: radial distortion coefficients :math:`(k_1, k_2)`\n"
          "* 2-3: tangential distortion coefficients :math:`(p_1, p_2)`\n"
          "* 4-7: radial distortion coefficients :math:`(k_3, k_4, k_5, k_6)`\n\n"
          "Each 3D point :math:`(x, y, z)` is projected in the following way:\n"
          ":math:`(u, v) = (c_x, c_y) + diag(f_x, f_y) \\cdot (radial \\cdot (x_n, y_n) + tangential)`\n\n"
          "where:\n\n"
          ":math:`radial = \\frac{1 + k_1 \\cdot r^2 + k_2 \\cdot r^4 + k_3 \\cdot r^6}{1 + k_4 \\cdot r^2 + k_5 "
          "\\cdot r^4 + k_6 \\cdot r^6}`\n"
          ":math:`tangential.x = 2 \\cdot p_1 \\cdot x_n \\cdot y_n + p_2 \\cdot (r^2 + 2 \\cdot x_n^2)`\n"
          ":math:`tangential.y = p_1 \\cdot (r^2 + 2 \\cdot y_n^2) + 2 \\cdot p_2 \\cdot x_n \\cdot y_n`\n"
          ":math:`x_n = x/z`\n"
          ":math:`y_n = y/z`\n"
          ":math:`r = \\sqrt{x_n^2 + y_n^2}`")
          .def(nb::init<>())
          .def(nb::init<Distortion::Model, const std::vector<float>&>(), nb::arg("model"),
               nb::arg("parameters") = std::vector<float>{})
          .def_rw("model", &Distortion::model, "Distortion model type, see :class:`Distortion.Model`")
          .def_rw("parameters", &Distortion::parameters, "Array of distortion parameters depending on model")
          .def("__str__",
               [](const Distortion& distortion) {
                 return nb::str("(model={}, parameters={})").format(distortion.model, distortion.parameters);
               })
          .def("__repr__",
               [](const Distortion& distortion) { return nb::str("cuvslam.Distortion{}").format(distortion); });

  nb::enum_<Distortion::Model>(distortion_cls, "Model", "Distortion model types for camera calibration")
      .value("Pinhole", Distortion::Model::Pinhole, "No distortion (0 parameters)")
      .value("Brown", Distortion::Model::Brown, "Brown distortion model with 3 radial and 2 tangential coefficients")
      .value("Fisheye", Distortion::Model::Fisheye, "Fisheye distortion model (equidistant) with 4 coefficients")
      .value("Polynomial", Distortion::Model::Polynomial,
             "Polynomial distortion model with 8 coefficients, order: (k1, k2, p1, p2, k3, k4, k5, k6)");

  auto cam_cls =
      nb::class_<Camera>(m, "Camera",
                         "Camera calibration parameters.\n\n"
                         "Describes intrinsic and extrinsic parameters of a camera and per-camera settings.\n"
                         "For camera coordinate system, top left pixel has (0, 0) coordinate (y is down, x is right).")
          .def(nb::init<>())
          // WARNING: the order of init arguments in this definition must coincide with the order in the structure
          .def(nb::init<IntArray<2>, Array<2>, Array<2>, const Pose&, const Distortion&, int32_t, int32_t, int32_t,
                        int32_t>(),
               nb::kw_only(), nb::arg("size"), nb::arg("principal"), nb::arg("focal"),
               nb::arg("rig_from_camera") = Pose{}, nb::arg("distortion") = Distortion{}, nb::arg("border_top") = 0,
               nb::arg("border_bottom") = 0, nb::arg("border_left") = 0, nb::arg("border_right") = 0)
          .def_rw("rig_from_camera", &Camera::rig_from_camera,
                  "Transformation from the camera coordinate frame to the rig coordinate frame")
          .def_rw("distortion", &Camera::distortion, "Distortion parameters, see :class:`Distortion`")
          .def_rw("border_top", &Camera::border_top, "Top border to ignore in pixels (0 to use full frame)")
          .def_rw("border_bottom", &Camera::border_bottom, "Bottom border to ignore in pixels (0 to use full frame)")
          .def_rw("border_left", &Camera::border_left, "Left border to ignore in pixels (0 to use full frame)")
          .def_rw("border_right", &Camera::border_right, "Right border to ignore in pixels (0 to use full frame)")
          .def("__str__",
               [](const Camera& cam) {
                 return nb::str("(size={}, principal={}, focal={}, rig_from_cam={}, distortion={})")
                     .format(cam.size, cam.principal, cam.focal, cam.rig_from_camera, cam.distortion);
               })
          .def("__repr__", [](const Camera& cam) { return nb::str("cuvslam.Camera{}").format(cam); });
  bind_array_accessors(cam_cls, "size", &Camera::size, "Size of the camera (width, height)");
  bind_array_accessors(cam_cls, "principal", &Camera::principal, "Principal point (cx, cy)");
  bind_array_accessors(cam_cls, "focal", &Camera::focal, "Focal length (fx, fy)");

  nb::class_<ImuCalibration>(m, "ImuCalibration", "IMU Calibration parameters")
      .def(nb::init<>())
      .def(nb::init<const Pose&, float, float, float, float, float>(), nb::kw_only(), nb::arg("rig_from_imu") = Pose{},
           nb::arg("gyroscope_noise_density"), nb::arg("gyroscope_random_walk"), nb::arg("accelerometer_noise_density"),
           nb::arg("accelerometer_random_walk"), nb::arg("frequency"))
      .def_rw("rig_from_imu", &ImuCalibration::rig_from_imu,
              "Transformation from IMU coordinate frame to the rig coordinate frame")
      .def_rw("gyroscope_noise_density", &ImuCalibration::gyroscope_noise_density,
              "Gyroscope noise density in :math:`rad/(s*sqrt(hz))`")
      .def_rw("accelerometer_noise_density", &ImuCalibration::accelerometer_noise_density,
              "Accelerometer noise density in :math:`m/(s^2*sqrt(hz))`")
      .def_rw("gyroscope_random_walk", &ImuCalibration::gyroscope_random_walk,
              "Gyroscope random walk in :math:`rad/(s^2*sqrt(hz))`")
      .def_rw("accelerometer_random_walk", &ImuCalibration::accelerometer_random_walk,
              "Accelerometer random walk in :math:`m/(s^3*sqrt(hz))`")
      .def_rw("frequency", &ImuCalibration::frequency, "IMU frequency in :math:`hz`")
      .def("__repr__", [](const ImuCalibration& imu) {
        return nb::str("cuvslam.ImuCalibration(rig_from_imu={})").format(imu.rig_from_imu);
      });

  nb::class_<Rig>(m, "Rig", "Rig consisting of cameras and 0 or 1 IMU sensors")
      .def(nb::init<const std::vector<Camera>&, const std::vector<ImuCalibration>&>(),
           nb::arg("cameras") = std::vector<Camera>{}, nb::arg("imus") = std::vector<ImuCalibration>{})
      .def_rw("cameras", &Rig::cameras, "List of cameras in the rig, see :class:`Camera`")
      .def_rw("imus", &Rig::imus, "List of IMU sensors in the rig (0 or 1 only), see :class:`ImuCalibration`")
      .def("__repr__",
           [](const Rig& rig) { return nb::str("cuvslam.Rig(cameras={}, imus={})").format(rig.cameras, rig.imus); });

  nb::class_<PoseStamped>(m, "PoseStamped", "Pose with timestamp")
      .def(nb::init<>())
      .def_rw("timestamp_ns", &PoseStamped::timestamp_ns, "Pose timestamp in nanoseconds")
      .def_rw("pose", &PoseStamped::pose, "Pose (transformation between two coordinate frames)")
      .def("__repr__", [](const PoseStamped& ps) {
        return nb::str("cuvslam.PoseStamped(timestamp={}, pose={})").format(ps.timestamp_ns, ps.pose);
      });

  nb::class_<PoseWithCovariance>(m, "PoseWithCovariance", "Pose with covariance matrix")
      .def(nb::init<>())
      .def_ro("pose", &PoseWithCovariance::pose, "Pose (transformation between two coordinate frames)")
      .def_ro("covariance_xyz_rpy", &PoseWithCovariance::covariance_xyz_rpy,
              "6x6 covariance matrix for the pose (row-major)\n"
              "The orientation parameters use a fixed-axis representation.\n"
              "The parameters: (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)")
      .def("__repr__", [](const PoseWithCovariance& pwc) {
        return nb::str("cuvslam.PoseWithCovariance(pose={}, covariance_xyz_rpy={})")
            .format(pwc.pose, pwc.covariance_xyz_rpy);
      });

  nb::class_<PoseEstimate>(
      m, "PoseEstimate",
      "Rig pose estimate from the tracker. The pose is world_from_rig where:\n\n"
      "The rig coordinate frame is user-defined and depends on the extrinsic parameters of the cameras.\n"
      "The world coordinate frame is an arbitrary 3D coordinate frame, corresponding to the rig frame at the start of "
      "tracking.")
      .def(nb::init<>())
      .def_ro("timestamp_ns", &PoseEstimate::timestamp_ns, "Timestamp of the pose estimate in nanoseconds")
      .def_ro("world_from_rig", &PoseEstimate::world_from_rig, "Rig pose in the world coordinate frame")
      .def("__repr__", [](const PoseEstimate& pose) {
        return nb::str("cuvslam.PoseEstimate(timestamp={}, world_from_rig={})")
            .format(pose.timestamp_ns, pose.world_from_rig);
      });

  nb::class_<Observation>(m, "Observation", "2D observation of a landmark in an image")
      .def(nb::init<>())
      .def_rw("id", &Observation::id, "Unique ID of the observed landmark")
      .def_rw("u", &Observation::u, "Horizontal pixel coordinate of the observation")
      .def_rw("v", &Observation::v, "Vertical pixel coordinate of the observation")
      .def_rw("camera_index", &Observation::camera_index, "Index of the camera that made this observation")
      .def("__repr__", [](const Observation& obs) {
        return nb::str("cuvslam.Observation(id={}, u={}, v={}, camera_index={})")
            .format(obs.id, obs.u, obs.v, obs.camera_index);
      });

  nb::class_<Landmark>(m, "Landmark", "3D landmark point")
      .def(nb::init<>())
      .def_rw("id", &Landmark::id, "Unique ID of the landmark")
      .def_rw("coords", &Landmark::coords, "3D coordinates of the landmark in the world coordinate frame")
      .def("__repr__",
           [](const Landmark& l) { return nb::str("cuvslam.Landmark(id={}, coords={})").format(l.id, l.coords); });

  nb::class_<ImuMeasurement>(m, "ImuMeasurement")
      .def(nb::init<>())
      .def(nb::init<int64_t, Array<3>, Array<3>>(), nb::kw_only(), nb::arg("timestamp_ns"),
           nb::arg("linear_accelerations"), nb::arg("angular_velocities"))
      .def_rw("timestamp_ns", &ImuMeasurement::timestamp_ns, "Timestamp of the IMU measurement in nanoseconds")
      .def_rw("linear_accelerations", &ImuMeasurement::linear_accelerations, "Linear accelerations in :math:`m/s^2`")
      .def_rw("angular_velocities", &ImuMeasurement::angular_velocities, "Angular velocities in :math:`rad/s`")
      .def("__repr__", [](const ImuMeasurement& imu) {
        return nb::str("cuvslam.ImuMeasurement(timestamp_ns={}, linear_accelerations={}, angular_velocities={})")
            .format(imu.timestamp_ns, imu.linear_accelerations, imu.angular_velocities);
      });

  // Data structures from Odometry class

  // Odometry class must be defined before its nested enums/classes are bound to it.
  auto odom_cls = nb::class_<Odometry>(
      m, "Odometry",
      "Estimates the rig pose from synchronized camera images and optional depth/IMU data.\n\n"
      "Use directly for odometry-only workflows, or access Tracker's instance through its `odometry` property.");

  nb::enum_<Odometry::MulticameraMode>(odom_cls, "MulticameraMode", "Multicamera tracking modes")
      .value("Performance", Odometry::MulticameraMode::Performance, "Optimized for speed")
      .value("Precision", Odometry::MulticameraMode::Precision, "Optimized for accuracy")
      .value("Moderate", Odometry::MulticameraMode::Moderate, "Balance between speed and accuracy");

  nb::enum_<Odometry::OdometryMode>(odom_cls, "OdometryMode", "Sensor configurations supported by odometry")
      .value("Multicamera", Odometry::OdometryMode::Multicamera,
             "Uses multiple synchronized cameras, all cameras need to have frustum overlap with at least one another. "
             "Simplest case: stereo camera pair.")
      .value("Inertial", Odometry::OdometryMode::Inertial,
             "Uses stereo camera and IMU measurements. A single stereo-camera with a single IMU is supported.")
      .value("RGBD", Odometry::OdometryMode::RGBD,
             "Uses RGB-D camera for tracking. A single RGB-D camera is supported. RGB & Depth images must be aligned.")
      .value("Mono", Odometry::OdometryMode::Mono, "Uses a single camera, tracking is accurate up to scale.")
      .value("Multisensor", Odometry::OdometryMode::Multisensor,
             "EXPERIMENTAL unified multi-sensor mode. Tracking may be inaccurate or fail for some sensor "
             "configurations and scenes. Supports any mix of plain RGB cameras, RGB-D cameras (any subset of the rig), "
             "and a single optional IMU. IMU fusion turns on automatically when the rig contains an IMU.\n\n"
             "Requirements (constructor raises ValueError otherwise):\n"
             " - Build must have cuNLS enabled (-DUSE_CUNLS=ON).\n"
             " - The rig must contain at least one RGB-D camera listed in MultisensorSettings.depth_camera_ids, or at "
             "least one camera pair with overlapping frustums. A single RGB-D camera is valid, with or without an "
             "IMU.\n"
             " - The current solver supports pinhole cameras; other camera models are not supported.\n"
             " - Depth images passed to track() must be 2D uint16 ndarrays (cuVSLAM's C++ contract also accepts "
             "FLOAT32, but the current PyCuVSLAM binding only exposes uint16); each depth image's camera_index must "
             "appear in MultisensorSettings.depth_camera_ids.");

  // Multisensor Settings class — describes available sensors at construction time for OdometryMode::Multisensor.
  nb::class_<Odometry::MultisensorSettings>(
      odom_cls, "MultisensorSettings",
      "Settings for the EXPERIMENTAL unified Multisensor odometry mode. Tracking may be inaccurate or fail for some "
      "sensor configurations and scenes.\n\n"
      "Multisensor supports any mix of plain RGB cameras, RGB-D cameras (any subset of the rig), and a single optional "
      "IMU. The fields here describe depth handling; IMU presence is auto-detected from `Rig.imus`.")
      .def(nb::init<std::vector<int32_t>, float, bool>(), nb::kw_only(),
           nb::arg("depth_camera_ids") = Odometry::MultisensorSettings{}.depth_camera_ids,
           nb::arg("depth_scale_factor") = Odometry::MultisensorSettings{}.depth_scale_factor,
           nb::arg("enable_depth_stereo_tracking") = Odometry::MultisensorSettings{}.enable_depth_stereo_tracking)
      .def_rw("depth_camera_ids", &Odometry::MultisensorSettings::depth_camera_ids,
              "Camera indices that supply depth images. Empty list = no depth (multi-RGB only).")
      .def_rw("depth_scale_factor", &Odometry::MultisensorSettings::depth_scale_factor,
              "Scale factor for depth measurements (applied uniformly to every depth camera).")
      .def_rw("enable_depth_stereo_tracking", &Odometry::MultisensorSettings::enable_depth_stereo_tracking,
              "Allow stereo 2D tracking between depth-aligned cameras and other cameras. "
              "Default: True (multisensor mode benefits from cross-camera 2D tracks).")
      .def("__repr__", [](const Odometry::MultisensorSettings& s) {
        return nb::str(
                   "cuvslam.Odometry.MultisensorSettings(depth_camera_ids={}, depth_scale_factor={}, "
                   "enable_depth_stereo_tracking={})")
            .format(s.depth_camera_ids, s.depth_scale_factor, s.enable_depth_stereo_tracking);
      });

  // RGBD Settings class
  nb::class_<Odometry::RGBDSettings>(odom_cls, "RGBDSettings", "Settings for RGB-D odometry mode")
      .def(nb::init<float, int, bool>(), nb::kw_only(),
           nb::arg("depth_scale_factor") = Odometry::RGBDSettings{}.depth_scale_factor,
           nb::arg("depth_camera_id") = Odometry::RGBDSettings{}.depth_camera_id,
           nb::arg("enable_depth_stereo_tracking") = Odometry::RGBDSettings{}.enable_depth_stereo_tracking)
      .def_rw("depth_scale_factor", &Odometry::RGBDSettings::depth_scale_factor, "Scale factor for depth measurements")
      .def_rw("depth_camera_id", &Odometry::RGBDSettings::depth_camera_id,
              "ID of the camera that the depth image is aligned with")
      .def_rw("enable_depth_stereo_tracking", &Odometry::RGBDSettings::enable_depth_stereo_tracking,
              "Whether to enable stereo tracking between depth-aligned camera and other cameras")
      .def("__repr__", [](const Odometry::RGBDSettings& settings) {
        return nb::str(
                   "cuvslam.Odometry.RGBDSettings(depth_scale_factor={}, depth_camera_id={}, "
                   "enable_depth_stereo_tracking={})")
            .format(settings.depth_scale_factor, settings.depth_camera_id, settings.enable_depth_stereo_tracking);
      });

  nb::class_<Odometry::Config>(odom_cls, "Config", "Odometry configuration parameters")
      // WARNING: the order of init arguments in this definition must coincide with the order in the structure
      .def(nb::init<Odometry::MulticameraMode, Odometry::OdometryMode, bool, bool, bool, bool, bool, bool, bool, bool,
                    float, std::string_view, bool, const Odometry::RGBDSettings&, const Odometry::MultisensorSettings&,
                    float, float>(),
           nb::kw_only(), nb::arg("multicam_mode") = Odometry::Config{}.multicam_mode,
           nb::arg("odometry_mode") = Odometry::Config{}.odometry_mode, nb::arg("use_gpu") = Odometry::Config{}.use_gpu,
           nb::arg("async_sba") = Odometry::Config{}.async_sba,
           nb::arg("use_motion_model") = Odometry::Config{}.use_motion_model,
           nb::arg("use_denoising") = Odometry::Config{}.use_denoising,
           nb::arg("rectified_stereo_camera") = Odometry::Config{}.rectified_stereo_camera,
           nb::arg("enable_observations_export") = kPyDefaultEnableObservationsExport,
           nb::arg("enable_landmarks_export") = kPyDefaultEnableLandmarksExport,
           nb::arg("enable_final_landmarks_export") = Odometry::Config{}.enable_final_landmarks_export,
           nb::arg("max_frame_delta_s") = Odometry::Config{}.max_frame_delta_s,
           nb::arg("debug_dump_directory") = Odometry::Config{}.debug_dump_directory,
           nb::arg("debug_imu_mode") = Odometry::Config{}.debug_imu_mode,
           nb::arg("rgbd_settings") = Odometry::Config{}.rgbd_settings,
           nb::arg("multisensor_settings") = Odometry::Config{}.multisensor_settings,
           nb::arg("min_depth") = Odometry::Config{}.min_depth, nb::arg("max_depth") = Odometry::Config{}.max_depth)
      .def_rw("multicam_mode", &Odometry::Config::multicam_mode, "See :class:`Odometry.MulticameraMode`")
      .def_rw("odometry_mode", &Odometry::Config::odometry_mode, "See :class:`Odometry.OdometryMode`")
      .def_rw("use_gpu", &Odometry::Config::use_gpu, "Enable to use GPU acceleration")
      .def_rw("async_sba", &Odometry::Config::async_sba, "Enable to run bundle adjustment asynchronously")
      .def_rw("use_motion_model", &Odometry::Config::use_motion_model, "Enable to use motion model for pose prediction")
      .def_rw("use_denoising", &Odometry::Config::use_denoising, "Enable to apply denoising to input images")
      .def_rw("rectified_stereo_camera", &Odometry::Config::rectified_stereo_camera,
              "Enable if stereo cameras are rectified and horizontally aligned")
      .def_rw("enable_observations_export", &Odometry::Config::enable_observations_export,
              "Enable to export landmark observations in images during tracking")
      .def_rw("enable_landmarks_export", &Odometry::Config::enable_landmarks_export,
              "Enable to export landmarks during tracking")
      .def_rw("enable_final_landmarks_export", &Odometry::Config::enable_final_landmarks_export,
              "Enable to export final landmarks. Also sets enable_landmarks_export and enable_observations_export.")
      .def_rw("max_frame_delta_s", &Odometry::Config::max_frame_delta_s,
              "Maximum time difference between frames in seconds")
      .def_rw("debug_dump_directory", &Odometry::Config::debug_dump_directory,
              "Directory for debug data dumps. If empty, no debug data will be dumped")
      .def_rw("debug_imu_mode", &Odometry::Config::debug_imu_mode, "Enable IMU debug mode")
      .def_rw("rgbd_settings", &Odometry::Config::rgbd_settings,
              "Settings for RGB-D odometry mode. See :class:`Odometry.RGBDSettings`")
      .def_rw("multisensor_settings", &Odometry::Config::multisensor_settings,
              "Settings for Multisensor odometry mode. See :class:`Odometry.MultisensorSettings`")
      .def_rw("min_depth", &Odometry::Config::min_depth,
              "Minimum scene depth (meters) sampled along the epipolar curve for L2R tracking. "
              "Used in every odometry mode that tracks between overlapping camera pairs (Multicamera, Inertial, "
              "RGBD, Multisensor); ignored in Mono. Any negative value (e.g. -1) auto-detects from baseline.")
      .def_rw("max_depth", &Odometry::Config::max_depth,
              "Maximum scene depth (meters) sampled along the epipolar curve for L2R tracking. "
              "Used in the same modes as min_depth. Any negative value (e.g. -1) auto-detects from baseline.");

  // Odometry::State binding
  nb::class_<Odometry::State>(odom_cls, "State",
                              "Odometry state snapshot. Contains pose, observations, landmarks, etc.\n"
                              "Consumed by :meth:`Slam.track`.")
      .def(nb::init<>())
      .def_rw("frame_id", &Odometry::State::frame_id, "Frame id of the state")
      .def_rw("timestamp_ns", &Odometry::State::timestamp_ns, "Timestamp in nanoseconds")
      .def_rw("delta", &Odometry::State::delta, "Delta pose (Pose)")
      .def_rw("keyframe", &Odometry::State::keyframe, "Whether this frame is a keyframe")
      .def_rw("warming_up", &Odometry::State::warming_up, "Whether tracker is in warming up phase")
      .def_rw("gravity", &Odometry::State::gravity, "Optional gravity vector (if available)")
      .def_rw("observations", &Odometry::State::observations, "List of 2D landmark observations (Observation)")
      .def_rw("landmarks", &Odometry::State::landmarks, "List of 3D landmarks (Landmark)")
      .def_rw("context", &Odometry::State::context, "Context of the state")
      .def("__repr__", [](const Odometry::State& s) {
        return nb::str(
                   "Odometry.State(frame_id={}, timestamp_ns={}, delta={}, keyframe={}, warming_up={}, gravity={}, "
                   "observations=[...], landmarks=[...])")
            .format(s.frame_id, s.timestamp_ns, s.delta, s.keyframe ? "True" : "False", s.warming_up ? "True" : "False",
                    s.gravity.has_value() ? s.gravity.value() : Odometry::Gravity{});
      });

  // Internals binding - per-frame parameter overrides (stateless)
  nb::class_<cuvslam::internal::Internals>(
      odom_cls, "Internals",
      "Per-frame tracking options (stateless).\n\n"
      "These options apply only to the current track() call and do not affect subsequent frames.\n"
      "Instantiate with Internals() and override only the fields you need.")
      .def(nb::init<>())
      // Feature Selection Settings
      .def_rw("num_desired_tracks", &cuvslam::internal::Internals::num_desired_tracks,
              "Number of desired feature tracks for this frame")
      .def_rw("border_top", &cuvslam::internal::Internals::border_top, "Top border to ignore in pixels")
      .def_rw("border_bottom", &cuvslam::internal::Internals::border_bottom, "Bottom border to ignore in pixels")
      .def_rw("border_left", &cuvslam::internal::Internals::border_left, "Left border to ignore in pixels")
      .def_rw("border_right", &cuvslam::internal::Internals::border_right, "Right border to ignore in pixels")
      .def_rw("box3_prefilter", &cuvslam::internal::Internals::box3_prefilter, "Enable box filter preprocessing")
      .def_rw("ransac_filter", &cuvslam::internal::Internals::ransac_filter, "Enable RANSAC filtering")
      // Keyframe Settings
      .def_rw("kf_survivor_from_last", &cuvslam::internal::Internals::kf_survivor_from_last,
              "Keyframe selection threshold: survivor tracks percentage (0-100)")
      .def_rw("kf_max_timedelta_between_kfs_s", &cuvslam::internal::Internals::kf_max_timedelta_between_kfs_s,
              "Maximum time delta between consecutive keyframes (seconds)")
      .def_rw("kf_override_frame_selection", &cuvslam::internal::Internals::kf_override_frame_selection,
              "Optional per-frame keyframe decision. None uses automatic keyframe selection, True forces keyframe, "
              "False forces non-keyframe.")
      // PNP Settings
      .def_rw("vo_pnp_lambda", &cuvslam::internal::Internals::vo_pnp_lambda,
              "Visual PNP Levenberg-Marquardt damping factor")
      .def_rw("vo_pnp_huber", &cuvslam::internal::Internals::vo_pnp_huber, "Visual PNP Huber robustifier scale")
      .def_rw("vo_pnp_max_iteration", &cuvslam::internal::Internals::vo_pnp_max_iteration,
              "Maximum visual PNP solver iterations")
      .def_rw("vo_pnp_recalculate_cov", &cuvslam::internal::Internals::vo_pnp_recalculate_cov,
              "Whether visual PNP recomputes covariance after a successful solve")
      .def_rw("vo_pnp_filter_new_observations", &cuvslam::internal::Internals::vo_pnp_filter_new_observations,
              "Whether visual PNP filters observations per camera")
      .def_rw("vo_pnp_max_obs_per_camera", &cuvslam::internal::Internals::vo_pnp_max_obs_per_camera,
              "Maximum visual PNP observations per camera")
      .def_rw("vo_pnp_point_z_thresh", &cuvslam::internal::Internals::vo_pnp_point_z_thresh,
              "Minimum visual PNP landmark z-depth")
      .def_rw("vo_pnp_min_observations", &cuvslam::internal::Internals::vo_pnp_min_observations,
              "Minimum observations required for visual PNP")
      .def_rw("vo_pnp_cost_thresh", &cuvslam::internal::Internals::vo_pnp_cost_thresh,
              "Visual PNP convergence threshold")
      .def_rw("inertial_stereo_pnp_lambda", &cuvslam::internal::Internals::inertial_stereo_pnp_lambda,
              "Inertial-mode stereo fallback PNP Levenberg-Marquardt damping factor")
      .def_rw("inertial_stereo_pnp_huber", &cuvslam::internal::Internals::inertial_stereo_pnp_huber,
              "Inertial-mode stereo fallback PNP Huber robustifier scale")
      .def_rw("inertial_stereo_pnp_max_iteration", &cuvslam::internal::Internals::inertial_stereo_pnp_max_iteration,
              "Maximum inertial-mode stereo fallback PNP solver iterations")
      .def_rw("inertial_stereo_pnp_recalculate_cov", &cuvslam::internal::Internals::inertial_stereo_pnp_recalculate_cov,
              "Whether inertial-mode stereo fallback PNP recomputes covariance after a successful solve")
      .def_rw("inertial_stereo_pnp_filter_new_observations",
              &cuvslam::internal::Internals::inertial_stereo_pnp_filter_new_observations,
              "Whether inertial-mode stereo fallback PNP filters observations per camera")
      .def_rw("inertial_stereo_pnp_max_obs_per_camera",
              &cuvslam::internal::Internals::inertial_stereo_pnp_max_obs_per_camera,
              "Maximum inertial-mode stereo fallback PNP observations per camera")
      .def_rw("inertial_stereo_pnp_point_z_thresh", &cuvslam::internal::Internals::inertial_stereo_pnp_point_z_thresh,
              "Minimum inertial-mode stereo fallback PNP landmark z-depth")
      .def_rw("inertial_stereo_pnp_min_observations",
              &cuvslam::internal::Internals::inertial_stereo_pnp_min_observations,
              "Minimum observations required for inertial-mode stereo fallback PNP")
      .def_rw("inertial_stereo_pnp_cost_thresh", &cuvslam::internal::Internals::inertial_stereo_pnp_cost_thresh,
              "Inertial-mode stereo fallback PNP convergence threshold")
      // Inertial PNP Settings
      .def_rw("imu_pnp_robustifier_scale", &cuvslam::internal::Internals::imu_pnp_robustifier_scale,
              "Inertial PNP robustifier scale")
      .def_rw("imu_pnp_max_iteration", &cuvslam::internal::Internals::imu_pnp_max_iteration,
              "Maximum inertial PNP solver iterations")
      .def_rw("imu_pnp_min_observations", &cuvslam::internal::Internals::imu_pnp_min_observations,
              "Minimum observations required for inertial PNP")
      // ICP Settings
      .def_rw("icp_lambda", &cuvslam::internal::Internals::icp_lambda, "ICP Levenberg-Marquardt damping factor")
      .def_rw("icp_huber_vis", &cuvslam::internal::Internals::icp_huber_vis,
              "ICP visual reprojection Huber robustifier scale")
      .def_rw("icp_huber_depth", &cuvslam::internal::Internals::icp_huber_depth, "ICP depth Huber robustifier scale")
      .def_rw("icp_max_iteration", &cuvslam::internal::Internals::icp_max_iteration,
              "Maximum ICP solver iterations without depth pyramid")
      .def_rw("icp_cost_thresh", &cuvslam::internal::Internals::icp_cost_thresh, "ICP convergence threshold")
      .def_rw("icp_min_scale_level", &cuvslam::internal::Internals::icp_min_scale_level,
              "Finest ICP pyramid level to process")
      .def_rw("icp_max_scale_level", &cuvslam::internal::Internals::icp_max_scale_level,
              "Coarsest ICP pyramid level to start from")
      .def_rw("icp_num_iters_per_scale", &cuvslam::internal::Internals::icp_num_iters_per_scale,
              "ICP iterations per pyramid level")
      .def_rw("icp_blending_alpha", &cuvslam::internal::Internals::icp_blending_alpha,
              "Blending weight between visual and depth ICP residuals")
      .def("__repr__", [](const cuvslam::internal::Internals& o) {
        return nb::str("Internals(num_desired_tracks={}, kf_survivor_from_last={}, ...)")
            .format(o.num_desired_tracks, o.kf_survivor_from_last);
      });

  // Odometry class methods
  odom_cls.def(nb::init<const Rig&, const Odometry::Config&>(), nb::arg("rig"), nb::arg("cfg") = Odometry::Config{})
      .def(
          "track",
          [](Odometry& self, int64_t timestamp, const std::vector<nb::ndarray<nb::ro>>& images,
             const std::optional<std::vector<nb::ndarray<nb::ro>>>& masks = std::nullopt,
             const std::optional<std::vector<nb::ndarray<nb::ro>>>& depths = std::nullopt,
             const std::optional<cuvslam::internal::Internals>& internals = std::nullopt) -> PoseEstimate {
            if (masks.has_value()) {
              THROW_INVALID_ARG_IF(masks->size() != images.size() && !masks->empty(),
                                   "If the masks vector is not empty, its size must match the images vector size. "
                                   "Got masks size: " +
                                       std::to_string(masks->size()) +
                                       ", images size: " + std::to_string(images.size()));
            }

            auto image_set = ImageSetFromNDArrays(images, timestamp, ArrayType::Image);
            auto mask_set = masks.has_value() ? ImageSetFromNDArrays(masks.value(), timestamp, ArrayType::Mask)
                                              : Odometry::ImageSet();
            auto depth_set = depths.has_value() ? ImageSetFromNDArrays(depths.value(), timestamp, ArrayType::Depth)
                                                : Odometry::ImageSet();
            return self.Track(image_set, mask_set, depth_set, internals.has_value() ? &*internals : nullptr);
          },
          nb::arg("timestamp"), nb::arg("images"), nb::arg("masks") = nb::none(), nb::arg("depths") = nb::none(),
          nb::arg("internals") = nb::none(),
          "Track a rig pose using current image frame.\n\n"
          "Synchronously tracks current image frame and returns a PoseEstimate.\n\n"
          "By default, this function uses visual odometry to compute a pose.\n"
          "In Inertial mode (or Multisensor mode with an IMU configured in the rig), if visual odometry "
          "tracker fails to compute a pose, the function returns the position calculated from a user-provided "
          "IMU data.\n"
          "If after several calls of :meth:`track` visual odometry is not able to recover, "
          "then invalid pose will be returned.\n"
          "The track will output poses in the same coordinate system until a loss of tracking.\n\n"
          "All cameras must be synchronized. If a camera rig provides 'almost synchronized' frames, the timestamps "
          "should be within 1 millisecond.\n\n"
          "Images (masks, depth images, etc.) can be numpy arrays or tensors, both GPU (CUDA) and CPU.\n"
          "All data must be of the same type (either GPU or CPU).\n"
          "This is not the same as `odom_config.use_gpu` - if odometry uses GPU for computations,\n"
          "images etc. can still be either CPU or GPU arrays/tensors.\n\n"
          "The images etc. must be in the same order as cameras in the rig.\n"
          "If data for a camera is not available, pass empty array or tensor for that camera image.\n\n"
          "Parameters:\n"
          "    timestamp: Image timestamp in nanoseconds\n"
          "    images: List of numpy arrays containing the camera images\n"
          "    masks: Optional list of numpy arrays containing masks for the images\n"
          "    depths: Optional list of numpy arrays containing depth images\n"
          "    options: Optional per-frame options that override default settings for this call only\n\n"
          "Returns:\n"
          "    :class:`PoseEstimate` object with the computed pose. If tracking fails, `is_valid` will be False.")
      .def(
          "register_imu_measurement",
          [](Odometry& self, uint32_t sensor_index, const ImuMeasurement& imu_measurement) {
            self.RegisterImuMeasurement(sensor_index, imu_measurement);
          },
          nb::arg("sensor_index"), nb::arg("imu_measurement"),
          "Register an IMU measurement.\n\n"
          "Requires Inertial mode, or Multisensor mode with an IMU configured in the rig.\n"
          "If visual odometry loses camera position, it briefly continues execution using user-provided\n"
          "IMU measurements while trying to recover the position.\n"
          "IMU sensors and cameras clocks must be synchronized, :meth:`track` and "
          ":meth:`register_imu_measurement` must be called in non-decreasing timestamp order and externally "
          "serialized on the same tracker. Do not call them concurrently from different threads. If IMU samples "
          "are captured on a separate thread, buffer them and submit them in timestamp order from the same "
          "sequence that calls :meth:`track`, or protect all calls with caller-owned synchronization.")
      .def(
          "get_last_observations",
          [](const Odometry& self, uint32_t camera_index) -> std::vector<Observation> {
            return self.GetLastObservations(camera_index);
          },
          nb::arg("camera_index"),
          "Get an array of observations from the last VO frame.\n\n"
          "Requires `enable_observations_export=True` in :class:`Odometry.Config`.")
      .def(
          "get_last_landmarks", [](Odometry& self) -> std::vector<Landmark> { return self.GetLastLandmarks(); },
          "Get an array of landmarks from the last VO frame.\n\n"
          "Landmarks are 3D points in the last camera frame.\n"
          "Requires `enable_landmarks_export=True` in :class:`Odometry.Config`.")
      .def(
          "get_last_gravity",
          [](const Odometry& self) -> nb::object {
            auto gravity = self.GetLastGravity();
            if (gravity.has_value()) {
              return nb::cast(gravity.value());
            } else {
              return nb::none();
            }
          },
          "Get gravity vector in the last VO frame.\n\n"
          "Returns `None` if gravity is not yet available.\n"
          "Requires Inertial mode, or Multisensor mode with an IMU configured in the rig.")
      .def(
          "get_final_landmarks",
          [](const Odometry& self) -> std::unordered_map<uint64_t, Vector3f> { return self.GetFinalLandmarks(); },
          "Get all final landmarks from all frames.\n\n"
          "Landmarks are 3D points in the odometry start frame.\n"
          "Requires `enable_final_landmarks_export=True` in :class:`Odometry.Config`.")
      .def(
          "get_state",
          [](const Odometry& self) -> Odometry::State {
            Odometry::State state;
            self.GetState(state);
            return state;
          },
          "Get the current tracker state (pose, observations, landmarks, etc) as a State object.\n\n"
          "Returns:\n    Odometry.State: The current tracker state snapshot.")
      .def(
          "get_primary_cameras", [](const Odometry& self) -> std::vector<uint8_t> { return self.GetPrimaryCameras(); },
          "Returns a list of primary cameras.\n\n"
          "Primary cameras are the ones where observations are always present.\n"
          "The list is required to initialize Slam.")
      .def(
          "apply_expert_parameters",
          [](Odometry& self, const nb::dict& parameters) {
            std::vector<cuvslam::internal::InternalParameter> params;
            params.reserve(parameters.size());
            // Materialise strings so string_views remain valid for the duration of the call.
            std::vector<std::string> keys, values;
            keys.reserve(parameters.size());
            values.reserve(parameters.size());
            for (auto [k, v] : parameters) {
              keys.push_back(nb::cast<std::string>(k));
              values.push_back(nb::cast<std::string>(v));
              params.push_back({keys.back(), values.back()});
            }
            self.ApplyPersistentInternalParameters(params);
          },
          nb::arg("parameters"),
          "Apply expert parameters by string key/value pairs.\n\n"
          "Allows tuning internal runtime settings after tracker creation. Unknown keys log a\n"
          "warning and are ignored. Invalid values raise RuntimeError.\n\n"
          "Supported keys:\n\n"
          "  SBA (all modes):\n"
          "    ``sba.num_sba_frames``, ``sba.num_inertial_sba_frames``, ``sba.num_fixed_sba_frames``,\n"
          "    ``sba.num_sba_iterations``, ``sba.robustifier_scale``, ``sba.use_sba_winsorizer``\n\n"
          "  StateMachine / IMU gravity (Inertial mode only):\n"
          "    ``sm.gravity_update_period_ns``, ``sm.max_integration_time_ns``,\n"
          "    ``sm.min_num_kf_for_gravity``, ``sm.min_time_period_ns``, ``sm.max_time_period_ns``\n\n"
          "Note: ``sba.async`` and ``sba.mode`` are construction-time only; set them in\n"
          ":class:`Odometry.Config` before creating the tracker.\n\n"
          "Example::\n\n"
          "    odometry.apply_expert_parameters({\n"
          "        'sba.num_sba_frames': '5',\n"
          "        'sba.robustifier_scale': '0.3',\n"
          "    })\n\n"
          "Args:\n"
          "    parameters (dict[str, str]): Map of key/value string pairs to apply.");

  // Slam class must be defined before its nested enums/classes are bound to it.
  auto slam_cls = nb::class_<Slam>(
      m, "Slam",
      "Builds and optimizes a reusable map from odometry results, including loop closure, map persistence, and "
      "relocalization.\n\n"
      "Use directly for manual orchestration, or access Tracker's instance through its `slam` property.");

  nb::enum_<Slam::DataLayer>(slam_cls, "DataLayer", "Data layer for SLAM")
      .value("Landmarks", Slam::DataLayer::Landmarks, "Landmarks that are visible in the current frame")
      .value("Map", Slam::DataLayer::Map, "Landmarks of the map")
      .value("LoopClosure", Slam::DataLayer::LoopClosure,
             "Map's landmarks that are visible in the last loop closure event");
  // currently we don't expose EnableReadingData()/DisableReadingData() to python,
  // so we only need layer names for ReadLandmarks() binding
  // .value("PoseGraph", Slam::DataLayer::PoseGraph, "Pose Graph");

  nb::class_<Slam::Config>(slam_cls, "Config", "SLAM configuration parameters")
      .def(nb::init<std::string_view, bool, bool, bool, bool, bool, float, float, uint32_t, uint32_t, uint32_t,
                    uint32_t>(),
           nb::kw_only(), nb::arg("map_cache_path") = Slam::Config{}.map_cache_path,
           nb::arg("use_gpu") = Slam::Config{}.use_gpu, nb::arg("sync_mode") = Slam::Config{}.sync_mode,
           nb::arg("enable_reading_internals") = true,  // enable by default; in Python convenience is a priority
           nb::arg("planar_constraints") = Slam::Config{}.planar_constraints,
           nb::arg("gt_align_mode") = Slam::Config{}.gt_align_mode,
           nb::arg("map_cell_size") = Slam::Config{}.map_cell_size,
           nb::arg("max_landmarks_distance") = Slam::Config{}.max_landmarks_distance,
           nb::arg("max_map_size") = Slam::Config{}.max_map_size,
           nb::arg("throttling_time_ms") = Slam::Config{}.throttling_time_ms,
           nb::arg("retention_time_ms") = Slam::Config{}.retention_time_ms,
           nb::arg("delay_warning_queue_size") = Slam::Config{}.delay_warning_queue_size)
      .def_rw("map_cache_path", &Slam::Config::map_cache_path,
              "If empty, map is kept in memory only. Else, map is synced to disk (LMDB) at this path, allowing "
              "large-scale maps; if the path already exists it will be overwritten. To load an existing map, use "
              "LocalizeInMap(). To save map, use SaveMap().")
      .def_rw("use_gpu", &Slam::Config::use_gpu, "Whether to use GPU acceleration")
      .def_rw("sync_mode", &Slam::Config::sync_mode,
              "If true, localization and mapping run in the same thread as visual odometry")
      .def_rw("enable_reading_internals", &Slam::Config::enable_reading_internals,
              "Enable reading internal data from SLAM")
      .def_rw("planar_constraints", &Slam::Config::planar_constraints,
              "Modify poses so camera moves on a horizontal plane")
      .def_rw("gt_align_mode", &Slam::Config::gt_align_mode, "Special mode for visual map building with ground truth")
      .def_rw("map_cell_size", &Slam::Config::map_cell_size,
              "Size of map cell (0 to auto-calculate from camera baseline)")
      .def_rw("max_landmarks_distance", &Slam::Config::max_landmarks_distance,
              "Maximum distance from camera to landmark for inclusion in map")
      .def_rw("max_map_size", &Slam::Config::max_map_size,
              "Maximum number of poses in SLAM pose graph (0 for unlimited)")
      .def_rw("throttling_time_ms", &Slam::Config::throttling_time_ms,
              "Minimum time between loop closure events in milliseconds")
      .def_rw("retention_time_ms", &Slam::Config::retention_time_ms,
              "How long the past is preserved. Maximum time to keep odometries delta history to be able "
              "to process LocalizeInMap within timestamps from past.")
      .def_rw("delay_warning_queue_size", &Slam::Config::delay_warning_queue_size,
              "Length of the SLAM input queue at which cuVSLAM warns that SLAM is falling behind odometry. "
              "Diagnostic only: exceeding it does not change tracking behavior, it only prints a warning (requires "
              "verbosity Warning or higher, see set_verbosity). Default: 10 queued commands.")
      .def("__repr__", [](const Slam::Config& cfg) {
        return nb::str(
                   "cuvslam.Slam.Config(map_cache_path={}, use_gpu={}, sync_mode={}, enable_reading_internals={}, "
                   "planar_constraints={}, gt_align_mode={}, map_cell_size={}, max_landmarks_distance={}, "
                   "max_map_size={}, throttling_time_ms={}, retention_time_ms={}, delay_warning_queue_size={})")
            .format(cfg.map_cache_path, cfg.use_gpu, cfg.sync_mode, cfg.enable_reading_internals,
                    cfg.planar_constraints, cfg.gt_align_mode, cfg.map_cell_size, cfg.max_landmarks_distance,
                    cfg.max_map_size, cfg.throttling_time_ms, cfg.retention_time_ms, cfg.delay_warning_queue_size);
      });

  nb::class_<Slam::LocalizationSettings>(slam_cls, "LocalizationSettings", "Localization settings")
      // .def(nb::init<>())
      .def(
          "__init__",
          [](Slam::LocalizationSettings* self, float horizontal_search_radius, float vertical_search_radius,
             float horizontal_step, float vertical_step, float angular_step_rads) {
            new (self) Slam::LocalizationSettings{};
            self->horizontal_search_radius = horizontal_search_radius;
            self->vertical_search_radius = vertical_search_radius;
            self->horizontal_step = horizontal_step;
            self->vertical_step = vertical_step;
            self->angular_step_rads = angular_step_rads;
          },
          nb::kw_only(), nb::arg("horizontal_search_radius"), nb::arg("vertical_search_radius"),
          nb::arg("horizontal_step"), nb::arg("vertical_step"), nb::arg("angular_step_rads"))
      .def_rw("horizontal_search_radius", &Slam::LocalizationSettings::horizontal_search_radius,
              "Horizontal search radius in meters")
      .def_rw("vertical_search_radius", &Slam::LocalizationSettings::vertical_search_radius,
              "Vertical search radius in meters")
      .def_rw("horizontal_step", &Slam::LocalizationSettings::horizontal_step, "Horizontal step in meters")
      .def_rw("vertical_step", &Slam::LocalizationSettings::vertical_step, "Vertical step in meters")
      .def_rw("angular_step_rads", &Slam::LocalizationSettings::angular_step_rads,
              "Angular step around vertical axis in radians")
      .def("__repr__", [](const Slam::LocalizationSettings& settings) {
        return nb::str(
                   "cuvslam.Slam.LocalizationSettings(horizontal_search_radius={}, "
                   "vertical_search_radius={}, horizontal_step={}, vertical_step={}, "
                   "angular_step_rads={})")
            .format(settings.horizontal_search_radius, settings.vertical_search_radius, settings.horizontal_step,
                    settings.vertical_step, settings.angular_step_rads);
      });

  nb::class_<Slam::Metrics>(slam_cls, "Metrics", "SLAM metrics")
      .def(nb::init<>())
      .def_rw("timestamp_ns", &Slam::Metrics::timestamp_ns, "Timestamp of measurements in nanoseconds")
      .def_rw("lc_status", &Slam::Metrics::lc_status, "Loop closure status")
      .def_rw("pgo_status", &Slam::Metrics::pgo_status, "Pose graph optimization status")
      .def_rw("lc_selected_landmarks_count", &Slam::Metrics::lc_selected_landmarks_count,
              "Count of landmarks selected for loop closure")
      .def_rw("lc_tracked_landmarks_count", &Slam::Metrics::lc_tracked_landmarks_count,
              "Count of landmarks tracked in loop closure")
      .def_rw("lc_pnp_landmarks_count", &Slam::Metrics::lc_pnp_landmarks_count,
              "Count of landmarks in PnP for loop closure")
      .def_rw("lc_good_landmarks_count", &Slam::Metrics::lc_good_landmarks_count, "Count of landmarks in loop closure")
      .def("__repr__", [](const Slam::Metrics& metrics) {
        return nb::str(
                   "cuvslam.Slam.Metrics(timestamp_ns={}, lc_status={}, pgo_status={}, "
                   "lc_selected_landmarks_count={}, lc_tracked_landmarks_count={}, "
                   "lc_pnp_landmarks_count={}, lc_good_landmarks_count={})")
            .format(metrics.timestamp_ns, metrics.lc_status, metrics.pgo_status, metrics.lc_selected_landmarks_count,
                    metrics.lc_tracked_landmarks_count, metrics.lc_pnp_landmarks_count,
                    metrics.lc_good_landmarks_count);
      });

  // Slam pose graph data structures
  nb::class_<Slam::PoseGraphNode>(slam_cls, "PoseGraphNode", "Node in a pose graph")
      .def(nb::init<>())
      .def_rw("id", &Slam::PoseGraphNode::id, "Node identifier")
      .def_rw("node_pose", &Slam::PoseGraphNode::node_pose, "Node pose")
      .def("__repr__", [](const Slam::PoseGraphNode& node) {
        return nb::str("cuvslam.PoseGraphNode(id={}, node_pose={})").format(node.id, node.node_pose);
      });

  auto pg_edge_cls =
      nb::class_<Slam::PoseGraphEdge>(slam_cls, "PoseGraphEdge", "Edge in a pose graph")
          .def(nb::init<>())
          .def_rw("node_from", &Slam::PoseGraphEdge::node_from, "Source node ID")
          .def_rw("node_to", &Slam::PoseGraphEdge::node_to, "Target node ID")
          .def_rw("transform", &Slam::PoseGraphEdge::transform, "Transformation from source to target node")
          .def("__repr__", [](const Slam::PoseGraphEdge& edge) {
            return nb::str("cuvslam.PoseGraphEdge(node_from={}, node_to={}, transform={})")
                .format(edge.node_from, edge.node_to, edge.transform);
          });
  bind_array_accessors(pg_edge_cls, "covariance", &Slam::PoseGraphEdge::covariance,
                       "Covariance matrix of the transformation");

  nb::class_<Slam::PoseGraph>(slam_cls, "PoseGraph", "Pose graph structure")
      .def(nb::init<>())
      .def_rw("nodes", &Slam::PoseGraph::nodes, "List of nodes in the graph")
      .def_rw("edges", &Slam::PoseGraph::edges, "List of edges in the graph")
      .def("__repr__", [](const Slam::PoseGraph& pg) {
        return nb::str("cuvslam.PoseGraph(nodes={}, edges={})").format(pg.nodes, pg.edges);
      });

  // Slam landmarks data structures
  nb::class_<Slam::Landmark>(slam_cls, "Landmark",
                             "Landmark with additional information.\n\n"
                             "Contains unique identifier, weight, and 3D coordinates in the world frame.")
      .def(nb::init<>())
      .def_rw("id", &Slam::Landmark::id, "Unique ID of the landmark")
      .def_rw("weight", &Slam::Landmark::weight, "Landmark weight (selection/reliability metric)")
      .def_rw("coords", &Slam::Landmark::coords, "3D coordinates of the landmark in the world coordinate frame")
      .def("__repr__", [](const Slam::Landmark& l) {
        return nb::str("cuvslam.Slam.Landmark(id={}, weight={}, coords={})").format(l.id, l.weight, l.coords);
      });

  nb::class_<Slam::Landmarks>(slam_cls, "Landmarks", "Collection of landmarks with a capture timestamp.")
      .def(nb::init<>())
      .def_rw("timestamp_ns", &Slam::Landmarks::timestamp_ns, "Timestamp of landmarks in nanoseconds")
      .def_rw("landmarks", &Slam::Landmarks::landmarks, "List of landmarks, see :class:`Slam.Landmark`")
      .def("__repr__", [](const Slam::Landmarks& ls) {
        return nb::str("cuvslam.Slam.Landmarks(timestamp_ns={}, landmarks=[{} items])")
            .format(ls.timestamp_ns, ls.landmarks.size());
      });

  // Slam class methods
  slam_cls
      .def(nb::init<const Rig&, const std::vector<uint8_t>&, const Slam::Config&>(), nb::arg("rig"),
           nb::arg("primary_cameras"), nb::arg("config") = Slam::Config{})
      .def("track", &Slam::Track, nb::arg("state"), nb::arg("gt_pose") = nullptr,
           "Update SLAM pose based on Odometry state and past SLAM loop closures.\n\n"
           "This method should be called after each successful Odometry.track() call.\n"
           "Parameters:\n"
           "    state: Odometry state containing all tracking data\n"
           "    gt_pose: Optional ground truth pose. Should be provided if `gt_align_mode` is enabled, otherwise "
           "should be None.")
      .def("get_pose", &Slam::GetPose,
           "Get the current SLAM rig pose in the world frame.\n\n"
           "Returns the most recent pose computed by SLAM. Before the first keyframe is processed the pose is the\n"
           "identity transform.\n"
           "Returns:\n"
           "    Current rig pose in world frame")
      .def(
          "get_all_slam_poses",
          [](const Slam& self, uint32_t max_poses_count) -> std::vector<PoseStamped> {
            std::vector<PoseStamped> poses;
            self.GetAllSlamPoses(poses, max_poses_count);
            return poses;
          },
          nb::arg("max_poses_count") = 0,
          "Get all SLAM poses for each frame.\n\n"
          "Parameters:\n"
          "    max_poses_count: Maximum number of poses to return (0 for all)\n"
          "Returns:\n"
          "    List of poses with timestamps\n"
          "This call could be blocked by slam thread.")
      .def(
          "save_map",
          [](Slam& self, const std::string_view& folder_name, nb::callable callback) {
            self.SaveMap(folder_name, [callback](bool success) {
              nb::gil_scoped_acquire gil;
              callback(success);
            });
          },
          nb::arg("folder_name"), nb::arg("callback"),
          "Save SLAM database (map) to a folder.\n\n"
          "This folder will be created if it does not exist.\n"
          "**WARNING**: *Contents of the folder will be overwritten.*\n"
          "This method works asynchronously depending on the `sync_mode` parameter in `Slam.Config`.\n"
          "In both cases, the callback will be called with a flag indicating if the map was saved successfully.\n\n"
          "Parameters:\n"
          "    folder_name: Folder name where SLAM database will be saved\n"
          "    callback: Function to be called when save is complete (takes bool success parameter)")
      .def(
          "localize_in_map",
          [](Slam& self, const std::string_view& folder_name, int64_t timestamp_ns, const Pose& guess_pose,
             const std::vector<nb::ndarray<nb::ro>>& images, const Slam::LocalizationSettings& settings,
             nb::callable start_cb, nb::callable finish_cb) {
            auto image_set = ImageSetFromNDArrays(images, timestamp_ns, ArrayType::Image);
            self.LocalizeInMap(
                folder_name, timestamp_ns, guess_pose, image_set, settings,
                [start_cb] {
                  nb::gil_scoped_acquire gil;
                  start_cb();
                },
                [finish_cb](const Result<Pose>& result) {
                  nb::gil_scoped_acquire gil;
                  finish_cb(result.data, result.error_message);
                });
          },
          nb::arg("folder_name"), nb::arg("timestamp_ns"), nb::arg("guess_pose"), nb::arg("images"),
          nb::arg("settings"), nb::arg("start_cb"), nb::arg("finish_cb"),
          "Localize in the existing database (map).\n\n"
          "If ``Slam.Config.sync_mode`` is false, work is queued for the background slam thread and the callback runs "
          "when localization finishes (possibly on another thread than the caller). If ``Slam.Config.sync_mode`` is "
          "true, localization runs immediately before this call returns.\n"
          "Finds the rig pose in the saved map. If successful, replace current map with saved one.\n"
          "`finish_cb` receives the localization result or an error message.\n"
          "Parameters:\n"
          "    folder_name: Folder containing the saved SLAM map\n"
          "    timestamp_ns: Timestamp for the image frame in nanoseconds\n"
          "    guess_pose: Initial guess for rig pose at the image timestamp\n"
          "    images: List of numpy arrays containing the camera images\n"
          "    settings: Localization settings\n"
          "    start_cb: Called when localization begins (no arguments)\n"
          "    finish_cb: Called when localization completes (receives pose result and error message)")
      .def(
          "get_landmarks",
          [](Slam& self, Slam::DataLayer layer) -> Slam::Landmarks {
            self.EnableReadingData(layer, 100000);
            auto landmarks = self.ReadLandmarks(layer);
            return landmarks ? *landmarks : Slam::Landmarks{};
          },
          nb::arg("layer"),
          "Get landmarks for a given data layer of SLAM.\n\n"
          "Parameters:\n"
          "    layer: Data layer to read (see `Slam.DataLayer`)\n"
          "Returns:\n"
          "    Landmarks for a given data layer")
      .def(
          "get_pose_graph",
          [](Slam& self) -> Slam::PoseGraph {
            self.EnableReadingData(Slam::DataLayer::PoseGraph, 1000);
            auto pose_graph = self.ReadPoseGraph();
            return pose_graph ? *pose_graph : Slam::PoseGraph{};
          },
          "Get pose graph consisting of all keyframes and their connections including loop closures.\n\n"
          "Returns:\n"
          "    Pose graph with nodes and edges")
      .def(
          "get_slam_metrics",
          [](const Slam& self) -> Slam::Metrics {
            Slam::Metrics metrics;
            self.GetSlamMetrics(metrics);
            return metrics;
          },
          "Get SLAM metrics.\n\n"
          "Returns:\n"
          "    SLAM metrics")
      .def(
          "get_loop_closure_poses",
          [](const Slam& self) -> std::vector<PoseStamped> {
            std::vector<PoseStamped> poses;
            self.GetLoopClosurePoses(poses);
            return poses;
          },
          "Get list of last 10 loop closure poses with timestamps.\n\n"
          "Returns:\n"
          "    List of poses with timestamps");

  auto tracker_cls = nb::class_<Tracker>(m, "Tracker",
                                         "Coordinates cuVSLAM Odometry and optional SLAM.\n\n"
                                         "Submit frames and IMU measurements through this class. Use the `odometry` "
                                         "and `slam` properties for module-specific queries and operations.");

  tracker_cls
      .def(
          "__init__",
          [](Tracker* self, const Rig& rig, const std::optional<Odometry::Config>& odom_config,
             const std::optional<Slam::Config>& slam_config) {
            // An omitted config means the Python default, which is not the C++ default.
            const Odometry::Config odometry_config = odom_config.value_or(PyDefaultOdometryConfig());
            new (self) Tracker(rig, odometry_config, slam_config.has_value() ? &*slam_config : nullptr);
          },
          nb::arg("rig"), nb::arg("odom_config") = nb::none(), nb::arg("slam_config") = nb::none(),
          "Initialize the cuVSLAM system. Each `Odometry.OdometryMode` has specific `rig` requirements.\n\n"
          "Parameters:\n"
          "    rig: Camera rig configuration\n"
          "    odom_config: Optional odometry configuration (uses defaults if omitted)\n"
          "    slam_config: Optional SLAM configuration (disables SLAM if None). `gt_align_mode` requires standalone "
          "Odometry and Slam instances.")
      .def(
          "track",
          [](Tracker& self, int64_t timestamp, const std::vector<nb::ndarray<nb::ro>>& images,
             const std::optional<std::vector<nb::ndarray<nb::ro>>>& masks,
             const std::optional<std::vector<nb::ndarray<nb::ro>>>& depths)
              -> std::tuple<PoseEstimate, std::optional<Pose>> {
            if (masks.has_value()) {
              THROW_INVALID_ARG_IF(masks->size() != images.size() && !masks->empty(),
                                   "If the masks vector is not empty, its size must match the images vector size. "
                                   "Got masks size: " +
                                       std::to_string(masks->size()) +
                                       ", images size: " + std::to_string(images.size()));
            }

            const auto image_set = ImageSetFromNDArrays(images, timestamp, ArrayType::Image);
            const auto mask_set =
                masks.has_value() ? ImageSetFromNDArrays(*masks, timestamp, ArrayType::Mask) : Odometry::ImageSet();
            const auto depth_set =
                depths.has_value() ? ImageSetFromNDArrays(*depths, timestamp, ArrayType::Depth) : Odometry::ImageSet();
            auto result = self.Track(image_set, mask_set, depth_set);
            return {result.odometry, result.slam};
          },
          nb::arg("timestamp"), nb::arg("images"), nb::arg("masks") = nb::none(), nb::arg("depths") = nb::none(),
          "Track a rig pose using current image frame.\n\n"
          "This method combines odometry and SLAM processing in a single call.\n\n"
          "In inertial mode, if visual odometry tracker fails to compute a pose, the function returns the position "
          "calculated from a user-provided IMU data.\n"
          "If after several calls of Track() visual odometry is not able to recover, then invalid pose will be "
          "returned.\n"
          "Odometry will output poses in the same coordinate frame until a loss of tracking.\n\n"
          "To get SLAM poses, SLAM must be enabled in the constructor by providing a non-null `slam_config`.\n"
          "SLAM poses may have loop closure (LC) jumps when LC is detected and pose graph is optimized.\n"
          "SLAM poses cannot be adjusted retroactively, so use `slam.get_all_slam_poses()` to get a smooth "
          "trajectory up to the latest frame.\n"
          "Also, in asynchronous mode, LC is done in a separate work thread to keep `track` call fast, so SLAM poses "
          "are not updated immediately.\n\n"
          "All cameras must be synchronized. If a camera rig provides 'almost synchronized' frames, the timestamps "
          "should be within 1 millisecond.\n\n"
          "Images (masks, depth images, etc.) can be numpy arrays or tensors, both GPU (CUDA) and CPU.\n"
          "All data must be of the same type (either GPU or CPU).\n"
          "This is not the same as `odom_config.use_gpu` - if odometry uses GPU for computations,\n"
          "images etc. can still be either CPU or GPU arrays/tensors.\n\n"
          "The images etc. must be in the same order as cameras in the rig.\n"
          "If data for a camera is not available, pass empty array or tensor for that camera image.\n\n"
          "Parameters:\n"
          "    timestamp: Images timestamp in nanoseconds\n"
          "    images: List of numpy arrays or tensors containing the camera images\n"
          "    masks: Optional list of numpy arrays or tensors containing masks for the images\n"
          "    depths: Optional list of numpy arrays or tensors containing depth images\n\n"
          "Returns:\n"
          "    PoseEstimate: The computed pose estimate from Odometry. On failure `world_from_rig` will be `None`.\n"
          "    Pose: If SLAM is enabled, the computed pose estimate from SLAM, otherwise `None`.\n\n"
          "Raises:\n"
          "    ValueError: If data checks fail (e.g. timestamps are out of order, images sizes are inconsistent, "
          "etc.).")
      .def(
          "register_imu_measurement",
          [](Tracker& self, uint32_t sensor_index, const ImuMeasurement& imu_measurement) {
            self.RegisterImuMeasurement(sensor_index, imu_measurement);
          },
          nb::arg("sensor_index"), nb::arg("imu_measurement"),
          "Register an IMU measurement with the tracker.\n\n"
          "Requires Inertial odometry mode, or Multisensor odometry mode with an IMU configured in the rig.\n\n"
          "If visual odometry loses camera position, it briefly continues execution using user-provided IMU "
          "measurements while trying to recover the position. :meth:`register_imu_measurement` should be called in "
          "between image acquisitions however many IMU measurements there are:\n\n"
          "- tracker.track\n"
          "- tracker.register_imu_measurement\n"
          "- ...\n"
          "- tracker.register_imu_measurement\n"
          "- tracker.track\n\n"
          "Higher frequency IMU measurements are recommended.\n\n"
          "IMU sensors and cameras clocks must be synchronized. :meth:`track` and "
          ":meth:`register_imu_measurement` must be called in non-decreasing timestamp order; camera frame "
          "timestamps must be strictly increasing.\n\n"
          "Parameters:\n"
          "    sensor_index: Sensor index; must be 0, as only one sensor is supported now\n"
          "    imu_measurement: IMU measurement to register\n\n"
          "Raises:\n"
          "    ValueError: If IMU fusion is disabled or if called out of the order of timestamps.")
      .def("is_slam_enabled", &Tracker::IsSlamEnabled, "Return whether this tracker owns a SLAM instance.")
      .def_prop_ro(
          "odometry", [](const Tracker& self) -> const Odometry& { return self.GetOdometry(); },
          nb::rv_policy::reference_internal,
          "The underlying odometry instance, see :class:`cuvslam.Odometry`.\n\n"
          "Do not call ``track()`` on an Odometry obtained from Tracker; use ``Tracker.track()`` so SLAM receives "
          "every successful odometry state.")
      .def_prop_ro(
          "slam", [](Tracker& self) -> Slam& { return self.GetSlam(); }, nb::rv_policy::reference_internal,
          "The underlying SLAM instance, see :class:`cuvslam.Slam`.\n\n"
          "Do not call ``track()`` on a Slam obtained from Tracker; use ``Tracker.track()`` so odometry and SLAM "
          "remain synchronized.\n\n"
          "Raises:\n"
          "    RuntimeError: If SLAM is not enabled.");
}

}  // namespace cuvslam

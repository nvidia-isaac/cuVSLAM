
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

#include <cctype>
#include <iostream>

#include "edex/edex_types.h"
#include "edex/file_name_tools.h"

namespace cuvslam::edex::internal {

static const char* EDEX_VERSION_VALUE = "0.9";

// json fields
static const char* EDEX_VERSION = "version";
static const char* EDEX_FRAME_START = "frame_start";
static const char* EDEX_FRAME_END = "frame_end";
static const char* EDEX_CAMERAS = "cameras";
static const char* EDEX_CAMERA_TRANSFORM = "transform";
static const char* EDEX_SEQUENCE = "sequence";
static const char* EDEX_DEPTH_SEQUENCE = "depth_sequence";  // optional
static const char* EDEX_POINTS_2D = "points2d";
static const char* EDEX_POINTS_3D = "points3d";
static const char* EDEX_RIG_POSITIONS = "rig_positions";
static const char* EDEX_CAM_INTRINSICS = "intrinsics";
static const char* EDEX_CAM_SIZE = "size";
static const char* EDEX_CAM_FOCAL = "focal";
static const char* EDEX_CAM_PRINCIPAL = "principal";
static const char* EDEX_CAM_DISTORTION_MODEL = "distortion_model";  // brown5k, fisheye, pinhole, polynomial
static const char* EDEX_CAM_DISTORTION_PARAMS = "distortion_params";
static const char* EDEX_ROTATION = "rotation";
static const char* EDEX_TRANSLATION = "translation";
static const char* EDEX_CAM_DEPTH_ID = "depth_id";
static const char* EDEX_KEY_FRAMES = "key_frames";                                // optional
static const char* EDEX_FAILED_FRAMES = "failed_frames";                          // optional
static const char* EDEX_IMU = "imu";                                              // optional
static const char* EDEX_IMU_LOG = "measurements";                                 // optional
static const char* EDEX_IMU_COORD_SYSTEM = "coordinate_system";                   // optional
static const char* EDEX_IMU_ACCELEROMETER_NOISE_DENSITY = "accel_noise_density";  // optional
static const char* EDEX_IMU_ACCELEROMETER_RANDOM_WALK = "accel_random_walk";      // optional
static const char* EDEX_IMU_GYROSCOPE_NOISE_DENSITY = "gyro_noise_density";       // optional
static const char* EDEX_IMU_GYROSCOPE_RANDOM_WALK = "gyro_random_walk";           // optional
static const char* EDEX_IMU_FREQUENCY = "frequency";                              // optional
static const char* EDEX_FRAME_META = "frame_metadata";                            // optional
static const char* EDEX_FPS = "fps";                                              // optional

template <int _SIZE>
bool isVector(const Json::Value& v) {
  bool result = (v.isArray() && v.size() == _SIZE);

  for (int i = 0; result && i < _SIZE; i++) {
    result = v[i].isDouble();
  }

  return result;
}

template <int _SIZE, class Derived>
void ReadFloatVector(const Json::Value& v, Eigen::MatrixBase<Derived>* result) {
  static_assert(Derived::ColsAtCompileTime == 1, "result must be a vector");
  static_assert(_SIZE <= Derived::RowsAtCompileTime, "result should have at least _SIZE elements");
  assert(isVector<_SIZE>(v));

  for (int i = 0; i < _SIZE; i++) {
    (*result)(i) = v[i].asFloat();
  }

  constexpr auto TAIL = Derived::RowsAtCompileTime - _SIZE;
  (*result).tail(TAIL).setZero();
}

void ReadFloatArray(const Json::Value& v, std::vector<float>* result) {
  assert(v.isArray());
  const int n = v.size();
  result->resize(n);
  for (int i = 0; i < n; ++i) {
    assert(v[i].isDouble());
    (*result)[i] = v[i].asFloat();
  }
}

inline void ReadIntVector2(const Json::Value& v, Vector2T* result) {
  assert(isVector<2>(v));
  (*result)[0] = v[0].asInt();
  (*result)[1] = v[1].asInt();
}

inline void WriteIntVector2(Json::Value& result, const Eigen::Ref<const Vector2T>& v) {
  result.clear();

  result[0] = static_cast<Json::Int>(v.x());
  result[1] = static_cast<Json::Int>(v.y());

  assert(isVector<2>(result));
}

template <int _SIZE, class Derived>
void WriteVector(Json::Value& result, const Eigen::MatrixBase<Derived>& v) {
  static_assert(Derived::ColsAtCompileTime == 1, "");
  static_assert(_SIZE <= Derived::RowsAtCompileTime, "we should have enough data to write.");
  result.clear();

  for (int i = 0; i < _SIZE; i++) {
    result[i] = v(i);
  }

  assert(isVector<_SIZE>(result));
}

void WriteFloatVector(Json::Value& result, const std::vector<float>& v) {
  const int n = static_cast<int>(v.size());
  result.clear();

  for (int i = 0; i < n; i++) {
    result[i] = v[i];
  }
}

inline bool CheckKey(const Json::Value root, const std::vector<std::string>& keys, bool optional = false) {
  for (const auto& k : keys) {
    if (!root.isMember(k)) {
      if (!optional) {
        TraceError("\"%s\" key is expected", k.c_str());
      }
      return false;
    }
  }

  return true;
}

inline void ReadPoints3D(const Json::Value& root, Tracks3DMap& tracks3D) {
  Json::Value::Members members = root.getMemberNames();

  for (Json::Value::Members::const_iterator i = members.begin(); i != members.end(); i++) {
    const std::string& pointIdStr = *i;
    const Json::Value& v = root[pointIdStr];
    const TrackId id = std::stoi(pointIdStr);

    tracks3D[id].x() = v[0].asFloat();
    tracks3D[id].y() = v[1].asFloat();
    tracks3D[id].z() = v[2].asFloat();
  }
}

inline void WritePoints3D(const Tracks3DMap& tracks3D, Json::Value& root) {
  for (Tracks3DMap::const_iterator i = tracks3D.begin(); i != tracks3D.end(); i++) {
    const TrackId& t = i->first;
    const Vector3T& v = i->second;
    root[std::to_string(t)][0] = v.x();
    root[std::to_string(t)][1] = v.y();
    root[std::to_string(t)][2] = v.z();
  }

  // illuminate value = null in json.
  if (root.isNull()) {
    root = Json::objectValue;
  }
}

inline bool ReadSequence(const Json::Value& root, const Timeline& timeline, Sequence& sequence) {
  if (root.isArray()) {
    for (const Json::Value& image : root) {
      sequence.push_back(image.asString());
    }
  } else {
    sequence.push_back(root.asString());
  }

  const size_t nFiles = sequence.size();

  // there are two possible cases. Single file = start mask or multiple = direct list.
  if (nFiles == timeline.nFrames()) {
    // here we have direct list
    return true;
  }
  if (nFiles != 1) {
    TraceError("Found %zu images in edex that is not corresponded to the header (%zu frames)", nFiles,
               timeline.nFrames());
    return false;
  }
  // assume its first frame. generate others.
  FrameId startFrameId;
  const std::string fileName = sequence[0];  // start frame

  if (!filepath::ExtractFrameIdFromFileName(fileName, startFrameId)) {
    TraceError("Can't extract frame id from file name %s", fileName.c_str());
    return false;
  }

  TimeControl tc(timeline);

  if (startFrameId != tc.currentFrame()) {
    TraceError("startFrameId does not match tc.currentFrame");
    return false;
  }

  // skip first extracted frame in first loop
  while (tc.nextFrame()) {
    std::string replacedFileName;

    if (!(filepath::ReplaceFrameIdInFileName(fileName, tc.currentFrame(), replacedFileName))) {
      return false;
    }

    sequence.push_back(replacedFileName);
  }

  assert(sequence.size() == timeline.nFrames());
  return true;
}

inline void WriteSequence(const Cameras& cameras, Json::Value& root) {
  const size_t nCams = cameras.size();

  for (Json::ArrayIndex i = 0; i < nCams; ++i) {
    const Sequence& sequence = cameras[i].sequence;

    for (const auto& f : sequence) {
      root[i].append(f);
    }

    // illuminate value = null in json.
    if (root[i].isNull()) {
      root[i] = Json::arrayValue;
    }
  }
}

inline bool VerifyDistortionParameters(const std::string& model, size_t num_params) {
  if (model == "pinhole") {
    return num_params == 0;
  } else if (model == "fisheye") {
    return num_params == 4;
  } else if (model == "brown5k") {
    return num_params == 5;
  } else if (model == "polynomial") {
    return num_params == 8;
  }
  return false;
}

inline bool ReadIntrinsics(const Json::Value& root, Intrinsics* intrinsics) {
  ReadIntVector2(root[EDEX_CAM_SIZE], &(intrinsics->resolution));
  ReadFloatVector<2>(root[EDEX_CAM_FOCAL], &(intrinsics->focal));
  ReadFloatVector<2>(root[EDEX_CAM_PRINCIPAL], &(intrinsics->principal));

  if (!root.isMember(EDEX_CAM_DISTORTION_MODEL)) {
    TraceError("Distortion model not found");
    return false;
  }
  if (root.isMember(EDEX_CAM_DISTORTION_PARAMS)) {
    ReadFloatArray(root[EDEX_CAM_DISTORTION_PARAMS], &(intrinsics->distortion_params));
  }

  intrinsics->distortion_model = root[EDEX_CAM_DISTORTION_MODEL].asString();

  if (!VerifyDistortionParameters(intrinsics->distortion_model, intrinsics->distortion_params.size())) {
    TraceError("Wrong distortion type or parameters number");
    return false;
  }
  return true;
}

inline void WriteIntrinsics(const Intrinsics& intrinsics, Json::Value& root) {
  WriteIntVector2(root[EDEX_CAM_SIZE], intrinsics.resolution);
  WriteVector<2>(root[EDEX_CAM_FOCAL], intrinsics.focal);
  WriteVector<2>(root[EDEX_CAM_PRINCIPAL], intrinsics.principal);
  root[EDEX_CAM_DISTORTION_MODEL] = intrinsics.distortion_model;
  WriteFloatVector(root[EDEX_CAM_DISTORTION_PARAMS], intrinsics.distortion_params);
}

inline bool ReadTrack2D(const Json::Value& root, Cameras& cameras) {
  using namespace Json;

  assert(root.isObject());

  const size_t nCams = cameras.size();

  Value::Members frames = root.getMemberNames();

  for (Value::Members::const_iterator f = frames.begin(); f != frames.end(); ++f) {
    const std::string& frameIdStr = *f;

    const Value& camTracks2dRoot = root[frameIdStr];  // in current frame in current cam

    if (!camTracks2dRoot.isArray() || camTracks2dRoot.size() != nCams) {
      TraceError("Wrong track2d format");
      return false;
    }

    for (Json::ArrayIndex i = 0; i < nCams; ++i) {
      const Value& tracks2dRoot = camTracks2dRoot[i];  // in current frame

      Value::Members tracks = tracks2dRoot.getMemberNames();
      const size_t nTracks = tracks2dRoot.size();
      const FrameId& frame = std::stoi(frameIdStr);

      TrackIdVector2TPairVector& frameTrack2d = cameras[i].tracks2D[frame];
      frameTrack2d.resize(nTracks);
      size_t j = 0;

      for (Value::Members::const_iterator t = tracks.begin(); t != tracks.end(); ++t) {
        const std::string& trackIdStr = *t;
        const TrackId& trackId = std::stoi(trackIdStr);
        const Value& trackRoot = tracks2dRoot[trackIdStr];
        frameTrack2d[j].first = trackId;
        frameTrack2d[j].second.x() = trackRoot[0].asFloat();
        frameTrack2d[j].second.y() = trackRoot[1].asFloat();
        ++j;
      }

      assert(j == nTracks);
    }
  }

  return true;
}
inline void WriteTrack2D(const Cameras& cameras, Json::Value& root) {
  root = Json::objectValue;

  const size_t nCams = cameras.size();

  // extract all frameIds from all cameras (they could be spread)
  std::set<FrameId> frameIds;

  for (size_t i = 0; i < nCams; ++i) {
    const Tracks2DVectorsMap& tracks2D = cameras[i].tracks2D;

    for (Tracks2DVectorsMap::const_iterator j = tracks2D.begin(); j != tracks2D.end(); ++j) {
      const FrameId frameId = j->first;
      frameIds.insert(frameId);
    }
  }

  for (const FrameId& frameId : frameIds) {
    const std::string frameIdStr = std::to_string(frameId);
    root[frameIdStr] = Json::arrayValue;

    for (Json::ArrayIndex i = 0; i < nCams; ++i) {
      const TrackIdVector2TPairVector& frameTrack2d = cameras[i].tracks2D.at(frameId);
      const size_t nTrack = frameTrack2d.size();

      Json::Value array;

      for (size_t j = 0; j < nTrack; j++) {
        const TrackIdVector2TPair& pair = frameTrack2d[j];
        const TrackId& trackId = pair.first;
        Json::Value p;
        p[0] = pair.second.x();
        p[1] = pair.second.y();
        array[std::to_string(trackId)] = p;
      }

      root[frameIdStr][i] = array;
    }
  }
}

inline void ReadDepthID(const Json::Value& root, DepthId& depth_id) {
  depth_id = std::optional<uint8_t>(root.asUInt());
}

inline void ReadIsometry(const Json::Value& root, Isometry3T& isometry) {
  auto& m = isometry.matrix();
  m = Matrix4T::Identity();

  assert(root.isArray());

  for (Json::ArrayIndex i = 0; i < 3; ++i) {
    const Json::Value& row = root[i];
    assert(row.isArray());
    assert(row.size() == 4);
    for (Json::ArrayIndex j = 0; j < 4; ++j) {
      m(i, j) = row[j].asFloat();
    }
  }
}
inline void WriteIsometry(const Isometry3T& isometry, Json::Value& root) {
  const auto& m = isometry.matrix();

  for (Json::ArrayIndex i = 0; i < 3; ++i) {
    Json::Value& row = root[i];

    for (Json::ArrayIndex j = 0; j < 4; ++j) {
      row[j] = m(i, j);
    }

    assert(row.isArray());
  }

  assert(root.isArray());
}

inline void ReadPositions(const Json::Value& root, CameraMap& positions, const RotationStyle rs) {
  Json::Value::Members members = root.getMemberNames();

  for (Json::Value::Members::const_iterator i = members.begin(); i != members.end(); i++) {
    const std::string& frameIdStr = *i;
    const FrameId id = std::stoi(frameIdStr);

    const Json::Value& cameraPos = root[frameIdStr];
    const Json::Value& rotationRoot = cameraPos[EDEX_ROTATION];
    const Json::Value& translationRoot = cameraPos[EDEX_TRANSLATION];

    if (rs == RotationStyle::EulerDegrees) {
      Vector3T rotation(rotationRoot[0].asFloat(), rotationRoot[1].asFloat(), rotationRoot[2].asFloat());
      Translation3T translation(translationRoot[0].asFloat(), translationRoot[1].asFloat(),
                                translationRoot[2].asFloat());
      positions[id] = translation * QuaternionT(rotation, AngleUnits::Degree);
    } else if (rs == RotationStyle::Quaternion) {
      // Quaternion ctor takes w, x, y, z but storage order is x, y, z, w (same as memory layout)
      QuaternionT rotation(rotationRoot[3].asFloat(), rotationRoot[0].asFloat(), rotationRoot[1].asFloat(),
                           rotationRoot[2].asFloat());
      Translation3T translation(translationRoot[0].asFloat(), translationRoot[1].asFloat(),
                                translationRoot[2].asFloat());
      positions[id] = translation * rotation;
    } else if (rs == RotationStyle::RotationMatrix) {
      Eigen::Matrix3f m;
      m(0, 0) = rotationRoot[0].asFloat();
      m(0, 1) = rotationRoot[1].asFloat();
      m(0, 2) = rotationRoot[2].asFloat();
      m(1, 0) = rotationRoot[3].asFloat();
      m(1, 1) = rotationRoot[4].asFloat();
      m(1, 2) = rotationRoot[5].asFloat();
      m(2, 0) = rotationRoot[6].asFloat();
      m(2, 1) = rotationRoot[7].asFloat();
      m(2, 2) = rotationRoot[8].asFloat();

      QuaternionT rotation(m);
      Translation3T translation(translationRoot[0].asFloat(), translationRoot[1].asFloat(),
                                translationRoot[2].asFloat());
      positions[id] = translation * rotation;
    } else {
      assert(false && "Wrong RotationStyle specified");
    }
  }
}

inline void DecomposeIsometryRotStable(const Isometry3T& isometry, const AngleVector3T& rotPrev, Vector3T& rot) {
  const AngleVector3T rotation = getEulerRotationStable(isometry, rotPrev);
  rot.x() = rotation.x().getValue(AngleUnits::Degree);
  rot.y() = rotation.y().getValue(AngleUnits::Degree);
  rot.z() = rotation.z().getValue(AngleUnits::Degree);
}

inline void WritePositions(const CameraMap& positions, Json::Value& root, const RotationStyle rs) {
  AngleVector3T rotPrev(0, 0, 0);

  for (CameraMap::const_iterator i = positions.begin(); i != positions.end(); i++) {
    const FrameId& frame = i->first;
    const Isometry3T& pos = i->second;

    Json::Value& cameraPos = root[std::to_string(frame)];
    Json::Value& rotationRoot = cameraPos[EDEX_ROTATION];
    Json::Value& translationRoot = cameraPos[EDEX_TRANSLATION];

    if (rs == RotationStyle::EulerDegrees) {
      Vector3T rotation;
      DecomposeIsometryRotStable(pos, rotPrev, rotation);
      rotPrev = AngleVector(rotation, AngleUnits::Degree);
      rotationRoot[0] = rotation.x();
      rotationRoot[1] = rotation.y();
      rotationRoot[2] = rotation.z();
    } else if (rs == RotationStyle::Quaternion) {
      QuaternionT rotation(pos.linear());
      rotationRoot[0] = rotation.x();
      rotationRoot[1] = rotation.y();
      rotationRoot[2] = rotation.z();
      rotationRoot[3] = rotation.w();
    } else if (rs == RotationStyle::RotationMatrix) {
      const Eigen::Matrix3f& m = pos.linear();

      rotationRoot[0] = m(0, 0);
      rotationRoot[1] = m(0, 1);
      rotationRoot[2] = m(0, 2);
      rotationRoot[3] = m(1, 0);
      rotationRoot[4] = m(1, 1);
      rotationRoot[5] = m(1, 2);
      rotationRoot[6] = m(2, 0);
      rotationRoot[7] = m(2, 1);
      rotationRoot[8] = m(2, 2);
    } else {
      assert(false && "Wrong RotationStyle specified");
    }

    const Vector3T translation = pos.translation();
    translationRoot[0] = translation.x();
    translationRoot[1] = translation.y();
    translationRoot[2] = translation.z();
  }

  // illuminate value = null in json.
  if (root.isNull()) {
    root = Json::objectValue;
  }
}

inline bool ReadCameras(const Json::Value& root, Cameras& cameras) {
  if (!root.isArray()) {
    TraceError("cameras section is not array.");
    return false;
  }

  const size_t nCams = root.size();
  cameras.resize(nCams);

  for (Json::ArrayIndex i = 0; i < nCams; ++i) {
    const Json::Value camRoot = root[i];

    if (!CheckKey(camRoot, {EDEX_CAM_INTRINSICS, EDEX_CAMERA_TRANSFORM})) {
      return false;
    }

    Camera& c = cameras[i];
    try {
      if (!ReadIntrinsics(camRoot[EDEX_CAM_INTRINSICS], &(c.intrinsics))) {
        return false;
      }
      ReadIsometry(camRoot[EDEX_CAMERA_TRANSFORM], c.transform);
      if (camRoot.isMember(EDEX_CAM_DEPTH_ID)) {
        ReadDepthID(camRoot[EDEX_CAM_DEPTH_ID], c.depth_id);
        c.has_depth = true;
      }
    } catch (const Json::Exception&) {
      return false;
    }
  }
  return true;
}

inline bool ReadIMU(const Json::Value& root, IMU& imu) {
  if (!CheckKey(root, {EDEX_IMU_LOG, EDEX_CAMERA_TRANSFORM})) {
    return false;
  }

  try {
    ReadIsometry(root[EDEX_CAMERA_TRANSFORM], imu.transform);

    if (root.isMember(EDEX_IMU_ACCELEROMETER_NOISE_DENSITY)) {
      imu.accelerometer_noise_density = root[EDEX_IMU_ACCELEROMETER_NOISE_DENSITY].asFloat();
    }
    if (root.isMember(EDEX_IMU_ACCELEROMETER_RANDOM_WALK)) {
      imu.accelerometer_random_walk = root[EDEX_IMU_ACCELEROMETER_RANDOM_WALK].asFloat();
    }
    if (root.isMember(EDEX_IMU_GYROSCOPE_NOISE_DENSITY)) {
      imu.gyroscope_noise_density = root[EDEX_IMU_GYROSCOPE_NOISE_DENSITY].asFloat();
    }
    if (root.isMember(EDEX_IMU_GYROSCOPE_RANDOM_WALK)) {
      imu.gyroscope_random_walk = root[EDEX_IMU_GYROSCOPE_RANDOM_WALK].asFloat();
    }
    if (root.isMember(EDEX_IMU_FREQUENCY)) {
      imu.frequency = root[EDEX_IMU_FREQUENCY].asFloat();
    }

    const Json::Value& img_log = root[EDEX_IMU_LOG];
    if (!img_log.isString()) {
      TraceError("Wrong imu log");
      return false;
    }
    imu.imu_log_path_ = img_log.asString();

    if (root.isMember(EDEX_IMU_COORD_SYSTEM)) {
      const Json::Value& value = root[EDEX_IMU_COORD_SYSTEM];
      if (!value.isString()) {
        TraceError("Wrong \"%s\"", EDEX_IMU_COORD_SYSTEM);
        return false;
      }
      auto coordinate_system = value.asString();
      std::transform(coordinate_system.begin(), coordinate_system.end(), coordinate_system.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (coordinate_system == "ros") {
        imu.coordinate_system = CoordinateSystem::ROS;
      } else if (coordinate_system == "nvidia_isaac") {
        imu.coordinate_system = CoordinateSystem::ROS;
      } else if (coordinate_system == "cuvslam") {
        imu.coordinate_system = CoordinateSystem::CUVSLAM;
      } else if (coordinate_system == "opencv") {
        imu.coordinate_system = CoordinateSystem::OPENCV;
      } else {
        TraceError("Coordinate system \"%s\" assigned to \"%s\" is unknown", coordinate_system.c_str(),
                   EDEX_IMU_COORD_SYSTEM);
        return false;
      }
    }
  } catch (const Json::Exception&) {
    return false;
  }

  return true;
}

inline void WriteCameras(const Cameras& cameras, Json::Value& root) {
  const size_t nCams = cameras.size();

  for (Json::ArrayIndex i = 0; i < nCams; ++i) {
    const Camera& c = cameras[i];
    Json::Value& cameraRoot = root[i];

    WriteIntrinsics(c.intrinsics, cameraRoot[EDEX_CAM_INTRINSICS]);
    WriteIsometry(c.transform, cameraRoot[EDEX_CAMERA_TRANSFORM]);
  }

  // illuminate value = null in json.
  if (root.isNull()) {
    root = Json::objectValue;
  }
}

inline bool ReadKeyFrames(const Json::Value& root, FrameSet& keyFrames, const Timeline& timeline) {
  (void)timeline;

  assert(keyFrames.size() == 0);

  if (!root.isArray()) {
    return false;
  }

  for (const auto& frameIdValue : root) {
    const FrameId frameId = frameIdValue.asUInt();
    assert(timeline.checkFrameId(frameId));
    keyFrames.insert(frameId);
  }

  assert(keyFrames.size() == root.size());
  return true;
}
inline void WriteFrames(const FrameSet& keyFrames, Json::Value& root) {
  root = Json::arrayValue;

  for (const auto& frameId : keyFrames) {
    root.append(static_cast<Json::UInt>(frameId));
  }
}

}  // namespace cuvslam::edex::internal

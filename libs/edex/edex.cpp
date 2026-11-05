
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

#include "edex/edex.h"

#include "common/include_json.h"

#include "edex/edex_internal.h"

namespace cuvslam::edex {

using namespace cuvslam::edex::internal;

bool loadPoses(const std::string& file_name, Isometry3TVector& poses) {
  FILE* fp = std::fopen(file_name.c_str(), "r");

  if (!fp) {
    return false;
  }

  while (!std::feof(fp)) {
    Isometry3T pose;
    Matrix4T& m = pose.matrix();
    float fm[12];

    bool read_ok = true;
    for (size_t k = 0; k < 12; ++k) {
      if (std::fscanf(fp, "%f", &fm[k]) != 1) {
        read_ok = false;
        break;
      }
    }

    if (!read_ok) {
      break;
    }

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        m(i, j) = fm[i * 4 + j];
      }
    }
    pose.makeAffine();

    if (m.allFinite()) {
      poses.push_back(pose);
    }
  }

  std::fclose(fp);
  return !poses.empty();
}

bool writePoses(const std::string& file_name, const Isometry3TVector& poses) {
  FILE* fp = std::fopen(file_name.c_str(), "w");
  if (!fp) {
    return false;
  }
  for (const auto& pose : poses) {
    std::fprintf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f\n", pose.matrix()(0, 0), pose.matrix()(0, 1),
                 pose.matrix()(0, 2), pose.matrix()(0, 3), pose.matrix()(1, 0), pose.matrix()(1, 1),
                 pose.matrix()(1, 2), pose.matrix()(1, 3), pose.matrix()(2, 0), pose.matrix()(2, 1),
                 pose.matrix()(2, 2), pose.matrix()(2, 3));
  }
  std::fclose(fp);
  return true;
}

EdexFile::EdexFile() : rotStyle_(RotationStyle::RotationMatrix) {}

EdexFile::EdexFile(const RotationStyle rs) : rotStyle_(rs) {}

bool EdexFile::read(const std::string& fileName) {
  std::lock_guard<std::mutex> lock(getMutex());
  Json::Value root;

  try {
    JsonUtils::readJson(fileName, root);
  } catch (const std::exception& e) {
    TraceError("%s", e.what());
    return false;
  }

  if (!root.isArray() && root.size() != 2) {
    TraceError("Edex file is not a JSON array [header, body] object. File is %s", fileName.c_str());
    return false;
  }

  const Json::Value header = root[0];
  const Json::Value body = root[1];

  if (!readHeader(header)) {
    TraceError("Wrong header. File is %s\n", fileName.c_str());
    return false;
  }

  if (!readBody(body)) {
    TraceError("Wrong body. File is %s\n", fileName.c_str());
    return false;
  }

  return true;
}

bool EdexFile::write(const std::string& fileName) const {
  std::lock_guard<std::mutex> lock(getMutex());
  std::ofstream jsonFile(fileName, std::ofstream::binary);

  if (!jsonFile.is_open()) {
    return false;
  }

  Json::Value root;
  writeHeader(root[0]);
  writeBody(root[1]);

  Json::StyledStreamWriter writer;
  std::ostringstream out;
  writer.write(out, root);
  jsonFile << out.str();

  if (!jsonFile.good()) {
    return false;
  }

  return true;
}
bool EdexFile::readHeader(const Json::Value& header) {
  if (!CheckKey(header, {EDEX_VERSION, EDEX_FRAME_START, EDEX_FRAME_END, EDEX_CAMERAS})) {
    return false;
  }

  version_ = header[EDEX_VERSION].asString();

  if (version_ != EDEX_VERSION_VALUE) {
    TraceError("Wrong edex version. Required edex %s, found %s", EDEX_VERSION_VALUE, version_.c_str());
    return false;
  }

  timeline_.set(header[EDEX_FRAME_START].asUInt(), header[EDEX_FRAME_END].asUInt());

  if (!ReadCameras(header[EDEX_CAMERAS], cameras_)) {
    TraceError("Can't read %s section from EDEX file", EDEX_CAMERAS);
    return false;
  }
  if (header.isMember(EDEX_IMU)) {
    if (!ReadIMU(header[EDEX_IMU], imu_)) {
      TraceError("Can't read %s section from EDEX file", EDEX_IMU);
      return false;
    }
  }

  return true;
}
bool EdexFile::readBody(const Json::Value& body) {
  if (!CheckKey(body, {EDEX_SEQUENCE})) {
    return false;
  }

  uint8_t nDepthCams = 0;
  bool depth_in_edex = CheckKey(body, {EDEX_DEPTH_SEQUENCE}, true);

  // read sequences
  const Json::Value sequenceRoot = body[EDEX_SEQUENCE];
  Cameras& cameras = cameras_;
  const size_t nCams = cameras.size();

  if (!sequenceRoot.isArray() || sequenceRoot.size() != nCams) {
    TraceError("%s expected to be array of %zu", EDEX_SEQUENCE, nCams);
    return false;
  }

  if (depth_in_edex) {
    const Json::Value sequenceDepth = body[EDEX_DEPTH_SEQUENCE];
    nDepthCams = std::count_if(cameras.begin(), cameras.end(), [](const Camera& cam) { return cam.has_depth; });
    if (!sequenceDepth.isArray() || sequenceDepth.size() != nDepthCams) {
      TraceError("%s expected to be array of %d", EDEX_DEPTH_SEQUENCE, nDepthCams);
      return false;
    }
  }

  for (Json::ArrayIndex i = 0; i < nCams; ++i) {
    if (!ReadSequence(sequenceRoot[i], timeline_, cameras[i].sequence)) {
      TraceError("Wrong image sequence.");
      return false;
    }
    if (cameras[i].has_depth) {
      const Json::Value sequenceDepth = body[EDEX_DEPTH_SEQUENCE];
      if (!ReadSequence(sequenceDepth[cameras[i].depth_id.value()], timeline_, cameras[i].depth_sequence)) {
        TraceError("Wrong depth sequence.");
        return false;
      }
    }
  }

  // read 2d points
  if (body.isMember(EDEX_POINTS_2D)) {
    const Json::Value points2d = body[EDEX_POINTS_2D];
    if (!points2d.isObject()) {
      TraceError("points 2d must be map of frames");
      return false;
    }
    if (!ReadTrack2D(points2d, cameras)) {
      TraceError("Wrong track2d format.");
      return false;
    }
  }

  if (body.isMember(EDEX_RIG_POSITIONS)) {
    ReadPositions(body[EDEX_RIG_POSITIONS], rigPositions_, rotStyle_);
  }
  if (body.isMember(EDEX_POINTS_3D)) {
    ReadPoints3D(body[EDEX_POINTS_3D], tracks3D_);
  }

  // check optional parameters
  if (body.isMember(EDEX_KEY_FRAMES)) {
    if (!ReadKeyFrames(body[EDEX_KEY_FRAMES], keyFrames_, timeline_)) {
      TraceError("Wrong key frames exist but wrong.");
      return false;
    }
  }
  const bool fps_present = body.isMember(EDEX_FPS);
  if (body.isMember(EDEX_FRAME_META)) {
    if (fps_present) {
      TraceError("Frame meta has timestamps. No fps is required.");
      return false;
    }
    const Json::Value& frame_meta_log = body[EDEX_FRAME_META];
    if (!frame_meta_log.isString()) {
      TraceError("Wrong frame metadata log");
      return false;
    }
    frame_meta_log_path_ = frame_meta_log.asString();
  } else {
    if (!fps_present) {
      TraceError("fps is required.");
      return false;
    }
    const Json::Value& fps = body[EDEX_FPS];
    if (!fps.isDouble()) {
      TraceError("Wrong FPS");
      return false;
    }
    fps_ = fps.asDouble();
  }

  return true;
}
void EdexFile::writeHeader(Json::Value& header) const {
  header[EDEX_VERSION] = EDEX_VERSION_VALUE;

  // read general parameters
  header[EDEX_FRAME_START] = static_cast<Json::UInt>(timeline_.firstFrame());
  header[EDEX_FRAME_END] = static_cast<Json::UInt>(timeline_.lastFrame());

  WriteCameras(cameras_, header[EDEX_CAMERAS]);
}
void EdexFile::writeBody(Json::Value& body) const {
  WriteSequence(cameras_, body[EDEX_SEQUENCE]);
  WriteTrack2D(cameras_, body[EDEX_POINTS_2D]);
  WritePoints3D(tracks3D_, body[EDEX_POINTS_3D]);
  WritePositions(rigPositions_, body[EDEX_RIG_POSITIONS], rotStyle_);
  WriteFrames(keyFrames_, body[EDEX_KEY_FRAMES]);
  WriteFrames(failedFrames_, body[EDEX_FAILED_FRAMES]);
}

std::mutex& EdexFile::getMutex() const {
  static std::mutex mutex;
  return mutex;
}

}  // namespace cuvslam::edex

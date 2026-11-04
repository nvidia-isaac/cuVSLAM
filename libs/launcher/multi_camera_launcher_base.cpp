
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

#include "launcher/multi_camera_launcher_base.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

#include "gflags/gflags.h"

// FIG-shaping flags. The launcher owns these because every launcher that builds a FIG
// (stereo, multi-camera, RGBD, multisensor, VIO) needs the same knobs and we want one
// canonical set instead of per-launcher duplicates. The camera library itself stays free
// of any gflags dependency: it consumes a plain camera::FigSettings struct supplied here.
DEFINE_bool(allow_stereo_track_for_depth, false,
            "Allow stereo 2D tracking between depth-aligned cameras and other cameras. "
            "Read by every launcher that builds a FrustumIntersectionGraph.");
DEFINE_string(fig_depth_camera_ids, "auto",
              "Cameras supplying depth: 'auto' (rig advertisement via getCamerasWithDepth), "
              "'none', or comma-separated ids (e.g. '0,2'). Used by every launcher to size the "
              "FIG and to filter runtime depth ingestion.");

DEFINE_int32(slam_camera, -1, "Select single camera for slam. If -1, all cameras will be used");

namespace cuvslam::launcher {
namespace {

// Resolves a depth-cameras specification string against the rig.
// Accepted values:
//   "auto" — every camera the rig advertises via ICameraRig::getCamerasWithDepth()
//   "none" — empty list (no depth)
//   comma-separated camera ids (e.g. "0,2") — explicit list, validated against rig.num_cameras
// Throws std::runtime_error on out-of-range or duplicate ids.
// Lives next to the -fig_depth_camera_ids flag definition above so the CLI string format and
// the parser stay in sync; the camera library stays free of any flag-shaped helpers.
std::vector<CameraId> ResolveDepthCameraIds(const ICameraRig& rig, std::string_view spec) {
  std::vector<CameraId> out;

  if (spec == "none") {
    return out;
  }

  if (spec == "auto") {
    return rig.getCamerasWithDepth();
  }

  // Explicit comma-separated list.
  std::stringstream ss{std::string(spec)};
  std::string tok;
  std::unordered_set<int32_t> seen;
  const auto num_cams = static_cast<int32_t>(rig.getCamerasNum());
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) {
      continue;
    }
    int32_t id = 0;
    try {
      id = std::stoi(tok);
    } catch (const std::exception&) {
      throw std::runtime_error("depth camera ids: cannot parse '" + tok + "' as a camera id");
    }
    if (id < 0 || id >= num_cams) {
      throw std::runtime_error("depth camera ids: id " + std::to_string(id) + " out of range (rig has " +
                               std::to_string(num_cams) + " cameras)");
    }
    if (!seen.insert(id).second) {
      throw std::runtime_error("depth camera ids: duplicate id " + std::to_string(id));
    }
    out.push_back(static_cast<CameraId>(id));
  }
  return out;
}

}  // namespace

MultiCameraBaseLauncher::MultiCameraBaseLauncher(ICameraRig& cameraRig, const odom::Settings& svo_settings,
                                                 bool auto_allow_stereo_track_for_depth)
    : BaseLauncher(cameraRig, svo_settings) {
  depth_ids_ = ResolveDepthCameraIds(cameraRig_, FLAGS_fig_depth_camera_ids);

  camera::FigSettings fig_settings{
      /*.mode=*/svo_settings_.sof_settings.multicam_mode,
      /*.depth_ids=*/depth_ids_,
      /*.allow_stereo_track_for_depth=*/FLAGS_allow_stereo_track_for_depth || auto_allow_stereo_track_for_depth,
      /*.manual_setup=*/svo_settings_.sof_settings.multicam_setup,
  };
  fig = camera::FrustumIntersectionGraph(rig, fig_settings);
  const std::vector<CameraId>& prim_cams = fig.primary_cameras();
  {
    std::stringstream s;
    s << "Fig setup: ";
    for (const CameraId& prim_id : prim_cams) {
      s << static_cast<int>(prim_id) << ": [";
      for (const CameraId& sec_id : fig.secondary_cameras(prim_id)) {
        s << static_cast<int>(sec_id) << ", ";
      }
      s << "] | ";
    }
    TraceMessage("%s", s.str().c_str());
  }

  if (!fig.is_valid()) {
    throw std::runtime_error(
        "Bad calibration. cuVSLAM is supposed to work with at least"
        " one stereo pair available.");
  }

  if (FLAGS_slam_camera != -1) {
    auto it = std::find(prim_cams.begin(), prim_cams.end(), FLAGS_slam_camera);
    if (it == prim_cams.end()) {
      std::stringstream ss;
      ss << "FLAGS_slam_camera must be one of: {";
      for (CameraId cam_id : prim_cams) {
        ss << static_cast<int>(cam_id) << " ";
      }
      ss << "}";
      throw std::runtime_error("Bad calibration. cuVSLAM is supposed to work with at least one stereo pair available.");
    }
    assert(*it == FLAGS_slam_camera);
    const CameraId manual_selected_slam_camera = *it;
    slam_cameras_ = {manual_selected_slam_camera};
  } else {
    slam_cameras_ = prim_cams;
  }
}

bool MultiCameraBaseLauncher::isDepthCamera(CameraId id) const {
  return std::find(depth_ids_.begin(), depth_ids_.end(), id) != depth_ids_.end();
}
}  // namespace cuvslam::launcher


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

#include <string>
#include <unordered_map>
#include <vector>

#include "camera/rig.h"
#include "common/camera_id.h"
#include "params/params.h"

namespace cuvslam::camera {

enum class MulticameraMode {
  // each secondary camera must be connected to only one primary camera
  Performance,
  // all cameras are primary cameras
  Precision,
  // primary camera(s) are cameras with highest degree, intersecting cameras are secondary
  Moderate,
  // primary/secondary cameras are set manually; no checks are done
  Manual
};

using MulticamManualSetup = std::vector<std::vector<CameraId>>;

MulticameraMode ParseMulticameraMode(const std::string& mode,
                                     MulticameraMode default_mode = MulticameraMode::Performance);

// User-facing settings that shape the FrustumIntersectionGraph topology. Grouped into a single
// struct so the camera library can stay free of any global-flag plumbing while still letting
// callers (launchers, tools, tests) drive every knob independently.
struct FigSettings {
  MulticameraMode mode = MulticameraMode::Performance;
  // Cameras that supply depth. When non-empty, the FIG can optionally drop secondary edges that
  // land on these cameras (controlled by allow_stereo_track_for_depth) and guarantees that every
  // depth camera ends up as a primary, even when it has no stereo partners.
  std::vector<CameraId> depth_ids;
  // If false, secondary edges that land on a depth camera are removed (the depth camera is only
  // tracked monocularly + via depth). If true, the FIG keeps those cross-camera tracks.
  // Default: true — matches Multisensor mode's enable_depth_stereo_tracking default
  // (multisensor benefits from cross-camera 2D tracks). Note: RGBDSettings defaults this to
  // false in the public API; that path explicitly forwards the per-mode value into
  // FigSettings, so the default here only affects callers who don't set the field.
  bool allow_stereo_track_for_depth = true;
  // Required when mode == MulticameraMode::Manual; ignored otherwise.
  MulticamManualSetup manual_setup;
};

// Building a camera graph without the need for manual assumption hard-coding of the camera configuration.
// It leverages camera intrinsic, distortion and extrinsic calibrations in order to check overlapping between
// the camera frustums and define the stereo pairs.
class FrustumIntersectionGraph {
public:
  // description of a camera in camera configuration in the context of a graph representation
  struct CameraGraphNode {
    struct ConnectedCam {
      ConnectedCam(CameraId id_, float fir_) : id(id_), frustrim_intersection_ratio(fir_){};
      CameraId id;
      float frustrim_intersection_ratio;  // 0 < FIR < 1;
    };
    // the vector of camera_id_js that forms a stereo pair with camera_id_i
    std::vector<ConnectedCam> stereo_camera_pairs;
    // the number of camera_id_js that forms a stereo_pair with camera_id_i
    size_t degree_of_vertex = 0;
  };

  FrustumIntersectionGraph() = default;

  FrustumIntersectionGraph(const camera::Rig& rig, const FigSettings& settings);

  FrustumIntersectionGraph(const std::vector<CameraGraphNode>& graph, const FigSettings& settings);

  // primary camera is tracked on every frame
  const std::vector<CameraId>& primary_cameras() const;

  // we track primary -> secondary only on keyframes
  const std::vector<CameraId>& secondary_cameras(CameraId primary_camera) const;

  bool is_valid() const;

private:
  void set_precision_mode(const std::vector<CameraGraphNode>& graph);
  void set_performance_mode(const std::vector<CameraGraphNode>& graph);
  void set_moderate_mode(const std::vector<CameraGraphNode>& graph);
  void set_manual_mode(const std::vector<CameraGraphNode>& graph, const MulticamManualSetup& manual_setup);

  std::vector<CameraId> primary_cameras_;
  std::vector<CameraId> depth_ids_;
  std::unordered_map<CameraId, std::vector<CameraId>> secondary_from_primary_;
};

}  // namespace cuvslam::camera

// Spellings accepted for MulticameraMode in parameter files and on the command line. Manual is
// omitted on purpose: it needs an accompanying camera setup that no scalar value can express.
CUVSLAM_PARAM_ENUM_BEGIN(cuvslam::camera::MulticameraMode)
CUVSLAM_PARAM_ENUM_VALUE("performance", Performance)
CUVSLAM_PARAM_ENUM_VALUE("precision", Precision)
CUVSLAM_PARAM_ENUM_VALUE("moderate", Moderate)
CUVSLAM_PARAM_ENUM_END()

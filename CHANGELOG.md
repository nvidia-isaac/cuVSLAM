# Changelog

## Unreleased

### Changed

- `Tracker` now takes a required `Tracker::Mode` (`cuvslam.Tracker.Mode` in Python) as its second
  constructor argument, choosing between odometry alone or odometry with SLAM, and between running
  the bundle adjuster and SLAM on background threads (realtime) or in the calling thread (offline).
  The mode has to agree with `Odometry::Config::async_sba` and `Slam::Config::sync_mode`: `Tracker`
  checks them and throws `std::invalid_argument` on a mismatch rather than overwriting them. A
  `slam_config` is now rejected in the odometry-only modes

### Added

- Visual place recognition, for the kidnapped robot problem: `Slam` keeps one frame descriptor per pose graph node,
  and `Slam::RecognizePlaceByFrame` reports the node a single frame was observed from, and that node's current world
  pose, with no pose guess. `Slam::AddFrameToVprMap` fills the map while mapping and `Slam::SaveMap` writes it as
  part of the SLAM map. `Slam::Config::vpr_mode` picks the backend: `Simple` (thumbnails) and `Bow`
  (in-tree ORB and a binary bag of words), neither with a third-party dependency, `DBoW2` (build with
  `USE_DBOW2`, needs OpenCV) or `AnyLoc` (build with `USE_ONNXRUNTIME`, needs a DINOv2 ONNX model in
  `Slam::Config::vpr_model_path`). All of it is available from Python; see the README
- `Slam::LocalizeInMap` now takes an optional `guess_pose`. Without one it verifies the places the saved map's own
  place recognition map proposes, which is what lets a robot that has no idea where it is relocalize at all;
  `LocalizationSettings::vpr_candidates` bounds how many it tries
- `Slam::Config::vpr_map_path`: loads a place recognition map on its own, for recognition without landmarks
- `cuvslam_vpr_reporter`: scores a place recognition backend on two recordings of the same route and reports the
  recognition and false positive rates with an HTML and PDF report
- `cuvslam_export_dinov2`: exports the DINOv2 value-facet ONNX model that the AnyLoc backend reads
- `Slam::Config::delay_warning_queue_size`: warns in verbose mode when more than the configured number of commands
  are queued to the SLAM thread, meaning SLAM falls behind odometry

### Fixed

- Unsynchronized reads of the SLAM engine during map localization (`LocalizeInMapCmd::Execute`)
- Unnecessary mutex contention in `AsyncSlam::GetSlamPose()`

## [17.0.0] - 2026-07-21

Adds cuNLS-based multisensor fusion, improves tracking and SLAM robustness, and expands evaluation tooling.

### Added

- Multisensor odometry mode for mixed RGB/RGB-D camera rigs with optional IMU fusion
- C++ and Python multisensor settings and TartanGround and RealSense examples

### Changed

- Enabled cuNLS by default
- Improved cross-camera feature tracking with rig-aware reprojection and reduced per-frame odometry allocations
- Split `Slam::Track()` into a void tracking call and `Slam::GetPose()`
- Updated Rerun SDK integration and example visualizations

### Fixed

- Inertial tracking during image blackouts
- Races in map save/load and asynchronous SLAM localization updates, other SLAM fixes

### Security

- Added overflow checks to CUDA allocation-size calculations

## [16.0.0] - 2026-06-02

Bugfixes in IMU integration, SLAM map loading, etc.

### Added

- Load-map documentation
- Missing tests for Python bindings

### Changed

- Refactored string-based settings to use enums
- Posegraph code cleanup
- Improved build support for native ARM targets

### Fixed

- Two IMU integration bugs causing stereo+IMU to underperform stereo-only
- SLAM jump after map load
- Cross-stream memory visibility race on Blackwell (sm_121)
- CUDA architecture selection logic

## [15.0.0] - 2026-03-02

Initial open-source release.

### Added

- Cached map-to-disk SLAM mode
- RGBD pipeline optimization
- Rerun visualization of internal cuVSLAM data
- Examples moved into the repository (from a previously separate pycuvslam repo)
- Troubleshooting guide (`TROUBLESHOOTING.md`)

### Changed

- New NVIDIA Community License
- Switched to semantic versioning (from this release onward)
- SLAM internal refactoring is in progress
- Removed JPEG/PNG dependencies from libcuvslam
- Opened cuVSLAM API to accept user-provided `cudaStream_t`
- CMake 4+ build compatibility

### Removed

- C API (superseded by C++ API)
- Several internal methods and unused parameters from public SLAM API

### Fixed

- Memory leaks
- RGBD depth mask bug
- Multiple other bugfixes

### Security

- Updated libpng version

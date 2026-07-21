# Changelog

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

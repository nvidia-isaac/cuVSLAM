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

- Internal parameters are now addressed by name. `Odometry::SetParameter()` and `GetParameters()` in C++, and
  `set_parameter()`, `set_parameters()` and `get_parameters()` in Python, cover every
  solver setting that `Internals` and `ApplyPersistentInternalParameters()` reached, plus everything they did not.
  `GetParameters()` reports each parameter's value, default, type, description and where the value came from, so a run's
  configuration can be recorded with its results
- `Odometry::TrackHints`, passed to `Track()`, for values that genuinely differ from frame to frame. It currently holds
  only `override_keyframe`
- `cuvslam_api_launcher`: `--params <file>` for a flat `key: value` parameter file, repeatable `-Pkey=value` overrides,
  and `--list_params` to print every parameter with its type, default and description. The file format belongs to the
  tool; the library reads no files, so Python callers can use any parser and pass the result to `set_parameters()`
- `Slam::Config::delay_warning_queue_size`: warns in verbose mode when more than the configured number of commands
  are queued to the SLAM thread, meaning SLAM falls behind odometry

### Removed

- `cuvslam::internal::Internals`, `cuvslam::internal::InternalParameter`, `Odometry::ApplyPersistentInternalParameters()`
  and the `cuvslam2_internal.h` header. Use `SetParameter()` for the settings they carried and `TrackHints` for the
  keyframe override. Names gained a prefix matching the settings they address, so `num_desired_tracks` is
  `sof.num_desired_tracks` and `kf_survivor_from_last` is `kf.survivor_from_last`; `sba.*` and `sm.*` keys are unchanged
- Python: `Odometry.Internals` and `apply_expert_parameters()`, replaced as above, and the `cuvslam.utils` module,
  whose configuration loaders are superseded by `set_parameters()`. This also drops the `pyyaml` dependency
- `cuvslam_api_launcher`: `--config` and its YAML schema, replaced by `--params`; the `--expert_sba_*` flags, now
  ordinary parameters reachable with `-P`; and `libs/utils/cuvslam_yaml_config.*`

### Changed

- `sof::Settings::multicam_mode` now defaults to `Precision`, matching `Odometry::Config::multicam_mode`. The two
  disagreed, so every report listed `sof.multicam_mode` as non-default even for an untouched configuration. Internal
  tools that construct `sof::Settings` directly and do not set the mode now get `Precision` instead of `Moderate`
- Setting an unknown internal parameter, or a value that does not parse or falls outside its range, is now an error
  instead of a logged warning, and leaves the parameter at its previous value. Parameters a mode never reads are not
  exposed at all, so naming one is an error rather than being silently ignored

### Fixed

- `Odometry::Config::use_denoising` and `Rig::Camera::border_top`/`border_bottom`/`border_left`/`border_right` had no
  effect. Both are applied at construction, but `Track()` rebuilt the per-frame settings from type defaults, and
  feature tracking re-reads denoising and border values from those per-frame settings on every frame. Setting either
  now changes tracking as documented; runs that relied on them being ignored will see different results
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

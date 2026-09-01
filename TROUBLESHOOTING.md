# Troubleshooting cuVSLAM

## Overview

This document describes how to troubleshoot cuVSLAM when the system is installed and runs successfully but the computed
pose is not sufficiently accurate. For installation or build issues, see
[Install PyCuVSLAM](README.md#install-pycuvslam) and [Build cuVSLAM](README.md#build-cuvslam).

![Troubleshooting](doc/images/troubleshooting.png)

**cuVSLAM is not tunable — it works out of the box.** All internal algorithms use carefully engineered defaults or
adapt automatically to the environment and motion profile. No internal parameter tuning is required or expected.
When tracking is inaccurate, the root cause is almost always in the inputs, not the algorithm parameters:

1. Accurate intrinsic calibration.
2. Accurate extrinsic calibration for stereo or multi-camera setups.
3. Proper image synchronization.
4. Sharp images with low noise.
5. Scenes that include large, rigid, textured objects.
6. Avoidance of scenes with strong repetitive patterns or structures.
7. Correct sensor configuration (tracking mode, rig description).
8. Correct coordinate frames for input and output poses.
9. Sequential frames with small time/pose deltas (no dropped or missing frames).

Most cuVSLAM issues arise when one or more of these requirements are not satisfied.

See [Requirements](README.md#performance).

## Table of Contents

- [Environment](#environment)
- [Step 1: Verify you have a working setup](#step-1-verify-you-have-a-working-setup)
- [Step 2: Create a reproducible testing environment](#step-2-create-a-reproducible-testing-environment)
  - [Use recording](#use-recording)
  - [Verify you have no missing frames](#verify-you-have-no-missing-frames)
  - [Use blocking single-thread execution mode](#use-blocking-single-thread-execution-mode)
  - [Enable visualization and verbosity](#enable-visualization-and-verbosity)
- [Step 3: Inspect your images](#step-3-inspect-your-images)
- [Step 4: Mask static image areas](#step-4-mask-static-image-areas)
- [Step 5: Run mono tracking on each eye of the stereo camera](#step-5-run-mono-tracking-on-each-eye-of-the-stereo-camera)
  - [Try to use rectified images if possible](#try-to-use-rectified-images-if-possible)
  - [Disable SLAM and ground constraint](#disable-slam-and-ground-constraint)
- [Step 6: Verify synchronization](#step-6-verify-synchronization)
- [Step 7: Run stereo odometry tracking](#step-7-run-stereo-odometry-tracking)
- [Step 8: Adjust odometry settings](#step-8-adjust-odometry-settings)
  - [Improve input images](#improve-input-images)
  - [Pick the best odometry mode](#pick-the-best-odometry-mode)
  - [Adjust motion prediction](#adjust-motion-prediction)
  - [Tune image area for feature selection](#tune-image-area-for-feature-selection)
- [Step 9: IMU integration](#step-9-imu-integration)
  - [Debug IMU with debug_imu_mode](#debug-imu-with-debug_imu_mode)
  - [Debug IMU with blackout_oscillator](#debug-imu-with-blackout_oscillator)
- [Step 10: Multi-camera setups](#step-10-multi-camera-setups)
- [Step 11: Enable ground constraint](#step-11-enable-ground-constraint)
- [Step 12: Enable SLAM](#step-12-enable-slam)
- [Step 13: Adjust SLAM map parameters](#step-13-adjust-slam-map-parameters)
- [Step 14: Run in async mode](#step-14-run-in-async-mode)
- [EDEX file](#edex-file)
- [Tracker](#tracker)
- [Result visualizer](#result-visualizer)
- [Debug visualization](#debug-visualization)
- [Shuttle-mode debug approach](#shuttle-mode-debug-approach)

## Environment

cuVSLAM can be used either through the native C++ API or via one of the available wrappers:

1. [Isaac ROS cuVSLAM](README.md#ros2-support) — a ROS 2 wrapper around the C++ API
2. [PyCuVSLAM](README.md#install-pycuvslam) — a Python wrapper around the C++ API.
3. [Command-line tools](tools) built on the C++ API, for example:
    1. [tracker](#tracker) — a standalone command-line application for image-sequence tracking with real-time
       benchmarking and debugging capabilities.
    2. [cuvslam_api_launcher](tools/cuvslam_api_launcher/README.md) — a test utility for tracking, saving maps, and
       performing localization using the cuVSLAM API.
    3. [undistort](tools/undistort/README.md) — a tool for removing lens distortion from images using camera intrinsics
       from an [EDEX](#edex-file) file.
    4. [result_visualizer](#result-visualizer) — a script to visualize output [EDEX](#edex-file) files.

Each wrapper can introduce its own issues. Use wrapper-specific tools and techniques for troubleshooting. If they do
not resolve the problem, use the built-in option to dump all data received from the wrapper to a JSON
[EDEX](#edex-file) file along with an image list (TGA), then continue debugging with the C++ diagnostic tools.

**C++ API**

```cpp
cuvslam::Odometry::Config::debug_dump_directory
```

**Python API**

```python
cuvslam.Odometry.Config.debug_dump_directory
```

See [Python API Reference](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.debug_dump_directory).

**Isaac ROS**

```
enable_debug_mode
debug_dump_path
```

See [Isaac ROS cuVSLAM parameters][].

**Important:** After dumping [EDEX](#edex-file) and TGA images, use the [tracker](#tracker) and
[result_visualizer](#result-visualizer) tools.

## Step 1: Verify you have a working setup

Regardless of your environment (Isaac ROS, Python API, C++ API, or command-line tools), first confirm that you can run
the provided examples. If cuVSLAM works on the sample dataset but not on your data, identify what differs between them.

## Step 2: Create a reproducible testing environment

### Use recording

When troubleshooting, ensure you can reproduce the issue reliably. Prefer a recorded sequence over a live camera.
Consider:

1. [EDEX](#edex-file) + images - is recommended for troubleshooting.
2. ROS bag - See [Isaac ROS cuVSLAM launch file][]. configure image topics, extrinsics, frames, and topic
   synchronization carefully; these are common sources of error.
3. [ZED 2 SVO](examples/zed/recording/README.md) - good compression and sync useful for long recordings.

### Verify you have no missing frames

Sometimes, when the system is overloaded or when you try to decompress a recording too quickly, frames get dropped.

The cuVSLAM 2D engine expects the images to form a sequential video stream with only a small time delta between frames.
Missing frames can significantly degrade quality, up to “tracking lost.” They can also make tracking results
non-deterministic—something you definitely don’t want during troubleshooting.

To diagnose this, enable verbose logging and check the `max_frame_delta_s` parameter. You’ll see a console message
whenever a frame drop is detected.

**C++ API**

```cpp
cuvslam::Odometry::Config::max_frame_delta_s
```

**Python API**

[cuvslam.Odometry.Config.max_frame_delta_s](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.max_frame_delta_s)

**Isaac ROS**

```
image_jitter_threshold_ms
```

In the Isaac ROS cuVSLAM node, there is also a synchronizer that can align multiple image streams with non-equal
timestamps. Check Isaac ROS documentation and tune:

```
num_cameras
min_num_images
sync_matching_threshold_ms
```

See [Isaac ROS cuVSLAM parameters][].

### Use blocking single-thread execution mode

In all later steps until [Step 14](#step-14-run-in-async-mode), run tracking in reproducible (blocking, non-async) mode.
In this mode, SBA and SLAM run on the main thread and block until completion rather than on a background thread.
This makes computation more stable and tests more reproducible, but it does not support high frame rates.
Feed images to cuVSLAM frame by frame.

Pick one of the `Tracker` offline modes and set the two configuration fields to match; `Tracker`
rejects a mode that disagrees with them.

**C++ API**

```cpp
cuvslam::Odometry::Config odometry_config;
odometry_config.async_sba = false;
cuvslam::Slam::Config slam_config;
slam_config.sync_mode = true;
cuvslam::Tracker tracker{rig, cuvslam::Tracker::Mode::OdometryWithSlamOffline, odometry_config, &slam_config};
```

**Python API**

1. [cuvslam.Tracker.Mode](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Tracker.Mode) —
   `OdometryOnlyOffline` or `OdometryWithSlamOffline`
2. [cuvslam.Odometry.Config.async_sba](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.async_sba)
3. [cuvslam.Slam.Config.sync_mode](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Slam.Config.sync_mode)

`sync_mode` is a `Slam.Config` field, so it only applies to `OdometryWithSlamOffline`. The
odometry-only modes take no `slam_config` at all — pass one and the tracker rejects it.

**Isaac ROS**

ROS cannot run cuVSLAM in blocking mode because `ros2 bag play` does not support blocking playback. Troubleshooting
options are therefore more limited. Workarounds:

1. Use `--rate` to reduce the playback rate (e.g., 1 or 0.1).
   ```shell
   ros2 bag play --rate <value>
   ```
2. Dump to [EDEX](#edex-file). See [Environment](#environment). Use a slow `--rate` here as well.
3. Use the [bag2edex](tools/ros/bag2edex) offline frame-by-frame export tool (this tool is not actively supported)
   with a low rate so that blocking mode does not trigger "missing frame" messages.

- Set up the [Isaac ROS environment in Docker](https://nvidia-isaac-ros.github.io/getting_started/index.html):
- Run the [Isaac ROS cuVSLAM launch file][] (you can edit it to adjust camera count, TF frames, and topics).
- If the input rosbag contains older cuVSLAM topics, remap them as needed.

### Enable visualization and verbosity

Use the following recommended settings for troubleshooting:

**C++ API**

```cpp
cuvslam::SetVerbosity()
cuvslam::Odometry::Config::enable_observations_export = true;
cuvslam::Odometry::Config::enable_landmarks_export = true;
cuvslam::Odometry::Config::enable_final_landmarks_export = true;
cuvslam::Slam::Config::enable_reading_internals = true;
```

**Python API**

1. [set_verbosity()](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.set_verbosity)
2. [cuvslam.Odometry.Config.enable_observations_export](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.enable_observations_export)
3. [cuvslam.Odometry.Config.enable_landmarks_export](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.enable_landmarks_export)
4. [cuvslam.Odometry.Config.enable_final_landmarks_export](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.enable_final_landmarks_export)
5. [cuvslam.Slam.Config.enable_reading_internals](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Slam.Config.enable_reading_internals)

**Isaac ROS**

See [Isaac ROS cuVSLAM parameters][]. Enable the following:

1. verbosity
2. enable_slam_visualization
3. enable_observations_view
4. enable_landmarks_view
5. path_max_size (consider increasing this value)

## Step 3: Inspect your images

Start with the simplest scenario: global shutter, high-quality images (no blur, low noise, sharp, good lighting), slow
motion, no moving objects, no image compression, and no repetitive structure. A 10–15 second recording is a good
starting point.

For cuVSLAM, frame rate is often more important than resolution (for example, 100 fps at VGA can be better than 30 fps
at 2K). cuVSLAM uses grayscale input only. For RGB cameras, one channel may be noisier than the others. Use a viewer to
inspect the input sequence frame by frame. Camera artifacts are among the most common causes of cuVSLAM failure.
Typical issues include:

1. Rolling shutter artifacts
2. Excessive noise
3. Excessive blur
4. In-frame artifacts

Even when you have multiple recordings from the same robot, verify the input images before investing time in deeper
troubleshooting. Images may be blurry if the robot shut down immediately before recording.

Example of a single corrupted frame from a one-hour recording:

![Corrupted frame](doc/images/troubleshooting_corrupted_frame.png)

## Step 4: Mask static image areas

If all of your images contain a static region (e.g., car hood, forklift manipulator), mask it. cuVSLAM provides an API
for this.

In the simplest case, crop the images using the following camera parameters:

**C++ API**

* cuvslam::Camera::border_top
* cuvslam::Camera::border_bottom
* cuvslam::Camera::border_left
* cuvslam::Camera::border_right

**Python API**

* cuvslam.Camera.border_top
* cuvslam.Camera.border_bottom
* cuvslam.Camera.border_left
* cuvslam.Camera.border_right

* See [Python API Reference](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Camera).

**Isaac ROS**

* img_mask_top
* img_mask_bottom
* img_mask_left
* img_mask_right

See [Masking region tutorial](examples/tum/README.md#masking-regions-to-prevent-feature-selection)

An example where it makes sense to mask the robot:

![ReRun](doc/images/troubleshooting_mask.gif)

## Step 5: Run mono tracking on each eye of the stereo camera

Mono tracking does not depend on extrinsic calibration. Only focal length matters. Visualize 2D observations during mono
tracking. Isaac ROS cuVSLAM, PyCuVSLAM, and the C++ API provide tools for this.
See [Debug visualization](#debug-visualization).

### Try to use rectified images if possible

Some cameras can perform rectification and lens undistortion onboard. Isaac ROS also provides hardware-accelerated nodes
for these steps. cuVSLAM can operate on either raw or rectified images, but when troubleshooting, it’s best to remove
lens distortion from the pipeline first. If that isn’t possible (for example, if your cameras output only raw
images and you don’t have an external undistortion tool), cuVSLAM includes tools to manually verify distortion
parameters. Common issues include using the wrong distortion model or providing distortion coefficients in the wrong
order.

![Corrupted frame](doc/images/troubleshooting_raw_vs_rect.png)

### Disable SLAM and ground constraint

First, verify that odometry works well. Only then enable SLAM. SLAM can reduce translation drift by roughly 1% using loop
closures, but it cannot correct broken trajectories when odometry has lost tracking.

1. You should obtain stable 2D tracking:
    - Keyframes should be rare and occur only when the scene changes significantly.
    - All selected features should lie on corners.
    - No features should die unreasonably. If the opposite occurs (features die without reason and trigger many
      keyframes), improve image quality. A quick option within cuVSLAM is to enable `use_denoising` to remove
      image noise. Choose the least noisy channel, and adjust camera exposure and frame rate.

      See [Step 8: Adjust odometry settings](#step-8-adjust-odometry-settings)

      Example of good 2D tracking:
      ![Good 2D tracking](doc/images/troubleshoot_sof.gif)

2. You should obtain good mono tracking. Mono cannot recover scene scale, so ignore the translation part of the pose and
   focus on rotation. Rotation should be accurate. Without ground truth, use the
   [shuttle-mode debug approach](#shuttle-mode-debug-approach) to verify. If 2D features look good but rotation is
   wrong, check the focal parameter (units, etc.).

## Step 6: Verify synchronization

Verify that left and right (or multi-camera) images are correctly synchronized in time and in topic
configuration. In rare cases, the left and right images from a stereo camera may not be synchronized for a given frame,
even when their timestamps are identical or very close. This is more likely in highly parallel processing environments
(e.g., ROS). A simple way to verify this is through visual inspection. Using VFX compositing software (e.g., the
open-source tool [Natron](https://natrongithub.github.io)), you can overlay the left and right frames at 50%
transparency. This makes it easy to spot frames where “double images” move out of sync.

Example: Left and right eyes out of sync

![Example: Left and right eyes out of sync](doc/images/troubleshooting_desync.gif)

## Step 7: Run stereo odometry tracking

If mono left and mono right both work, run stereo tracking. Stereo reuses extrinsic calibration via the
`left_from_right` matrix. A common issue is this matrix being expressed in the wrong coordinate frame. If both mono
pipelines work but stereo does not or produces much worse rotations, correct the extrinsic calibration. Many cameras are
nearly parallel. You can try supplying an identity rotation with the correct baseline instead of your calibration—if
results improve significantly, the issue is likely calibration.
Inspect the landmark point cloud to validate 3D structure. See [Debug visualization](#debug-visualization).

It is also important to properly set `rectified_stereo_camera` parameter. Set it to true **only** when you have
rectified, horizontally aligned images without lens distortion.

**C++ API**

```cpp
cuvslam::Odometry::Config::rectified_stereo_camera
```

**Python API**

```python
cuvslam.Odometry.Config.rectified_stereo_camera
```

See [Python API Reference](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.rectified_stereo_camera).

**Isaac ROS**

```
rectified_images
```

See [Isaac ROS cuVSLAM parameters][].

## Step 8: Adjust odometry settings

### Improve input images

1. Find the best resolution and FPS for your motion.
   Start with VGA (640x480) at 100 FPS if the camera and processing pipeline can sustain it without dropping frames.
   VGA is usually enough for most cases. Higher FPS usually helps more than higher resolution when the camera or robot
   moves quickly, because adjacent frames have smaller pixel displacement and less motion blur. Higher resolution can
   help in distant or low-texture scenes because it preserves more usable features. Test both on the same short,
   repeatable recording.

   Check which camera modes are native sensor readouts and which modes are resized from a native sensor resolution.
   Resized modes behave like an image filter: they can sometimes improve tracking quality by smoothing noise or aliasing,
   but they can also blur corners and remove details that cuVSLAM could otherwise track. Treat each non-native resolution
   as a separate image-processing mode and validate it instead of assuming that lower resolution is only a cheaper
   version of the same image.

   To compare modes:
   1. Start from the camera mode that can deliver frames without drops.
   2. Run the same sequence at several resolutions and frame rates.
   3. Compare tracking loss, drift against ground truth or a known path, number and distribution of observations in the
      debug view, CPU/GPU load, and any dropped-frame messages. If ground truth is not available, estimate drift with a
      repeated forward/backward recording, also known as [shuttle mode](#shuttle-mode-debug-approach).
   4. Prefer the lowest resolution that still gives stable, well-distributed features, and the highest FPS that your
      capture and processing pipeline can sustain without dropping frames.
   5. If you resize or crop images, make sure the intrinsics match the resized or cropped image.
2. Find good brightness, contrast, and gamma settings.
   Good settings for cuVSLAM are usually the same settings a person would choose when looking at the image. If the image
   looks too dark, make it brighter. If texture is washed out, reduce exposure or contrast. Inspect the grayscale images
   that cuVSLAM actually receives, not only the camera preview. Avoid clipped highlights, crushed shadows, aggressive
   sharpening, and strong gamma correction. A good image should preserve corners and texture in both bright and dark
   regions. Use the same image-processing path during testing and deployment so tuning results remain comparable.
3. Decide whether to use auto exposure.
   After the camera has been adjusted for the scene, first try fixed exposure and gain. Slow brightness changes are
   usually acceptable, but flashing lights or abrupt exposure jumps can break tracking. If auto exposure is required,
   limit its range where possible, let it settle before recording, and inspect the sequence for exposure "pumping."
4. Avoid compression and image-processing artifacts.
   Prefer raw or lossless image streams for troubleshooting. MJPEG, H.264, low bitrate recording, debayering artifacts,
   temporal denoising, and excessive sharpening can create unstable features that are easy to miss in a normal video
   player. Inspect the exact recorded frames frame by frame.
5. Pick the least noisy channel if you have a color image.
   cuVSLAM tracks grayscale features. If your camera or wrapper allows selecting the grayscale source, compare the
   individual color channels on the same recording. Pick the channel with the best combination of sharp texture, low
   noise, and low compression artifacts; do not assume that the default RGB-to-grayscale conversion is optimal.
6. Try to enable or disable the `use_denoising` parameter.
   Denoising can help with noisy low-light images, but it can also blur small features and increase processing cost.
   Compare tracking with denoising enabled and disabled on the same recording, and use the debug visualization to verify
   that useful observations become more stable rather than simply fewer.

**C++ API**

```cpp
cuvslam::Odometry::Config::use_denoising
```

**Python API**

```python
cuvslam.Odometry.Config.use_denoising
```

See [Python API Reference](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.use_denoising).

**Isaac ROS**

```
enable_image_denoising
```

See [Isaac ROS cuVSLAM parameters][].

### Pick the best odometry mode

Approximate accuracy ranking, least to most accurate:

Mono → RGBD → Stereo Inertial → Stereo → Multicamera; Multisensor depends on its sensor set

Multisensor (see [examples/multisensor/](examples/multisensor/README.md)) does not slot at a fixed
position — its accuracy scales with the sensor set it is given. It requires at least one RGB-D
camera or one overlapping camera pair, a cuNLS-enabled build, and currently supports pinhole cameras only. Official
release wheels include cuNLS. Multisensor is experimental and may be inaccurate or fail for some sensor configurations
and scenes. A single RGB-D + IMU rig is roughly RGBD-class. A multi-stereo + RGB-D + IMU rig may approach or exceed
Multicamera under favorable conditions, but this is an empirical expectation rather than a guarantee; benchmark the
actual rig and environment. Use Multisensor when you have a mixed-camera-type rig or want IMU fusion on a non-stereo
configuration; otherwise prefer the mode that exactly matches your rig.

### Adjust motion prediction

Ensure timestamps such as timestamp_ns are correct and consistent. The motion model only
enables prediction of 2D observations in UV space. If prediction for the current frame is not correct, cuVSLAM will
automatically re-track the frame without prediction. The main goal for prediction is to remove outliers caused by
repetitive structures in the image. Try disabling and enabling this model to see whether motion prediction matches your
motion.

**C++ API**

```cpp
cuvslam::Odometry::Config::use_motion_model
```

**Python API**

```python
cuvslam.Odometry.Config.use_motion_model
```

See [Python API Reference](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Odometry.Config.use_motion_model).

**Isaac ROS**

Currently, this parameter is not exposed in the Isaac ROS cuVSLAM node, so you need to update the node source code
yourself.

### Tune image area for feature selection

See [Step 4: Mask static image areas](#step-4-mask-static-image-areas)

- For raw images (with high lens distortion), try to cut up to ~10% of image area near the border.
- For stereo images, try to cut up to 10% from the left border of the left eye and up to 10% from the right border of
  the right eye to keep only the overlapped area.

### Tune L2R depth range

Applies to every odometry mode that tracks between overlapping camera pairs — Multicamera, Inertial, RGBD and
Multisensor. Ignored in Mono mode, which has no L2R stage. cuVSLAM samples scene depth along each epipolar curve
to seed initial guesses for the left-to-right (L2R) LK tracker. The sampled range must match the rig: too tight
and near-camera features are dropped, too wide and the candidate list inflates and LK converges on decoys.

Always set these to the actual near/far limits of your scene when you know them — the auto-detected defaults
are a fallback for when the scene is unknown, not a target to leave in place. Tight bounds around the real depth
range give shorter candidate lists, faster tracking, and fewer spurious matches.

Auto-detected defaults (used only if you leave the values negative) are derived from the rig baseline:

- Small stereo (~5–10 cm baseline, indoor / robot arm): `[0.1 m, 20 m]`.
- KITTI-scale outdoor (~0.5 m baseline): `[7 m, 1000 m]`.

Symptoms of a mismatched range: consistently low L2R success on near-camera or far-away features, or a quality
drop after widening the range too much.

**C++ API**

```cpp
cuvslam::Odometry::Config::min_depth  // meters; any negative value (e.g. -1) auto-detects
cuvslam::Odometry::Config::max_depth  // meters; any negative value (e.g. -1) auto-detects
```

## Step 9: IMU integration

The cuVSLAM implementation of IMU fusion does not add extra accuracy in scenarios where visual input works, and the IMU can
even slightly decrease accuracy. The main reason to add IMU is robustness since IMU may help when visual tracking is
blocked for a short period of time, for example, several seconds. So if you are not experiencing robustness issues,
consider keeping IMU disabled.

If you use an IMU, verify that timestamp alignment (e.g., timestamp_ns) is correct so that the IMU and image streams are
properly synchronized.

Another common issue is using the wrong IMU measurement frame or an incorrect `imu_from_rig` extrinsic matrix.

To troubleshoot this, cuVSLAM provides a special mode: `debug_imu_mode`. When enabled, visual tracking is disabled and
tracking relies only on the IMU. Translation is also locked, so only rotation is estimated.

### Debug IMU with debug_imu_mode

1. Connect a live camera with IMU.
2. Enable debug_imu_mode.
3. Rotate the camera around all axes (roll, pitch, yaw).
4. Verify that the reported tracking rotation matches the physical camera rotation.
   If the rotations match, IMU alignment/extrinsics are likely correct—disable `debug_imu_mode` and continue with
   normal visual-inertial tracking. If they do not match, recheck the IMU frame convention and the `imu_from_rig` matrix.

### Debug IMU with blackout_oscillator

A final step to verify IMU operation is to simulate a visual blackout. The [tracker](#tracker) applications provide
`blackout_oscillator` and `blackout_oscillator_duration` options for this purpose. Using the
[result-visualizer](#result-visualizer) tool, verify that switching from visual tracking to IMU-only tracking, and then
back to visual, occurs smoothly.

Debug IMU with the signal-loss oscillator.
![IMU debug](doc/images/troubleshooting_imu_oscillator.gif)

## Step 10: Multi-camera setups

Multi-camera mode dramatically improves robustness, but not necessarily accuracy. In fact, accuracy can sometimes
decrease slightly in multi-camera mode. Forward- and rear-facing cameras typically provide better accuracy than side- or
upward-facing cameras, so fusing all cameras together can occasionally yield a less accurate result.

1. Before enabling multi-camera mode, optimize and validate each stereo pair independently.
2. Ensure extrinsics and time synchronization are correct across all cameras.

## Step 11: Enable ground constraint

If your motion is predominantly planar, after stereo works correctly, enable the ground constraint and verify that
vertical (up/down) drift is eliminated or greatly reduced. There are two ground constraint methods in cuVSLAM:

1. Odometry ground constraint - postprocessing step
   **C++ API**
   ```cpp
   class cuvslam::GroundConstraint()
   ```
   **Python API**
   It is currently not exposed.

   **Isaac ROS**
   ```
   enable_ground_constraint_in_odometry
   ```
   See [Isaac ROS cuVSLAM parameters][].

2. SLAM ground constraint
   ```cpp
   cuvslam::Slam::Config::planar_constraints
   ```
   **Python API**
   [cuvslam.Slam.Config.planar_constraints](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.Slam.Config.planar_constraints)

   **Isaac ROS**
   ```
   enable_ground_constraint_in_slam
   ```
   See [Isaac ROS cuVSLAM parameters][].

## Step 12: Enable SLAM

Enable SLAM only when you have a solid odometry trajectory. SLAM can reduce drift by roughly 1% using loop-closure
search. Inspect the landmark point cloud and use `sync_mode`, `enable_reading_internals`, and
`planar_constraints` as needed for your scenario.

![VIO vs SLAM](doc/images/troubleshooting_vio_vs_slam.png)

## Step 13: Adjust SLAM map parameters

Tune SLAM parameters such as:

1. map_cell_size
2. max_landmarks_distance
3. max_map_size

to match your environment and accuracy requirements.

## Step 14: Run in async mode

For real-time or high-throughput operation, switch to a realtime `Tracker::Mode` and set the two
configuration fields to match. Adjust these parameters to balance latency and throughput:

1. `Tracker::Mode::OdometryOnlyRealtime` or `Tracker::Mode::OdometryWithSlamRealtime`
2. `Odometry::Config::async_sba` (must be `true` in a realtime mode)
3. `Slam::Config::sync_mode` (must be `false` in a realtime mode)
4. `Slam::Config::throttling_time_ms`

Items 3 and 4 are `Slam::Config` fields, so they only apply to `OdometryWithSlamRealtime`. In
`OdometryOnlyRealtime`, omit `slam_config` entirely.

### Verify SLAM keeps up with odometry

In async mode SLAM runs on a background thread that is fed a queue of commands: keyframes produced by odometry, map
localization and map saving requests. If SLAM cannot keep up, this queue grows and the poses and loop closures SLAM
reports refer to an increasingly old point of the trajectory.

To diagnose this, enable verbose logging and check the `delay_warning_queue_size` parameter. You’ll see a console
message whenever more commands are queued to the SLAM thread than the configured number:

```text
[WARNING] SLAM is behind odometry: XXX commands are queued to the SLAM thread that is more than desired YYY.
          Check SLAM settings: reduce max_map_size or increase throttling_time_ms.
```

Note that a single slow command—loading a large map, for example—delays every keyframe queued behind it, so the
queue length is what matters, not only the number of keyframes.

If you see this warning, reduce the SLAM workload: lower `max_map_size`, increase `throttling_time_ms` to make loop
closures less frequent, or set `map_cache_path` so a large map is kept on disk instead of in memory.

**C++ API**

```cpp
cuvslam::Slam::Config::delay_warning_queue_size
```

**Python API**

[cuvslam.core.Slam.Config.delay_warning_queue_size](https://nvidia-isaac.github.io/cuVSLAM/python/api.html#cuvslam.core.Slam.Config.delay_warning_queue_size)

## EDEX file

An EDEX file is the standard JSON “scene file” for cuVSLAM. It describes the camera rig, image (and optional depth)
sequences, and optional IMU data, and can also hold tracking output (poses and observations/landmarks).
[tracker](tools/tracker/README.md) reads an input EDEX with scene description only and outputs an EDEX with poses and
observations/landmarks.

Example of KITTI 01 input EDEX:

```
[{
   "version": "0.9",
   "frame_start": 1,
   "frame_end": 4541,
   "cameras": [
      {
        "intrinsics": {
          "distortion_model": "pinhole",
          "distortion_params": [],
          "focal": [718.855999999, 718.855999999],
          "principal": [607.192799999, 185.2157],
          "size": [1241, 376]
        },
        "transform": [ [ 1, 0, 0, 0 ],
                       [ 0, 1, 0, 0 ],
                       [ 0, 0, 1, 0 ]]
      },
      {
        "intrinsics": {
          "distortion_model": "pinhole",
          "distortion_params": [],
          "focal": [718.855999999, 718.855999999],
          "principal": [607.192799999, 185.2157],
          "size": [1241, 376]
        },
         "transform": [ [ 1, 0, 0, 0.5371657188644179 ],
                        [ 0, 1, 0, 0 ],
                        [ 0, 0, 1, 0 ]]
      }
    ]
  },
  {
    "fps": 10,
    "points2d": {},
    "points3d": {},
    "rig_positions": {},
    "sequence": [["00/00.0.0001.png"], ["01/00.1.0001.png"]]
  }
]
```

Use [result_visualizer](#result-visualizer) to visualize and compare trajectories from multiple [tracker](#tracker)
runs.

## Tracker

The tracker is a standalone command-line application for image-sequence tracking with real-time benchmarking and
debugging capabilities. See [tracker](tools/tracker/README.md). Set `--out_edex` to dump the result trajectory. Use
[result visualizer](#result-visualizer) to inspect and compare results.

It’s also the easiest way to dig deeper using the C++ debugger.

Also, there is a pure Python version of this tool:
[track_kitti.py](https://github.com/nvidia-isaac/cuVSLAM/blob/main/examples/kitti/track_kitti.py) script (adjust topics
and calibration as needed) to run PyCuVSLAM and rerun visualization.

## Result visualizer

The result_visualizer script is a Python tool that visualizes output [EDEX](#edex-file) files (tracking results) from
the [tracker](#tracker).

**Location:** `tools/edex/result_visualizer/result_visualizer.py`

**Usage:**

```shell
python result_visualizer.py <edex_result_file> [edex_result_file ...]
```

Result visualizer output:
![Result Visualizer](doc/images/troubleshooting_result_visualizer.png)

**Example (compare two runs):**

```shell
python result_visualizer.py run1.edex run2.edex
```

## Debug visualization

Enable visualization support as described in [Step 2](#step-2-create-a-reproducible-testing-environment).

![ReRun](doc/images/troubleshooting_rerun.png)

- **C++**: Add the `-DUSE_RERUN=True` CMake flag and rebuild the [tracker](#tracker).
- **Python API**: See [Examples](examples/kitti/track_kitti.py) for how to run [Rerun](https://rerun.io/) visualization.
- **Isaac ROS**: Use [Foxglove](https://foxglove.dev/product/visualization) or [RViz](https://wiki.ros.org/rviz) to
  inspect images, topics, the TF tree, and any cuVSLAM output.

  See [Isaac ROS cuVSLAM topics][] for published cuVSLAM topics.

## Shuttle-mode debug approach

Run the tracker forward and then backward over the same sequence several times. If the trajectory is consistent in both
directions, odometry is likely correct. Large discrepancies indicate drift or calibration issues.

<!-- Long links -->

[Isaac ROS cuVSLAM topics]: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html#ros-topics-published

[Isaac ROS cuVSLAM launch file]: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam/blob/main/isaac_ros_visual_slam/launch/isaac_ros_visual_slam_core.launch.py

[Isaac ROS cuVSLAM parameters]: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html#ros-parameters


If you follow the above process and still cannot identify the issue, consider opening a GitHub issue.

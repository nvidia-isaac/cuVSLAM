---
name: cuvslam-troubleshoot
description: >
  Troubleshoot NVIDIA cuVSLAM (CUDA Visual SLAM) pose accuracy, tracking, build, and
  integration issues. Use when asked to debug cuVSLAM tracking failures, pose drift,
  lost tracking, calibration problems, image quality issues, IMU integration, multi-camera
  setups, SLAM loop closure, or build/install problems with cuVSLAM, PyCuVSLAM, or
  Isaac ROS cuVSLAM. Triggers on: "cuVSLAM not tracking", "visual odometry drift",
  "SLAM lost", "cuvslam build error", "PyCuVSLAM install", "Isaac ROS visual_slam",
  "camera calibration issue", "stereo tracking bad", "cuVSLAM pose inaccurate".
---

# cuVSLAM Troubleshooting

Diagnose and fix issues with NVIDIA cuVSLAM — visual odometry and SLAM.

**Source repo:** https://github.com/nvidia-isaac/cuVSLAM
**Full troubleshooting guide:** `../../TROUBLESHOOTING.md` (read when investigating pose/tracking issues)
**Known limitations:** `references/cuvslam-limitations.md` — check this first if the scene has dynamic objects, reflective surfaces, IR projector patterns, line-dominated environments, or agile/drone motion

## Quick Triage

Classify the issue into one of these categories, then follow the relevant section:

1. **Build / Install** → [Build Issues](#build-issues)
2. **Tracking lost or inaccurate pose** → [Tracking Issues](#tracking-issues)
3. **Integration (ROS2, Python API, C++ API)** → [Integration Issues](#integration-issues)

## Build Issues

### Requirements
- Ubuntu 22+ (x86_64 or aarch64/Jetson)
- CUDA Toolkit 12 or 13
- `apt install g++ cmake git git-lfs python3-dev`
- CMake 3.19+

### Build from source
```bash
cmake -S . -B build && cmake --build build --parallel $(nproc)
```

### PyCuVSLAM install
Pre-built wheels: https://github.com/nvidia-isaac/cuVSLAM/releases/latest

- x86_64: Ubuntu 22.04 / Python 3.10 has `cu12` and `cu13`; Ubuntu 24.04 / Python 3.12+ has `cu12` and `cu13`.
- Jetson Orin: Ubuntu 22.04 / Python 3.10 / `cu12` only.
- Jetson Thor: Ubuntu 24.04 / Python 3.12+ / `cu13` only.

From source (after building C++ lib):
```bash
CUVSLAM_BUILD_DIR=<path-to-build> pip install python/
```

### Common build failures
- **Missing CUDA**: Set `CUDAToolkit_ROOT=/usr/local/cuda` or install CUDA Toolkit
- **git-lfs not installed**: `apt install git-lfs && git lfs pull` (binary test data)
- **CMake too old**: Need 3.19+
- **Wheel ABI mismatch**: Match the Python tag, CUDA major, architecture, and Jetson family using the matrix above.

## Tracking Issues

Complete the three phases below **in order**. During Phases 1 and 2, run only information-gathering commands such as `scripts/inspect_rosbag.py`; do not convert bags, replay/launch data, or start tracker/replication steps until Phase 3.

---

### Phase 1: Information gathering

Collect everything needed before touching any tools.

#### 1a. Configuration file

**Always ask for the user's config before proceeding** — do not assume default parameters.

The config file must contain the cuVSLAM parameters the user had active when they observed the issue. Depending on their integration, this is one of:

| Integration | Config format |
|---|---|
| Isaac ROS cuVSLAM | ROS 2 parameters YAML (passed via `ros2 launch … params_file:=…`) |
| C++ API | Source snippet or struct dump showing `CUVSLAM_Configuration` fields |
| Python API | Source snippet showing `Odometry.Config` / `Slam.Config` fields |

If the user has not provided a config, say:

> "To root cause this accurately I need the cuVSLAM configuration you were using. Please share the parameters file (for Isaac ROS) or the relevant `Odometry.Config` / `CUVSLAM_Configuration` settings you had set when the issue occurred."

Key parameters to note once you have the config:
- `tracking_mode` (Isaac ROS) / `OdometryMode` (Python) — `0`/`Multicamera` is VO-only, while
  `1`/`Inertial` requires and uses an IMU. `Multisensor` is different: it automatically enables IMU fusion only when
  `Rig.imus` is non-empty, so the mode value alone does not determine IMU use. Verify the rig's IMU list alongside
  cuNLS availability, `depth_camera_ids`, and pinhole calibration.
  Knowing this up front directs the diagnosis toward vision, depth, or IMU root causes.
- `rectified_images` (Isaac ROS) / `rectified_stereo_camera` (C++ API) — wrong value flips the entire stereo pipeline
- `image_jitter_threshold_ms` — tolerated timestamp jitter between left/right images; too tight a value causes frames to be dropped silently
- `Tracker.Mode` plus the `async_sba` / `sync_mode` that must match it — affects reproducibility
- `enable_slam` / loop closure settings — SLAM should only be tuned after odometry is solid
- `denoise_input_images`, `imu_from_left` — sensor-specific flags
- Any border/masking parameters — may inadvertently block features

#### 1b. Dataset inspection

Identify the dataset format, then inspect it to confirm all required data is present. **Read/inspect only — inspection commands such as `scripts/inspect_rosbag.py` are allowed, but do not convert bags, replay/launch data, or run tracker/replication steps yet.**

**Rosbag (`.bag` / `.db3` + `metadata.yaml`):**

Run the bag inspection script. Add `--vio` if the config has `tracking_mode: 1`:

```bash
source /opt/ros/<distro>/setup.bash
python3 scripts/inspect_rosbag.py <path/to/bag_folder>          # VO mode
python3 scripts/inspect_rosbag.py <path/to/bag_folder> --vio    # VIO mode
```

The script prints:
- `ros_distro`, duration, every topic with message count and Hz
- Required-topic check: stereo image topics (×2), camera_info topics (×2), IMU topic if `--vio`; warns on missing topics or left/right image count mismatch
- Camera intrinsics from the first `camera_info` messages (resolution, fx/fy/cx/cy, distortion model, zero-distortion = rectified)
- Stereo baseline derived from the right camera's P matrix (`baseline = −P[3] / P[0]`), with a sensor-model guess (D435i ≈ 50 mm, D455 ≈ 95 mm)
- All `/tf_static` transforms, with the IMU-related ones highlighted (needed for VIO extrinsic verification)
- Available TF frames (needed for `rig_frame` in the bag2edex config)

If any required topic is missing, report it to the user before proceeding.

**EDEX directory:**

Check that the following files/folders exist inside the EDEX directory:

| Required | File/folder | Notes |
|---|---|---|
| Always | `stereo.edex` | Camera intrinsics, extrinsics, IMU params, frame list |
| Always | `images/` directory | Must contain `cam0.NNNNN.png` (and `cam1.NNNNN.png` for stereo) |
| Always | `frame_metadata.jsonl` | Per-frame timestamp and filename list |
| If VIO | `IMU.jsonl` | One IMU sample per line |

Open `stereo.edex` and verify:
- `frame_start` / `frame_end` match the actual number of image files
- Camera intrinsics look sane (focal length matches resolution, no zeros)
- Baseline (translation X of the second camera's `transform`) is non-zero and plausible (typically 0.05–0.10 m for typical stereo rigs)
- `distortion_model` is `"pinhole"` for rectified images — a `"fisheye"` model with zero params is a misconfiguration; see `references/edex-calibration-pitfalls.md`
- If VIO: `imu` block exists with non-zero `gyro_noise_density`, `accel_noise_density`, and a valid `transform`

If the EDEX came from an `isaac_ros_visual_slam` debug dump, note this — it will need to be patched before the tracker can use it (see Phase 3).

#### 1c. Ground truth

Ask the user whether they have ground truth trajectory data:

**If yes** — ask them to provide it. The tracker tool accepts a ground truth file directly:
```bash
CUVSLAM_DATASETS=<parent>/ CUVSLAM_OUTPUT=/tmp/out/ \
  ./build/bin/tracker \
    -edex <sequence_subfolder> \
    -edex_filename stereo.edex \
    -output_edex result.edex \
    -gt_file gt.txt
```
Ground truth enables ATE/RTE metrics in the tracker output and makes drift quantitative rather than qualitative.

**If no** — that is fine. As a fallback, per-frame translation differences from the estimated trajectory can surface large jumps or drift trends. For Isaac ROS / rosbag workflows, use the reference script `scripts/vo_pose_diff_recorder.py` to record these from the `/visual_slam/tracking/vo_pose` topic:
```bash
python3 scripts/vo_pose_diff_recorder.py --output /tmp/vo_pose_diff.txt
```
The output CSV (`frame_id, timestamp_ns, translation_diff_m`) can be plotted or scanned for anomalous spikes that indicate where tracking degrades.

---

### Phase 2: Prerequisites for accurate tracking

With the config and dataset in hand, verify these before touching the diagnostic workflow:

1. **Accurate intrinsic calibration** — verify distortion model and params match the actual camera. See `references/edex-calibration-pitfalls.md` for common EDEX mistakes (e.g. `"fisheye"` with zero params ≠ `"pinhole"`).
2. **Accurate extrinsic calibration** (stereo/multi-cam) — verify baseline axis and magnitude match the physical rig. See `references/edex-calibration-pitfalls.md` for wrong-axis baseline diagnosis.
3. **Proper image synchronization** — left/right timestamps must match per frame.
4. **Sharp images, low noise** — frame rate matters more than resolution.
5. **Textured, rigid scenes** — avoid repetitive patterns and dynamic objects.
6. **Correct parameters and coordinate frames** — check `rectified_stereo_camera` flag.
7. **Sequential frames with small time deltas** — no dropped or missing frames.

---

### Phase 3: Diagnostic workflow

**Only start this phase after Phases 1 and 2 are complete.** This is where replication, bag conversion, and tracker runs happen.

Read `../../TROUBLESHOOTING.md` for the full 14-step diagnostic flow. The summary below uses the same step numbers as that document.

#### Dataset path into the diagnostic workflow

**Rosbag (`.bag` / `.db3`):**
- Do NOT run the tracker tool directly on a rosbag.
- First reproduce the issue through **Isaac ROS cuVSLAM** (`isaac_ros_visual_slam`).
- Read `references/isaac-ros-visual-slam-usage.md` for installation, launch steps, key parameters, and how to verify it is running correctly.
- Use `scripts/isaac_ros_visual_slam_realsense_bag.launch.py` as the reference launch file for RealSense bag playback (Isaac ROS 4.3 — adapt for other versions). Pass `enable_debug_mode:=true` to capture a debug dump (EDEX + images) for offline analysis.
- **Quick check for pose jumps:** Before capturing a full debug dump, replay the bag at a very slow rate (0.05) with visualization disabled and check if the jumps disappear. Monitor `/visual_slam/tracking/vo_pose` (e.g. with `scripts/vo_pose_diff_recorder.py`) to detect jumps without RViz overhead.
  - **Jumps gone at slow rate** → the node cannot keep up at real-time speed (frame processing overrun, jitter). The root cause is a performance or timing issue, not calibration or data quality. Next steps: check CPU/GPU utilization during playback, reduce image resolution or publish rate, or tune `throttling_time_ms`.
  - **Jumps persist at slow rate** → the issue is in the data, calibration, or configuration. Proceed with the debug dump workflow below.
- **Container note:** If running inside the Isaac ROS Docker container, skill scripts (e.g. `vo_pose_diff_recorder.py`) are not mounted inside the container. Copy them to the shared workspace mount before use:
  ```bash
  cp scripts/vo_pose_diff_recorder.py ~/workspaces/isaac_ros-dev/
  # Then reference them inside the container as /workspaces/isaac_ros-dev/vo_pose_diff_recorder.py
  ```
- **Debug dump ownership:** The dump directory is written by the container as root. To work with the dump files, either run commands via `docker exec` or write output to a host-owned directory.
- **Patch the debug dump EDEX before running tracker** — the dump produced by `isaac_ros_visual_slam` has format incompatibilities with the tracker CLI. See `commands/debug-dump-patch.md` for the full list and the patch script `scripts/patch_debug_dump_edex.py`.
- Once the EDEX dump is captured and patched, proceed with the tracker-based workflow below.

**EDEX file (`.edex` + image folder):**
- The sequence can be run directly with the `tracker` tool — no ROS step needed.
- If the EDEX came from an `isaac_ros_visual_slam` debug dump, patch it first — see `commands/debug-dump-patch.md`.
- See `commands/tracker.md` for the full flag reference, all tracking modes, and worked examples.
- Inspect the output trajectory with `result_visualizer`.

**Converting a rosbag to EDEX for offline tracker use:**
- Use `commands/bag2edex.md` — install, config, and usage for the `cuvslam-tools` package (`tools/python_tools/`).
- Before converting, run `rosbag_extract_urdf -r <bag> -o /tmp/urdf_out -d <distro>` (see `commands/bag2edex.md`) to find the correct `rig_frame` for extrinsic extraction. Set `ros_distribution` in the config to match the bag's ROS distro.


**Steps 1–2: Reproduce the issue**

Follow the `#### Dataset path into the diagnostic workflow` section above for the correct path:

- **Rosbag**: First run at slow rate (0.05) in `isaac_ros_visual_slam` to confirm the issue appears. If jumps are gone at slow rate, the root cause is a real-time performance problem — investigate CPU/GPU load and stop here. If jumps persist, capture a debug dump, patch the EDEX, and continue with the tracker below.
- **EDEX**: Run the tracker directly.

Once you have a runnable EDEX, run the tracker in the **same mode the user was using** (match their `tracking_mode`, SLAM on/off, etc.):

```bash
CUVSLAM_DATASETS=<parent>/ CUVSLAM_OUTPUT=/tmp/out/ \
  ./build/bin/tracker -edex <sequence> -edex_filename stereo.edex \
  -output_edex result.edex -mode <user_mode>
```

Confirm the issue appears in the output (jumps, drift, lost tracking). Use the jump-analysis snippet in `commands/tracker.md` to count and locate jumps in the result EDEX.

Do not proceed to Steps 3–14 until the issue is confirmed reproducible here.

**Step 3: Inspect images** — Check for rolling shutter artifacts, blur, noise, corrupted frames. Frame rate > resolution for cuVSLAM.

**Step 4: Mask static areas** — Crop robot body / static objects from camera view using border parameters.

**Step 5: Run mono tracking per eye** — Isolates extrinsic calibration issues. Focus on rotation accuracy.

**Step 6: Verify synchronization** — Overlay left/right at 50% transparency to spot desync.

**Step 7: Run stereo** — If mono works but stereo doesn't → extrinsic calibration issue. Check `left_from_right` coordinate frame.

**Step 8: Fine-tune** — Resolution/FPS, brightness/contrast, denoising, motion model, masking. After tuning, validate with shuttle mode (tracker forward then backward over the same sequence — consistent trajectory = good odometry; large discrepancy = remaining drift or calibration error). See `### Shuttle-mode validation` below.

**Step 9: IMU** — Only add IMU for robustness (not accuracy). If the user is running VIO and the issue may be IMU-related, re-run the tracker in VO-only mode (`-mode multicamera`) with the same EDEX and compare: if VO-only is significantly better, the IMU is the root cause. Then use `debug_imu_mode` to validate IMU alignment, and verify the IMU-to-camera extrinsic transform, noise density values, and gravity direction for your mounting orientation.

  **Frame rate and VIO (sub-topic of Step 9)** — As part of diagnosis, try throttling the image publish rate and observe whether tracking improves or degrades. Use `scripts/stereo_image_throttle.py` (a ROS 2 relay node) with the SLAM node remapped to the `*_throttled` topics. Set `image_jitter_threshold_ms` to at least `ceil(3000 / target_hz)` to absorb timer scheduling jitter (e.g. 50 ms for 60 Hz, 100 ms for 30 Hz). See `commands/stereo-image-throttle.md` for the full deployment guide and launch-file snippet.

  Frame rate interacts with VIO differently than VO: at 30 Hz the IMU integrates over 33 ms (vs 11 ms at 90 Hz), accumulating √3 more drift per step and receiving ~6–7 IMU messages per image interval instead of ~2–3. This means throttling can make VIO worse even if it improves VO. Measured on a D435i dataset: reducing 90→30 Hz in VIO mode turned 368 jumps (max 4.1 m) into 995 jumps (max 16.5 m), while VO at 30 Hz had only 1 jump (73 mm). If throttling degrades VIO specifically, switch to VO mode (`tracking_mode: 0`) instead of trying to find a safe throttle rate.

**Step 10: Multi-camera** — Validate each stereo pair independently first.

**Step 11: Ground constraint** — Enable for planar motion to reduce vertical drift.

**Step 12: Enable SLAM** — Only after odometry is solid. SLAM reduces drift ~1% via loop closure.

**Step 13: Tune SLAM** — Tune loop closure and pose graph parameters.

**Step 14: Async mode** — Enable for real-time. Tune `throttling_time_ms`.

### Debug data dump

To capture diagnostic data for offline analysis:

```cpp
// C++ — dumps EDEX + images
config.debug_dump_directory = "/tmp/cuvslam_dump";
```
```python
# Python
odometry_config.debug_dump_directory = "/tmp/cuvslam_dump"
```
```yaml
# Isaac ROS params
enable_debug_mode: true
debug_dump_path: "/tmp/cuvslam_dump"
```

Then use the `tracker` tool and `result_visualizer` for offline analysis. If the dump was captured from `isaac_ros_visual_slam`, patch the EDEX before running the tracker — see `commands/debug-dump-patch.md`.

### Shuttle-mode validation
Run tracker forward and backward over the same sequence. Consistent trajectory = good odometry. Large discrepancy = drift or calibration issue.


## Integration Issues

### Isaac ROS cuVSLAM

See `references/isaac-ros-visual-slam-usage.md` for the full guide: installation, launch commands, key parameters, topic remappings, debug dump setup, verification steps, and RealSense-specific notes.

**Global localization** — Isaac ROS cuVSLAM does not support global localization without an external pose hint. Use [Isaac ROS Visual Global Localization](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_mapping_and_localization/blob/release-4.3/isaac_ros_visual_global_localization) for this.

### Python API (PyCuVSLAM)
- Docs: https://nvidia-isaac.github.io/cuVSLAM/python/
- Rerun visualization: see `examples/kitti/track_kitti.py`

### C++ API
- Docs: https://nvidia-isaac.github.io/cuVSLAM/cpp/
- Rerun support: rebuild with `-DUSE_RERUN=True`

## Tools

- **tracker** — standalone CLI for image-sequence tracking. See `commands/tracker.md` for the full flag reference, all tracking modes, shuttle-mode validation, and output analysis. The output EDEX always stores only camera 0 intrinsics — having one camera entry is normal, not a sign of mono mode.
- **bag2edex** — convert a ROS 2 bag to an EDEX sequence for offline tracker use. Uses the `cuvslam-tools` package (`tools/python_tools/`). See `commands/bag2edex.md` for install, config, and usage.
- **result_visualizer** — visualize EDEX output trajectories (`tools/edex/result_visualizer/`)
- **undistort** — remove lens distortion from images (`tools/undistort/`)
- **cuvslam_api_launcher** — test utility for tracking + map save/load (`tools/cuvslam_api_launcher/`)

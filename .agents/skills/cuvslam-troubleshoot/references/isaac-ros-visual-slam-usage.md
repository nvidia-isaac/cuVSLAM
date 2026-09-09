# Running Isaac ROS Visual SLAM

Practical reference for installing, launching, and verifying `isaac_ros_visual_slam`.
Full docs: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html

---

## Installation

**Binary (recommended):**
```bash
isaac-ros activate
sudo apt-get install -y ros-jazzy-isaac-ros-visual-slam
```

**From source (release-4.3):**
```bash
cd ${ISAAC_ROS_WS}/src
git clone -b release-4.3 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
rosdep install --from-paths isaac_ros_visual_slam/isaac_ros_visual_slam --ignore-src -y
colcon build --symlink-install --packages-up-to isaac_ros_visual_slam
source install/setup.bash
```

---

## Launching

**Minimal launch (images must already be published):**
```bash
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py
```

**With a RealSense camera (live):**
```bash
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
    launch_fragments=realsense_stereo_rect,visual_slam \
    interface_specs_file=${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_visual_slam/quickstart_interface_specs.json \
    base_frame=camera_link \
    camera_optical_frames="['camera_infra1_optical_frame', 'camera_infra2_optical_frame']"
```

**With a rosbag (RealSense, release-4.3):**

See `scripts/isaac_ros_visual_slam_realsense_bag.launch.py` for a reference launch file.
Key points for bag playback:
- Pass `--clock` so nodes use bag time
- Delay bag playback ~2 s to let the SLAM node initialize
- Use a slow rate during debugging to avoid frame drops:
```bash
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense_bag.launch.py \
    bag_path:=/path/to/bag rate:=0.1
```

---

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `num_cameras` | `2` | Number of input cameras (max 32) |
| `min_num_images` | `num_cameras` | Minimum synchronized images required per frame |
| `rectified_images` | `true` | Set `false` if publishing raw/distorted images |
| `tracking_mode` | `0` | `0`=VO only, `1`=VIO (IMU), `2`=RGBD |
| `enable_localization_n_mapping` | `true` | `false` = odometry only, no loop closure |
| `enable_image_denoising` | `false` | Enable for low-light/noisy cameras |
| `enable_ground_constraint_in_slam` | `false` | Constrain to horizontal plane for wheeled robots |
| `image_jitter_threshold_ms` | — | Tolerated timestamp jitter between left/right images |
| `sync_matching_threshold_ms` | — | Tolerated timestamp offset across cameras for sync |
| `base_frame` | `base_link` | Robot base frame for TF |
| `camera_optical_frames` | — | List of camera optical frame names (must match TF tree) |
| `imu_frame` | `imu` | IMU frame (VIO mode only) |
| `save_map_folder_path` | `""` | Path to save SLAM map |
| `load_map_folder_path` | `""` | Path to load a saved SLAM map |

---

## Topics

**Inputs (must be remapped to match your sensor topics):**

| Topic | Type | Notes |
|---|---|---|
| `visual_slam/image_{i}` | `sensor_msgs/Image` | Grayscale image from camera i |
| `visual_slam/camera_info_{i}` | `sensor_msgs/CameraInfo` | Calibration for camera i |
| `visual_slam/imu` | `sensor_msgs/Imu` | Required for VIO mode |

**Key outputs:**

| Topic | Type | Notes |
|---|---|---|
| `visual_slam/tracking/odometry` | `nav_msgs/Odometry` | Main pose output |
| `visual_slam/tracking/vo_pose` | `geometry_msgs/PoseStamped` | Per-frame VO pose (published only when tracking succeeds) |
| `visual_slam/tracking/slam_path` | `nav_msgs/Path` | Full SLAM trajectory |
| `visual_slam/status` | `VisualSlamStatus` | Tracking state, useful for diagnostics |

---

## Capturing a Debug Dump

Enable before launching to capture an EDEX + image sequence for offline analysis with the `tracker` tool:

```yaml
# In your ROS 2 params file
enable_debug_mode: true
debug_dump_path: "/tmp/cuvslam_dump"
```

The dump directory can be passed directly to the `tracker` tool for reproducible offline runs without ROS.

---

## Verifying It Is Running Correctly

**Check image input rate** — low rate is the most common cause of tracking loss:
```bash
ros2 topic hz --window 100 /camera/infra1/image_rect_raw
```

**Check tracking status:**
```bash
ros2 topic echo /visual_slam/status
```

**Visualize in RViz:**
```bash
rviz2 -d $(ros2 pkg prefix isaac_ros_visual_slam --share)/rviz/default.cfg.rviz
```
If poses don't appear, verify the **Fixed Frame** in RViz matches the `map_frame` parameter (default: `map`).

**Record per-frame pose differences** (no ground truth needed) — see `scripts/vo_pose_diff_recorder.py`.

---

## RealSense-Specific Notes

- IR emitter must be **disabled** if white dots appear on nearby surfaces — they cause feature tracking errors and drift.
- Images must be published as **raw uncompressed** (not compressed or re-encoded).
- `realsense-ros` firmware version must match the [RealSense setup docs](https://nvidia-isaac-ros.github.io/getting_started/sensors/realsense_setup.html).

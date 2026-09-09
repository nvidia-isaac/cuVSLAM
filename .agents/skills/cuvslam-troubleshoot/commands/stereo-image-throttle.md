# Stereo Image Throttle

`scripts/stereo_image_throttle.py` is a ROS 2 relay node that caps a stereo image pair to a
configurable rate.  It is useful when the camera publishes faster than the downstream
system needs (e.g. 90 Hz camera, 60 Hz SLAM budget) or when bandwidth is limited.

---

## VIO and frame rate interaction

Throttling image rate affects VIO and VO differently — **try it as part of diagnosis** and observe whether tracking improves or degrades.

In VIO mode the IMU bridges the gap between camera frames.  At 90 Hz the
IMU integrates for ~11 ms per interval (~2–3 IMU messages at 200 Hz); at 30 Hz
that grows to ~33 ms (~6–7 IMU messages) — a 3× longer interval and √3× more
accumulated IMU noise per step.

The table below was captured with a pre-fix VIO build (IMU `predict_pose` ignored
preintegration, `prior_acc` was 1e10 — see Step 9 in SKILL.md).  Absolute numbers
will improve with the fixed build, but the frame-rate interaction remains:

| Frame rate | Mode | Jumps > 5 cm | Max jump |
|---|---|---|---|
| 90 Hz | VIO live 1× (pre-fix) | 368 | 4.1 m |
| **30 Hz** | **VIO live 1× (pre-fix)** | **995** | **16.5 m** |
| 90 Hz | VO (no IMU) | 1 | 73 mm |

If throttling degrades VIO specifically, the IMU integration interval is the
likely cause.  Switch to **VO mode (`tracking_mode: 0`)** rather than searching
for a safe throttle rate — VO at 30 Hz is viable; VIO at 30 Hz may not be.

---

## image_jitter_threshold_ms

The timer-based relay occasionally fires ~one base-camera-period late,
creating inter-frame gaps larger than the nominal throttled period.
Set the SLAM node's `image_jitter_threshold_ms` to at least **ceil(3 × frame_period_ms)**,
i.e. `ceil(3000 / target_hz)`:

| target_hz | frame_period | recommended threshold (`ceil(3000 / hz)`) |
|---|---|---|
| 90 Hz | 11 ms | 34 ms |
| 60 Hz | 17 ms | 50 ms |
| 30 Hz | 33 ms | 100 ms |

`isaac_ros_visual_slam_realsense_bag_vio_throttled.launch.py` computes this automatically
as `ceil(3000 / target_hz)`.

---

## Deploying inside Isaac ROS container

The script must be reachable inside the Docker container.  It is installed
alongside `vo_pose_diff_recorder.py` in the isaac_ros_visual_slam package:

```bash
# From the host — copy once after cloning/updating the skill
cp scripts/stereo_image_throttle.py \
   ~/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam/share/isaac_ros_visual_slam/launch/
```

Inside the container it is then at:
```text
/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam/share/isaac_ros_visual_slam/launch/stereo_image_throttle.py
```

---

## Adding to a launch file

```python
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

_THROTTLE_SCRIPT = (
    '/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam/'
    'share/isaac_ros_visual_slam/launch/stereo_image_throttle.py'
)

# 1. Throttle node — run before bag playback starts
throttle_node = ExecuteProcess(
    cmd=[
        'python3', _THROTTLE_SCRIPT,
        '--ros-args',
        '-p', 'target_hz:=60.0',
        '-p', 'input_left:=/camera/infra1/image_rect_raw',
        '-p', 'input_right:=/camera/infra2/image_rect_raw',
    ],
    output='screen',
)

# 2. SLAM node — remap to throttled topics, set threshold to 3 × frame_period
visual_slam_node = ComposableNode(
    name='visual_slam_node',
    package='isaac_ros_visual_slam',
    plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
    parameters=[{
        'use_sim_time': True,
        'rectified_images': True,
        'tracking_mode': 1,              # VIO — keep ≥60 Hz
        'image_jitter_threshold_ms': 50.0,  # ceil(3000 / 60) = 50 ms for 60 Hz
        # ... other params ...
    }],
    remappings=[
        ('visual_slam/image_0',       '/camera/infra1/image_rect_raw_throttled'),
        ('visual_slam/camera_info_0', '/camera/infra1/camera_info'),
        ('visual_slam/image_1',       '/camera/infra2/image_rect_raw_throttled'),
        ('visual_slam/camera_info_1', '/camera/infra2/camera_info'),
        ('visual_slam/imu',           '/camera/imu'),
    ],
)
```

Camera info topics do not need throttling — the SLAM node matches them to images
by timestamp and the calibration data is effectively static.

---

## Reference launch file

`scripts/isaac_ros_visual_slam_realsense_bag.launch.py` is the reference bag-replay
launch file.  The 30Hz VIO variant used during the cuvslam_docking_img investigation
is at `~/workspaces/isaac_ros-dev/isaac_ros_visual_slam_realsense_bag_30hz_vio.launch.py`
(host workspace) — adapt from there for other rates and modes.

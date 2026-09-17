# REFERENCE SCRIPT — Isaac ROS release 4.3
# This file is for reference only.
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""
Launch Isaac ROS Visual SLAM (VIO) with stereo images throttled to target_hz.

Usage:
    ros2 launch <path>/isaac_ros_visual_slam_realsense_bag_vio_throttled.launch.py
    ros2 launch <path>/isaac_ros_visual_slam_realsense_bag_vio_throttled.launch.py \\
        target_hz:=60.0 bag_path:=/path/to/bag rate:=1.0

Arguments:
    target_hz   float  Throttled image rate in Hz (default 60.0).
                       ⚠ Do NOT set below 60 Hz for VIO — see Step 8a in SKILL.md.
    bag_path    str    Path to the ROS2 bag directory.
    rate        float  Bag playback rate (default 1.0).
    pose_diff_file str Output CSV for per-frame translation diffs.

image_jitter_threshold_ms is computed automatically as ceil(3 × frame_period_ms).
"""
import math
import os
import launch
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))


def _launch_setup(context, *args, **kwargs):
    target_hz   = float(LaunchConfiguration('target_hz').perform(context))
    bag_path    = LaunchConfiguration('bag_path').perform(context)
    rate        = LaunchConfiguration('rate').perform(context)
    pose_diff_file = LaunchConfiguration('pose_diff_file').perform(context)

    # image_jitter_threshold = 3 × frame_period to absorb timer scheduling jitter
    threshold_ms = math.ceil(3000.0 / target_hz)

    if target_hz < 60.0:
        import logging
        logging.getLogger('launch').warning(
            f'target_hz={target_hz:.0f} Hz is below 60 Hz. '
            f'VIO IMU integration interval grows to {1000.0/target_hz:.0f} ms — '
            'monitor for increased jumps and consider tracking_mode:=0 (VO only) if it degrades.'
        )

    # ---------- 30/60 Hz stereo throttle ----------
    throttle_node = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(_LAUNCH_DIR, 'stereo_image_throttle.py'),
            '--ros-args',
            '-p', f'target_hz:={target_hz}',
        ],
        output='screen',
    )

    # ---------- vo_pose diff recorder ----------
    pose_diff_recorder = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(_LAUNCH_DIR, 'vo_pose_diff_recorder.py'),
            '--output', pose_diff_file,
        ],
        output='screen',
    )

    # ---------- bag playback (delayed to let SLAM + throttle subscribe) ----------
    # Remap /tf from the bag to avoid TF_OLD_DATA spam: the bag contains pre-recorded
    # odometry transforms (odom → base_link) that conflict with the SLAM node's live TF
    # output.  /tf_static is kept so camera-frame definitions still reach the SLAM node.
    bag_play = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play', bag_path,
                    '--clock',
                    '--rate', rate,
                    '--remap', '/tf:=/tf_bag',
                ],
                output='screen',
            )
        ],
    )

    # ---------- visual SLAM node (VIO, throttled image topics) ----------
    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'use_sim_time': True,
            'rectified_images': True,
            'enable_image_denoising': False,
            'num_cameras': 2,
            'camera_optical_frames': [
                'camera_infra1_optical_frame',
                'camera_infra2_optical_frame',
            ],
            'base_frame': 'camera_link',
            # VIO mode — D435i BMI055 noise parameters
            'tracking_mode': 1,
            'imu_frame': 'camera_gyro_optical_frame',
            'gyro_noise_density': 0.000244,
            'gyro_random_walk': 0.000019393,
            'accel_noise_density': 0.001862,
            'accel_random_walk': 0.003,
            'calibration_frequency': 200.0,
            # Jitter threshold: 3 × frame period, computed from target_hz
            'image_jitter_threshold_ms': float(threshold_ms),
            'enable_debug_mode': False,
            'enable_slam_visualization': False,
            'enable_landmarks_view': False,
            'enable_observations_view': False,
        }],
        remappings=[
            # Subscribe to the throttled image stream
            ('visual_slam/image_0',       '/camera/infra1/image_rect_raw_throttled'),
            ('visual_slam/camera_info_0', '/camera/infra1/camera_info'),
            ('visual_slam/image_1',       '/camera/infra2/image_rect_raw_throttled'),
            ('visual_slam/camera_info_1', '/camera/infra2/camera_info'),
            ('visual_slam/imu',           '/camera/imu'),
        ],
    )

    visual_slam_container = ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    )

    return [visual_slam_container, throttle_node, pose_diff_recorder, bag_play]


def generate_launch_description():
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'target_hz', default_value='60.0',
            description='Throttled image rate (Hz). Keep ≥60 for VIO.',
        ),
        DeclareLaunchArgument(
            'bag_path',
            default_value='/workspaces/isaac_ros-dev/isaac_ros_assets/cuvslam_docking_img',
            description='Path to the ROS2 bag directory to replay',
        ),
        DeclareLaunchArgument(
            'rate', default_value='1.0',
            description='Bag playback rate multiplier',
        ),
        DeclareLaunchArgument(
            'pose_diff_file',
            default_value='/workspaces/isaac_ros-dev/isaac_ros_assets/cuvslam_docking_img/vo_pose_diff_vio_throttled.txt',
            description='Output CSV for per-frame translation diffs',
        ),
        OpaqueFunction(function=_launch_setup),
    ])

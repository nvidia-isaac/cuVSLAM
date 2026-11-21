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
Launch Isaac ROS Visual SLAM replaying a RealSense rosbag.

Usage:
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense_bag.launch.py
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense_bag.launch.py \
        bag_path:=/path/to/bag rate:=0.5 tracking_mode:=0
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense_bag.launch.py \
        enable_debug_mode:=true debug_dump_path:=/tmp/cuvslam_dump
"""
import os
import launch
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    # ---------- launch arguments ----------
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value='/workspaces/isaac_ros-dev/'
                      'isaac_ros_assets/cuvslam_docking_img',
        description='Path to the ROS2 bag directory to replay',
    )
    rate_arg = DeclareLaunchArgument(
        'rate',
        default_value='1.0',
        description='Bag playback rate multiplier (use 0.1 for slow debug playback)',
    )
    tracking_mode_arg = DeclareLaunchArgument(
        'tracking_mode',
        default_value='1',
        description='0=multicamera (VO only), 1=VIO (IMU fusion)',
    )
    enable_slam_arg = DeclareLaunchArgument(
        'enable_slam_visualization',
        default_value='True',
    )
    enable_debug_mode_arg = DeclareLaunchArgument(
        'enable_debug_mode',
        default_value='False',
        description='Dump EDEX + images for offline tracker analysis',
    )
    debug_dump_path_arg = DeclareLaunchArgument(
        'debug_dump_path',
        default_value='/tmp/cuvslam_dump',
        description='Directory for the debug dump (used when enable_debug_mode=true)',
    )
    pose_diff_file_arg = DeclareLaunchArgument(
        'pose_diff_file',
        default_value='/workspaces/isaac_ros-dev/isaac_ros_assets/cuvslam_docking_img/vo_pose_diff.txt',
        description='Output path for per-frame vo_pose translation diff CSV',
    )

    # ---------- bag playback ----------
    # --clock  : publish /clock so nodes use bag time
    # --delay  : give the SLAM node 2 s to initialize before messages arrive
    bag_play = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path'),
                    '--clock',
                    '--rate', LaunchConfiguration('rate'),
                ],
                output='screen',
            )
        ],
    )

    # ---------- vo_pose diff recorder ----------
    recorder_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'vo_pose_diff_recorder.py',
    )
    pose_diff_recorder = ExecuteProcess(
        cmd=[
            'python3', recorder_script,
            '--output', LaunchConfiguration('pose_diff_file'),
        ],
        output='screen',
    )

    # ---------- visual SLAM node ----------
    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            # Use /clock published by bag play
            'use_sim_time': True,

            # Images from the bag are already rectified
            'rectified_images': True,
            'enable_image_denoising': False,

            # Stereo camera frames (from RealSense TF tree in the bag)
            'num_cameras': 2,
            'camera_optical_frames': [
                'camera_infra1_optical_frame',
                'camera_infra2_optical_frame',
            ],
            'base_frame': 'camera_link',

            # IMU — D455 BMI088 noise parameters (empirically calibrated)
            'tracking_mode': LaunchConfiguration('tracking_mode'),
            'imu_frame': 'camera_gyro_optical_frame',
            'gyro_noise_density': 0.00017221024330235052,
            'gyro_random_walk': 9.595089650939567e-06,
            'accel_noise_density': 0.0014855959546680947,
            'accel_random_walk': 0.0001219997738886554,
            'calibration_frequency': 200.0,

            # Sync tolerance — 3× frame period at 90 fps (11.1 ms × 3 = 33.3 ms)
            'image_jitter_threshold_ms': 35.0,

            # Debug dump — set enable_debug_mode:=true to capture EDEX + images
            'enable_debug_mode': LaunchConfiguration('enable_debug_mode'),
            'debug_dump_path': LaunchConfiguration('debug_dump_path'),

            # Visualization
            'enable_slam_visualization': LaunchConfiguration('enable_slam_visualization'),
            'enable_landmarks_view': True,
            'enable_observations_view': True,
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', '/camera/infra1/camera_info'),
            ('visual_slam/image_1', '/camera/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', '/camera/infra2/camera_info'),
            ('visual_slam/imu', '/camera/imu'),
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

    return launch.LaunchDescription([
        bag_path_arg,
        rate_arg,
        tracking_mode_arg,
        enable_slam_arg,
        enable_debug_mode_arg,
        debug_dump_path_arg,
        pose_diff_file_arg,
        visual_slam_container,
        pose_diff_recorder,
        bag_play,
    ])

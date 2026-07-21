"""Bring up D435i rectified stereo and a time-aligned PX4 IMU for cuVSLAM."""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.events import Shutdown
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from isaac_ros_yopo_bringup.calibration import (
    assert_runtime_calibration_allowed,
    assert_runtime_imu_noise_allowed,
    load_calibration,
    load_imu_noise,
)


PACKAGE_NAME = "isaac_ros_yopo_bringup"
EXPECTED_VISUAL_SLAM_PREFIX = "/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam"
VISUAL_SLAM_PATCH_MARKER = b"ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1"


def _shutdown_if_process_exits(action, label: str) -> RegisterEventHandler:
    return RegisterEventHandler(
        OnProcessExit(
            target_action=action,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason=f"required process exited: {label}"),
                )
            ],
        )
    )


def _build_runtime_actions(context):
    calibration_path = LaunchConfiguration("calibration_file").perform(context)
    calibration = load_calibration(calibration_path)
    try:
        assert_runtime_calibration_allowed(calibration)
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    visual_slam_prefix = get_package_prefix("isaac_ros_visual_slam")
    if visual_slam_prefix != EXPECTED_VISUAL_SLAM_PREFIX:
        raise RuntimeError(
            "isaac_ros_visual_slam must resolve to the patched workspace overlay; "
            f"expected {EXPECTED_VISUAL_SLAM_PREFIX}, got {visual_slam_prefix}"
        )
    visual_slam_library = os.path.join(
        visual_slam_prefix,
        "lib",
        "libvisual_slam_node.so",
    )
    try:
        with open(visual_slam_library, "rb") as stream:
            patched_binary = VISUAL_SLAM_PATCH_MARKER in stream.read()
    except OSError as error:
        raise RuntimeError(
            f"cannot read installed Visual SLAM library: {visual_slam_library}"
        ) from error
    if not patched_binary:
        raise RuntimeError(
            "installed libvisual_slam_node.so does not contain the required "
            "IMU timestamp patch marker; rebuild isaac_ros_visual_slam"
        )

    noise_file = LaunchConfiguration("imu_noise_file").perform(context)
    noise = load_imu_noise(
        noise_file,
        calibration.imu_hardware_id,
        calibration.imu_rate_hz,
    )
    try:
        assert_runtime_imu_noise_allowed(noise)
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    camera_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace="camera",
        output="screen",
        parameters=[{
            "serial_no": calibration.camera_serial,
            "enable_infra1": True,
            "enable_infra2": True,
            "enable_color": False,
            "enable_depth": False,
            "depth_module.emitter_enabled": 0,
            "depth_module.profile": "640x360x90",
            "enable_gyro": False,
            "enable_accel": False,
            "unite_imu_method": 0,
            "publish_tf": True,
        }],
    )

    aligned_imu_relay = Node(
        package=PACKAGE_NAME,
        executable="aligned_imu_relay",
        name="aligned_fcu_imu_relay",
        output="screen",
        parameters=[{
            "input_topic": calibration.imu_input_topic,
            "output_topic": calibration.imu_output_topic,
            "expected_input_frame_id": calibration.imu_source_frame,
            "output_frame_id": calibration.imu_runtime_frame,
            "imu_to_camera_offset_ns": calibration.imu_to_camera_offset_ns,
            "expected_rate_hz": calibration.imu_rate_hz,
            "rate_tolerance_ratio": 0.15,
            "maximum_gap_ratio": 5.0,
            "diagnostic_period_sec": 1.0,
            "stale_after_sec": 2.0,
            "startup_timeout_sec": 15.0,
            "maximum_receipt_time_residual_sec": 0.25,
            "hardware_id": calibration.imu_hardware_id,
        }],
    )

    translation = calibration.tf_translation_m
    quaternion = calibration.tf_rotation_xyzw
    calibrated_imu_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="calibrated_fcu_imu_static_transform",
        output="screen",
        arguments=[
            "--x", f"{translation[0]:.17g}",
            "--y", f"{translation[1]:.17g}",
            "--z", f"{translation[2]:.17g}",
            "--qx", f"{quaternion[0]:.17g}",
            "--qy", f"{quaternion[1]:.17g}",
            "--qz", f"{quaternion[2]:.17g}",
            "--qw", f"{quaternion[3]:.17g}",
            "--frame-id", calibration.tf_parent_frame,
            "--child-frame-id", calibration.tf_child_frame,
        ],
    )

    visual_slam_component = ComposableNode(
        package="isaac_ros_visual_slam",
        plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
        name="visual_slam_node",
        parameters=[{
            "enable_image_denoising": False,
            "rectified_images": True,
            "num_cameras": 2,
            "enable_imu_fusion": True,
            "gyro_noise_density": noise.gyroscope_noise_density,
            "gyro_random_walk": noise.gyroscope_random_walk,
            "accel_noise_density": noise.accelerometer_noise_density,
            "accel_random_walk": noise.accelerometer_random_walk,
            "calibration_frequency": calibration.imu_rate_hz,
            "image_jitter_threshold_ms": 22.0,
            "imu_jitter_threshold_ms": 10.0,
            "base_frame": "camera_link",
            "imu_frame": calibration.imu_runtime_frame,
            "camera_optical_frames": [
                calibration.left_camera_frame,
                calibration.right_camera_frame,
            ],
            "enable_ground_constraint_in_odometry": False,
            "enable_ground_constraint_in_slam": False,
            "enable_localization_n_mapping": False,
            "enable_slam_visualization": False,
            "enable_landmarks_view": False,
            "enable_observations_view": False,
            "publish_map_to_odom_tf": False,
        }],
        remappings=[
            ("visual_slam/image_0", "/camera/infra1/image_rect_raw"),
            ("visual_slam/camera_info_0", "/camera/infra1/camera_info"),
            ("visual_slam/image_1", "/camera/infra2/image_rect_raw"),
            ("visual_slam/camera_info_1", "/camera/infra2/camera_info"),
            ("visual_slam/imu", calibration.imu_output_topic),
        ],
    )
    visual_slam_container = ComposableNodeContainer(
        name="visual_slam_launch_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[visual_slam_component],
        output="screen",
    )
    runtime_health_monitor = Node(
        package=PACKAGE_NAME,
        executable="runtime_health_monitor",
        name="d435i_cuvslam_runtime_health_monitor",
        output="screen",
        parameters=[{
            "calibration_file": calibration_path,
            "startup_timeout_sec": 35.0,
            "camera_info_stale_after_sec": 2.0,
            "odometry_stale_after_sec": 2.0,
            "expected_odometry_frame_id": "odom",
            "expected_odometry_child_frame_id": "camera_link",
        }],
    )

    initialization_state = {"ready": False, "output_buffer": ""}
    marker_text = VISUAL_SLAM_PATCH_MARKER.decode("ascii")

    def on_visual_slam_output(event):
        text = event.text
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        if initialization_state["ready"]:
            return []
        initialization_state["output_buffer"] += str(text)
        if marker_text not in initialization_state["output_buffer"]:
            initialization_state["output_buffer"] = initialization_state[
                "output_buffer"
            ][-2 * len(marker_text):]
            return []
        initialization_state["ready"] = True
        return [
            LogInfo(
                msg=(
                    "[PASS] Patched cuVSLAM tracker was constructed; "
                    "continuing runtime health and odometry checks."
                )
            )
        ]

    def on_visual_slam_initialization_timeout(_context):
        if initialization_state["ready"]:
            return []
        get_logger(PACKAGE_NAME).error(
            "[STOP] Patched cuVSLAM tracker did not initialize within 30 seconds."
        )
        return [
            EmitEvent(
                event=Shutdown(reason="cuVSLAM initialization readiness timeout"),
            )
        ]

    initialization_handler = RegisterEventHandler(
        OnProcessIO(
            target_action=visual_slam_container,
            on_stdout=on_visual_slam_output,
            on_stderr=on_visual_slam_output,
        )
    )
    initialization_timeout = TimerAction(
        period=30.0,
        actions=[OpaqueFunction(function=on_visual_slam_initialization_timeout)],
    )

    messages = [
        LogInfo(msg=f"Calibration: {calibration.calibration_id}"),
        LogInfo(msg=f"Calibration status: {calibration.status}"),
        LogInfo(
            msg=(
                f"Expected D435i serial={calibration.camera_serial}, "
                f"firmware={calibration.camera_firmware}"
            )
        ),
        LogInfo(
            msg=(
                "IMU alignment: raw stamp + "
                f"{calibration.imu_to_camera_offset_ns} ns; "
                f"{calibration.tf_parent_frame} -> {calibration.tf_child_frame}"
            )
        ),
        LogInfo(
            msg=(
                f"IMU noise: {noise.calibration_id}; "
                f"project_status={noise.project_status}; method={noise.method}; "
                f"source={noise.source_artifact}; "
                f"allan_validated={noise.validated}"
            )
        ),
        LogInfo(
            msg=(
                "Operating mode: odometry-only; mapping, loop closure, ground "
                "constraints, internal visualization, and map->odom TF are disabled"
            )
        ),
    ]
    if not noise.validated:
        messages.append(
            LogInfo(
                msg=(
                    "IMU noise provenance: project-approved Kalibr input weights; "
                    "independent Allan validation is optional and is not claimed"
                )
            )
        )

    required_processes = (
        (camera_node, "RealSense D435i"),
        (aligned_imu_relay, "aligned FCU IMU relay"),
        (calibrated_imu_tf, "calibrated FCU IMU static TF"),
        (visual_slam_container, "cuVSLAM component container"),
        (runtime_health_monitor, "calibrated runtime health monitor"),
    )
    process_exit_handlers = [
        _shutdown_if_process_exits(action, label)
        for action, label in required_processes
    ]
    actions = messages + process_exit_handlers + [
        initialization_handler,
        camera_node,
        aligned_imu_relay,
        calibrated_imu_tf,
        visual_slam_container,
        runtime_health_monitor,
        initialization_timeout,
    ]
    return actions


def generate_launch_description() -> LaunchDescription:
    default_calibration = str(
        os.path.join(
            get_package_share_directory(PACKAGE_NAME),
            "config",
            "d435i_243622070369_fcu_imu.yaml",
        )
    )
    default_imu_noise = str(
        os.path.join(
            get_package_share_directory(PACKAGE_NAME),
            "config",
            "px4_imu_noise_unvalidated.yaml",
        )
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "calibration_file",
            default_value=default_calibration,
            description="Version-controlled factory-rectified camera/FCU IMU calibration.",
        ),
        DeclareLaunchArgument(
            "imu_noise_file",
            default_value=default_imu_noise,
            description=(
                "Version-controlled PX4 IMU noise model with independent project "
                "approval and Allan provenance status."
            ),
        ),
        OpaqueFunction(function=_build_runtime_actions),
    ])

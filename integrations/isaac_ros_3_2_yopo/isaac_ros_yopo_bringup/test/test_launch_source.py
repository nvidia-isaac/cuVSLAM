import ast
import os
import unittest


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH_PATH = os.path.join(PACKAGE_ROOT, "launch", "d435i_fcu_imu_cuvslam.launch.py")


class LaunchSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(LAUNCH_PATH, "r", encoding="utf-8") as stream:
            cls.source = stream.read()
        cls.tree = ast.parse(cls.source)

    def test_disables_d435i_imu(self):
        self.assertIn('"enable_gyro": False', self.source)
        self.assertIn('"enable_accel": False', self.source)
        self.assertIn('"unite_imu_method": 0', self.source)
        self.assertNotIn('"/camera/imu"', self.source)

    def test_uses_official_visual_slam_component(self):
        self.assertIn(
            'plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode"',
            self.source,
        )
        self.assertIn('"enable_imu_fusion": True', self.source)
        self.assertIn('"rectified_images": True', self.source)

    def test_explicitly_overrides_both_camera_frames(self):
        self.assertIn('"camera_optical_frames"', self.source)
        self.assertIn("calibration.left_camera_frame", self.source)
        self.assertIn("calibration.right_camera_frame", self.source)

    def test_visual_slam_consumes_only_aligned_imu(self):
        self.assertIn(
            '("visual_slam/imu", calibration.imu_output_topic)',
            self.source,
        )
        self.assertNotIn(
            '("visual_slam/imu", calibration.imu_input_topic)',
            self.source,
        )

    def test_noise_file_has_a_versioned_default_and_cli_values_are_not_accepted(self):
        required = {"imu_noise_file"}
        declarations = {}
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = getattr(node.func, "id", "")
            if function_name != "DeclareLaunchArgument" or not node.args:
                continue
            if not isinstance(node.args[0], ast.Constant):
                continue
            declarations[node.args[0].value] = {
                keyword.arg for keyword in node.keywords
            }
        self.assertTrue(required.issubset(declarations))
        for name in required:
            self.assertIn("default_value", declarations[name])
        self.assertIn('"px4_imu_noise_unvalidated.yaml"', self.source)
        for forbidden in (
            'DeclareLaunchArgument(\n            "gyro_noise_density"',
            'DeclareLaunchArgument(\n            "gyro_random_walk"',
            'DeclareLaunchArgument(\n            "accel_noise_density"',
            'DeclareLaunchArgument(\n            "accel_random_walk"',
        ):
            self.assertNotIn(forbidden, self.source)

    def test_requires_patched_workspace_overlay(self):
        self.assertIn(
            "/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam",
            self.source,
        )
        self.assertIn("ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1", self.source)
        self.assertIn("libvisual_slam_node.so", self.source)

    def test_requires_successful_tracker_initialization(self):
        self.assertIn("OnProcessIO", self.source)
        self.assertIn("initialization_timeout = TimerAction", self.source)
        self.assertIn("Patched cuVSLAM tracker was constructed", self.source)

    def test_runtime_approval_is_versioned_without_cli_overrides(self):
        self.assertIn("assert_runtime_calibration_allowed", self.source)
        self.assertIn("assert_runtime_imu_noise_allowed", self.source)
        self.assertNotIn('"allow_candidate_calibration"', self.source)
        self.assertNotIn('"allow_unvalidated_imu_noise"', self.source)

    def test_production_launch_is_fixed_to_odometry_only(self):
        for parameter in (
            "enable_ground_constraint_in_odometry",
            "enable_ground_constraint_in_slam",
            "enable_localization_n_mapping",
            "enable_slam_visualization",
            "enable_landmarks_view",
            "enable_observations_view",
            "publish_map_to_odom_tf",
        ):
            self.assertIn(f'"{parameter}": False', self.source)
        self.assertNotIn('LaunchConfiguration("enable_visualization")', self.source)
        self.assertNotIn('DeclareLaunchArgument(\n            "enable_visualization"', self.source)

    def test_runtime_monitor_is_required(self):
        self.assertIn('executable="runtime_health_monitor"', self.source)
        self.assertIn(
            '(runtime_health_monitor, "calibrated runtime health monitor")',
            self.source,
        )
        self.assertIn('"camera_info_stale_after_sec": 2.0', self.source)
        self.assertIn('"odometry_stale_after_sec": 2.0', self.source)

    def test_gap_threshold_has_margin_over_the_recorded_maximum(self):
        self.assertIn('"maximum_gap_ratio": 5.0', self.source)


if __name__ == "__main__":
    unittest.main()

import ast
import math
import os
from types import SimpleNamespace
import unittest


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITOR_PATH = os.path.join(
    PACKAGE_ROOT,
    "isaac_ros_yopo_bringup",
    "runtime_health_monitor.py",
)
SETUP_PATH = os.path.join(PACKAGE_ROOT, "setup.py")
PACKAGE_XML_PATH = os.path.join(PACKAGE_ROOT, "package.xml")


class RuntimeHealthMonitorSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MONITOR_PATH, "r", encoding="utf-8") as stream:
            cls.source = stream.read()
        with open(SETUP_PATH, "r", encoding="utf-8") as stream:
            cls.setup_source = stream.read()
        with open(PACKAGE_XML_PATH, "r", encoding="utf-8") as stream:
            cls.package_xml = stream.read()
        cls.tree = ast.parse(cls.source)
        finite_function = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "odometry_measurements_are_finite"
        )
        namespace = {"Odometry": object, "math": math}
        exec(
            compile(
                ast.Module(body=[finite_function], type_ignores=[]),
                MONITOR_PATH,
                "exec",
            ),
            namespace,
        )
        cls.odometry_is_finite = staticmethod(
            namespace["odometry_measurements_are_finite"]
        )

    def test_subscribes_to_all_runtime_health_inputs_with_sensor_qos(self):
        for topic in (
            "/camera/infra1/camera_info",
            "/camera/infra2/camera_info",
            "/camera/imu",
            "/visual_slam/tracking/odometry",
        ):
            self.assertIn(topic, self.source)
        self.assertEqual(4, self.source.count("qos_profile_sensor_data,"))
        self.assertIn("from nav_msgs.msg import Odometry", self.source)

    def test_compares_full_versioned_camera_info_contract(self):
        for field in (
            "camera.width",
            "camera.height",
            "camera.distortion_model",
            "camera.left_d",
            "camera.left_k",
            "camera.left_r",
            "camera.left_p",
            "camera.right_d",
            "camera.right_k",
            "camera.right_r",
            "camera.right_p",
        ):
            self.assertIn(field, self.source)
        self.assertIn("float(value) == reference", self.source)

    def test_allows_only_the_known_right_frame_reuse(self):
        self.assertIn(
            "camera.right_recorded_frame_bug",
            self.source,
        )
        self.assertIn("if frame_id not in allowed_right_frames", self.source)
        self.assertIn("if frame_id != camera.left_frame", self.source)

    def test_any_d435i_imu_sample_is_fatal(self):
        function = self._function("_on_forbidden_camera_imu")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        self.assertTrue(
            any(getattr(call.func, "attr", "") == "_fail" for call in calls)
        )

    def test_requires_first_and_continuous_odometry_and_camera_info(self):
        self.assertIn('"left_camera_info": None', self.source)
        self.assertIn('"right_camera_info": None', self.source)
        self.assertIn('"odometry": None', self.source)
        self.assertIn("startup_timeout_sec", self.source)
        self.assertIn("camera_info_stale_after_sec", self.source)
        self.assertIn("odometry_stale_after_sec", self.source)

    def test_odometry_must_be_structurally_valid(self):
        self.assertIn("expected_odometry_frame_id", self.source)
        self.assertIn("expected_odometry_child_frame_id", self.source)
        self.assertIn("stamp_to_nanoseconds", self.source)
        self.assertIn("odometry timestamp is not strictly increasing", self.source)
        self.assertIn("odometry_measurements_are_finite", self.source)
        self.assertIn("non-finite pose, twist, or covariance", self.source)

    def test_odometry_finite_gate_checks_pose_twist_and_covariance(self):
        vector = lambda x=0.0, y=0.0, z=0.0: SimpleNamespace(x=x, y=y, z=z)
        message = SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=vector(),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                covariance=[0.0] * 36,
            ),
            twist=SimpleNamespace(
                twist=SimpleNamespace(linear=vector(), angular=vector()),
                covariance=[0.0] * 36,
            ),
        )
        self.assertTrue(self.odometry_is_finite(message))
        for mutate in (
            lambda: setattr(message.pose.pose.position, "x", math.nan),
            lambda: setattr(message.pose.pose.orientation, "w", math.inf),
            lambda: message.pose.covariance.__setitem__(35, -math.inf),
            lambda: setattr(message.twist.twist.angular, "z", math.nan),
            lambda: message.twist.covariance.__setitem__(35, math.inf),
        ):
            message.pose.pose.position.x = 0.0
            message.pose.pose.orientation.w = 1.0
            message.pose.covariance[35] = 0.0
            message.twist.twist.angular.z = 0.0
            message.twist.covariance[35] = 0.0
            mutate()
            self.assertFalse(self.odometry_is_finite(message))

    def test_failure_is_a_nonzero_process_exit_contract(self):
        function = self._function("_fail")
        self.assertTrue(
            any(isinstance(node, ast.Raise) for node in ast.walk(function))
        )
        self.assertIn("raise RuntimeError(self._fatal_reason)", self.source)

    def test_installs_console_entry_point_and_odometry_dependency(self):
        self.assertIn('"runtime_health_monitor = "', self.setup_source)
        self.assertIn(
            '"isaac_ros_yopo_bringup.runtime_health_monitor:main"',
            self.setup_source,
        )
        self.assertIn("<exec_depend>nav_msgs</exec_depend>", self.package_xml)

    def test_source_is_parseable_and_respects_python_line_length(self):
        self.assertIsInstance(self.tree, ast.Module)
        long_lines = [
            index
            for index, line in enumerate(self.source.splitlines(), start=1)
            if len(line) > 99
        ]
        self.assertEqual([], long_lines)

    def _function(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == name:
                    return node
        self.fail(f"function {name!r} was not found")


if __name__ == "__main__":
    unittest.main()

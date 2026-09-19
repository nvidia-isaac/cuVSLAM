import ast
import math
import os
from types import SimpleNamespace
import unittest


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELAY_PATH = os.path.join(
    PACKAGE_ROOT,
    "isaac_ros_yopo_bringup",
    "aligned_imu_relay.py",
)
LAUNCH_PATH = os.path.join(PACKAGE_ROOT, "launch", "d435i_fcu_imu_cuvslam.launch.py")


def make_imu_message():
    return SimpleNamespace(
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=[0.0] * 9,
        angular_velocity=SimpleNamespace(x=0.1, y=0.2, z=0.3),
        angular_velocity_covariance=[0.0] * 9,
        linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=9.81),
        linear_acceleration_covariance=[0.0] * 9,
    )


class RelaySourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(RELAY_PATH, "r", encoding="utf-8") as stream:
            cls.relay_source = stream.read()
        with open(LAUNCH_PATH, "r", encoding="utf-8") as stream:
            cls.launch_source = stream.read()
        relay_tree = ast.parse(cls.relay_source, filename=RELAY_PATH)
        finite_function = next(
            node
            for node in relay_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "imu_measurements_are_finite"
        )
        function_module = ast.Module(body=[finite_function], type_ignores=[])
        namespace = {"Imu": object, "math": math}
        exec(compile(function_module, RELAY_PATH, "exec"), namespace)
        cls.measurements_are_finite = staticmethod(
            namespace["imu_measurements_are_finite"]
        )

    def test_input_and_output_use_sensor_data_qos(self):
        self.assertGreaterEqual(
            self.relay_source.count("qos_profile_sensor_data"),
            3,
        )

    def test_clock_gate_uses_host_system_time(self):
        self.assertIn("time.time_ns()", self.relay_source)
        self.assertNotIn("self.get_clock().now().nanoseconds", self.relay_source)
        self.assertIn(
            '"maximum_receipt_time_residual_sec": 0.25',
            self.launch_source,
        )

    def test_parameters_are_declared_read_only(self):
        self.assertIn(
            "ParameterDescriptor(description=description, read_only=True)",
            self.relay_source,
        )

    def test_sustained_unhealthy_stream_exits_nonzero(self):
        self.assertIn("ConsecutiveFailureGate(3)", self.relay_source)
        self.assertIn("raise RuntimeError(fatal_reason)", self.relay_source)

    def test_callback_rejects_nonfinite_measurements(self):
        self.assertIn(
            "if not imu_measurements_are_finite(message):",
            self.relay_source,
        )
        self.assertGreaterEqual(
            self.relay_source.count('"nonfinite_measurement"'),
            3,
        )

    def test_finite_gate_accepts_a_complete_finite_message(self):
        self.assertTrue(self.measurements_are_finite(make_imu_message()))

    def test_finite_gate_checks_vectors_and_all_covariances(self):
        vector_fields = (
            ("orientation", "x"),
            ("orientation", "y"),
            ("orientation", "z"),
            ("orientation", "w"),
            ("angular_velocity", "x"),
            ("angular_velocity", "y"),
            ("angular_velocity", "z"),
            ("linear_acceleration", "x"),
            ("linear_acceleration", "y"),
            ("linear_acceleration", "z"),
        )
        for vector_name, component_name in vector_fields:
            with self.subTest(field=f"{vector_name}.{component_name}"):
                message = make_imu_message()
                setattr(getattr(message, vector_name), component_name, math.nan)
                self.assertFalse(self.measurements_are_finite(message))

        covariance_fields = (
            "orientation_covariance",
            "angular_velocity_covariance",
            "linear_acceleration_covariance",
        )
        for covariance_name in covariance_fields:
            for index in range(9):
                with self.subTest(field=covariance_name, index=index):
                    message = make_imu_message()
                    getattr(message, covariance_name)[index] = math.inf
                    self.assertFalse(self.measurements_are_finite(message))

    def test_relay_never_subscribes_to_d435i_imu(self):
        self.assertNotIn("/camera/imu", self.relay_source)


if __name__ == "__main__":
    unittest.main()

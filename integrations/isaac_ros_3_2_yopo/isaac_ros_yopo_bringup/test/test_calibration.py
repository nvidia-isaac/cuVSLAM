import copy
import hashlib
import os
import tempfile
import unittest

import yaml

from isaac_ros_yopo_bringup.calibration import (
    assert_runtime_calibration_allowed,
    load_calibration,
    load_imu_noise,
)


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_PATH = os.path.join(
    PACKAGE_ROOT,
    "config",
    "d435i_243622070369_fcu_imu.yaml",
)
UNVALIDATED_NOISE_PATH = os.path.join(
    PACKAGE_ROOT,
    "config",
    "px4_imu_noise_unvalidated.yaml",
)


class RuntimeCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_calibration(CALIBRATION_PATH)

    def test_identifies_the_fixed_camera(self):
        self.assertEqual("243622070369", self.calibration.camera_serial)
        self.assertEqual("5.15.1.55", self.calibration.camera_firmware)

    def test_uses_distinct_physical_camera_frames(self):
        self.assertEqual(
            "camera_infra1_optical_frame",
            self.calibration.left_camera_frame,
        )
        self.assertEqual(
            "camera_infra2_optical_frame",
            self.calibration.right_camera_frame,
        )

    def test_fixes_time_offset_sign_and_integer_rounding(self):
        self.assertAlmostEqual(
            -0.001737986760008108,
            self.calibration.kalibr_timeshift_cam_imu_sec,
        )
        self.assertEqual(1_737_987, self.calibration.imu_to_camera_offset_ns)

    def test_uses_dedicated_runtime_imu_frame_and_topic(self):
        self.assertEqual("base_link", self.calibration.imu_source_frame)
        self.assertEqual("fcu_imu", self.calibration.imu_runtime_frame)
        self.assertEqual("/mavros/imu/data_raw", self.calibration.imu_input_topic)
        self.assertEqual(
            "/fcu/imu/data_raw_aligned",
            self.calibration.imu_output_topic,
        )

    def test_transform_direction_is_camera_parent_to_imu_child(self):
        self.assertEqual(
            "camera_infra1_optical_frame",
            self.calibration.tf_parent_frame,
        )
        self.assertEqual("fcu_imu", self.calibration.tf_child_frame)
        self.assertEqual(
            (0.027362927932, 0.052851887430, -0.062141618519),
            self.calibration.tf_translation_m,
        )
        self.assertAlmostEqual(
            1.0,
            sum(value * value for value in self.calibration.tf_rotation_xyzw),
            places=12,
        )

    def test_uses_measured_not_requested_imu_rate(self):
        self.assertEqual(170.0, self.calibration.imu_rate_hz)

    def test_factory_camera_info_is_versioned_exactly(self):
        camera = self.calibration.camera
        self.assertEqual((0.0,) * 5, camera.left_d)
        self.assertEqual((0.0,) * 5, camera.right_d)
        self.assertEqual(
            (
                323.1030578613281,
                0.0,
                319.8547058105469,
                0.0,
                323.1030578613281,
                184.5459442138672,
                0.0,
                0.0,
                1.0,
            ),
            camera.left_k,
        )
        self.assertEqual(-16.174942016601562, camera.right_p[3])
        self.assertEqual(camera.left_frame, camera.right_recorded_frame_bug)

    def test_candidate_requires_an_explicit_runtime_override(self):
        with self.assertRaisesRegex(ValueError, "runtime candidate"):
            assert_runtime_calibration_allowed(self.calibration, False)
        assert_runtime_calibration_allowed(self.calibration, True)

    def test_approved_status_is_allowed_without_override(self):
        calibration = self._load_modified(status="approved")
        assert_runtime_calibration_allowed(calibration, False)

    def test_rejected_status_cannot_be_overridden(self):
        calibration = self._load_modified(status="rejected")
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            assert_runtime_calibration_allowed(calibration, True)

    def test_rejects_unknown_calibration_status(self):
        with self.assertRaisesRegex(ValueError, "unsupported calibration status"):
            self._load_modified(status="looks_good")

    def test_rejects_duplicate_yaml_keys(self):
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as stream:
            source = stream.read()
        source = source.replace(
            "schema_version: 1\n",
            "schema_version: 1\nschema_version: 1\n",
            1,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
            delete=False,
        ) as stream:
            stream.write(source)
            path = stream.name
        try:
            with self.assertRaisesRegex(ValueError, "duplicate YAML key"):
                load_calibration(path)
        finally:
            os.unlink(path)

    def test_rejects_boolean_camera_rate(self):
        with self.assertRaisesRegex(ValueError, "YAML integer or floating-point"):
            self._load_modified(camera={"rate_hz": True})

    def test_rejects_projection_baseline_disagreement(self):
        with self.assertRaisesRegex(ValueError, "projection matrix"):
            self._load_modified(camera={"right": {"P": [
                323.1030578613281, 0.0, 319.8547058105469, -15.0,
                0.0, 323.1030578613281, 184.5459442138672, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]}})

    def test_rejects_explicit_null_in_required_string(self):
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        data["provenance"]["camera_serial"] = None
        path = self._temporary_yaml(data)
        try:
            with self.assertRaises(ValueError):
                load_calibration(path)
        finally:
            os.unlink(path)

    @classmethod
    def _load_modified(cls, **changes):
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        cls._deep_update(data, changes)
        path = cls._temporary_yaml(data)
        try:
            return load_calibration(path)
        finally:
            os.unlink(path)

    @classmethod
    def _deep_update(cls, target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                cls._deep_update(target[key], value)
            else:
                target[key] = value

    @staticmethod
    def _temporary_yaml(data):
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
            delete=False,
        ) as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
            return stream.name


class ImuNoiseCalibrationTest(unittest.TestCase):
    def test_loads_bundled_values_only_as_unvalidated(self):
        noise = load_imu_noise(
            UNVALIDATED_NOISE_PATH,
            "px4-highres-imu-105-ttyTHS2",
            170.0,
        )
        self.assertFalse(noise.validated)
        self.assertEqual("kalibr_input_assumption", noise.method)
        self.assertEqual(0.06, noise.gyroscope_noise_density)
        self.assertEqual(0.001, noise.gyroscope_random_walk)
        self.assertEqual(0.09, noise.accelerometer_noise_density)
        self.assertEqual(0.05, noise.accelerometer_random_walk)

    def test_rejects_noise_for_another_hardware_id(self):
        with self.assertRaises(ValueError):
            load_imu_noise(
                UNVALIDATED_NOISE_PATH,
                "another-imu",
                170.0,
            )

    def test_rejects_noise_from_a_different_sample_rate(self):
        with self.assertRaises(ValueError):
            load_imu_noise(
                UNVALIDATED_NOISE_PATH,
                "px4-highres-imu-105-ttyTHS2",
                200.0,
            )

    def test_rejects_known_input_tuple_claimed_as_validated_allan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._validated_noise_yaml(directory, preserve_parameters=True)
            with self.assertRaises(ValueError):
                load_imu_noise(
                    path,
                    "px4-highres-imu-105-ttyTHS2",
                    170.0,
                )

    def test_loads_traceable_validated_allan_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._validated_noise_yaml(directory)
            noise = load_imu_noise(
                path,
                "px4-highres-imu-105-ttyTHS2",
                170.0,
            )
            self.assertTrue(noise.validated)
            self.assertEqual("allan_deviation", noise.method)

    def test_rejects_validated_allan_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._validated_noise_yaml(directory)
            with open(path, "r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
            data["method"]["source_sha256"] = "0" * 64
            with open(path, "w", encoding="utf-8") as stream:
                yaml.safe_dump(data, stream, sort_keys=False)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_imu_noise(
                    path,
                    "px4-highres-imu-105-ttyTHS2",
                    170.0,
                )

    @staticmethod
    def _validated_noise_yaml(directory, preserve_parameters=False):
        artifact_name = "px4_stationary_imu.csv"
        artifact_path = os.path.join(directory, artifact_name)
        artifact_contents = b"timestamp_ns,gx,gy,gz,ax,ay,az\n1,0,0,0,0,0,9.81\n"
        with open(artifact_path, "wb") as stream:
            stream.write(artifact_contents)

        with open(UNVALIDATED_NOISE_PATH, "r", encoding="utf-8") as stream:
            data = copy.deepcopy(yaml.safe_load(stream))
        data["calibration_id"] = "px4_allan_test"
        data["validated"] = True
        data["method"] = {
            "name": "allan_deviation",
            "source_artifact": artifact_name,
            "source_sha256": hashlib.sha256(artifact_contents).hexdigest(),
        }
        if not preserve_parameters:
            data["parameters"] = {
                "gyroscope_noise_density": 0.01,
                "gyroscope_random_walk": 0.002,
                "accelerometer_noise_density": 0.1,
                "accelerometer_random_walk": 0.01,
            }
        path = os.path.join(directory, "validated_allan.yaml")
        with open(path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        return path


if __name__ == "__main__":
    unittest.main()

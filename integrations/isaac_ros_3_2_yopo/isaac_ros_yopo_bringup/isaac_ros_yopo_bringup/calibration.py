"""Load and validate the audited D435i/PX4 runtime calibration record."""

from dataclasses import dataclass
import hashlib
import math
import os
import re
from typing import Any, Mapping, Sequence, Tuple

import yaml

from .time_alignment import normalize_absolute_topic, validate_frame_id


APPROVED_CALIBRATION_STATUS = "approved"
CANDIDATE_CALIBRATION_STATUS = (
    "runtime_candidate_pending_allan_and_independent_repeatability"
)
REJECTED_CALIBRATION_STATUS = "rejected"
SUPPORTED_CALIBRATION_STATUSES = {
    APPROVED_CALIBRATION_STATUS,
    CANDIDATE_CALIBRATION_STATUS,
    REJECTED_CALIBRATION_STATUS,
}
APPROVED_PROJECT_STATUS = "approved"
CANDIDATE_PROJECT_STATUS = "candidate"
REJECTED_PROJECT_STATUS = "rejected"
SUPPORTED_PROJECT_STATUSES = {
    APPROVED_PROJECT_STATUS,
    CANDIDATE_PROJECT_STATUS,
    REJECTED_PROJECT_STATUS,
}


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be hashable") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class StereoCameraCalibration:
    width: int
    height: int
    rate_hz: float
    distortion_model: str
    left_frame: str
    right_frame: str
    right_recorded_frame_bug: str
    left_d: Tuple[float, ...]
    left_k: Tuple[float, ...]
    left_r: Tuple[float, ...]
    left_p: Tuple[float, ...]
    right_d: Tuple[float, ...]
    right_k: Tuple[float, ...]
    right_r: Tuple[float, ...]
    right_p: Tuple[float, ...]


@dataclass(frozen=True)
class RuntimeCalibration:
    calibration_id: str
    status: str
    camera_serial: str
    camera_firmware: str
    camera: StereoCameraCalibration
    imu_input_topic: str
    imu_output_topic: str
    imu_source_frame: str
    imu_runtime_frame: str
    imu_rate_hz: float
    imu_hardware_id: str
    imu_to_camera_offset_ns: int
    kalibr_timeshift_cam_imu_sec: float
    tf_parent_frame: str
    tf_child_frame: str
    tf_translation_m: Tuple[float, float, float]
    tf_rotation_xyzw: Tuple[float, float, float, float]
    transform_row_major: Tuple[float, ...]

    @property
    def left_camera_frame(self) -> str:
        return self.camera.left_frame

    @property
    def right_camera_frame(self) -> str:
        return self.camera.right_frame


@dataclass(frozen=True)
class ImuNoiseCalibration:
    schema_version: int
    calibration_id: str
    project_status: str
    validated: bool
    sensor_hardware_id: str
    sample_rate_hz: float
    method: str
    source_artifact: str
    source_sha256: str
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a YAML integer or floating-point value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_sequence(value: Any, length: int, name: str) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    if len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(
        _finite_number(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _determinant_3x3(matrix: Sequence[float]) -> float:
    return (
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
    )


def _quaternion_matrix(quaternion: Sequence[float]) -> Tuple[float, ...]:
    x, y, z, w = quaternion
    return (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


def _validate_rotation(matrix: Sequence[float], quaternion: Sequence[float]) -> None:
    rows = (matrix[0:3], matrix[3:6], matrix[6:9])
    for row_index in range(3):
        for column_index in range(3):
            dot = sum(
                rows[item][row_index] * rows[item][column_index]
                for item in range(3)
            )
            expected = 1.0 if row_index == column_index else 0.0
            if not math.isclose(dot, expected, abs_tol=1.0e-8):
                raise ValueError("T_parent_child rotation is not orthonormal")
    if not math.isclose(_determinant_3x3(matrix), 1.0, abs_tol=1.0e-8):
        raise ValueError("T_parent_child rotation determinant is not +1")

    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, abs_tol=1.0e-8):
        raise ValueError("rotation_xyzw is not a unit quaternion")
    quaternion_matrix = _quaternion_matrix(quaternion)
    if any(
        not math.isclose(actual, expected, abs_tol=2.0e-9)
        for actual, expected in zip(matrix, quaternion_matrix)
    ):
        raise ValueError("rotation_xyzw and T_parent_child disagree")


def _validate_factory_camera(camera: Mapping[str, Any]) -> StereoCameraCalibration:
    if camera.get("model") != "Intel RealSense D435I":
        raise ValueError("camera.model must identify the calibrated Intel RealSense D435I")
    if camera.get("runtime_intrinsics_source") != "realsense_camera_info":
        raise ValueError("runtime camera intrinsics must come from RealSense CameraInfo")
    width = _positive_integer(camera.get("width"), "camera.width")
    height = _positive_integer(camera.get("height"), "camera.height")
    if width != 640 or height != 360:
        raise ValueError("calibration is only valid for 640x360 rectified images")
    rate_hz = _finite_number(camera.get("rate_hz"), "camera.rate_hz")
    if not math.isclose(rate_hz, 90.0, abs_tol=1.0e-12):
        raise ValueError("calibration is only valid for 90 Hz rectified images")
    distortion_model = _nonempty_string(
        camera.get("distortion_model"),
        "camera.distortion_model",
    )
    if distortion_model != "plumb_bob":
        raise ValueError("factory rectified distortion_model must be plumb_bob")

    left = _mapping(camera.get("left"), "camera.left")
    right = _mapping(camera.get("right"), "camera.right")
    left_frame = validate_frame_id(_nonempty_string(left.get("frame_id"), "camera.left.frame_id"))
    right_frame = validate_frame_id(
        _nonempty_string(right.get("frame_id"), "camera.right.frame_id")
    )
    if left_frame == right_frame:
        raise ValueError("left and right physical camera frames must differ")
    right_recorded_frame_bug = validate_frame_id(
        _nonempty_string(
            right.get("recorded_camera_info_frame_bug"),
            "camera.right.recorded_camera_info_frame_bug",
        )
    )
    if right_recorded_frame_bug != left_frame:
        raise ValueError(
            "the documented right CameraInfo frame bug must reuse the left optical frame"
        )

    left_k = _finite_sequence(left.get("K"), 9, "camera.left.K")
    right_k = _finite_sequence(right.get("K"), 9, "camera.right.K")
    left_p = _finite_sequence(left.get("P"), 12, "camera.left.P")
    right_p = _finite_sequence(right.get("P"), 12, "camera.right.P")
    identity_rotation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    left_d = _finite_sequence(left.get("D"), 5, "camera.left.D")
    right_d = _finite_sequence(right.get("D"), 5, "camera.right.D")
    left_r = _finite_sequence(left.get("R"), 9, "camera.left.R")
    right_r = _finite_sequence(right.get("R"), 9, "camera.right.R")
    for side_name, distortion, rectification in (
        ("left", left_d, left_r),
        ("right", right_d, right_r),
    ):
        if any(abs(value) > 1.0e-12 for value in distortion):
            raise ValueError(f"camera.{side_name}.D must be zero for rectified images")
        if any(
            not math.isclose(actual, expected, abs_tol=1.0e-12)
            for actual, expected in zip(rectification, identity_rotation)
        ):
            raise ValueError(f"camera.{side_name}.R must be identity")
    if any(
        not math.isclose(actual, expected, abs_tol=1.0e-12)
        for actual, expected in zip(left_k, right_k)
    ):
        raise ValueError("factory rectified left and right K matrices must match")
    if left_k[0] <= 0.0 or left_k[4] <= 0.0:
        raise ValueError("factory rectified focal lengths must be positive")
    if not 0.0 <= left_k[2] < width or not 0.0 <= left_k[5] < height:
        raise ValueError("factory rectified principal point lies outside the image")
    if not math.isclose(left_k[8], 1.0, abs_tol=1.0e-12):
        raise ValueError("camera K homogeneous scale must be 1")

    expected_left_p = (
        left_k[0], left_k[1], left_k[2], 0.0,
        left_k[3], left_k[4], left_k[5], 0.0,
        left_k[6], left_k[7], left_k[8], 0.0,
    )
    if any(
        not math.isclose(actual, expected, abs_tol=1.0e-12)
        for actual, expected in zip(left_p, expected_left_p)
    ):
        raise ValueError("camera.left.P disagrees with the rectified K matrix")

    stereo = _mapping(camera.get("stereo"), "camera.stereo")
    baseline = _finite_number(stereo.get("baseline_m"), "camera.stereo.baseline_m")
    if baseline <= 0.0:
        raise ValueError("camera.stereo.baseline_m must be positive")
    if not math.isclose(-right_p[3] / right_p[0], baseline, abs_tol=1.0e-12):
        raise ValueError("right projection matrix and stereo baseline disagree")
    expected_right_p = list(expected_left_p)
    expected_right_p[3] = -right_p[0] * baseline
    if any(
        not math.isclose(actual, expected, abs_tol=1.0e-12)
        for actual, expected in zip(right_p, expected_right_p)
    ):
        raise ValueError("camera.right.P disagrees with K and baseline")

    right_from_left = _finite_sequence(
        stereo.get("T_right_left"),
        16,
        "camera.stereo.T_right_left",
    )
    expected_right_from_left = (
        1.0, 0.0, 0.0, -baseline,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    if any(
        not math.isclose(actual, expected, abs_tol=1.0e-12)
        for actual, expected in zip(right_from_left, expected_right_from_left)
    ):
        raise ValueError("T_right_left disagrees with the factory stereo baseline")
    return StereoCameraCalibration(
        width=width,
        height=height,
        rate_hz=rate_hz,
        distortion_model=distortion_model,
        left_frame=left_frame,
        right_frame=right_frame,
        right_recorded_frame_bug=right_recorded_frame_bug,
        left_d=left_d,
        left_k=left_k,
        left_r=left_r,
        left_p=left_p,
        right_d=right_d,
        right_k=right_k,
        right_r=right_r,
        right_p=right_p,
    )


def load_calibration(path) -> RuntimeCalibration:
    """Load one version-controlled runtime calibration and reject ambiguity."""
    with open(path, "r", encoding="utf-8") as stream:
        root = _mapping(yaml.load(stream, Loader=UniqueKeySafeLoader), "calibration")

    if root.get("schema_version") != 1:
        raise ValueError("unsupported calibration schema_version")
    calibration_id = _nonempty_string(root.get("calibration_id"), "calibration_id")
    status = _nonempty_string(root.get("status"), "status")
    if status not in SUPPORTED_CALIBRATION_STATUSES:
        raise ValueError(f"unsupported calibration status: {status}")

    provenance = _mapping(root.get("provenance"), "provenance")
    camera_serial = _nonempty_string(
        provenance.get("camera_serial"),
        "provenance.camera_serial",
    )
    camera_firmware = _nonempty_string(
        provenance.get("camera_firmware"),
        "provenance.camera_firmware",
    )
    for field in (
        "joint_calibration_bag",
        "kalibr_camchain_result",
        "camera_model",
        "imu_clock_source",
        "fcu_bridge",
    ):
        _nonempty_string(provenance.get(field), f"provenance.{field}")

    camera = _mapping(root.get("camera"), "camera")
    camera_calibration = _validate_factory_camera(camera)

    imu = _mapping(root.get("imu"), "imu")
    input_topic = normalize_absolute_topic(
        _nonempty_string(imu.get("source_topic"), "imu.source_topic")
    )
    output_topic = normalize_absolute_topic(
        _nonempty_string(imu.get("aligned_topic"), "imu.aligned_topic")
    )
    if input_topic == output_topic:
        raise ValueError("raw and aligned IMU topics must differ")
    source_frame = validate_frame_id(
        _nonempty_string(imu.get("source_frame_id"), "imu.source_frame_id")
    )
    runtime_frame = validate_frame_id(
        _nonempty_string(imu.get("runtime_frame_id"), "imu.runtime_frame_id")
    )
    imu_rate_hz = _finite_number(imu.get("measured_rate_hz"), "imu.measured_rate_hz")
    if imu_rate_hz <= 0.0:
        raise ValueError("imu.measured_rate_hz must be positive")
    hardware_id = _nonempty_string(imu.get("hardware_id"), "imu.hardware_id")

    time_alignment = _mapping(imu.get("time_alignment"), "imu.time_alignment")
    offset_ns = time_alignment.get("imu_to_camera_offset_ns")
    if isinstance(offset_ns, bool) or not isinstance(offset_ns, int):
        raise ValueError("imu_to_camera_offset_ns must be an integer")
    kalibr_shift = _finite_number(
        time_alignment.get("kalibr_timeshift_cam_imu_sec"),
        "imu.time_alignment.kalibr_timeshift_cam_imu_sec",
    )
    if round(-kalibr_shift * 1_000_000_000) != offset_ns:
        raise ValueError("runtime offset has the wrong Kalibr sign or rounding")

    extrinsics = _mapping(root.get("extrinsics"), "extrinsics")
    parent_frame = validate_frame_id(
        _nonempty_string(extrinsics.get("parent_frame"), "extrinsics.parent_frame")
    )
    child_frame = validate_frame_id(
        _nonempty_string(extrinsics.get("child_frame"), "extrinsics.child_frame")
    )
    if parent_frame != camera_calibration.left_frame or child_frame != runtime_frame:
        raise ValueError("extrinsic frames disagree with camera/IMU runtime frames")
    translation = _finite_sequence(extrinsics.get("translation_m"), 3, "translation_m")
    quaternion = _finite_sequence(extrinsics.get("rotation_xyzw"), 4, "rotation_xyzw")
    transform = _finite_sequence(extrinsics.get("T_parent_child"), 16, "T_parent_child")
    if any(
        not math.isclose(actual, expected, abs_tol=1.0e-12)
        for actual, expected in zip(transform[12:16], (0.0, 0.0, 0.0, 1.0))
    ):
        raise ValueError("T_parent_child has an invalid homogeneous bottom row")
    matrix_rotation = (
        transform[0], transform[1], transform[2],
        transform[4], transform[5], transform[6],
        transform[8], transform[9], transform[10],
    )
    matrix_translation = (transform[3], transform[7], transform[11])
    if any(
        not math.isclose(actual, expected, abs_tol=1.0e-12)
        for actual, expected in zip(matrix_translation, translation)
    ):
        raise ValueError("translation_m and T_parent_child disagree")
    _validate_rotation(matrix_rotation, quaternion)

    return RuntimeCalibration(
        calibration_id=calibration_id,
        status=status,
        camera_serial=camera_serial,
        camera_firmware=camera_firmware,
        camera=camera_calibration,
        imu_input_topic=input_topic,
        imu_output_topic=output_topic,
        imu_source_frame=source_frame,
        imu_runtime_frame=runtime_frame,
        imu_rate_hz=imu_rate_hz,
        imu_hardware_id=hardware_id,
        imu_to_camera_offset_ns=offset_ns,
        kalibr_timeshift_cam_imu_sec=kalibr_shift,
        tf_parent_frame=parent_frame,
        tf_child_frame=child_frame,
        tf_translation_m=translation,
        tf_rotation_xyzw=quaternion,
        transform_row_major=transform,
    )


def assert_runtime_calibration_allowed(
    calibration: RuntimeCalibration,
) -> None:
    """Reject a calibration that has not received project runtime approval."""
    if calibration.status == APPROVED_CALIBRATION_STATUS:
        return
    if calibration.status == CANDIDATE_CALIBRATION_STATUS:
        raise ValueError("calibration is a runtime candidate and is not project-approved")
    if calibration.status == REJECTED_CALIBRATION_STATUS:
        raise ValueError("calibration status is rejected")
    raise ValueError(f"unsupported calibration status: {calibration.status}")


def assert_runtime_imu_noise_allowed(noise: ImuNoiseCalibration) -> None:
    """Require project approval independently from Allan provenance validation."""
    if noise.project_status == APPROVED_PROJECT_STATUS:
        return
    if noise.project_status == CANDIDATE_PROJECT_STATUS:
        raise ValueError("IMU noise model is a candidate and is not project-approved")
    if noise.project_status == REJECTED_PROJECT_STATUS:
        raise ValueError("IMU noise model is rejected")
    raise ValueError(f"unsupported IMU noise project_status: {noise.project_status}")


def load_imu_noise(
    path,
    expected_hardware_id: str,
    expected_rate_hz: float,
) -> ImuNoiseCalibration:
    """Load one traceable IMU noise model instead of independent CLI values."""
    with open(path, "r", encoding="utf-8") as stream:
        root = _mapping(
            yaml.load(stream, Loader=UniqueKeySafeLoader),
            "IMU noise calibration",
        )
    schema_version = root.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError("unsupported IMU noise schema_version")

    calibration_id = _nonempty_string(root.get("calibration_id"), "calibration_id")
    validated = root.get("validated")
    if not isinstance(validated, bool):
        raise ValueError("validated must be a boolean")
    if schema_version == 1:
        project_status = (
            APPROVED_PROJECT_STATUS if validated else CANDIDATE_PROJECT_STATUS
        )
    else:
        project_status = _nonempty_string(
            root.get("project_status"),
            "project_status",
        )
        if project_status not in SUPPORTED_PROJECT_STATUSES:
            raise ValueError(
                f"unsupported IMU noise project_status: {project_status}"
            )

    sensor = _mapping(root.get("sensor"), "sensor")
    hardware_id = _nonempty_string(sensor.get("hardware_id"), "sensor.hardware_id")
    if hardware_id != expected_hardware_id:
        raise ValueError(
            f"IMU noise hardware mismatch: expected {expected_hardware_id}, got {hardware_id}"
        )
    sample_rate_hz = _finite_number(
        sensor.get("sample_rate_hz"),
        "sensor.sample_rate_hz",
    )
    if sample_rate_hz <= 0.0:
        raise ValueError("sensor.sample_rate_hz must be positive")
    if abs(sample_rate_hz - expected_rate_hz) / expected_rate_hz > 0.05:
        raise ValueError("IMU noise sample rate differs from the runtime rate by more than 5%")

    method_record = _mapping(root.get("method"), "method")
    method = _nonempty_string(method_record.get("name"), "method.name")
    source_artifact = _nonempty_string(
        method_record.get("source_artifact"),
        "method.source_artifact",
    )
    source_sha256_value = method_record.get("source_sha256")
    if validated:
        source_sha256 = _nonempty_string(source_sha256_value, "method.source_sha256")
        if method != "allan_deviation":
            raise ValueError("validated IMU noise method must be allan_deviation")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", source_sha256):
            raise ValueError("validated method.source_sha256 must contain 64 hex digits")
        artifact_path = source_artifact
        if not os.path.isabs(artifact_path):
            artifact_path = os.path.join(os.path.dirname(os.path.abspath(path)), artifact_path)
        if not os.path.isfile(artifact_path):
            raise ValueError(f"validated Allan source artifact does not exist: {artifact_path}")
        digest = hashlib.sha256()
        with open(artifact_path, "rb") as artifact_stream:
            for chunk in iter(lambda: artifact_stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != source_sha256.lower():
            raise ValueError("validated Allan source artifact SHA-256 does not match")
    else:
        source_sha256 = "" if source_sha256_value is None else str(source_sha256_value).strip()

    expected_units = {
        "gyroscope_noise_density": "rad/(s*sqrt(Hz))",
        "gyroscope_random_walk": "rad/(s^2*sqrt(Hz))",
        "accelerometer_noise_density": "m/(s^2*sqrt(Hz))",
        "accelerometer_random_walk": "m/(s^3*sqrt(Hz))",
    }
    units = _mapping(root.get("units"), "units")
    for name, expected_unit in expected_units.items():
        if units.get(name) != expected_unit:
            raise ValueError(f"units.{name} must be {expected_unit}")

    parameters = _mapping(root.get("parameters"), "parameters")
    values = tuple(
        _finite_number(parameters.get(name), f"parameters.{name}")
        for name in expected_units
    )
    upper_bounds = (1.0, 1.0, 10.0, 10.0)
    if any(
        not 0.0 < value < upper_bound
        for value, upper_bound in zip(values, upper_bounds)
    ):
        raise ValueError("IMU noise parameters are non-finite, non-positive, or out of bounds")
    known_unvalidated = (
        (0.000244, 0.000019393, 0.001862, 0.003),
        (0.06, 0.001, 0.09, 0.05),
    )
    if validated and any(
        all(
            math.isclose(value, reference, rel_tol=1.0e-9, abs_tol=1.0e-12)
            for value, reference in zip(values, known)
        )
        for known in known_unvalidated
    ):
        raise ValueError("validated IMU noise cannot reuse a known default/input tuple")

    return ImuNoiseCalibration(
        schema_version=schema_version,
        calibration_id=calibration_id,
        project_status=project_status,
        validated=validated,
        sensor_hardware_id=hardware_id,
        sample_rate_hz=sample_rate_hz,
        method=method,
        source_artifact=source_artifact,
        source_sha256=source_sha256,
        gyroscope_noise_density=values[0],
        gyroscope_random_walk=values[1],
        accelerometer_noise_density=values[2],
        accelerometer_random_walk=values[3],
    )

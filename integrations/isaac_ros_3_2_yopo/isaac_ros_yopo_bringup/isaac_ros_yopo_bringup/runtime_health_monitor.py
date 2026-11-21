"""Fail closed when the calibrated D435i/cuVSLAM runtime contract breaks."""

import math
import os
import time
from typing import Dict, Optional, Sequence, Tuple

from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Imu

from .calibration import load_calibration
from .time_alignment import (
    TimestampContractError,
    normalize_absolute_topic,
    stamp_to_nanoseconds,
    validate_frame_id,
)


PACKAGE_NAME = "isaac_ros_yopo_bringup"
DEFAULT_CALIBRATION_FILE = os.path.join(
    get_package_share_directory(PACKAGE_NAME),
    "config",
    "d435i_243622070369_fcu_imu.yaml",
)


def odometry_measurements_are_finite(message: Odometry) -> bool:
    """Return whether every pose, twist, and covariance value is finite."""
    values = (
        message.pose.pose.position.x,
        message.pose.pose.position.y,
        message.pose.pose.position.z,
        message.pose.pose.orientation.x,
        message.pose.pose.orientation.y,
        message.pose.pose.orientation.z,
        message.pose.pose.orientation.w,
        *message.pose.covariance,
        message.twist.twist.linear.x,
        message.twist.twist.linear.y,
        message.twist.twist.linear.z,
        message.twist.twist.angular.x,
        message.twist.twist.angular.y,
        message.twist.twist.angular.z,
        *message.twist.covariance,
    )
    try:
        return all(math.isfinite(value) for value in values)
    except (TypeError, ValueError):
        return False


class RuntimeHealthMonitor(Node):
    """Continuously enforce live camera and cuVSLAM output invariants."""

    def __init__(self) -> None:
        super().__init__("d435i_cuvslam_runtime_health_monitor")

        calibration_file = str(
            self._read_only_parameter(
                "calibration_file",
                DEFAULT_CALIBRATION_FILE,
                "Version-controlled D435i/FCU runtime calibration YAML.",
            )
        ).strip()
        if not calibration_file:
            raise RuntimeError("calibration_file must be non-empty")
        self._calibration = load_calibration(calibration_file)

        self._left_topic = self._topic_parameter(
            "left_camera_info_topic",
            "/camera/infra1/camera_info",
            "Rectified left infrared CameraInfo topic.",
        )
        self._right_topic = self._topic_parameter(
            "right_camera_info_topic",
            "/camera/infra2/camera_info",
            "Rectified right infrared CameraInfo topic.",
        )
        self._camera_imu_topic = self._topic_parameter(
            "camera_imu_topic",
            "/camera/imu",
            "D435i IMU topic that must remain silent.",
        )
        self._odometry_topic = self._topic_parameter(
            "odometry_topic",
            "/visual_slam/tracking/odometry",
            "cuVSLAM tracking odometry topic.",
        )
        self._expected_odometry_frame = validate_frame_id(
            str(
                self._read_only_parameter(
                    "expected_odometry_frame_id",
                    "odom",
                    "Required cuVSLAM tracking odometry parent frame.",
                )
            )
        )
        self._expected_odometry_child_frame = validate_frame_id(
            str(
                self._read_only_parameter(
                    "expected_odometry_child_frame_id",
                    "camera_link",
                    "Required cuVSLAM tracking odometry child frame.",
                )
            )
        )
        topics = {
            self._left_topic,
            self._right_topic,
            self._camera_imu_topic,
            self._odometry_topic,
        }
        if len(topics) != 4:
            raise RuntimeError("runtime health topics must be distinct")

        self._diagnostic_period_sec = self._positive_parameter(
            "diagnostic_period_sec",
            1.0,
            "Health evaluation and diagnostic publication period.",
        )
        self._camera_info_stale_after_sec = self._positive_parameter(
            "camera_info_stale_after_sec",
            2.0,
            "Maximum silence allowed from either CameraInfo stream.",
        )
        self._odometry_stale_after_sec = self._positive_parameter(
            "odometry_stale_after_sec",
            2.0,
            "Maximum silence allowed from cuVSLAM tracking odometry.",
        )
        self._startup_timeout_sec = self._positive_parameter(
            "startup_timeout_sec",
            30.0,
            "Maximum wait for both CameraInfo streams and tracking odometry.",
        )
        if self._diagnostic_period_sec >= min(
            self._camera_info_stale_after_sec,
            self._odometry_stale_after_sec,
        ):
            raise RuntimeError("stale thresholds must exceed diagnostic_period_sec")
        if self._startup_timeout_sec <= max(
            self._camera_info_stale_after_sec,
            self._odometry_stale_after_sec,
        ):
            raise RuntimeError("startup_timeout_sec must exceed stale thresholds")
        if self._startup_timeout_sec > 120.0:
            raise RuntimeError("startup_timeout_sec must be <= 120 seconds")

        self._started_monotonic_ns = time.monotonic_ns()
        self._last_seen_monotonic_ns: Dict[str, Optional[int]] = {
            "left_camera_info": None,
            "right_camera_info": None,
            "odometry": None,
        }
        self._counts = {
            "left_camera_info": 0,
            "right_camera_info": 0,
            "odometry": 0,
            "forbidden_camera_imu": 0,
        }
        self._ready_logged = False
        self._right_frame_bug_logged = False
        self._fatal_reason: Optional[str] = None
        self._last_odometry_stamp_ns: Optional[int] = None

        self._diagnostic_publisher = self.create_publisher(
            DiagnosticArray,
            "/diagnostics",
            10,
        )
        self._left_subscription = self.create_subscription(
            CameraInfo,
            self._left_topic,
            self._on_left_camera_info,
            qos_profile_sensor_data,
        )
        self._right_subscription = self.create_subscription(
            CameraInfo,
            self._right_topic,
            self._on_right_camera_info,
            qos_profile_sensor_data,
        )
        self._camera_imu_subscription = self.create_subscription(
            Imu,
            self._camera_imu_topic,
            self._on_forbidden_camera_imu,
            qos_profile_sensor_data,
        )
        self._odometry_subscription = self.create_subscription(
            Odometry,
            self._odometry_topic,
            self._on_odometry,
            qos_profile_sensor_data,
        )
        self._diagnostic_timer = self.create_timer(
            self._diagnostic_period_sec,
            self._evaluate_health,
        )

        self.get_logger().info(
            "Runtime health monitor armed for calibration "
            f"{self._calibration.calibration_id}; D435i IMU samples are forbidden"
        )

    def _read_only_parameter(self, name: str, default_value, description: str):
        descriptor = ParameterDescriptor(description=description, read_only=True)
        return self.declare_parameter(name, default_value, descriptor).value

    def _topic_parameter(self, name: str, default_value: str, description: str) -> str:
        value = str(
            self._read_only_parameter(name, default_value, description)
        ).strip()
        return normalize_absolute_topic(value)

    def _positive_parameter(
        self,
        name: str,
        default_value: float,
        description: str,
    ) -> float:
        value = self._read_only_parameter(name, default_value, description)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise RuntimeError(f"{name} must be finite and positive")
        return result

    def _fail(self, reason: str) -> None:
        if self._fatal_reason is None:
            self._fatal_reason = reason
            self.get_logger().fatal(f"{reason}; shutting down runtime bringup")
        raise RuntimeError(self._fatal_reason)

    @staticmethod
    def _sequence_matches(
        actual: Sequence[float],
        expected: Tuple[float, ...],
    ) -> bool:
        return len(actual) == len(expected) and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) == reference
            for value, reference in zip(actual, expected)
        )

    def _camera_info_mismatch(
        self,
        message: CameraInfo,
        side: str,
    ) -> Optional[str]:
        camera = self._calibration.camera
        if message.width != camera.width or message.height != camera.height:
            return (
                f"{side} CameraInfo dimensions mismatch: expected "
                f"{camera.width}x{camera.height}, got "
                f"{message.width}x{message.height}"
            )
        if message.distortion_model != camera.distortion_model:
            return (
                f"{side} CameraInfo distortion_model mismatch: expected "
                f"{camera.distortion_model!r}, got {message.distortion_model!r}"
            )

        expected_values = {
            "D": camera.left_d if side == "left" else camera.right_d,
            "K": camera.left_k if side == "left" else camera.right_k,
            "R": camera.left_r if side == "left" else camera.right_r,
            "P": camera.left_p if side == "left" else camera.right_p,
        }
        actual_values = {
            "D": message.d,
            "K": message.k,
            "R": message.r,
            "P": message.p,
        }
        for field_name, expected in expected_values.items():
            if not self._sequence_matches(actual_values[field_name], expected):
                return f"{side} CameraInfo {field_name} mismatch"

        frame_id = message.header.frame_id
        if side == "left":
            if frame_id != camera.left_frame:
                return (
                    "left CameraInfo frame mismatch: expected "
                    f"{camera.left_frame!r}, got {frame_id!r}"
                )
        else:
            allowed_right_frames = {
                camera.right_frame,
                camera.right_recorded_frame_bug,
            }
            if frame_id not in allowed_right_frames:
                return (
                    "right CameraInfo frame mismatch: expected physical frame "
                    f"{camera.right_frame!r} or known RealSense reuse "
                    f"{camera.right_recorded_frame_bug!r}, got {frame_id!r}"
                )
        return None

    def _accept_camera_info(self, message: CameraInfo, side: str) -> None:
        mismatch = self._camera_info_mismatch(message, side)
        if mismatch is not None:
            self._fail(mismatch)
        if (
            side == "right"
            and message.header.frame_id
            == self._calibration.camera.right_recorded_frame_bug
            and not self._right_frame_bug_logged
        ):
            self.get_logger().warning(
                "right CameraInfo reuses the left optical frame as expected for "
                "this RealSense firmware; cuVSLAM uses the explicit physical frames"
            )
            self._right_frame_bug_logged = True
        key = f"{side}_camera_info"
        self._counts[key] += 1
        self._last_seen_monotonic_ns[key] = time.monotonic_ns()

    def _on_left_camera_info(self, message: CameraInfo) -> None:
        self._accept_camera_info(message, "left")

    def _on_right_camera_info(self, message: CameraInfo) -> None:
        self._accept_camera_info(message, "right")

    def _on_forbidden_camera_imu(self, _message: Imu) -> None:
        self._counts["forbidden_camera_imu"] += 1
        self._fail(
            "received a D435i IMU sample although camera gyro/accel must be disabled"
        )

    def _on_odometry(self, message: Odometry) -> None:
        if message.header.frame_id != self._expected_odometry_frame:
            self._fail(
                "odometry parent frame mismatch: expected "
                f"{self._expected_odometry_frame!r}, got "
                f"{message.header.frame_id!r}"
            )
        if message.child_frame_id != self._expected_odometry_child_frame:
            self._fail(
                "odometry child frame mismatch: expected "
                f"{self._expected_odometry_child_frame!r}, got "
                f"{message.child_frame_id!r}"
            )
        try:
            stamp_ns = stamp_to_nanoseconds(
                message.header.stamp.sec,
                message.header.stamp.nanosec,
            )
        except TimestampContractError as error:
            self._fail(f"invalid odometry timestamp: {error}")
        if stamp_ns == 0:
            self._fail("odometry timestamp is zero")
        if (
            self._last_odometry_stamp_ns is not None
            and stamp_ns <= self._last_odometry_stamp_ns
        ):
            self._fail(
                "odometry timestamp is not strictly increasing: "
                f"{stamp_ns} <= {self._last_odometry_stamp_ns}"
            )
        if not odometry_measurements_are_finite(message):
            self._fail("odometry contains a non-finite pose, twist, or covariance value")
        self._last_odometry_stamp_ns = stamp_ns
        self._counts["odometry"] += 1
        self._last_seen_monotonic_ns["odometry"] = time.monotonic_ns()

    def _age_sec(self, key: str, now_monotonic_ns: int) -> Optional[float]:
        last_seen = self._last_seen_monotonic_ns[key]
        if last_seen is None:
            return None
        return (now_monotonic_ns - last_seen) / 1_000_000_000.0

    def _publish_diagnostic(
        self,
        level: int,
        summary: str,
        ages: Dict[str, Optional[float]],
    ) -> None:
        status = DiagnosticStatus()
        status.level = level
        status.name = f"{self.get_fully_qualified_name()}: calibrated runtime"
        status.message = summary
        status.hardware_id = self._calibration.camera_serial
        values = {
            "calibration_id": self._calibration.calibration_id,
            "left_camera_info_topic": self._left_topic,
            "right_camera_info_topic": self._right_topic,
            "camera_imu_topic": self._camera_imu_topic,
            "odometry_topic": self._odometry_topic,
            "expected_odometry_frame": self._expected_odometry_frame,
            "expected_odometry_child_frame": self._expected_odometry_child_frame,
            "last_odometry_stamp_ns": self._last_odometry_stamp_ns,
            "known_right_frame_reuse_observed": self._right_frame_bug_logged,
        }
        values.update(self._counts)
        values.update({
            f"{name}_age_sec": "not_received" if age is None else f"{age:.6f}"
            for name, age in ages.items()
        })
        status.values = [
            KeyValue(key=key, value=str(value)) for key, value in values.items()
        ]
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = self.get_clock().now().to_msg()
        diagnostics.status = [status]
        self._diagnostic_publisher.publish(diagnostics)

    def _evaluate_health(self) -> None:
        if self._fatal_reason is not None:
            raise RuntimeError(self._fatal_reason)

        now_monotonic_ns = time.monotonic_ns()
        startup_age_sec = (
            now_monotonic_ns - self._started_monotonic_ns
        ) / 1_000_000_000.0
        ages = {
            name: self._age_sec(name, now_monotonic_ns)
            for name in self._last_seen_monotonic_ns
        }
        stale = []
        for name in ("left_camera_info", "right_camera_info"):
            if (
                ages[name] is not None
                and ages[name] > self._camera_info_stale_after_sec
            ):
                stale.append(name)
        if (
            ages["odometry"] is not None
            and ages["odometry"] > self._odometry_stale_after_sec
        ):
            stale.append("odometry")
        if stale:
            summary = "required runtime streams are stale: " + ", ".join(stale)
            self._publish_diagnostic(DiagnosticStatus.ERROR, summary, ages)
            self._fail(summary)

        missing = [name for name, age in ages.items() if age is None]
        if missing:
            summary = "waiting for required streams: " + ", ".join(missing)
            self._publish_diagnostic(DiagnosticStatus.STALE, summary, ages)
            if startup_age_sec > self._startup_timeout_sec:
                self._fail(
                    "required runtime streams did not appear before startup timeout: "
                    + ", ".join(missing)
                )
            return

        self._publish_diagnostic(
            DiagnosticStatus.OK,
            "calibrated D435i CameraInfo and cuVSLAM odometry are healthy",
            ages,
        )
        if not self._ready_logged:
            self.get_logger().info(
                "[PASS] Both CameraInfo streams match calibration and cuVSLAM "
                "tracking odometry is live"
            )
            self._ready_logged = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = RuntimeHealthMonitor()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

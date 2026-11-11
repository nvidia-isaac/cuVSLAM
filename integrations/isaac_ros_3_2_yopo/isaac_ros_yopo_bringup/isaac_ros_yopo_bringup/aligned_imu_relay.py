"""Publish a strictly monotonic, time-aligned copy of the PX4 IMU stream."""

from collections import deque
import math
import time
from typing import Deque, Dict, Optional

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from .time_alignment import (
    ConsecutiveFailureGate,
    StampOrder,
    StrictStampGuard,
    TimestampContractError,
    add_offset_nanoseconds,
    clock_residual_nanoseconds,
    clone_with_aligned_stamp,
    normalize_absolute_topic,
    stamp_to_nanoseconds,
    validate_frame_id,
)


def imu_measurements_are_finite(message: Imu) -> bool:
    """Return whether every IMU measurement and covariance value is finite."""
    values = (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
        *message.orientation_covariance,
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
        *message.angular_velocity_covariance,
        message.linear_acceleration.x,
        message.linear_acceleration.y,
        message.linear_acceleration.z,
        *message.linear_acceleration_covariance,
    )
    try:
        return all(math.isfinite(value) for value in values)
    except (TypeError, ValueError):
        return False


class AlignedImuRelay(Node):
    """Apply one audited constant clock offset without modifying IMU samples."""

    def __init__(self) -> None:
        super().__init__("aligned_fcu_imu_relay")

        self._input_topic = normalize_absolute_topic(
            self._read_only_parameter(
                "input_topic",
                "/mavros/imu/data_raw",
                "Absolute raw MAVROS IMU topic.",
            )
        )
        self._output_topic = normalize_absolute_topic(
            self._read_only_parameter(
                "output_topic",
                "/fcu/imu/data_raw_aligned",
                "Absolute aligned IMU topic consumed by cuVSLAM.",
            )
        )
        if self._input_topic == self._output_topic:
            raise RuntimeError("input_topic and output_topic must differ")

        self._expected_input_frame = validate_frame_id(
            self._read_only_parameter(
                "expected_input_frame_id",
                "base_link",
                "Frame that names the calibrated MAVROS IMU measurement axes.",
            )
        )
        self._output_frame = validate_frame_id(
            self._read_only_parameter(
                "output_frame_id",
                "fcu_imu",
                "Dedicated runtime name for the same calibrated IMU axes.",
            )
        )
        self._offset_ns = self._read_only_parameter(
            "imu_to_camera_offset_ns",
            1_737_987,
            "Integer offset in t_aligned = t_imu_raw + offset.",
        )
        if isinstance(self._offset_ns, bool) or not isinstance(self._offset_ns, int):
            raise RuntimeError("imu_to_camera_offset_ns must be an integer")
        if abs(self._offset_ns) > 100_000_000:
            raise RuntimeError("imu_to_camera_offset_ns exceeds the 100 ms safety bound")

        self._expected_rate_hz = float(
            self._read_only_parameter(
                "expected_rate_hz",
                170.0,
                "Measured external IMU rate used by the health diagnostic.",
            )
        )
        self._rate_tolerance_ratio = float(
            self._read_only_parameter(
                "rate_tolerance_ratio",
                0.15,
                "Allowed relative stamp-rate error before diagnostics warn.",
            )
        )
        self._maximum_gap_ratio = float(
            self._read_only_parameter(
                "maximum_gap_ratio",
                3.0,
                "Maximum accepted stamp gap as a multiple of the expected period.",
            )
        )
        self._diagnostic_period_sec = float(
            self._read_only_parameter(
                "diagnostic_period_sec",
                1.0,
                "Diagnostic publication period.",
            )
        )
        self._stale_after_sec = float(
            self._read_only_parameter(
                "stale_after_sec",
                2.0,
                "Maximum input silence before diagnostics report an error.",
            )
        )
        self._startup_timeout_sec = float(
            self._read_only_parameter(
                "startup_timeout_sec",
                15.0,
                "Maximum startup wait for the first publishable FCU IMU sample.",
            )
        )
        self._maximum_receipt_time_residual_sec = float(
            self._read_only_parameter(
                "maximum_receipt_time_residual_sec",
                0.25,
                "Maximum raw IMU stamp residual against host system time.",
            )
        )
        self._hardware_id = str(
            self._read_only_parameter(
                "hardware_id",
                "px4-highres-imu-105-ttyTHS2",
                "Stable external IMU hardware/transport identifier.",
            )
        ).strip()
        numeric_values = (
            self._expected_rate_hz,
            self._rate_tolerance_ratio,
            self._maximum_gap_ratio,
            self._diagnostic_period_sec,
            self._stale_after_sec,
            self._startup_timeout_sec,
            self._maximum_receipt_time_residual_sec,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise RuntimeError("relay numeric parameters must be finite")
        if self._expected_rate_hz <= 0.0:
            raise RuntimeError("expected_rate_hz must be positive")
        if not 0.0 < self._rate_tolerance_ratio < 1.0:
            raise RuntimeError("rate_tolerance_ratio must be in (0, 1)")
        if not 1.0 < self._maximum_gap_ratio <= 10.0:
            raise RuntimeError("maximum_gap_ratio must be in (1, 10]")
        if self._diagnostic_period_sec <= 0.0:
            raise RuntimeError("diagnostic_period_sec must be positive")
        if self._stale_after_sec <= self._diagnostic_period_sec:
            raise RuntimeError("stale_after_sec must exceed diagnostic_period_sec")
        if not self._stale_after_sec < self._startup_timeout_sec <= 120.0:
            raise RuntimeError("startup_timeout_sec must exceed stale_after_sec and be <= 120")
        if not 0.0 < self._maximum_receipt_time_residual_sec <= 1.0:
            raise RuntimeError("maximum_receipt_time_residual_sec must be in (0, 1]")
        if not self._hardware_id:
            raise RuntimeError("hardware_id must be non-empty")

        self._guard = StrictStampGuard()
        self._counts: Dict[str, int] = {
            "received": 0,
            "published": 0,
            "zero_stamp": 0,
            "invalid_stamp": 0,
            "duplicate": 0,
            "nonmonotonic": 0,
            "frame_mismatch": 0,
            "nonfinite_measurement": 0,
            "clock_domain_mismatch": 0,
            "aligned_out_of_range": 0,
        }
        self._recent_rejects: Dict[str, int] = {
            name: 0 for name in self._counts if name not in {"received", "published"}
        }
        history_size = max(32, int(math.ceil(self._expected_rate_hz * 3.0)))
        self._accepted_stamp_ns: Deque[int] = deque(maxlen=history_size)
        self._last_input_stamp_ns: Optional[int] = None
        self._last_output_stamp_ns: Optional[int] = None
        self._last_clock_residual_ns: Optional[int] = None
        self._last_receive_monotonic_ns: Optional[int] = None
        self._last_publish_monotonic_ns: Optional[int] = None
        self._maximum_recent_gap_ns = 0
        self._rate_gap_failure_gate = ConsecutiveFailureGate(3)
        self._reject_failure_gate = ConsecutiveFailureGate(3)
        self._started_monotonic_ns = time.monotonic_ns()

        self._output_publisher = self.create_publisher(
            Imu,
            self._output_topic,
            qos_profile_sensor_data,
        )
        self._diagnostic_publisher = self.create_publisher(
            DiagnosticArray,
            "/diagnostics",
            10,
        )
        self._input_subscription = self.create_subscription(
            Imu,
            self._input_topic,
            self._on_imu,
            qos_profile_sensor_data,
        )
        self._diagnostic_timer = self.create_timer(
            self._diagnostic_period_sec,
            self._publish_diagnostics,
        )

        self.get_logger().info(
            "Aligned FCU IMU relay ready: "
            f"{self._input_topic} [{self._expected_input_frame}] -> "
            f"{self._output_topic} [{self._output_frame}], "
            f"timestamp offset {self._offset_ns:+d} ns"
        )

    def _read_only_parameter(self, name: str, default_value, description: str):
        descriptor = ParameterDescriptor(description=description, read_only=True)
        return self.declare_parameter(name, default_value, descriptor).value

    def _reject(self, reason: str, message: str) -> None:
        self._counts[reason] += 1
        self._recent_rejects[reason] += 1
        count = self._counts[reason]
        if count == 1 or count & (count - 1) == 0:
            self.get_logger().warning(f"{message}; rejected {reason} count={count}")

    def _on_imu(self, message: Imu) -> None:
        now_monotonic_ns = time.monotonic_ns()
        self._counts["received"] += 1
        self._last_receive_monotonic_ns = now_monotonic_ns

        if message.header.frame_id != self._expected_input_frame:
            self._reject(
                "frame_mismatch",
                "input frame mismatch: "
                f"expected {self._expected_input_frame!r}, got {message.header.frame_id!r}",
            )
            return

        if not imu_measurements_are_finite(message):
            self._reject(
                "nonfinite_measurement",
                "input IMU contains a non-finite measurement or covariance value",
            )
            return

        try:
            input_stamp_ns = stamp_to_nanoseconds(
                message.header.stamp.sec,
                message.header.stamp.nanosec,
            )
        except TimestampContractError as error:
            self._reject("invalid_stamp", str(error))
            return
        if input_stamp_ns == 0:
            self._reject("zero_stamp", "input timestamp is zero")
            return

        try:
            clock_residual_ns = clock_residual_nanoseconds(
                input_stamp_ns,
                time.time_ns(),
            )
        except TimestampContractError as error:
            self._reject("clock_domain_mismatch", str(error))
            return
        self._last_clock_residual_ns = clock_residual_ns
        if abs(clock_residual_ns) > round(
            self._maximum_receipt_time_residual_sec * 1_000_000_000.0
        ):
            self._reject(
                "clock_domain_mismatch",
                "raw IMU timestamp is not in the ROS system-time domain: "
                f"residual={clock_residual_ns / 1e9:.6f}s",
            )
            return

        order = self._guard.classify(input_stamp_ns)
        if order is StampOrder.DUPLICATE:
            self._reject("duplicate", f"duplicate input timestamp {input_stamp_ns}")
            return
        if order is StampOrder.NONMONOTONIC:
            self._reject(
                "nonmonotonic",
                "input timestamp moved backwards: "
                f"{input_stamp_ns} <= {self._guard.last_accepted_ns}",
            )
            return

        try:
            output_stamp_ns = add_offset_nanoseconds(input_stamp_ns, self._offset_ns)
            if output_stamp_ns == 0:
                raise TimestampContractError("aligned timestamp is zero")
            output = clone_with_aligned_stamp(
                message,
                self._output_frame,
                self._offset_ns,
            )
        except TimestampContractError as error:
            self._reject("aligned_out_of_range", str(error))
            return

        if self._last_input_stamp_ns is not None:
            gap_ns = input_stamp_ns - self._last_input_stamp_ns
            self._maximum_recent_gap_ns = max(self._maximum_recent_gap_ns, gap_ns)
        self._output_publisher.publish(output)
        self._guard.commit(input_stamp_ns)
        self._last_input_stamp_ns = input_stamp_ns
        self._last_output_stamp_ns = output_stamp_ns
        self._last_publish_monotonic_ns = now_monotonic_ns
        self._accepted_stamp_ns.append(input_stamp_ns)
        self._counts["published"] += 1

    def _stamp_rate_hz(self) -> Optional[float]:
        if len(self._accepted_stamp_ns) < 2:
            return None
        duration_ns = self._accepted_stamp_ns[-1] - self._accepted_stamp_ns[0]
        if duration_ns <= 0:
            return None
        return (
            (len(self._accepted_stamp_ns) - 1)
            * 1_000_000_000.0
            / duration_ns
        )

    def _publish_diagnostics(self) -> None:
        now_monotonic_ns = time.monotonic_ns()
        startup_age_sec = (
            now_monotonic_ns - self._started_monotonic_ns
        ) / 1_000_000_000.0
        if self._last_receive_monotonic_ns is None:
            receive_age_sec = startup_age_sec
        else:
            receive_age_sec = (
                now_monotonic_ns - self._last_receive_monotonic_ns
            ) / 1_000_000_000.0
        if self._last_publish_monotonic_ns is None:
            publish_age_sec = startup_age_sec
        else:
            publish_age_sec = (
                now_monotonic_ns - self._last_publish_monotonic_ns
            ) / 1_000_000_000.0
        stamp_rate_hz = self._stamp_rate_hz()
        maximum_allowed_gap_ns = round(
            self._maximum_gap_ratio * 1_000_000_000.0 / self._expected_rate_hz
        )

        recent_errors = sum(
            self._recent_rejects[name]
            for name in (
                "invalid_stamp",
                "nonmonotonic",
                "frame_mismatch",
                "nonfinite_measurement",
                "clock_domain_mismatch",
                "aligned_out_of_range",
                "zero_stamp",
            )
        )
        recent_rejects = recent_errors + self._recent_rejects["duplicate"]
        gap_is_bad = self._maximum_recent_gap_ns > maximum_allowed_gap_ns
        rate_is_bad = stamp_rate_hz is not None and (
            abs(stamp_rate_hz - self._expected_rate_hz) / self._expected_rate_hz
            > self._rate_tolerance_ratio
        )
        fatal_reason = None
        if self._counts["received"] == 0:
            level = DiagnosticStatus.STALE
            summary = "waiting for raw FCU IMU"
            if startup_age_sec > self._startup_timeout_sec:
                level = DiagnosticStatus.ERROR
                summary = "raw FCU IMU did not appear before startup timeout"
                fatal_reason = summary
        elif receive_age_sec > self._stale_after_sec:
            level = DiagnosticStatus.ERROR
            summary = "raw FCU IMU input is stale"
            fatal_reason = summary
        elif self._counts["published"] == 0:
            level = DiagnosticStatus.ERROR
            summary = "raw IMU is present but no aligned sample has been published"
            if startup_age_sec > self._startup_timeout_sec:
                fatal_reason = summary
        elif publish_age_sec > self._stale_after_sec:
            level = DiagnosticStatus.ERROR
            summary = "aligned FCU IMU output is stale"
            fatal_reason = summary
        elif recent_errors:
            level = DiagnosticStatus.ERROR
            summary = "unsafe IMU samples were rejected"
        elif self._recent_rejects["duplicate"]:
            level = DiagnosticStatus.WARN
            summary = "duplicate IMU samples were rejected"
        elif gap_is_bad:
            level = DiagnosticStatus.WARN
            summary = "aligned IMU stamp gap exceeded the configured limit"
        elif rate_is_bad:
            level = DiagnosticStatus.WARN
            summary = "aligned IMU stamp rate is outside tolerance"
        elif stamp_rate_hz is None:
            level = DiagnosticStatus.WARN
            summary = "waiting for enough aligned samples to measure rate"
        else:
            level = DiagnosticStatus.OK
            summary = "aligned FCU IMU stream is healthy"

        reject_failure = self._reject_failure_gate.observe(recent_rejects > 0)
        rate_gap_failure = self._rate_gap_failure_gate.observe(gap_is_bad or rate_is_bad)
        if fatal_reason is None and reject_failure:
            level = DiagnosticStatus.ERROR
            summary = "unsafe IMU samples were rejected for three diagnostics"
            fatal_reason = summary
        elif fatal_reason is None and rate_gap_failure:
            level = DiagnosticStatus.ERROR
            summary = "aligned IMU rate/gap remained unhealthy for three diagnostics"
            fatal_reason = summary

        status = DiagnosticStatus()
        status.level = level
        status.name = f"{self.get_fully_qualified_name()}: aligned FCU IMU"
        status.message = summary
        status.hardware_id = self._hardware_id
        values = {
            "input_topic": self._input_topic,
            "output_topic": self._output_topic,
            "expected_input_frame": self._expected_input_frame,
            "output_frame": self._output_frame,
            "imu_to_camera_offset_ns": self._offset_ns,
            "offset_equation": "t_aligned = t_imu_raw + offset",
            "expected_rate_hz": f"{self._expected_rate_hz:.6f}",
            "maximum_receipt_time_residual_sec": (
                f"{self._maximum_receipt_time_residual_sec:.6f}"
            ),
            "last_clock_residual_ms": (
                "not_available"
                if self._last_clock_residual_ns is None
                else f"{self._last_clock_residual_ns / 1e6:.6f}"
            ),
            "stamp_rate_hz": (
                "not_available" if stamp_rate_hz is None else f"{stamp_rate_hz:.6f}"
            ),
            "last_receive_age_sec": f"{receive_age_sec:.6f}",
            "last_publish_age_sec": f"{publish_age_sec:.6f}",
            "maximum_recent_stamp_gap_ms": f"{self._maximum_recent_gap_ns / 1e6:.6f}",
            "maximum_allowed_stamp_gap_ms": f"{maximum_allowed_gap_ns / 1e6:.6f}",
            "consecutive_rate_gap_warning_cycles": self._rate_gap_failure_gate.count,
            "consecutive_reject_warning_cycles": self._reject_failure_gate.count,
            "last_input_stamp_ns": self._last_input_stamp_ns,
            "last_output_stamp_ns": self._last_output_stamp_ns,
        }
        values.update(self._counts)
        status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]

        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = self.get_clock().now().to_msg()
        diagnostics.status = [status]
        self._diagnostic_publisher.publish(diagnostics)

        for name in self._recent_rejects:
            self._recent_rejects[name] = 0
        self._maximum_recent_gap_ns = 0
        if fatal_reason is not None:
            self.get_logger().fatal(f"{fatal_reason}; shutting down runtime bringup")
            raise RuntimeError(fatal_reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = AlignedImuRelay()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

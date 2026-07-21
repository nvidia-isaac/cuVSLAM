"""Pure contracts used by the aligned external-IMU relay."""

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


NANOSECONDS_PER_SECOND = 1_000_000_000
MAX_TIME_SECOND = 2_147_483_647
MAX_TIME_NANOSECONDS = (
    MAX_TIME_SECOND * NANOSECONDS_PER_SECOND + NANOSECONDS_PER_SECOND - 1
)


class TimestampContractError(ValueError):
    """Raised when a timestamp cannot satisfy the runtime clock contract."""


class StampOrder(Enum):
    """Relationship between an input stamp and the last accepted stamp."""

    ACCEPT = "accept"
    DUPLICATE = "duplicate"
    NONMONOTONIC = "nonmonotonic"


def _require_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimestampContractError(f"{name} must be an integer")
    return value


def stamp_to_nanoseconds(sec: int, nanosec: int) -> int:
    """Convert a non-negative ROS system-time stamp using integer arithmetic."""
    sec = _require_integer("sec", sec)
    nanosec = _require_integer("nanosec", nanosec)
    if sec < 0:
        raise TimestampContractError("sec must be non-negative for system time")
    if not 0 <= nanosec < NANOSECONDS_PER_SECOND:
        raise TimestampContractError("nanosec must be in [0, 1000000000)")
    if sec > MAX_TIME_SECOND:
        raise TimestampContractError("sec exceeds builtin_interfaces/Time range")
    return sec * NANOSECONDS_PER_SECOND + nanosec


def add_offset_nanoseconds(stamp_ns: int, offset_ns: int) -> int:
    """Add a signed constant offset and enforce builtin_interfaces/Time bounds."""
    stamp_ns = _require_integer("stamp_ns", stamp_ns)
    offset_ns = _require_integer("offset_ns", offset_ns)
    if not 0 <= stamp_ns <= MAX_TIME_NANOSECONDS:
        raise TimestampContractError("input timestamp is outside ROS Time range")
    aligned_ns = stamp_ns + offset_ns
    if aligned_ns < 0:
        raise TimestampContractError("aligned timestamp is negative")
    if aligned_ns > MAX_TIME_NANOSECONDS:
        raise TimestampContractError("aligned timestamp exceeds ROS Time range")
    return aligned_ns


def split_nanoseconds(stamp_ns: int) -> Tuple[int, int]:
    """Split checked nanoseconds into canonical ROS Time fields."""
    stamp_ns = _require_integer("stamp_ns", stamp_ns)
    if not 0 <= stamp_ns <= MAX_TIME_NANOSECONDS:
        raise TimestampContractError("timestamp is outside ROS Time range")
    return divmod(stamp_ns, NANOSECONDS_PER_SECOND)


def clock_residual_nanoseconds(stamp_ns: int, reference_ns: int) -> int:
    """Return a checked signed residual against the active ROS system clock."""
    stamp_ns = _require_integer("stamp_ns", stamp_ns)
    reference_ns = _require_integer("reference_ns", reference_ns)
    if not 0 <= stamp_ns <= MAX_TIME_NANOSECONDS:
        raise TimestampContractError("sensor timestamp is outside ROS Time range")
    if not 0 <= reference_ns <= MAX_TIME_NANOSECONDS:
        raise TimestampContractError("reference timestamp is outside ROS Time range")
    return stamp_ns - reference_ns


def validate_frame_id(frame_id: str) -> str:
    """Validate an unambiguous tf2 frame identifier."""
    if not isinstance(frame_id, str):
        raise ValueError("frame_id must be a string")
    frame_id = frame_id.strip()
    if not frame_id or frame_id.startswith("/") or any(char.isspace() for char in frame_id):
        raise ValueError("frame_id must be non-empty, relative, and contain no whitespace")
    return frame_id


def normalize_absolute_topic(topic: str) -> str:
    """Validate and normalize an absolute ROS topic used by this integration."""
    if not isinstance(topic, str):
        raise ValueError("topic must be a string")
    topic = topic.strip()
    if not topic.startswith("/"):
        raise ValueError("topic must be absolute")
    parts = [part for part in topic.split("/") if part]
    if not parts or any(any(char.isspace() for char in part) for part in parts):
        raise ValueError("topic must contain a name and no whitespace")
    return "/" + "/".join(parts)


def clone_with_aligned_stamp(message: Any, output_frame_id: str, offset_ns: int) -> Any:
    """Deep-copy an IMU-like message and modify only its header stamp and frame."""
    output_frame_id = validate_frame_id(output_frame_id)
    input_ns = stamp_to_nanoseconds(
        message.header.stamp.sec,
        message.header.stamp.nanosec,
    )
    output_ns = add_offset_nanoseconds(input_ns, offset_ns)
    output_sec, output_nanosec = split_nanoseconds(output_ns)

    output = copy.deepcopy(message)
    output.header.stamp.sec = output_sec
    output.header.stamp.nanosec = output_nanosec
    output.header.frame_id = output_frame_id
    return output


@dataclass
class StrictStampGuard:
    """Track the last accepted input stamp without advancing on rejected samples."""

    last_accepted_ns: Optional[int] = None

    def classify(self, stamp_ns: int) -> StampOrder:
        stamp_ns = _require_integer("stamp_ns", stamp_ns)
        if self.last_accepted_ns is None or stamp_ns > self.last_accepted_ns:
            return StampOrder.ACCEPT
        if stamp_ns == self.last_accepted_ns:
            return StampOrder.DUPLICATE
        return StampOrder.NONMONOTONIC

    def commit(self, stamp_ns: int) -> None:
        stamp_ns = _require_integer("stamp_ns", stamp_ns)
        if self.classify(stamp_ns) is not StampOrder.ACCEPT:
            raise TimestampContractError("cannot commit a non-increasing timestamp")
        self.last_accepted_ns = stamp_ns


@dataclass
class ConsecutiveFailureGate:
    """Trip only after a condition remains unhealthy for a fixed number of checks."""

    threshold: int
    count: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, int)
            or self.threshold <= 0
        ):
            raise ValueError("threshold must be a positive integer")

    def observe(self, unhealthy: bool) -> bool:
        if unhealthy:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.threshold

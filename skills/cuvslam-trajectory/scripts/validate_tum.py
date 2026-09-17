#!/usr/bin/env python3

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

"""Validate a trajectory in TUM ``timestamp tx ty tz qx qy qz qw`` format."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory does not satisfy the replay contract."""


def validate_tum(
    path: Path,
    *,
    expected_rows: int | None = None,
    min_coverage: float = 0.9,
    quaternion_tolerance: float = 1e-3,
    require_motion: bool = True,
) -> dict[str, Any]:
    """Validate *path* and return JSON-serializable trajectory statistics."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise TrajectoryValidationError(f"trajectory does not exist: {path}")
    if not 0.0 < min_coverage <= 1.0:
        raise TrajectoryValidationError("min_coverage must be in (0, 1]")

    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 8:
            raise TrajectoryValidationError(
                f"line {line_number}: expected 8 fields, found {len(fields)}"
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise TrajectoryValidationError(f"line {line_number}: non-numeric field") from exc
        if not all(math.isfinite(value) for value in values):
            raise TrajectoryValidationError(f"line {line_number}: non-finite value")
        if rows and values[0] <= rows[-1][0]:
            raise TrajectoryValidationError(
                f"line {line_number}: timestamp {values[0]} is not strictly increasing"
            )
        quaternion_norm = math.sqrt(sum(value * value for value in values[4:8]))
        if abs(quaternion_norm - 1.0) > quaternion_tolerance:
            raise TrajectoryValidationError(
                f"line {line_number}: quaternion norm {quaternion_norm:.9g} is not near 1"
            )
        rows.append(values)

    if not rows:
        raise TrajectoryValidationError("trajectory contains no data rows")
    if expected_rows is not None:
        if expected_rows <= 0:
            raise TrajectoryValidationError("expected_rows must be positive")
        coverage = len(rows) / expected_rows
        if coverage < min_coverage:
            raise TrajectoryValidationError(
                f"trajectory coverage {coverage:.3%} is below required {min_coverage:.3%} "
                f"({len(rows)}/{expected_rows} rows)"
            )
    else:
        coverage = None

    path_length = 0.0
    cumulative_rotation = 0.0
    for previous, current in zip(rows, rows[1:]):
        path_length += math.sqrt(sum((current[index] - previous[index]) ** 2 for index in range(1, 4)))
        dot = abs(sum(previous[index] * current[index] for index in range(4, 8)))
        cumulative_rotation += 2.0 * math.acos(min(1.0, max(-1.0, dot)))

    if (
        require_motion
        and len(rows) > 1
        and path_length <= 1e-6
        and cumulative_rotation <= 1e-6
    ):
        raise TrajectoryValidationError("trajectory is static")

    return {
        "path": str(path),
        "rows": len(rows),
        "expected_rows": expected_rows,
        "coverage": coverage,
        "first_timestamp": rows[0][0],
        "last_timestamp": rows[-1][0],
        "duration_seconds": rows[-1][0] - rows[0][0],
        "path_length": path_length,
        "cumulative_rotation_radians": cumulative_rotation,
        "valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument("--quaternion-tolerance", type=float, default=1e-3)
    parser.add_argument("--allow-static", action="store_true")
    args = parser.parse_args()

    try:
        result = validate_tum(
            args.trajectory,
            expected_rows=args.expected_rows,
            min_coverage=args.min_coverage,
            quaternion_tolerance=args.quaternion_tolerance,
            require_motion=not args.allow_static,
        )
    except TrajectoryValidationError as exc:
        parser.exit(1, f"TUM validation failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Trajectory math: SE(3) conversion, interpolation, alignment, and error metrics.

Quaternions are ``x, y, z, w`` throughout, matching both cuVSLAM and SciPy.

Dataset-independent by construction: nothing here reads a recording. Anything that
has to know an on-disk layout lives in a dataset package such as ``sessions.euroc``.
``errors`` and ``align`` are used only by ``validation.evaluate_localization``.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def matrix(p):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(p["rotation"]).as_matrix()
    m[:3, 3] = p["translation"]
    return m


def pose(m):
    return {"translation": m[:3, 3].tolist(), "rotation": Rotation.from_matrix(m[:3, :3]).as_quat().tolist()}


def load_trajectory(path):
    rows = json.loads(Path(path).read_text())
    return np.array([r["timestamp_ns"] for r in rows], dtype=np.int64), np.array([matrix(r) for r in rows])


def interpolate(timestamps, transforms, query):
    """Interpolate translation (linear) and orientation (Slerp) at ``query`` times.

    Queries outside the trajectory span clamp to the nearest endpoint rather than
    extrapolating, so a replay that starts slightly late still yields a usable pose for
    the first keyframes. Supply a replay covering all retained keyframes to avoid it.
    """
    # Subtract the integer nanosecond origin before converting to floating seconds.
    t0 = int(timestamps[0])
    times = (timestamps - t0).astype(float) * 1e-9
    query_s = (np.asarray(query, dtype=np.int64) - t0).astype(float) * 1e-9
    result = np.tile(np.eye(4), (len(query_s), 1, 1))
    for axis in range(3):
        result[:, axis, 3] = np.interp(query_s, times, transforms[:, axis, 3])
    result[:, :3, :3] = Slerp(times, Rotation.from_matrix(transforms[:, :3, :3]))(
        np.clip(query_s, times[0], times[-1])
    ).as_matrix()
    return result


def align(source, target):
    """Rigid position alignment; never fit scale."""
    a, b = np.mean(source, 0), np.mean(target, 0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(vt.T @ u.T)
    m = np.eye(4)
    m[:3, :3] = vt.T @ sign @ u.T
    m[:3, 3] = b - m[:3, :3] @ a
    return m


def errors(estimate, truth):
    translation = np.linalg.norm(estimate[:, :3, 3] - truth[:, :3, 3], axis=1)
    rotation = (
        Rotation.from_matrix(np.transpose(truth[:, :3, :3], (0, 2, 1)) @ estimate[:, :3, :3]).magnitude() * 180 / np.pi
    )
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(translation**2))),
        "translation_median_m": float(np.median(translation)),
        "translation_p95_m": float(np.quantile(translation, 0.95)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p95_deg": float(np.quantile(rotation, 0.95)),
    }

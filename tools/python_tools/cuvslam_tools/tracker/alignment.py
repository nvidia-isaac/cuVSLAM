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

"""Trajectory alignment and absolute (ATE / ARE RMSE) error metrics.

NumPy only, so the module imports without a cuvslam build. The metrics follow
the usual absolute-trajectory-error definition (as in evo_ape --align, plus
--correct_scale for monocular runs): align the whole estimated trajectory to
ground truth once, then take the RMSE of the position and orientation errors.
"""

from typing import Sequence, Tuple

import numpy as np


def umeyama_alignment(src: np.ndarray, dst: np.ndarray,
                      with_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares transform mapping src points onto dst points.

    Umeyama, "Least-squares estimation of transformation parameters between two
    point patterns", IEEE TPAMI 13(4), 1991.

    Args:
        src: Nx3 points to transform
        dst: Nx3 target points
        with_scale: Also estimate a uniform scale (Sim(3)); otherwise rigid (SE(3))

    Returns:
        (R, t, s) such that dst_i ~= s * R @ src_i + t, with det(R) = +1
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Point sets must both be Nx3, got {src.shape} and {dst.shape}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    U, d, Vt = np.linalg.svd(dst_centered.T @ src_centered / len(src))
    # Without this sign flip, planar or noisy point sets can yield a reflection.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    scale = 1.0
    src_variance = np.mean(np.sum(src_centered ** 2, axis=1))
    if with_scale and src_variance > 0:
        scale = float(np.sum(d * np.diag(S)) / src_variance)

    t = dst_mean - scale * R @ src_mean
    return R, t, scale


def rotation_angles_deg(rotations: np.ndarray) -> np.ndarray:
    """Rotation angle in degrees of each 3x3 matrix in an Nx3x3 array."""
    cos = (np.trace(rotations, axis1=1, axis2=2) - 1) / 2
    skew = rotations - rotations.transpose(0, 2, 1)
    sin = np.linalg.norm(skew[:, [2, 0, 1], [1, 2, 0]], axis=1) / 2
    return np.degrees(np.arctan2(sin, cos))


def absolute_trajectory_errors(gt_transforms: Sequence[np.ndarray], est_transforms: Sequence[np.ndarray],
                               with_scale: bool = False) -> Tuple[float, float]:
    """Translation RMSE [m] and rotation RMSE [deg] after aligning est to gt.

    Args:
        gt_transforms: Ground-truth 4x4 poses
        est_transforms: Estimated 4x4 poses, paired with gt_transforms by index
        with_scale: Align with Sim(3) instead of SE(3), for scale-ambiguous (monocular) tracking

    Returns:
        (ate_rmse, are_rmse), or (0.0, 0.0) for an empty trajectory. are_rmse is NaN when the
        positions are stationary or collinear, since rotation about them is unobservable.
    """
    if len(gt_transforms) != len(est_transforms):
        raise ValueError(f"Trajectories differ in length: {len(gt_transforms)} vs {len(est_transforms)}")
    if not gt_transforms:
        return 0.0, 0.0

    gt = np.asarray(gt_transforms)
    est = np.asarray(est_transforms)
    gt_positions, est_positions = gt[:, :3, 3], est[:, :3, 3]
    R, t, scale = umeyama_alignment(est_positions, gt_positions, with_scale)

    position_errors = gt_positions - (scale * est_positions @ R.T + t)
    ate_rmse = np.sqrt(np.mean(np.sum(position_errors ** 2, axis=1)))

    covariance = (gt_positions - gt_positions.mean(axis=0)).T @ (est_positions - est_positions.mean(axis=0))
    if np.linalg.matrix_rank(covariance) < 2:
        return float(ate_rmse), float("nan")
    rotation_errors = gt[:, :3, :3].transpose(0, 2, 1) @ R @ est[:, :3, :3]
    are_rmse = np.sqrt(np.mean(rotation_angles_deg(rotation_errors) ** 2))
    return float(ate_rmse), float(are_rmse)

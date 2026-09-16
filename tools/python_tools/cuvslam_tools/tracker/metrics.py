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

"""Functions for calculating tracking and performance metrics."""

import numpy as np
from typing import Any, Dict, List, Optional

from cuvslam_tools.tracker.pose_utils import pose_to_transform


def translation_error(pose_error: np.ndarray) -> float:
    """Calculate translation error from pose error matrix.

    Args:
        pose_error: 4x4 transformation matrix representing pose error

    Returns:
        Translation error magnitude
    """
    return float(np.linalg.norm(pose_error[:3, 3]))


def rotation_error(pose_error: np.ndarray) -> float:
    """Calculate rotation error from pose error matrix in degrees.

    Args:
        pose_error: 4x4 transformation matrix representing pose error

    Returns:
        Rotation error in degrees
    """
    a = (np.trace(pose_error[:3, :3]) - 1) / 2
    a = min(max(a, -1), 1)
    return np.degrees(np.arccos(a))


def transform_between_pointclouds(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Find rigid transform between two point clouds using Kabsch algorithm.
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    Args:
        A: First point cloud as Nx3 array of points
        B: Second point cloud as Nx3 array of points

    Returns:
        4x4 rigid transformation matrix that transforms points in A to align with B

    Raises:
        ValueError: If point clouds have different shapes
    """
    if A.shape != B.shape:
        raise ValueError("Point clouds must have same shape")

    # Calculate centroids
    A_mean = np.mean(A, axis=0)
    B_mean = np.mean(B, axis=0)

    # Center point clouds
    Am = A - A_mean
    Bm = B - B_mean

    # Calculate optimal rotation using SVD
    H = Am.T @ Bm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Calculate translation
    t = -R @ A_mean + B_mean

    # Build transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t

    return transform


def calc_kabsch_rms_metric(gt_transforms: List[np.ndarray],
                           result_transforms: List[np.ndarray]) -> float:
    """Calculate Kabsch (RMS) metric between ground truth and result trajectories."""
    # Extract translation points from transforms
    points_gt = np.array([transform[:3, 3] for transform in gt_transforms])
    points_result = np.array([transform[:3, 3] for transform in result_transforms])

    # Find optimal transform between point clouds
    transform = transform_between_pointclouds(points_result, points_gt)

    # Calculate RMS error
    sum_squared_error = 0.0
    count = 0
    for i in range(len(gt_transforms)):
        if i >= len(result_transforms):
            break

        # Transform result point using alignment transform
        vo = transform @ np.append(result_transforms[i][:3, 3], 1)  # Make homogeneous
        gt = gt_transforms[i][:3, 3]

        # Calculate squared error
        error = vo[:3] - gt  # Only take x,y,z components
        sum_squared_error += np.dot(error, error)  # squared norm
        count += 1

    if count:
        rms = np.sqrt(sum_squared_error / count)
    else:
        rms = 0.0

    return rms


def get_frame_mapping(total_frames: int,
                      frame_metadata: Dict,
                      num_loops: int = 0,
                      repeat_type: str = "none") -> Dict[int, int]:
    """Map processed frame_id -> GT index for the run.

    Shuttle: compare against the final backward pass (reversed -> forward GT order).
    Repeat: compare against the final forward pass (already in GT order).
    None / single pass: identity mapping.
    """
    repeat_type = (repeat_type or "none").lower()
    if num_loops <= 0 or repeat_type == "none":
        return {fid: fid for fid in frame_metadata.keys()}

    if repeat_type == "shuttle":
        backward_frames = sorted([
            fid for fid, meta in frame_metadata.items()
            if meta['loop'] == num_loops - 1 and not meta['forward']
        ])
        return {fid: total_frames - 1 - i for i, fid in enumerate(backward_frames)}

    # repeat: last forward pass, frames in forward order
    forward_frames = sorted([
        fid for fid, meta in frame_metadata.items()
        if meta['loop'] == num_loops - 1 and meta['forward']
    ])
    return {fid: i for i, fid in enumerate(forward_frames)}

def trajectory_distances(poses: List[np.ndarray]) -> np.ndarray:
    """Calculate cumulative distances along trajectory.

    Args:
        poses: List of 4x4 transformation matrices

    Returns:
        Array of cumulative distances
    """
    distances: List[float] = [0.0]  # First pose has zero distance
    for i in range(1, len(poses)):
        # Calculate distance between current and previous pose
        pose_delta = np.linalg.inv(poses[i-1]) @ poses[i]
        distances.append(distances[-1] + float(np.linalg.norm(pose_delta[:3, 3])))
    return np.array(distances)


def last_frame_from_segment_length(
    distances: np.ndarray,
    first_frame: int,
    segment_length: float
) -> int:
    """Find last frame index that gives desired segment length.

    Args:
        distances: Array of cumulative distances
        first_frame: Starting frame index
        segment_length: Desired segment length

    Returns:
        Last frame index or -1 if segment length cannot be achieved
    """
    for i in range(first_frame + 1, len(distances)):
        if distances[i] > distances[first_frame] + segment_length:
            return i
    return -1


def calculate_sequence_errors(
    poses: Dict[int, Any],
    gt_transforms: List[np.ndarray],
    stat,
    frame_metadata: Dict[int, Dict],
    use_segments: bool = False,
    segment_lengths: List[int] = [],
    num_loops: int = 0,
    repeat_type: str = "none",
    frame_mapping: Optional[Dict[int, int]] = None,
) -> None:
    """Calculate tracking errors compared to ground truth.

    Args:
        poses: Dictionary of estimated poses
        gt_transforms: List of ground truth transformation matrices
        stat: Statistics object to update
        frame_metadata: Frame metadata dictionary
        use_segments: Whether to use segment-based error calculation
        segment_lengths: List of segment lengths for error calculation
        num_loops: Number of tracking loops
    """
    gt_transforms_filtered = []
    pose_transforms = []

    gt_length = 0
    trans_error = 0
    rot_error = 0
    n_error_segments = 0
    total_frames = len(gt_transforms)
    if frame_mapping is None:
        frame_mapping = get_frame_mapping(total_frames, frame_metadata, num_loops, repeat_type)

    for frame_id_pose, frame_id_gt in frame_mapping.items():
        if frame_id_gt >= len(gt_transforms):
            continue
        # Lost/filtered frames aren't in `poses`; .get lets the None guard skip them.
        pose = poses.get(frame_id_pose)
        gt_transform = gt_transforms[frame_id_gt]
        if pose is None or gt_transform is None:
            continue
        gt_transforms_filtered.append(gt_transform)
        pose_transforms.append(pose_to_transform(pose))

    kabsch_rms_metric = calc_kabsch_rms_metric(gt_transforms_filtered, pose_transforms)  # m
    total_frames = len(gt_transforms_filtered)

    # Ensure stat owns a fresh list so non-segment runs don't share the Stat class default.
    stat.seg_err_points = []

    if not use_segments:
        # Calculate ATE and ARE errors for entire sequence
        for i in range(total_frames):
            pose_gt = gt_transforms_filtered[i]
            if i > 0:
                pose_gt_prev = gt_transforms_filtered[i - 1]
                gt_rel = np.linalg.inv(pose_gt_prev) @ pose_gt
                gt_length += np.linalg.norm(gt_rel[:3, 3])

            pose_result = pose_transforms[i]
            pose_error = np.linalg.inv(pose_gt) @ pose_result

            trans_error += translation_error(pose_error)
            rot_error += rotation_error(pose_error)

        if gt_length == 0 or total_frames == 0:
            print("Warning: Cannot calculate errors - zero trajectory length or no frames")
            return

        t_avg = trans_error * 100 / (gt_length * total_frames)  # %
        r_avg = rot_error / total_frames  # deg

    else:
        # Calculate ATE and ARE errors for each segment

        # TODO: Move to the configuration
        step_size = 10  # every second since KITTI camera is 10 fps
        # pre-compute distances (from ground truth as reference)
        dist = trajectory_distances(gt_transforms_filtered)

        # Per-(start_frame, segment_length) error points, retained for the
        # cross-sequence error-vs-path-length aggregate plot in the report.
        seg_err_points: List[Dict[str, float]] = []

        for first_frame in range(0, total_frames, step_size):
            for length in segment_lengths:
                if length <= 0:
                    continue  # guard against bad cfg entries; divide-by-zero below
                last_frame = last_frame_from_segment_length(dist, first_frame, length)

                # continue, if sequence is not long enough
                if last_frame == -1:
                    continue

                # compute rotational and translational errors
                pose_delta_gt = np.linalg.inv(gt_transforms_filtered[first_frame]) @ gt_transforms_filtered[last_frame]
                pose_delta_result = np.linalg.inv(pose_transforms[first_frame]) @ pose_transforms[last_frame]
                pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt

                t_rel = translation_error(pose_error) / length  # dimensionless
                r_per_m = rotation_error(pose_error) / length  # deg/m

                trans_error += t_rel
                rot_error += r_per_m
                n_error_segments += 1
                seg_err_points.append({
                    'length': float(length),
                    't_pct': t_rel * 100.0,
                    'r_deg_per_m': r_per_m,
                })

        stat.seg_err_points = seg_err_points

        if n_error_segments == 0:
            print("Warning: Cannot calculate errors - no valid segments")
            t_avg = np.nan
            r_avg = np.nan
        else:
            t_avg = trans_error * 100 / n_error_segments  # % / m
            r_avg = rot_error / n_error_segments  # deg / m

    # Update statistics
    stat.gt_av_translation_error = t_avg
    stat.gt_av_rotation_error = r_avg
    stat.gt_n_error_segments = n_error_segments
    stat.gt_simple_error = kabsch_rms_metric

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

import unittest

import numpy as np

from cuvslam_tools.tracker.alignment import absolute_trajectory_errors, umeyama_alignment


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = np.radians(degrees)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def pose(R, t):
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform


def random_trajectory(n=50, seed=0):
    rng = np.random.default_rng(seed)
    positions = np.cumsum(rng.normal(size=(n, 3)), axis=0)
    return [pose(rotation(rng.normal(size=3), rng.uniform(0, 180)), p) for p in positions]


class TestUmeyamaAlignment(unittest.TestCase):
    def test_mirrored_points_still_yield_a_rotation(self):
        src = np.random.default_rng(1).normal(size=(30, 3))
        dst = src * [-1, 1, 1]

        R, _, _ = umeyama_alignment(src, dst)

        self.assertAlmostEqual(np.linalg.det(R), 1.0)


class TestAbsoluteTrajectoryErrors(unittest.TestCase):
    def test_world_frame_offset_is_aligned_out(self):
        gt = random_trajectory()
        R_offset, t_offset = rotation([1, 2, 3], 70), np.array([5.0, -2.0, 1.0])
        for scale, with_scale in ((1.0, False), (0.3, True)):
            est = [pose(R_offset @ T[:3, :3], scale * R_offset @ T[:3, 3] + t_offset) for T in gt]

            ate, are = absolute_trajectory_errors(gt, est, with_scale=with_scale)

            self.assertAlmostEqual(ate, 0.0, places=9)
            self.assertAlmostEqual(are, 0.0, places=5)

        ate_unscaled, are_unscaled = absolute_trajectory_errors(gt, est, with_scale=False)
        self.assertGreater(ate_unscaled, 1.0)
        self.assertAlmostEqual(are_unscaled, 0.0, places=5)

    def test_orientation_error_is_measured_after_alignment(self):
        gt = random_trajectory()
        R_offset = rotation([0, 0, 1], 40)
        est = [pose(R_offset @ T[:3, :3] @ rotation([1, 0, 0], 5), R_offset @ T[:3, 3]) for T in gt]

        ate, are = absolute_trajectory_errors(gt, est)

        self.assertAlmostEqual(ate, 0.0, places=9)
        self.assertAlmostEqual(are, 5.0, places=6)


if __name__ == "__main__":
    unittest.main()

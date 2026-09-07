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

"""Shared conversion primitives that are not tied to one dataset.

The association and ground-truth-sampling primitives are covered alongside the
first dataset that used them, in test_tum_conversion.py. This file covers the
ones added for datasets whose poses arrive as matrices in a non-camera frame.
"""

import math
import unittest

from cuvslam_tools.dataset_preparation import rgbd


def _rotation_about_z(degrees):
    angle = math.radians(degrees)
    return [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]


class TestMatrixToQuaternion(unittest.TestCase):
    def test_round_trips_through_quaternion_to_matrix(self):
        for degrees in (1.0, 45.0, 90.0, 179.0, 180.0, 270.0, 359.0):
            with self.subTest(degrees=degrees):
                rotation = _rotation_about_z(degrees)
                recovered = rgbd.quaternion_to_matrix(rgbd.matrix_to_quaternion(rotation))
                for row in range(3):
                    for column in range(3):
                        self.assertAlmostEqual(recovered[row][column], rotation[row][column], places=9)

    def test_half_turn_stays_accurate(self):
        # At 180 degrees the trace is -1, so a trace-only formula divides by zero
        # or loses every digit. Each axis exercises a different branch.
        for axis, expected in enumerate(("x", "y", "z")):
            rotation = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            rotation[axis][axis] = 1.0
            with self.subTest(axis=expected):
                quaternion = rgbd.matrix_to_quaternion(rotation)
                self.assertAlmostEqual(math.hypot(*quaternion), 1.0, places=12)
                self.assertAlmostEqual(abs(quaternion[axis]), 1.0, places=9)
                recovered = rgbd.quaternion_to_matrix(quaternion)
                for row in range(3):
                    for column in range(3):
                        self.assertAlmostEqual(recovered[row][column], rotation[row][column], places=9)

class TestBodyFromCamera(unittest.TestCase):
    def _trajectory(self):
        # Pure rotation about z, so a lateral camera offset sweeps an arc and
        # cannot cancel out of the relative poses.
        return [
            (0, [0.0, 0.0, 0.0], rgbd.matrix_to_quaternion(_rotation_about_z(0.0))),
            (1_000_000_000, [0.0, 0.0, 0.0], rgbd.matrix_to_quaternion(_rotation_about_z(90.0))),
        ]

    def test_absent_transform_leaves_the_trajectory_alone(self):
        trajectory = self._trajectory()
        timestamps = [0, 1_000_000_000]
        lines = rgbd.relative_ground_truth_lines(trajectory, timestamps)
        values = [float(value) for value in lines[1].split()]
        # The body never translates, so without an offset neither does the camera.
        self.assertAlmostEqual(values[3], 0.0, places=9)
        self.assertAlmostEqual(values[7], 0.0, places=9)

    def test_offset_camera_traces_an_arc_under_rotation(self):
        trajectory = self._trajectory()
        timestamps = [0, 1_000_000_000]
        # Camera sits 1 m along +x of the body frame.
        offset = (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (1.0, 0.0, 0.0))
        lines = rgbd.relative_ground_truth_lines(trajectory, timestamps, body_from_camera=offset)
        values = [float(value) for value in lines[1].split()]
        translation = [values[3], values[7], values[11]]
        # A 90 degree turn about a point 1 m away moves the camera by sqrt(2).
        self.assertAlmostEqual(math.hypot(*translation), math.sqrt(2.0), places=6)

    def test_first_row_is_the_identity_with_or_without_a_transform(self):
        trajectory = self._trajectory()
        offset = (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (1.0, 2.0, 3.0))
        for transform in (None, offset):
            with self.subTest(transform=transform is not None):
                lines = rgbd.relative_ground_truth_lines(
                    trajectory, [0, 1_000_000_000], body_from_camera=transform
                )
                self.assertEqual(
                    [float(value) for value in lines[0].split()],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                )

    def test_empty_frame_list_is_rejected(self):
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "empty sequence"):
            rgbd.relative_ground_truth_lines(self._trajectory(), [])


if __name__ == "__main__":
    unittest.main()

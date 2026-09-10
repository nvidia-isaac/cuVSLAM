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

from cuvslam_tools.bag2edex.rosbag_image_extraction import get_distortion_model
from cuvslam_tools.common.edex import DistortionModel


class TestGetDistortionModel(unittest.TestCase):
    def test_plumb_bob_is_reordered_to_brown5k(self):
        # ROS gives (k1, k2, p1, p2, k3), edex wants (k1, k2, k3, p1, p2).
        model, params = get_distortion_model(
            "plumb_bob", np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        )

        self.assertEqual(model, DistortionModel.BROWN5K)
        np.testing.assert_allclose(params, [0.1, 0.2, 0.5, 0.3, 0.4])

    def test_plumb_bob_rejects_wrong_coefficient_count(self):
        # Cover short and long inputs, and check that the all-zero shortcut
        # does not swallow the malformed length.
        cases = {
            "short": np.array([0.1, 0.2, 0.3, 0.4]),
            "long": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "short_all_zero": np.zeros(4),
            "long_all_zero": np.zeros(6),
        }
        for label, params in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    get_distortion_model("plumb_bob", params)

    def test_rational_polynomial_keeps_ros_order(self):
        coefficients = np.arange(1, 9, dtype=np.float64) / 10.0

        model, params = get_distortion_model("rational_polynomial", coefficients)

        self.assertEqual(model, DistortionModel.POLYNOMIAL)
        np.testing.assert_allclose(params, coefficients)

    def test_equidistant_keeps_ros_order(self):
        coefficients = np.array([0.1, 0.2, 0.3, 0.4])

        model, params = get_distortion_model("equidistant", coefficients)

        self.assertEqual(model, DistortionModel.FISHEYE)
        np.testing.assert_allclose(params, coefficients)

    def test_all_zero_coefficients_become_pinhole(self):
        model, params = get_distortion_model("plumb_bob", np.zeros(5))

        self.assertEqual(model, DistortionModel.PINHOLE)
        self.assertEqual(len(params), 0)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(AssertionError):
            get_distortion_model("kannala_brandt", np.array([0.1, 0.2, 0.3, 0.4]))


if __name__ == "__main__":
    unittest.main()

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
import cuvslam as vslam


class TestImageFormat(unittest.TestCase):
    def setUp(self):
        # Create a simple camera rig for testing
        camera = vslam.Camera(size=(640, 480), focal=(320.0, 320.0),
                              principal=(320.0, 240.0))
        self.rig = vslam.Rig([camera])
        self.tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryOnlyRealtime)
        self.timestamp = 1000

    def test_valid_2d_mono_image(self):
        # Valid mono image (2D uint8 array)
        valid_mono = np.zeros((480, 640), dtype=np.uint8)
        _, _ = self.tracker.track(self.timestamp, [valid_mono])

    def test_valid_3d_mono_image(self):
        # Valid mono image as 3D array with 1 channel
        valid_mono_3d = np.zeros((480, 640, 1), dtype=np.uint8)
        _, _ = self.tracker.track(self.timestamp, [valid_mono_3d])

    def test_valid_rgb_image(self):
        # Valid RGB image (3D uint8 array with 3 channels)
        valid_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _ = self.tracker.track(self.timestamp, [valid_rgb])

    def test_invalid_dimensions(self):
        # Test 1D array (invalid)
        invalid_1d = np.zeros(640, dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.tracker.track(self.timestamp, [invalid_1d])

        # Test 4D array (invalid)
        invalid_4d = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.tracker.track(self.timestamp, [invalid_4d])

    def test_invalid_dtype(self):
        # Test invalid data type (float32 instead of uint8)
        invalid_dtype = np.zeros((480, 640), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.tracker.track(self.timestamp, [invalid_dtype])  # type: ignore

    def test_invalid_channels(self):
        # Test invalid number of channels (2, 4, etc. instead of 1 or 3)
        invalid_channels = np.zeros((480, 640, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.tracker.track(self.timestamp, [invalid_channels])

        invalid_channels = np.zeros((480, 640, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.tracker.track(self.timestamp, [invalid_channels])

    def test_non_contiguous_array(self):
        # Test non-contiguous array
        # Create a contiguous array and then make a non-contiguous view
        contiguous = np.zeros((480 * 2, 640 * 2), dtype=np.uint8)
        non_contiguous = contiguous[::2, ::2]  # Every 2nd element
        self.assertFalse(non_contiguous.flags.c_contiguous)
        self.assertEqual(non_contiguous.shape, (480, 640))
        with self.assertRaises(ValueError):
            self.tracker.track(self.timestamp, [non_contiguous])


if __name__ == "__main__":
    unittest.main()

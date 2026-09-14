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

from cuvslam_tools.tracker.edex_reader import EdexReader


def _edex(cameras, metadata=None):
    """Build the two-section stereo.edex structure the reader parses."""
    return [{"cameras": cameras}, metadata or {}]


def _camera(depth_id=None, depth_scale_factor=None):
    camera = {"intrinsics": {}, "transform": []}
    if depth_id is not None:
        camera["depth_id"] = depth_id
    if depth_scale_factor is not None:
        camera["depth_scale_factor"] = depth_scale_factor
    return camera


def _reader(camera_id_map=None, camera_ids=None):
    """A reader with only the state the depth parsers touch.

    The constructor loads an EDEX, its frame metadata and its IMU stream from
    disk, none of which the parsers under test read.
    """
    reader = EdexReader.__new__(EdexReader)
    reader.camera_id_map = camera_id_map
    reader.camera_ids = camera_ids
    return reader


class TestDepthDescription(unittest.TestCase):
    def test_a_rig_without_depth_reports_no_depth_cameras(self):
        ids, scale, _ = EdexReader._parse_depth_description(_edex([_camera(), _camera()]))

        self.assertEqual(ids, [])
        self.assertEqual(scale, 1.0)

    def test_every_depth_camera_is_reported_not_just_the_first(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000),
                        _camera(depth_id=1, depth_scale_factor=5000)])

        ids, scale, _ = EdexReader._parse_depth_description(config)

        self.assertEqual(ids, [0, 1])
        self.assertEqual(scale, 5000.0)

    def test_depth_cameras_that_disagree_on_scale_are_rejected(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000),
                        _camera(depth_id=1, depth_scale_factor=1000)])

        with self.assertRaisesRegex(ValueError, "different depth_scale_factor"):
            EdexReader._parse_depth_description(config)

    def test_npy_depth_overrides_the_declared_scale(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000)],
                       {"depth_sequence": [["depth/000000.npy"]]})

        _, scale, _ = EdexReader._parse_depth_description(config)

        self.assertEqual(scale, 1000.0)


class TestRgbdSettings(unittest.TestCase):
    def test_a_single_depth_camera_is_accepted(self):
        settings = _reader()._parse_rgbd_settings(_edex([_camera(depth_id=0, depth_scale_factor=5000)]))

        self.assertEqual(settings.depth_camera_id, 0)
        self.assertEqual(settings.depth_scale_factor, 5000.0)

    def test_a_rig_without_depth_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "depth_id"):
            _reader()._parse_rgbd_settings(_edex([_camera()]))

    def test_several_depth_cameras_name_the_mode_that_handles_them(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000),
                        _camera(depth_id=1, depth_scale_factor=5000)])

        with self.assertRaisesRegex(ValueError, "multisensor"):
            _reader()._parse_rgbd_settings(config)


class TestMultisensorSettings(unittest.TestCase):
    def test_every_depth_camera_reaches_the_settings(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000),
                        _camera(depth_id=1, depth_scale_factor=5000)])

        settings = _reader()._parse_multisensor_settings(config)

        self.assertEqual(settings.depth_camera_ids, [0, 1])
        self.assertEqual(settings.depth_scale_factor, 5000.0)

    def test_a_rig_of_plain_rgb_cameras_is_accepted(self):
        settings = _reader()._parse_multisensor_settings(_edex([_camera(), _camera()]))

        self.assertEqual(settings.depth_camera_ids, [])

    def test_cross_camera_depth_tracking_stays_on_by_default(self):
        settings = _reader()._parse_multisensor_settings(_edex([_camera(depth_id=0)]))

        self.assertTrue(settings.enable_depth_stereo_tracking)

    def test_depth_ids_follow_a_filtered_rig(self):
        config = _edex([_camera(), _camera(depth_id=1, depth_scale_factor=5000)])

        settings = _reader(camera_id_map={1: 0}, camera_ids=[1])._parse_multisensor_settings(config)

        self.assertEqual(settings.depth_camera_ids, [0])

    def test_a_depth_camera_filtered_out_of_the_rig_is_rejected(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000), _camera()])

        with self.assertRaisesRegex(ValueError, "not included in camera_ids"):
            _reader(camera_id_map={1: 0}, camera_ids=[1])._parse_multisensor_settings(config)


if __name__ == "__main__":
    unittest.main()

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

from cuvslam_tools.tracker import depth_config


def _edex(cameras, metadata=None):
    """Build the two-section stereo.edex structure the parser reads."""
    return [{"cameras": cameras}, metadata or {}]


def _camera(depth_id=None, depth_scale_factor=None):
    camera = {"intrinsics": {}, "transform": []}
    if depth_id is not None:
        camera["depth_id"] = depth_id
    if depth_scale_factor is not None:
        camera["depth_scale_factor"] = depth_scale_factor
    return camera


class TestParseDepthDescription(unittest.TestCase):
    def test_a_rig_without_depth_reports_no_depth_cameras(self):
        description = depth_config.parse_depth_description(_edex([_camera(), _camera()]))

        self.assertEqual(description.camera_ids, ())
        self.assertEqual(description.scale_factor, 1.0)

    def test_every_depth_camera_is_reported_not_just_the_first(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000),
                        _camera(depth_id=1, depth_scale_factor=5000)])

        description = depth_config.parse_depth_description(config)

        self.assertEqual(description.camera_ids, (0, 1))
        self.assertEqual(description.scale_factor, 5000.0)

    def test_depth_cameras_that_disagree_on_scale_are_rejected(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000),
                        _camera(depth_id=1, depth_scale_factor=1000)])

        with self.assertRaisesRegex(ValueError, "different depth_scale_factor"):
            depth_config.parse_depth_description(config)

    def test_npy_depth_overrides_the_declared_scale(self):
        config = _edex([_camera(depth_id=0, depth_scale_factor=5000)],
                       {"depth_sequence": [["depth/000000.npy"]]})

        description = depth_config.parse_depth_description(config)

        self.assertEqual(description.scale_factor, depth_config.NPY_DEPTH_SCALE_FACTOR)

    def test_a_malformed_rig_section_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "'cameras' list is empty"):
            depth_config.parse_depth_description(_edex([]))


class TestSingleDepthCameraId(unittest.TestCase):
    def test_one_depth_camera_is_accepted(self):
        description = depth_config.parse_depth_description(
            _edex([_camera(depth_id=0, depth_scale_factor=5000)])
        )

        self.assertEqual(depth_config.single_depth_camera_id(description), 0)

    def test_a_rig_without_depth_is_rejected(self):
        description = depth_config.parse_depth_description(_edex([_camera()]))

        with self.assertRaisesRegex(ValueError, "depth_id"):
            depth_config.single_depth_camera_id(description)

    def test_several_depth_cameras_name_the_mode_that_handles_them(self):
        description = depth_config.parse_depth_description(
            _edex([_camera(depth_id=0, depth_scale_factor=5000),
                   _camera(depth_id=1, depth_scale_factor=5000)])
        )

        with self.assertRaisesRegex(ValueError, "multisensor"):
            depth_config.single_depth_camera_id(description)


class TestRemapDepthCameraIds(unittest.TestCase):
    def test_an_unfiltered_rig_keeps_its_ids(self):
        self.assertEqual(depth_config.remap_depth_camera_ids([0, 1], None), [0, 1])

    def test_ids_follow_a_filtered_rig(self):
        self.assertEqual(depth_config.remap_depth_camera_ids([1], {1: 0}, [1]), [0])

    def test_a_depth_camera_filtered_out_of_the_rig_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not included in camera_ids"):
            depth_config.remap_depth_camera_ids([0], {1: 0}, [1])


if __name__ == "__main__":
    unittest.main()

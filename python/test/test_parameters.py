"""Tests for the internal parameter API and per-frame hints."""

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

import cuvslam as vslam


def _stereo_rig():
    camera = vslam.Camera()
    camera.size = [640, 480]
    camera.focal = [500.0, 500.0]
    camera.principal = [320.0, 240.0]

    right = vslam.Camera()
    right.size = camera.size
    right.focal = camera.focal
    right.principal = camera.principal
    right.rig_from_camera.translation = [0.1, 0.0, 0.0]
    return vslam.Rig([camera, right])


class TestParameters(unittest.TestCase):
    def setUp(self):
        cfg = vslam.Odometry.Config()
        cfg.async_sba = False
        self.odometry = vslam.Odometry(_stereo_rig(), cfg)

    def _value(self, key):
        return self.odometry.get_parameters()[key]['value']

    def test_get_parameters_describes_every_parameter(self):
        params = self.odometry.get_parameters()
        self.assertTrue(params)
        for key, info in params.items():
            self.assertTrue(key)
            for field in ('value', 'default', 'type', 'doc', 'source'):
                self.assertIn(field, info, f'{key} is missing {field}')
                self.assertTrue(info[field], f'{key} has an empty {field}')
            self.assertEqual(info['source'], 'default',
                             f'{key} was never assigned, so it must report as a default')

    def test_set_parameter_changes_the_value_and_records_the_source(self):
        self.odometry.set_parameter('sof.num_desired_tracks', '300')

        info = self.odometry.get_parameters()['sof.num_desired_tracks']
        self.assertEqual(info['value'], '300')
        self.assertEqual(info['source'], 'api')
        self.assertNotEqual(info['default'], '300')

    def test_set_parameters_applies_a_dict(self):
        self.odometry.set_parameters({
            'sba.num_sba_iterations': '9',
            'sba.robustifier_scale': '0.25',
        })
        self.assertEqual(self._value('sba.num_sba_iterations'), '9')
        self.assertEqual(self._value('sba.robustifier_scale'), '0.25')

    def test_unambiguous_suffix_is_accepted(self):
        self.odometry.set_parameter('num_desired_tracks', '250')
        self.assertEqual(self._value('sof.num_desired_tracks'), '250')

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            self.odometry.set_parameter('sof.no_such_knob', '1')

    def test_ambiguous_suffix_raises(self):
        # Both vo_pnp and icp have lambda, and both run in this mode.
        with self.assertRaises(ValueError):
            self.odometry.set_parameter('lambda', '0.1')

    def test_bad_value_raises_and_keeps_the_previous_one(self):
        before = self._value('sof.num_desired_tracks')
        with self.assertRaises(RuntimeError):
            self.odometry.set_parameter('sof.num_desired_tracks', 'many')
        self.assertEqual(self._value('sof.num_desired_tracks'), before)

    def test_out_of_range_value_raises_and_keeps_the_previous_one(self):
        before = self._value('icp.blending_alpha')
        with self.assertRaises(RuntimeError):
            self.odometry.set_parameter('icp.blending_alpha', '2.5')
        self.assertEqual(self._value('icp.blending_alpha'), before)

    def test_parameters_the_mode_never_reads_are_not_exposed(self):
        # This rig has no IMU, so the state machine and inertial solvers never run.
        with self.assertRaises(ValueError):
            self.odometry.set_parameter('sm.min_num_kf_for_gravity', '30')
        for key in self.odometry.get_parameters():
            self.assertFalse(key.startswith('sm.'), f'{key} should not be listed without an IMU')
            self.assertFalse(key.startswith('imu_pnp.'), f'{key} should not be listed without an IMU')

    def test_enum_parameter_is_set_by_name(self):
        self.odometry.set_parameter('sof.tracker', 'klt')
        self.assertEqual(self._value('sof.tracker'), 'klt')
        with self.assertRaises(RuntimeError):
            self.odometry.set_parameter('sof.tracker', 'sideways')

    def test_construction_config_is_reported_as_the_current_value(self):
        cfg = vslam.Odometry.Config()
        cfg.async_sba = False
        cfg.use_denoising = True
        rig = _stereo_rig()
        rig.cameras[0].border_top = 20
        odometry = vslam.Odometry(rig, cfg)

        params = odometry.get_parameters()
        self.assertEqual(params['sof.box3_prefilter']['value'], 'true')
        self.assertEqual(params['sof.border_top']['value'], '20')


class TestTrackHints(unittest.TestCase):
    def test_defaults_to_no_override(self):
        self.assertIsNone(vslam.Odometry.TrackHints().override_keyframe)

    def test_constructor_keyword_and_attribute(self):
        hints = vslam.Odometry.TrackHints(override_keyframe=True)
        self.assertTrue(hints.override_keyframe)

        hints.override_keyframe = False
        self.assertFalse(hints.override_keyframe)

        hints.override_keyframe = None
        self.assertIsNone(hints.override_keyframe)

    def test_repr_mentions_the_field(self):
        self.assertIn('override_keyframe', repr(vslam.Odometry.TrackHints()))


if __name__ == '__main__':
    unittest.main()

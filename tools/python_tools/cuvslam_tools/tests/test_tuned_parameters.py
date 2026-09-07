"""Tests for recording a run's non-default parameters in its report."""

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

from cuvslam_tools.tracker.runner import non_default_parameters


def _param(value, default, source='default'):
    return {'value': value, 'default': default, 'type': 'int32', 'doc': 'a knob', 'source': source}


class TestNonDefaultParameters(unittest.TestCase):
    def test_keeps_nothing_when_everything_is_default(self):
        self.assertEqual(non_default_parameters({
            'sof.num_desired_tracks': _param('450', '450'),
            'sba.num_sba_iterations': _param('7', '7'),
        }), {})

    def test_keeps_only_the_differences(self):
        self.assertEqual(non_default_parameters({
            'sof.num_desired_tracks': _param('300', '450', source='api'),
            'sba.num_sba_iterations': _param('7', '7'),
            'sba.robustifier_scale': _param('0.25', '0.5', source='file'),
        }), {'sof.num_desired_tracks': '300', 'sba.robustifier_scale': '0.25'})

    def test_keeps_values_derived_at_construction(self):
        # box3_prefilter comes from Odometry.Config.use_denoising, so it differs from the default
        # while still reporting a source of 'default'. Filtering on source would drop it, which
        # would leave a report claiming a run used defaults when it did not.
        self.assertEqual(non_default_parameters({
            'sof.box3_prefilter': _param('true', 'false', source='default'),
        }), {'sof.box3_prefilter': 'true'})

    def test_handles_an_empty_parameter_set(self):
        self.assertEqual(non_default_parameters({}), {})


if __name__ == '__main__':
    unittest.main()

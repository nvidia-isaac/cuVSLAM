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

import importlib.metadata
import unittest
import warnings
from unittest import mock

import cuvslam as vslam


def _fake_distributions(*names):
    def distribution(name):
        if name not in names:
            raise importlib.metadata.PackageNotFoundError(name)
        return mock.Mock(metadata={'Name': name.replace('_', '-')})
    return mock.patch('importlib.metadata.distribution', side_effect=distribution)


class TestApi(unittest.TestCase):
    def test_public_api(self):
        self.assertIn('Tracker', dir(vslam))
        self.assertIn('Odometry', dir(vslam))
        self.assertIn('Slam', dir(vslam))

    def test_core_api_is_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            core = vslam.core
        self.assertIs(core.Odometry, vslam.Odometry)
        self.assertIs(core.Slam, vslam.Slam)

    def test_warns_about_several_distributions(self):
        with _fake_distributions('cuvslam', 'cuvslam_cu12'):
            with self.assertWarnsRegex(RuntimeWarning, r'\(cuvslam, cuvslam-cu12\)'):
                vslam._warn_if_multiple_distributions()

    def test_single_distribution_does_not_warn(self):
        with _fake_distributions('cuvslam_cu13'), warnings.catch_warnings():
            warnings.simplefilter('error')
            vslam._warn_if_multiple_distributions()

    def test_loads_cuda_libraries_with_pathfinder(self):
        pathfinder = mock.Mock(DynamicLibNotFoundError=LookupError)
        pathfinder.load_nvidia_dynamic_lib.side_effect = [None, LookupError('cusparse'), None]
        with mock.patch.dict('sys.modules', {'cuda': mock.Mock(pathfinder=pathfinder), 'cuda.pathfinder': pathfinder}):
            vslam._load_cuda_libraries()
        self.assertEqual([c.args[0] for c in pathfinder.load_nvidia_dynamic_lib.call_args_list],
                         ['cublas', 'cusparse', 'cusolver'])

if __name__ == "__main__":
    unittest.main()

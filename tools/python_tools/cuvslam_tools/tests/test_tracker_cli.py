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

import argparse
import builtins
import unittest
from unittest import mock

from cuvslam_tools.tracker import cli


class TestTrackerCli(unittest.TestCase):
    def test_help_parser_does_not_import_cuvslam_bindings(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "cuvslam":
                raise AssertionError("cuvslam should not be imported while building tracker help")
            return original_import(name, *args, **kwargs)

        parser = argparse.ArgumentParser(prog="cuvslam_tracker")
        with mock.patch.object(builtins, "__import__", side_effect=guarded_import):
            cli.add_tracker_arguments(parser)
            with self.assertRaises(SystemExit) as cm:
                parser.parse_args(["--help"])

        self.assertEqual(cm.exception.code, 0)

    def test_tracker_parser_keeps_binding_enums_as_strings_until_tracking(self):
        parser = argparse.ArgumentParser(prog="cuvslam_tracker")
        cli.add_tracker_arguments(parser)

        args = parser.parse_args([])

        self.assertEqual(args.multicam_mode, "performance")
        self.assertEqual(args.odometry_mode, "multicamera")

    def test_accepted_odometry_modes_match_the_dataset_registry(self):
        # run_eval.sh passes registry argument tuples straight to this parser, so a
        # mode the registry allows and the parser rejects fails only in CI.
        from cuvslam_tools import dataset_registry

        parser = argparse.ArgumentParser()
        cli.add_tracker_arguments(parser)
        choices = parser._option_string_actions["--odometry_mode"].choices

        self.assertEqual(set(choices), set(dataset_registry.ODOMETRY_MODES))


if __name__ == "__main__":
    unittest.main()

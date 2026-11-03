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
import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from cuvslam_tools.tracker import cli


def _tracker_parser() -> argparse.ArgumentParser:
    """Build the parser cuvslam_tracker uses."""
    parser = argparse.ArgumentParser(prog="cuvslam_tracker")
    cli.add_tracker_arguments(parser)
    cli.add_loop_closure_arguments(parser)
    return parser


class TestTrackerCli(unittest.TestCase):
    def setUp(self):
        # Hide the caller's CUVSLAM_ANYLOC_ENGINE, the --vpr_model_path default.
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("CUVSLAM_ANYLOC_ENGINE", None)

    def test_help_parser_does_not_import_cuvslam_bindings(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "cuvslam":
                raise AssertionError("cuvslam should not be imported while building tracker help")
            return original_import(name, *args, **kwargs)

        parser = argparse.ArgumentParser(prog="cuvslam_tracker")
        with mock.patch.object(builtins, "__import__", side_effect=guarded_import):
            cli.add_tracker_arguments(parser)
            cli.add_loop_closure_arguments(parser)
            with self.assertRaises(SystemExit) as cm:
                parser.parse_args(["--help"])

        self.assertEqual(cm.exception.code, 0)

    def test_tracker_parser_keeps_binding_enums_as_strings_until_tracking(self):
        parser = _tracker_parser()

        args = parser.parse_args([])

        self.assertEqual(args.multicam_mode, "performance")
        self.assertEqual(args.odometry_mode, "multicamera")
        self.assertEqual(args.loop_closure_mode, "lsi")
        self.assertFalse(args.use_vpr)
        self.assertEqual(args.vpr_model_path, "")
        self.assertIsNone(args.max_map_size, "no flag leaves the SLAM default in place")

    def test_loop_closure_and_vpr_flags_are_settable(self):
        parser = _tracker_parser()

        args = parser.parse_args([
            "--loop_closure_mode", "vpr",
            "--use_vpr", "true",
            "--vpr_model_path", "/tmp/anyloc.engine",
            "--max_map_size", "0",
        ])

        self.assertEqual(args.loop_closure_mode, "vpr")
        self.assertTrue(args.use_vpr)
        self.assertEqual(args.vpr_model_path, "/tmp/anyloc.engine")
        self.assertEqual(args.max_map_size, 0)

    def test_loop_closure_and_vpr_modes_are_case_insensitive(self):
        # Docs spell the modes like the binding enums, e.g. VPR.
        for flag, value in (("--loop_closure_mode", "LSI"), ("--loop_closure_mode", "VPR"), ("--loop_closure_mode", "Vpr")):
            with self.subTest(flag=flag, value=value):
                args = _tracker_parser().parse_args([flag, value])

                self.assertEqual(getattr(args, flag.lstrip("-")), value.lower())

    def test_loop_closure_mode_rejects_unknown_values(self):
        parser = _tracker_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--loop_closure_mode", "not_a_mode"])

    def _main_usage_error(self, argv: list) -> str:
        """Run cli.main, expect a usage error before tracking, and return stderr."""
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in ("cuvslam", "cuvslam_tools.tracker.runner"):
                raise AssertionError(f"{name} was imported, so the arguments were not rejected before tracking")
            return original_import(name, *args, **kwargs)

        stderr = io.StringIO()
        with mock.patch.object(builtins, "__import__", side_effect=guarded_import), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli.main(argv)

        self.assertEqual(cm.exception.code, 2)
        return stderr.getvalue()

    def test_vpr_loop_closure_without_a_backend_fails_before_tracking(self):
        error = self._main_usage_error(["--dataset", "seq", "--use_slam", "true", "--loop_closure_mode", "vpr"])

        self.assertIn("needs place recognition", error)
        self.assertIn("--use_vpr", error)

    def test_negative_max_map_size_fails_before_tracking(self):
        error = self._main_usage_error(["--dataset", "seq", "--use_slam", "true", "--max_map_size", "-1"])

        self.assertIn("--max_map_size", error)

    def test_anyloc_without_a_model_file_fails_before_tracking(self):
        for flags in (["--use_vpr", "true"], ["--use_vpr", "true", "--vpr_model_path", "/nonexistent/anyloc.engine"]):
            with self.subTest(flags=flags):
                error = self._main_usage_error(["--dataset", "seq", "--use_slam", "true", *flags])

                self.assertIn("--vpr_model_path", error)
                self.assertIn("anyloc_engine", error)

    def test_vpr_model_path_defaults_to_cuvslam_anyloc_engine(self):
        with mock.patch.dict(os.environ, {"CUVSLAM_ANYLOC_ENGINE": "/build/bin/anyloc.engine"}):
            args = _tracker_parser().parse_args([])

        self.assertEqual(args.vpr_model_path, "/build/bin/anyloc.engine")

    def test_anyloc_with_a_model_file_passes_validation(self):
        with tempfile.NamedTemporaryFile(suffix=".engine") as model:
            parser = _tracker_parser()
            args = parser.parse_args(["--use_slam", "true", "--loop_closure_mode", "vpr", "--use_vpr", "true",
                                      "--vpr_model_path", model.name])

            cli.validate_loop_closure_arguments(parser, args)  # parser.error() would raise SystemExit

    def test_loop_closure_flags_without_slam_fail_before_tracking(self):
        for flags in (["--loop_closure_mode", "vpr", "--use_vpr", "true"], ["--use_vpr", "true"]):
            with self.subTest(flags=flags):
                error = self._main_usage_error(["--dataset", "seq", *flags])

                self.assertIn("--use_slam true", error)

    def test_runtime_error_from_tracking_is_a_usage_error(self):
        # Slam raises RuntimeError for a VPR backend it cannot run.
        runner = types.ModuleType("cuvslam_tools.tracker.runner")
        unavailable = "VPR backend AnyLoc is not available: cuVSLAM was built with USE_TENSORRT=OFF."
        runner.track = mock.Mock(side_effect=RuntimeError(unavailable))
        stderr = io.StringIO()
        with tempfile.NamedTemporaryFile(suffix=".engine") as model, \
                mock.patch.dict(sys.modules, {"cuvslam_tools.tracker.runner": runner}), \
                mock.patch.object(cli, "_normalize_tracker_binding_args"), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--dataset", "seq", "--use_slam", "true", "--loop_closure_mode", "vpr",
                          "--use_vpr", "true", "--vpr_model_path", model.name])

        self.assertEqual(cm.exception.code, 2)
        self.assertIn(unavailable, stderr.getvalue())
        runner.track.assert_called_once()

    def test_stats_record_the_loop_closures(self):
        stat = mock.Mock(num_loop_closures=3, loop_closure_mode="vpr")

        saved = cli.stat_to_dict(stat)

        self.assertEqual(saved["num_loop_closures"], 3)
        self.assertEqual(saved["loop_closure_mode"], "vpr")

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

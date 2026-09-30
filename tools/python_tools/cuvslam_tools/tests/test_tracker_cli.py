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
        self.assertEqual(args.loop_closure_mode, "default")
        self.assertEqual(args.vpr_mode, "off")
        self.assertEqual(args.vpr_model_path, "")
        self.assertIsNone(args.max_map_size, "no flag leaves the SLAM default in place")

    def test_loop_closure_and_vpr_flags_are_settable(self):
        parser = _tracker_parser()

        args = parser.parse_args([
            "--loop_closure_mode", "vpr",
            "--vpr_mode", "bow",
            "--vpr_model_path", "/tmp/dinov2.onnx",
            "--max_map_size", "0",
        ])

        self.assertEqual(args.loop_closure_mode, "vpr")
        self.assertEqual(args.vpr_mode, "bow")
        self.assertEqual(args.vpr_model_path, "/tmp/dinov2.onnx")
        self.assertEqual(args.max_map_size, 0)

    def test_loop_closure_and_vpr_modes_are_case_insensitive(self):
        # The place recognition tools and docs spell the modes like the binding enums, e.g. Bow and AnyLoc.
        for flag, value in (("--loop_closure_mode", "Default"), ("--loop_closure_mode", "Vpr"),
                            ("--vpr_mode", "Off"), ("--vpr_mode", "Simple"), ("--vpr_mode", "Bow"),
                            ("--vpr_mode", "DBoW2"), ("--vpr_mode", "AnyLoc")):
            with self.subTest(flag=flag, value=value):
                args = _tracker_parser().parse_args([flag, value])

                self.assertEqual(getattr(args, flag.lstrip("-")), value.lower())

    def test_loop_closure_mode_rejects_unknown_values(self):
        parser = _tracker_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--loop_closure_mode", "not_a_mode"])

    def test_vpr_mode_rejects_unknown_values(self):
        parser = _tracker_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--vpr_mode", "not_a_backend"])

    def test_vpr_loop_closure_with_a_backend_passes_validation(self):
        parser = _tracker_parser()
        args = parser.parse_args(["--use_slam", "true", "--loop_closure_mode", "vpr", "--vpr_mode", "bow"])

        cli.validate_loop_closure_arguments(parser, args)  # parser.error() would raise SystemExit

    def _main_usage_error(self, argv: list) -> str:
        """Run cli.main, expect a usage error before the tracker is loaded, and return the error output."""
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

        self.assertIn("needs a place recognition backend", error)
        self.assertIn("--vpr_mode", error)

    def test_negative_max_map_size_fails_before_tracking(self):
        error = self._main_usage_error(["--dataset", "seq", "--use_slam", "true", "--max_map_size", "-1"])

        self.assertIn("--max_map_size", error)

    def test_anyloc_without_a_model_file_fails_before_tracking(self):
        for flags in (["--vpr_mode", "anyloc"], ["--vpr_mode", "AnyLoc", "--vpr_model_path", "/nonexistent/dinov2.onnx"]):
            with self.subTest(flags=flags):
                error = self._main_usage_error(["--dataset", "seq", "--use_slam", "true", *flags])

                self.assertIn("--vpr_model_path", error)

    def test_anyloc_with_a_model_file_passes_validation(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as model:
            parser = _tracker_parser()
            args = parser.parse_args(["--use_slam", "true", "--loop_closure_mode", "vpr", "--vpr_mode", "anyloc",
                                      "--vpr_model_path", model.name])

            cli.validate_loop_closure_arguments(parser, args)  # parser.error() would raise SystemExit

    def test_loop_closure_flags_without_slam_fail_before_tracking(self):
        for flags in (["--loop_closure_mode", "vpr", "--vpr_mode", "bow"], ["--vpr_mode", "bow"]):
            with self.subTest(flags=flags):
                error = self._main_usage_error(["--dataset", "seq", *flags])

                self.assertIn("--use_slam true", error)

    def test_runtime_error_from_tracking_is_a_usage_error(self):
        # Constructing SLAM with a place recognition backend it cannot run raises RuntimeError.
        runner = types.ModuleType("cuvslam_tools.tracker.runner")
        runner.track = mock.Mock(side_effect=RuntimeError("DBoW2 is not available in this build"))
        stderr = io.StringIO()
        with mock.patch.dict(sys.modules, {"cuvslam_tools.tracker.runner": runner}), \
                mock.patch.object(cli, "_normalize_tracker_binding_args"), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--dataset", "seq", "--use_slam", "true",
                          "--loop_closure_mode", "vpr", "--vpr_mode", "dbow2"])

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("DBoW2 is not available in this build", stderr.getvalue())
        runner.track.assert_called_once()

    def test_stats_record_the_loop_closures(self):
        stat = mock.Mock(num_loop_closures=3, loop_closure_mode="vpr")

        saved = cli.stat_to_dict(stat)

        self.assertEqual(saved["num_loop_closures"], 3)
        self.assertEqual(saved["loop_closure_mode"], "vpr")


if __name__ == "__main__":
    unittest.main()

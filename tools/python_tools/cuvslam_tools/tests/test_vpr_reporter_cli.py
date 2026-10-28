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
import contextlib
import io
import tempfile
import types
import unittest
from unittest import mock

from cuvslam_tools.vpr_reporter import cli
from cuvslam_tools.vpr_reporter import execution
from cuvslam_tools.vpr_reporter import generate_report
from cuvslam_tools.vpr_reporter.execution import VprStat


class TestVprReporterCli(unittest.TestCase):
    def test_parser_builds_without_the_tracker_loop_closure_flags(self):
        # The reporter has its own VPR flags; the tracker's would clash or be ignored.
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--help"])

        self.assertEqual(cm.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--vpr_model_path", help_text)
        self.assertIn("--vpr_vocabulary_path", help_text)
        self.assertNotIn("--loop_closure_mode", help_text)
        self.assertNotIn("--use_vpr", help_text)
        self.assertNotIn("--vpr_modes", help_text)

    def test_tracker_flags_are_rejected(self):
        # It always runs synchronous SLAM with place recognition; --dataset must not abbreviate --datasets_root.
        for argv in (["--use_slam", "true"], ["--sync_slam", "false"], ["--async_sba", "true"], ["--dataset", "x"]):
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as cm:
                    cli.main(["--map_sequence", "map", "--query_sequence", "query"] + argv)
                self.assertEqual(cm.exception.code, 2)
                self.assertIn(f"unrecognized arguments: {' '.join(argv)}", stderr.getvalue())

    def test_missing_vpr_vocabulary_fails_before_the_evaluation(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as output_dir, \
                mock.patch.object(execution, "evaluate_pairs") as evaluate_pairs, contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--map_sequence", "map", "--query_sequence", "query", "--output_dir", output_dir,
                          "--vpr_vocabulary_path", "/nonexistent/kitti.vocabulary"])

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("/nonexistent/kitti.vocabulary is not a file", stderr.getvalue())
        self.assertIn("cuvslam_vpr_vocabulary", stderr.getvalue())
        evaluate_pairs.assert_not_called()


class TestVprSlamConfig(unittest.TestCase):
    def test_vocabulary_is_set_only_when_given(self):
        vslam = types.SimpleNamespace(Slam=types.SimpleNamespace(Config=types.SimpleNamespace))
        args = argparse.Namespace(vpr_score_threshold=None, vpr_model_path="", vpr_vocabulary_path="")

        self.assertFalse(hasattr(execution._slam_config(vslam, args), "vpr_vocabulary_path"))
        args.vpr_vocabulary_path = "kitti.vocabulary"
        self.assertEqual(execution._slam_config(vslam, args).vpr_vocabulary_path, "kitti.vocabulary")


class TestVprSummary(unittest.TestCase):
    def test_total_median_error_averages_the_two_middle_errors(self):
        # Pooled over pairs, defined like the per-pair median (np.median).
        stats = [VprStat(n_true_positives=2, errors_m=[4.0, 1.0]),
                 VprStat(n_true_positives=2, errors_m=[3.0, 2.0])]
        self.assertEqual(generate_report.calc_summary("Total", stats)["median_error_m"], 2.5)

    def test_total_median_error_without_errors_is_zero(self):
        self.assertEqual(generate_report.calc_summary("Total", [VprStat()])["median_error_m"], 0.0)


if __name__ == "__main__":
    unittest.main()

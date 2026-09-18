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
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from cuvslam_tools.reporter import cli
from cuvslam_tools.reporter import execution
from cuvslam_tools.reporter import generate_report


class TestReporterCli(unittest.TestCase):
    def test_resolve_config_path_uses_datasets_root_for_relative_config(self):
        with tempfile.TemporaryDirectory() as datasets_root:
            config_path = Path(datasets_root) / "kitti" / "kitti-vio_slam_gt.cfg"
            config_path.parent.mkdir()
            config_path.write_text("{}", encoding="utf-8")

            resolved = cli._resolve_config_path("kitti/kitti-vio_slam_gt.cfg", datasets_root)

        self.assertEqual(resolved, config_path)


class TestReporterOutputDirectory(unittest.TestCase):
    """The registry may evaluate one reporter config in several odometry modes."""

    def _run_report(self, root, odometry_mode):
        datasets_root = Path(root) / "datasets"
        config_path = datasets_root / "kitti" / "kitti-vio_slam_gt.cfg"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"segment_lengths": [1], "sequence_cfgs": []}), encoding="utf-8")

        args = argparse.Namespace(
            test_config="kitti/kitti-vio_slam_gt.cfg",
            datasets_root=str(datasets_root),
            output_root=str(Path(root) / "stats"),
            output_dir="",
            odometry_mode=odometry_mode,
            max_workers=1,
            pdf=False,
            report_comments=[],
        )
        with mock.patch.object(execution, "run_parallel_tracking", return_value=[]), \
             mock.patch.object(generate_report, "save_stats_to_json"), \
             mock.patch.object(generate_report, "generate_report") as report:
            output_dir = cli.run_report(args)

        return Path(output_dir).parent, report.call_args.kwargs["config_name"]

    def test_each_odometry_mode_gets_its_own_directory(self):
        with tempfile.TemporaryDirectory() as root:
            stereo, _ = self._run_report(root, "multicamera")
            multisensor, _ = self._run_report(root, "multisensor")

        self.assertEqual(stereo.name, "kitti-vio_slam_gt-multicamera")
        self.assertEqual(multisensor.name, "kitti-vio_slam_gt-multisensor")

    def test_the_mode_suffix_leaves_the_kpi_prefix_alone(self):
        with tempfile.TemporaryDirectory() as root:
            multisensor, _ = self._run_report(root, "multisensor")

        self.assertEqual(multisensor.name.split("-")[0], "kitti")

    def test_the_report_is_named_for_the_run(self):
        with tempfile.TemporaryDirectory() as root:
            _, config_name = self._run_report(root, "rgbd")

        self.assertEqual(config_name, "kitti-vio_slam_gt-rgbd")


class TestGitSourceMetadata(unittest.TestCase):
    def test_metadata_is_read_from_git(self):
        outputs = ["1234567890abcdef", "feature/branch", "2026-08-31 17:08:04 +0000"]
        with mock.patch.object(generate_report.subprocess, "run") as run:
            run.side_effect = [mock.Mock(stdout=f"{value}\n") for value in outputs]
            metadata = generate_report._git_source_metadata("/cuvslam")

        self.assertEqual(metadata, ("1234567890abcdef", "feature/branch", "2026-08-31/17:08:04/+0000", ""))

    def test_unreadable_repository_degrades_to_unknown_with_a_reason(self):
        error = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
        error.stderr = "fatal: detected dubious ownership in repository at /cuvslam\n"
        with mock.patch.object(generate_report.subprocess, "run", side_effect=error):
            commit_sha, branch_name, commit_ts, warning = generate_report._git_source_metadata("/cuvslam")

        self.assertEqual((commit_sha, branch_name, commit_ts), ("unknown", "unknown", "unknown"))
        self.assertIn("detected dubious ownership", warning)

    def test_missing_git_degrades_to_unknown_with_a_reason(self):
        with mock.patch.object(generate_report.subprocess, "run", side_effect=FileNotFoundError("git")):
            commit_sha, branch_name, commit_ts, warning = generate_report._git_source_metadata("/cuvslam")

        self.assertEqual((commit_sha, branch_name, commit_ts), ("unknown", "unknown", "unknown"))
        self.assertIn("git is not installed", warning)

    def test_commit_date_failure_keeps_the_resolved_fields_and_the_reason(self):
        error = subprocess.CalledProcessError(128, ["git", "show"])
        error.stderr = "fatal: bad object HEAD\n"
        with mock.patch.object(generate_report.subprocess, "run") as run:
            run.side_effect = [mock.Mock(stdout="1234567890abcdef\n"), mock.Mock(stdout="feature/branch\n"), error]
            commit_sha, branch_name, commit_ts, warning = generate_report._git_source_metadata("/cuvslam")

        self.assertEqual((commit_sha, branch_name, commit_ts), ("1234567890abcdef", "feature/branch", "unknown"))
        self.assertIn("commit date: fatal: bad object HEAD", warning)

    def test_both_secondary_failures_are_reported(self):
        branch_error = subprocess.CalledProcessError(128, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
        branch_error.stderr = "fatal: no branch\n"
        date_error = subprocess.CalledProcessError(128, ["git", "show"])
        date_error.stderr = "fatal: bad object HEAD\n"
        with mock.patch.object(generate_report.subprocess, "run") as run:
            run.side_effect = [mock.Mock(stdout="1234567890abcdef\n"), branch_error, date_error]
            commit_sha, branch_name, commit_ts, warning = generate_report._git_source_metadata("/cuvslam")

        self.assertEqual((commit_sha, branch_name, commit_ts), ("1234567890abcdef", "unknown", "unknown"))
        self.assertIn("branch name: fatal: no branch", warning)
        self.assertIn("commit date: fatal: bad object HEAD", warning)

    def test_git_failure_is_not_written_to_the_process_log(self):
        """A directory outside any repository must not leak git's fatal error to stderr."""
        script = (
            "from cuvslam_tools.reporter.generate_report import _git_source_metadata\n"
            "print(_git_source_metadata(__import__('sys').argv[1])[:3])\n"
        )
        env = dict(os.environ)
        package_root = Path(generate_report.__file__).resolve().parents[2]
        env["PYTHONPATH"] = os.pathsep.join([str(package_root), env.get("PYTHONPATH", "")])

        with tempfile.TemporaryDirectory() as non_repo_dir:
            env["MPLCONFIGDIR"] = non_repo_dir
            completed = subprocess.run(
                [sys.executable, "-c", script, non_repo_dir],
                capture_output=True,
                universal_newlines=True,
                env=env,
                check=True,
            )

        self.assertNotIn("fatal:", completed.stderr)
        self.assertIn("('unknown', 'unknown', 'unknown')", completed.stdout)


class TestReportProvenanceHeader(unittest.TestCase):
    """The report header must explain unknown provenance without the run log."""

    UNRESOLVED = ("unknown", "unknown", "unknown", "Provenance unavailable for /cuvslam: git is not installed")
    RESOLVED = ("1234567890abcdef", "feature/branch", "2026-08-31/17:08:04/+0000", "")

    def _render_html(self, metadata):
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(generate_report, "_git_source_metadata", return_value=metadata):
                generate_report.generate_report(output_dir, [], [])
            return (Path(output_dir) / "report.html").read_text(encoding="utf-8")

    def _render_pdf_html(self, metadata):
        captured = {}

        class CapturingHtml:
            def __init__(self, **kwargs):
                captured["html"] = kwargs["string"]

            def write_pdf(self, path):
                Path(path).write_bytes(b"")

        weasyprint = types.ModuleType("weasyprint")
        weasyprint.HTML = CapturingHtml
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.dict(sys.modules, {"weasyprint": weasyprint}):
                with mock.patch.object(generate_report, "_git_source_metadata", return_value=metadata):
                    generate_report.generate_report(output_dir, [], [], generate_pdf=True)
        return captured["html"]

    def test_warning_is_rendered_in_the_html_report(self):
        html = self._render_html(self.UNRESOLVED)

        self.assertIn("git is not installed", html)

    def test_warning_is_rendered_in_the_pdf_report(self):
        pdf_html = self._render_pdf_html(self.UNRESOLVED)

        self.assertIn("git is not installed", pdf_html)

    def test_resolved_provenance_is_reported_without_a_warning_row(self):
        for name, rendered in (("html", self._render_html(self.RESOLVED)),
                               ("pdf", self._render_pdf_html(self.RESOLVED))):
            with self.subTest(report=name):
                self.assertIn("1234567890abcdef", rendered)
                self.assertIn("feature/branch", rendered)
                self.assertNotIn("Provenance", rendered)


class TestPdfGeneration(unittest.TestCase):
    def test_missing_pdf_dependency_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.dict(sys.modules, {"weasyprint": None}):
                with self.assertRaisesRegex(RuntimeError, r"install cuvslam-tools\[pdf\]"):
                    generate_report.generate_report(output_dir, [], [], generate_pdf=True)

    def test_pdf_rendering_error_fails_explicitly(self):
        class FailingHtml:
            def __init__(self, **_kwargs):
                pass

            def write_pdf(self, _path):
                raise OSError("renderer failed")

        weasyprint = types.ModuleType("weasyprint")
        weasyprint.HTML = FailingHtml
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.dict(sys.modules, {"weasyprint": weasyprint}):
                with self.assertRaisesRegex(RuntimeError, "PDF generation failed: renderer failed"):
                    generate_report.generate_report(output_dir, [], [], generate_pdf=True)


if __name__ == "__main__":
    unittest.main()

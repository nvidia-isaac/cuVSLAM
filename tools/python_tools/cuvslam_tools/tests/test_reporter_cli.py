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

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from cuvslam_tools.reporter import cli
from cuvslam_tools.reporter import generate_report


class TestReporterCli(unittest.TestCase):
    def test_resolve_config_path_uses_datasets_root_for_relative_config(self):
        with tempfile.TemporaryDirectory() as datasets_root:
            config_path = Path(datasets_root) / "kitti" / "kitti-vio_slam_gt.cfg"
            config_path.parent.mkdir()
            config_path.write_text("{}", encoding="utf-8")

            resolved = cli._resolve_config_path("kitti/kitti-vio_slam_gt.cfg", datasets_root)

        self.assertEqual(resolved, config_path)


class TestGitSourceMetadata(unittest.TestCase):
    def test_metadata_is_read_from_git(self):
        outputs = ["1234567890abcdef", "feature/branch", "2026-08-31 17:08:04 +0000"]
        with mock.patch.object(generate_report.subprocess, "run") as run:
            run.side_effect = [mock.Mock(stdout=f"{value}\n") for value in outputs]
            metadata = generate_report._git_source_metadata("/cuvslam")

        self.assertEqual(metadata, ("1234567890abcdef", "feature/branch", "2026-08-31/17:08:04/+0000"))

    def test_unreadable_repository_degrades_to_unknown(self):
        error = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
        error.stderr = "fatal: detected dubious ownership in repository at '/cuvslam'\n"
        with mock.patch.object(generate_report.subprocess, "run", side_effect=error):
            metadata = generate_report._git_source_metadata("/cuvslam")

        self.assertEqual(metadata, ("unknown", "unknown", "unknown"))

    def test_missing_git_degrades_to_unknown(self):
        with mock.patch.object(generate_report.subprocess, "run", side_effect=FileNotFoundError("git")):
            metadata = generate_report._git_source_metadata("/cuvslam")

        self.assertEqual(metadata, ("unknown", "unknown", "unknown"))

    def test_git_failure_is_not_written_to_the_process_log(self):
        """A directory outside any repository must not leak git's fatal error to stderr."""
        script = (
            "from cuvslam_tools.reporter.generate_report import _git_source_metadata\n"
            "print(_git_source_metadata(__import__('sys').argv[1]))\n"
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

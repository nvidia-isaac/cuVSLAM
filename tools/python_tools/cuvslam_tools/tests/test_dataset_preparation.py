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

"""Contract shared by every dataset preparation module."""

import importlib
import inspect
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from cuvslam_tools import dataset_registry
from cuvslam_tools.dataset_preparation import common

DATASETS = tuple(spec.dataset_id for spec in dataset_registry.provisionable_datasets())

PREPARATION_ROOT = Path(common.__file__).resolve().parent


class TestPreparationModuleContract(unittest.TestCase):
    def test_every_dataset_exposes_prepare_and_main(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                module = importlib.import_module(f"cuvslam_tools.dataset_preparation.{dataset}.prepare")

                self.assertTrue(callable(module.prepare))
                self.assertTrue(callable(module.main))
                parameters = inspect.signature(module.prepare).parameters
                self.assertLessEqual(
                    {"raw_dir", "output_dir", "force_download", "download_only"},
                    set(parameters),
                )
                for name, parameter in parameters.items():
                    self.assertIsNot(parameter.default, inspect.Parameter.empty, name)

    def test_provisioning_arguments_are_accepted_by_every_command(self):
        provisioning_arguments = ["--raw-dir", "/raw", "--output-dir", "/converted", "--force-download"]
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                module = importlib.import_module(f"cuvslam_tools.dataset_preparation.{dataset}.prepare")

                with mock.patch.object(module, "prepare") as prepare:
                    self.assertEqual(module.main(provisioning_arguments), 0)

                self.assertEqual(prepare.call_args.kwargs["raw_dir"], Path("/raw"))
                self.assertEqual(prepare.call_args.kwargs["output_dir"], Path("/converted"))
                self.assertTrue(prepare.call_args.kwargs["force_download"])

    def test_only_download_scripts_remain_under_dataset_preparation(self):
        shell_scripts = sorted(path.name for path in PREPARATION_ROOT.rglob("*.sh"))

        self.assertEqual(
            shell_scripts,
            ["download_euroc.sh", "download_kitti.sh", "download_tum.sh"],
        )

    def test_no_dataset_keeps_a_cli_wrapper_module(self):
        self.assertEqual(sorted(path.name for path in PREPARATION_ROOT.rglob("cli.py")), [])


class TestPreparationHelpers(unittest.TestCase):
    def test_defaults_follow_the_current_directory(self):
        with mock.patch.object(common.Path, "cwd", return_value=Path("/work")):
            self.assertEqual(common.default_raw_dir("kitti"), Path("/work/datasets/kitti/raw"))
            self.assertEqual(common.default_output_dir(), Path("/work/datasets/converted"))

    def test_explicit_paths_win_over_defaults(self):
        self.assertEqual(common.resolve_raw_dir(Path("/raw"), "kitti"), Path("/raw"))
        self.assertEqual(common.resolve_output_dir(Path("/converted")), Path("/converted"))

    def test_failing_download_script_is_reported_with_its_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "download_broken.sh"
            script.write_text("#!/usr/bin/env bash\nexit 3\n")

            with self.assertRaisesRegex(common.PreparationError, "download_broken.sh failed with exit code 3"):
                common.run_download_script(script)

    def test_missing_download_script_is_reported(self):
        with self.assertRaisesRegex(common.PreparationError, "download script not found"):
            common.run_download_script(Path("/nonexistent/download_missing.sh"))

    def test_empty_artifacts_are_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "empty.cfg").write_text("")

            with self.assertRaisesRegex(common.PreparationError, "converter did not produce"):
                common.require_nonempty_files(root, ["empty.cfg"], "converter")
            with self.assertRaisesRegex(common.PreparationError, "converter did not produce"):
                common.require_nonempty_files(root, ["absent.cfg"], "converter")

    def test_run_preparation_maps_failures_to_exit_codes(self):
        self.assertEqual(common.run_preparation(lambda: Path("/prepared")), 0)

        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            def failing():
                raise common.PreparationError("nothing to convert")

            self.assertEqual(common.run_preparation(failing), 1)

        self.assertIn("nothing to convert", stderr.getvalue())

    def test_run_preparation_reports_filesystem_errors(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            def failing():
                raise FileNotFoundError(2, "No such file or directory", "/raw/archive.zip")

            self.assertEqual(common.run_preparation(failing), 1)

        self.assertIn("archive.zip", stderr.getvalue())

    def test_dataset_file_resolves_next_to_the_module(self):
        with redirect_stdout(io.StringIO()):
            resolved = common.dataset_file(str(PREPARATION_ROOT / "tum" / "prepare.py"), "download_tum.sh")

        self.assertTrue(resolved.is_file())


if __name__ == "__main__":
    unittest.main()

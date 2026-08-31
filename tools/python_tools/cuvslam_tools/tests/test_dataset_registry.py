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

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cuvslam_tools import dataset_registry
from cuvslam_tools.dataset_registry import DatasetSpec, EvalSpec, RegistryError

KITTI_PREPARE = "cuvslam_tools.dataset_preparation.kitti.prepare"


def stereo_eval(config="kitti-vio_slam_gt.cfg", mode="multicamera", **overrides):
    fields = {
        "config": config,
        "args": (f"--odometry_mode={mode}", "--use_segments"),
        "suites": frozenset(dataset_registry.SUITES),
    }
    fields.update(overrides)
    return EvalSpec(**fields)


class TestShippedRegistry(unittest.TestCase):
    def test_shipped_registry_validates(self):
        dataset_registry.validate()

    def test_active_eval_records_are_exact(self):
        # run_eval.sh passes these straight to cuvslam_app, and the KPI history is
        # keyed on the derived prefixes, so any change here reindexes the history.
        expected = [
            (
                "kitti",
                "KITTI",
                "kitti/kitti-vio_slam_gt.cfg",
                "--odometry_mode=multicamera --rectified_stereo_camera=true "
                "--async_sba=false --multicam_mode=moderate --use_segments",
            ),
            (
                "euroc",
                "EUROC",
                "euroc/euroc-vio_slam.cfg",
                "--odometry_mode=inertial --rectified_stereo_camera=false "
                "--async_sba=false --multicam_mode=moderate --use_segments",
            ),
        ]

        actual = [
            (
                spec.dataset_id,
                record.kpi_prefix,
                dataset_registry.reporter_config_path(spec, record),
                " ".join(record.args),
            )
            for spec in dataset_registry.eval_datasets()
            for record in dataset_registry.eval_records(spec)
        ]

        self.assertEqual(actual, expected)

    def test_archive_and_mount_derive_from_the_dataset_id(self):
        for spec in dataset_registry.provisionable_datasets():
            self.assertEqual(spec.archive_name, f"{spec.dataset_id}.tar")
            self.assertEqual(spec.mount_name, spec.dataset_id)

    def test_only_kitti_and_euroc_are_eval_enabled(self):
        enabled = [spec.dataset_id for spec in dataset_registry.eval_datasets()]
        self.assertEqual(enabled, ["kitti", "euroc"])

    def test_listing_the_registry_imports_no_converter_dependencies(self):
        # In a subprocess, because sibling test modules import converters and
        # would otherwise leave their dependencies in sys.modules.
        probe = (
            "import sys; from cuvslam_tools import dataset_registry as r; r.validate();"
            "[print(m) for m in ('numpy', 'cv2', 'scipy', 'PIL', 'h5py', 'yaml') if m in sys.modules]"
        )
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
        self.assertEqual(completed.stdout.strip(), "", f"validating the registry imported {completed.stdout.split()}")

    def test_registry_is_runnable_as_a_module(self):
        # CI shell wrappers invoke `python3 -m cuvslam_tools.dataset_registry`.
        completed = subprocess.run(
            [sys.executable, "-m", "cuvslam_tools.dataset_registry", "eval-records"],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = [line.split("\t") for line in completed.stdout.strip().splitlines()]
        self.assertEqual([row[0] for row in rows], ["kitti", "euroc"])
        self.assertTrue(all(len(row) == 4 for row in rows), rows)

    def test_unknown_dataset_is_rejected_with_the_known_ids(self):
        # provision_dataset.sh relies on this to reject a stale workflow choice.
        with self.assertRaisesRegex(RegistryError, r"unknown dataset 'nope' \(known: euroc, kitti, tartan, tum\)"):
            dataset_registry.validate(["nope"])

    def test_preparation_modules_expose_a_callable_prepare(self):
        # Imports the converters, so it is the one test that pays that cost.
        for spec in dataset_registry.provisionable_datasets():
            self.assertTrue(callable(spec.load_prepare()), spec.dataset_id)


class TestValidationFailures(unittest.TestCase):
    def _validate(self, datasets):
        original = dataset_registry.DATASETS
        dataset_registry.DATASETS = datasets
        try:
            dataset_registry.validate()
        finally:
            dataset_registry.DATASETS = original

    def test_key_must_match_dataset_id(self):
        with self.assertRaisesRegex(RegistryError, "does not match dataset_id"):
            self._validate({"kitty": DatasetSpec("kitti", KITTI_PREPARE)})

    def test_dataset_id_must_be_lower_case(self):
        with self.assertRaisesRegex(RegistryError, "must be lower-case"):
            self._validate({"KITTI": DatasetSpec("KITTI", KITTI_PREPARE)})

    def test_missing_preparation_module_fails(self):
        with self.assertRaisesRegex(RegistryError, "preparation module not found"):
            self._validate({"kitti": DatasetSpec("kitti", "cuvslam_tools.dataset_preparation.nope.prepare")})

    def test_config_must_be_a_bare_filename(self):
        spec = DatasetSpec("kitti", KITTI_PREPARE, (stereo_eval(config="kitti/kitti-vio_slam_gt.cfg"),))
        with self.assertRaisesRegex(RegistryError, "bare filename"):
            self._validate({"kitti": spec})

    def test_config_must_end_with_cfg(self):
        spec = DatasetSpec("kitti", KITTI_PREPARE, (stereo_eval(config="kitti.txt"),))
        with self.assertRaisesRegex(RegistryError, "must end with .cfg"):
            self._validate({"kitti": spec})

    def test_unknown_odometry_mode_fails(self):
        spec = DatasetSpec("kitti", KITTI_PREPARE, (stereo_eval(mode="stereo"),))
        with self.assertRaisesRegex(RegistryError, "odometry_mode must be one of"):
            self._validate({"kitti": spec})

    def test_repeated_odometry_mode_fails(self):
        # cuvslam_app takes the last value, so a repeat could run an unchecked mode.
        record = stereo_eval(args=("--odometry_mode=multicamera", "--odometry_mode=rgbd"))
        with self.assertRaisesRegex(RegistryError, "exactly one --odometry_mode flag, found 2"):
            self._validate({"kitti": DatasetSpec("kitti", KITTI_PREPARE, (record,))})

    def test_missing_odometry_mode_fails(self):
        record = stereo_eval(args=("--use_segments",))
        with self.assertRaisesRegex(RegistryError, "exactly one --odometry_mode flag, found 0"):
            self._validate({"kitti": DatasetSpec("kitti", KITTI_PREPARE, (record,))})

    def test_collision_across_datasets_is_caught_when_scoped_to_one(self):
        # A collision belongs to the pair, so narrowing to one dataset must not hide it.
        datasets = {
            "kitti": DatasetSpec("kitti", KITTI_PREPARE, (stereo_eval(config="tartan-a.cfg"),)),
            "tartan": DatasetSpec(
                "tartan",
                "cuvslam_tools.dataset_preparation.tartan.prepare",
                (stereo_eval(config="tartan-b.cfg"),),
            ),
        }
        original = dataset_registry.DATASETS
        dataset_registry.DATASETS = datasets
        try:
            with self.assertRaisesRegex(RegistryError, "same KPI identity"):
                dataset_registry.validate(["kitti"])
        finally:
            dataset_registry.DATASETS = original

    def test_malformed_argument_tuple_fails(self):
        spec = DatasetSpec("kitti", KITTI_PREPARE, (stereo_eval(args=("odometry_mode=multicamera",)),))
        with self.assertRaisesRegex(RegistryError, "non-empty tuple of --flag"):
            self._validate({"kitti": spec})

    def test_unknown_suite_fails(self):
        spec = DatasetSpec("kitti", KITTI_PREPARE, (stereo_eval(suites=frozenset({"nightly"})),))
        with self.assertRaisesRegex(RegistryError, "unknown suite"):
            self._validate({"kitti": spec})

    def test_empty_suite_membership_fails(self):
        spec = DatasetSpec("kitti", KITTI_PREPARE, (stereo_eval(suites=frozenset()),))
        with self.assertRaisesRegex(RegistryError, "at least one suite"):
            self._validate({"kitti": spec})

    def test_unknown_gating_policy_fails(self):
        spec = DatasetSpec("kitti", KITTI_PREPARE, (stereo_eval(gating="blocking"),))
        with self.assertRaisesRegex(RegistryError, "gating must be one of"):
            self._validate({"kitti": spec})

    def test_two_evals_deriving_one_kpi_identity_fail(self):
        # Both configs reduce to the TARTAN prefix with the same mode, so one
        # would silently replace the other in the KPI report.
        colliding = DatasetSpec(
            "tartan",
            "cuvslam_tools.dataset_preparation.tartan.prepare",
            (stereo_eval(config="tartan-osmo-vo_slam.cfg"), stereo_eval(config="tartan-flaky-vo_slam.cfg")),
        )
        with self.assertRaisesRegex(RegistryError, "same KPI identity"):
            self._validate({"tartan": colliding})

    def test_underscore_prefix_keeps_the_flaky_report_distinct(self):
        distinct = DatasetSpec(
            "tartan",
            "cuvslam_tools.dataset_preparation.tartan.prepare",
            (stereo_eval(config="tartan-osmo-vo_slam.cfg"), stereo_eval(config="tartan_flaky-vo_slam.cfg")),
        )
        self._validate({"tartan": distinct})
        self.assertEqual(distinct.evals[0].kpi_prefix, "TARTAN")
        self.assertEqual(distinct.evals[1].kpi_prefix, "TARTAN_FLAKY")

    def test_unknown_dataset_lookup_lists_known_ids(self):
        with self.assertRaisesRegex(RegistryError, "known: euroc, kitti, tartan, tum"):
            dataset_registry.dataset("m3ed_spot")

    def test_unknown_suite_filter_fails(self):
        with self.assertRaisesRegex(RegistryError, "unknown suite"):
            dataset_registry.eval_records(dataset_registry.dataset("kitti"), "nightly")


class TestVerifyStaged(unittest.TestCase):
    def _stage(self, root, config, dataset_folder):
        (Path(root) / config).write_text(json.dumps({"dataset_folder": dataset_folder}), encoding="utf-8")

    def test_matching_dataset_folder_passes(self):
        spec = dataset_registry.dataset("kitti")
        with tempfile.TemporaryDirectory() as root:
            self._stage(root, "kitti-vio_slam_gt.cfg", "kitti/")
            dataset_registry.verify_staged(spec, Path(root))

    def test_mismatched_dataset_folder_fails(self):
        spec = dataset_registry.dataset("kitti")
        with tempfile.TemporaryDirectory() as root:
            self._stage(root, "kitti-vio_slam_gt.cfg", "kitti_gt/")
            with self.assertRaisesRegex(RegistryError, "declares dataset_folder"):
                dataset_registry.verify_staged(spec, Path(root))

    def test_missing_config_fails(self):
        spec = dataset_registry.dataset("kitti")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RegistryError, "staged config not found"):
                dataset_registry.verify_staged(spec, Path(root))

    def test_invalid_json_fails(self):
        spec = dataset_registry.dataset("kitti")
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "kitti-vio_slam_gt.cfg").write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "not valid JSON"):
                dataset_registry.verify_staged(spec, Path(root))

    def test_json_that_is_not_an_object_fails(self):
        spec = dataset_registry.dataset("kitti")
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "kitti-vio_slam_gt.cfg").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "must hold a JSON object, got list"):
                dataset_registry.verify_staged(spec, Path(root))


if __name__ == "__main__":
    unittest.main()

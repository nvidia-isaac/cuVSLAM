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
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cuvslam_tools.reporter import execution


class _Future:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def result(self):
        if self._exc:
            raise self._exc
        return self._result


class _FakeExecutor:
    def __init__(self, futures):
        self._futures = list(futures)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, *_args, **_kwargs):
        return self._futures.pop(0)


class TestReporterExecution(unittest.TestCase):
    def test_process_sequence_applies_reporter_config_fields(self):
        captured_args = []
        stat = object()

        class _Result:
            pass

        def track(args):
            captured_args.append(args)
            result = _Result()
            result.stat = stat
            return result

        args = argparse.Namespace(config_path="", repeat_type="none", num_loops=0)
        sequence = {
            "enable": True,
            "sequence_title": "seq-a",
            "sequence_folder": "seq-a-folder",
            "edex_file": "custom.edex",
            "gt_file_path": "poses/gt.txt",
            "cameras": [1, 0],
            "use_slam": True,
            "repeat_type": "REPEAT",
            "sequence_num_repeats": 3,
        }

        with mock.patch.object(execution, "_load_track", return_value=track):
            result = execution.process_sequence(sequence, args, "/datasets", "dataset")

        self.assertIs(result, stat)
        self.assertEqual(len(captured_args), 1)
        args_copy = captured_args[0]
        expected_dataset = os.path.join("/datasets", "dataset", "seq-a-folder")
        self.assertEqual(args_copy.sequence_title, "seq-a")
        self.assertEqual(args_copy.dataset, expected_dataset)
        self.assertEqual(args_copy.config_path, os.path.join(expected_dataset, "custom.edex"))
        self.assertEqual(args_copy.gt_path, "poses/gt.txt")
        self.assertFalse(args_copy.gt_from_shuttle)
        self.assertEqual(args_copy.camera_ids, [1, 0])
        self.assertTrue(args_copy.use_slam)
        self.assertEqual(args_copy.repeat_type, "repeat")
        self.assertEqual(args_copy.num_loops, 3)

    def test_process_sequence_forwards_shuttle_ground_truth_opt_in(self):
        captured_args = []
        stat = object()

        class _Result:
            pass

        def track(args):
            captured_args.append(args)
            result = _Result()
            result.stat = stat
            return result

        args = argparse.Namespace(config_path="", repeat_type="none", num_loops=0)
        sequence = {
            "enable": True,
            "sequence_title": "seq-a",
            "sequence_folder": "seq-a-folder",
            "gt_from_shuttle": True,
            "repeat_type": "Shuttle",
            "sequence_num_repeats": 1,
        }

        with mock.patch.object(execution, "_load_track", return_value=track):
            execution.process_sequence(sequence, args, "/datasets", "dataset")

        args_copy = captured_args[0]
        self.assertTrue(args_copy.gt_from_shuttle)
        self.assertIsNone(args_copy.gt_path)

    def test_process_sequence_defaults_edex_and_warns_on_empty_gt_path(self):
        captured_args = []
        stat = object()

        class _Result:
            pass

        def track(args):
            captured_args.append(args)
            result = _Result()
            result.stat = stat
            return result

        args = argparse.Namespace(config_path="/tmp/global.edex", repeat_type="none", num_loops=0)
        sequence = {
            "enable": True,
            "sequence_title": "seq-a",
            "sequence_folder": "seq-a-folder",
            "gt_file_path": "",
        }

        with mock.patch.object(execution, "_load_track", return_value=track):
            with self.assertWarnsRegex(UserWarning, "Ignoring empty gt_file_path"):
                result = execution.process_sequence(sequence, args, "/datasets", "dataset")

        self.assertIs(result, stat)
        args_copy = captured_args[0]
        expected_dataset = os.path.join("/datasets", "dataset", "seq-a-folder")
        self.assertEqual(args_copy.config_path, os.path.join(expected_dataset, "stereo.edex"))
        self.assertIsNone(args_copy.gt_path)
        self.assertIsNone(args_copy.camera_ids)

    def test_process_sequence_accepts_dataset_root_that_is_dataset_folder(self):
        captured_args = []
        stat = object()

        class _Result:
            pass

        def track(args):
            captured_args.append(args)
            result = _Result()
            result.stat = stat
            return result

        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_root = Path(temp_dir) / "kitti"
            (datasets_root / "00").mkdir(parents=True)

            args = argparse.Namespace(config_path="", repeat_type="none", num_loops=0)
            sequence = {
                "enable": True,
                "sequence_title": "KITTI-00-ODOM",
                "sequence_folder": "00",
                "edex_file": "stereo.edex",
            }

            with mock.patch.object(execution, "_load_track", return_value=track):
                result = execution.process_sequence(sequence, args, str(datasets_root), "kitti/")

        self.assertIs(result, stat)
        args_copy = captured_args[0]
        expected_dataset = os.path.join(str(datasets_root), "00")
        self.assertEqual(args_copy.dataset, expected_dataset)
        self.assertEqual(args_copy.config_path, os.path.join(expected_dataset, "stereo.edex"))

    def test_run_parallel_tracking_warns_about_unsupported_config_fields(self):
        reporter_config = {
            "dataset_folder": "dataset",
            "write_cache": True,
            "sequence_cfgs": [],
        }

        with self.assertWarnsRegex(UserWarning, "write_cache=true"):
            stats = execution.run_parallel_tracking(
                reporter_config,
                argparse.Namespace(),
                "/datasets",
                max_workers=1,
            )

        self.assertEqual(stats, [])

    def test_run_parallel_tracking_rejects_malformed_reporter_config(self):
        cases = [
            (
                [],
                "Reporter config must be a JSON object",
            ),
            (
                {"sequence_cfgs": []},
                "Reporter config missing required key: dataset_folder",
            ),
            (
                {"dataset_folder": "dataset"},
                "Reporter config missing required key: sequence_cfgs",
            ),
            (
                {"dataset_folder": "dataset", "sequence_cfgs": {}},
                "Reporter config key sequence_cfgs must be a list",
            ),
            (
                {"dataset_folder": 1, "sequence_cfgs": []},
                "Reporter config key dataset_folder must be a string",
            ),
            (
                {"dataset_folder": "dataset", "sequence_cfgs": ["seq-a"]},
                r"Reporter config sequence_cfgs\[0\] must be an object",
            ),
            (
                {"dataset_folder": "dataset", "sequence_cfgs": [{"sequence_title": "seq-a"}]},
                r"Reporter config sequence_cfgs\[0\] missing required key: sequence_folder",
            ),
            (
                {"dataset_folder": "dataset", "sequence_cfgs": [{"sequence_title": 1, "sequence_folder": "seq-a"}]},
                r"Reporter config sequence_cfgs\[0\] key sequence_title must be a string",
            ),
            (
                {"dataset_folder": "dataset", "sequence_cfgs": [{"sequence_title": "seq-a", "sequence_folder": 1}]},
                r"Reporter config sequence_cfgs\[0\] key sequence_folder must be a string",
            ),
            (
                {
                    "dataset_folder": "dataset",
                    "sequence_cfgs": [
                        {"sequence_title": "seq-a", "sequence_folder": "seq-a", "gt_from_shuttle": "false"}
                    ],
                },
                r"Reporter config sequence_cfgs\[0\] key gt_from_shuttle must be a boolean",
            ),
            (
                {
                    "dataset_folder": "dataset",
                    "sequence_cfgs": [{"sequence_title": "seq-a", "sequence_folder": "seq-a", "use_slam": 1}],
                },
                r"Reporter config sequence_cfgs\[0\] key use_slam must be a boolean",
            ),
            (
                {"dataset_folder": "dataset", "sequence_cfgs": [{"enable": "false"}]},
                r"Reporter config sequence_cfgs\[0\] key enable must be a boolean",
            ),
            (
                {"dataset_folder": "dataset", "sequence_cfgs": [{"enable": False, "gt_from_shuttle": "false"}]},
                r"Reporter config sequence_cfgs\[0\] key gt_from_shuttle must be a boolean",
            ),
        ]

        for reporter_config, error in cases:
            with self.subTest(reporter_config=reporter_config):
                with self.assertRaisesRegex(ValueError, error):
                    execution.run_parallel_tracking(
                        reporter_config,
                        argparse.Namespace(),
                        "/datasets",
                        max_workers=1,
                    )

    def test_run_parallel_tracking_skips_a_disabled_sequence_stub(self):
        # A disabled sequence may still be a stub: only flags it does carry are checked.
        stats = execution.run_parallel_tracking(
            {"dataset_folder": "dataset", "sequence_cfgs": [{"enable": False, "use_slam": True}]},
            argparse.Namespace(),
            "/datasets",
            max_workers=1,
        )

        self.assertEqual(stats, [])

    def test_run_parallel_tracking_raises_when_all_enabled_sequences_fail(self):
        reporter_config = {
            "dataset_folder": "dataset",
            "sequence_cfgs": [
                {"enable": True, "sequence_title": "seq-a", "sequence_folder": "seq-a"},
                {"enable": True, "sequence_title": "seq-b", "sequence_folder": "seq-b"},
            ],
        }

        def executor(*_args, **_kwargs):
            return _FakeExecutor(
                [
                    _Future(exc=RuntimeError("tracking failed")),
                    _Future(exc=RuntimeError("tracking failed")),
                ]
            )

        with mock.patch.object(execution, "ProcessPoolExecutor", executor):
            with self.assertRaisesRegex(
                RuntimeError,
                "Tracking failed for all enabled sequences: seq-a, seq-b",
            ):
                execution.run_parallel_tracking(
                    reporter_config,
                    argparse.Namespace(),
                    "/datasets",
                    max_workers=1,
                )

    def test_run_parallel_tracking_reraises_cuvslam_dependency_error(self):
        reporter_config = {
            "dataset_folder": "dataset",
            "sequence_cfgs": [
                {"enable": True, "sequence_title": "seq-a", "sequence_folder": "seq-a"},
            ],
        }

        def executor(*_args, **_kwargs):
            return _FakeExecutor([_Future(exc=RuntimeError(execution.CUVSLAM_REPORTER_DEPENDENCY_ERROR))])

        with mock.patch.object(execution, "ProcessPoolExecutor", executor):
            with self.assertRaisesRegex(RuntimeError, execution.CUVSLAM_REPORTER_DEPENDENCY_ERROR):
                execution.run_parallel_tracking(
                    reporter_config,
                    argparse.Namespace(),
                    "/datasets",
                    max_workers=1,
                )

    def test_run_parallel_tracking_raises_when_any_enabled_sequence_fails(self):
        reporter_config = {
            "dataset_folder": "dataset",
            "sequence_cfgs": [
                {"enable": True, "sequence_title": "seq-a", "sequence_folder": "seq-a"},
                {"enable": True, "sequence_title": "seq-b", "sequence_folder": "seq-b"},
            ],
        }

        def executor(*_args, **_kwargs):
            return _FakeExecutor([_Future(result=object()), _Future(result=None)])

        with mock.patch.object(execution, "ProcessPoolExecutor", executor):
            with self.assertRaisesRegex(
                RuntimeError,
                "Tracking failed for sequences: seq-b",
            ):
                execution.run_parallel_tracking(
                    reporter_config,
                    argparse.Namespace(),
                    "/datasets",
                    max_workers=1,
                )


if __name__ == "__main__":
    unittest.main()

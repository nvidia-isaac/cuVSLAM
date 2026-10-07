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

"""CPU-only regression tests: python -m unittest discover -s <skill-dir>/evals."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_trajectory as evaluation


def pose(angle=0.0, position=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    result = np.eye(4)
    vector = np.asarray(axis) * np.sin(np.radians(angle) / 2)
    result[:3, :3] = evaluation.quaternion_matrix(np.r_[vector, np.cos(np.radians(angle) / 2)])
    result[:3, 3] = position
    return result


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path

    def args(self, *extra):
        return evaluation.build_parser().parse_args([
            str(self.root / "estimate.txt"), str(self.root / "gt.txt"),
            "--estimate-frame", "cam0", "--gt-frame", "cam0",
            "--estimate-world", "world", "--gt-world", "world",
            "--output-dir", str(self.root / "output"), *extra,
        ])

    def test_quaternion_order_and_sign(self):
        half = np.sqrt(0.5)
        tum = self.write("tum.txt", f"1 0 0 0 0 0 {half} {half}\n2 0 0 0 0 0 {-half} {-half}\n")
        euroc = self.write("gt.csv", f"1000000000,0,0,0,{half},0,0,{half},0,0,0,0,0,0,0,0,0\n"
                            f"2000000000,0,0,0,{-half},0,0,{-half},0,0,0,0,0,0,0,0,0\n")
        estimated = evaluation.load_trajectory(tum, "tum")
        gt = evaluation.load_trajectory(euroc, "euroc")
        np.testing.assert_allclose(estimated.poses, gt.poses)
        np.testing.assert_allclose(estimated.poses[0, :3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-14)
        _, rotation = evaluation.pose_errors(estimated.poses, gt.poses)
        np.testing.assert_allclose(rotation, 0, atol=1e-12)

    def test_epoch_nanosecond_precision(self):
        value = "1403636580.123456789"
        self.assertEqual(evaluation.timestamp_ns(value), 1403636580123456789)
        self.assertEqual(evaluation.timestamp_ns("1403636580123456789", "ns"), 1403636580123456789)

    def test_reject_malformed_input(self):
        for text in ("1 0 0 0 0 0 0 0\n2 0 0 0 0 0 0 1\n",
                     "1 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n",
                     "1 nan 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evaluation.load_trajectory(self.write("bad.txt", text), "tum")

    def test_full_sensor_conversion_and_world_alignment(self):
        gt = np.array([pose(i * 25, (i, i*i/2, 0.2*i)) for i in range(6)])
        sensor = pose(70, (0.5, -0.3, 0.2), axis=(1, 0, 0))
        world = pose(-35, (4, -2, 1), axis=(0, 1, 0))
        estimated = evaluation.inverse(world) @ gt @ evaluation.inverse(sensor)
        converted = estimated @ sensor
        alignment, diagnostics = evaluation.fit_alignment(converted[:3], gt[:3], "initial-poses")
        np.testing.assert_allclose(alignment, world, atol=1e-12)
        np.testing.assert_allclose(alignment @ converted, gt, atol=1e-12)
        self.assertLess(diagnostics["fit_rotation_rmse_deg"], 1e-10)
        # Omitting a rotating lever arm is not repaired by one world transform.
        wrong, _ = evaluation.fit_alignment(estimated[:3], gt[:3], "initial-poses")
        errors, _ = evaluation.pose_errors(wrong @ estimated, gt)
        self.assertGreater(evaluation.rmse(errors), 0.1)

    def test_yaml_extrinsics_direction_and_lever_arm(self):
        body_est = pose(90, (1, 2, 3))
        body_gt = pose(30, (-2, 1, 0))
        for name, matrix in (("est.yaml", body_est), ("gt.yaml", body_gt)):
            self.write(name, json.dumps({"T_BS": {"rows": 4, "cols": 4, "data": matrix.flatten().tolist()}}))
        args = self.args("--estimate-sensor-yaml", str(self.root / "est.yaml"),
                         "--gt-sensor-yaml", str(self.root / "gt.yaml"), "--gt-frame", "imu0")
        transform, sources, warnings = evaluation.frame_conversion(args)
        np.testing.assert_allclose(body_est @ transform, body_gt, atol=1e-12)
        self.assertEqual(len(sources), 2)
        self.assertFalse(warnings)

    def test_missing_extrinsics_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "different physical frames"):
            evaluation.frame_conversion(self.args("--gt-frame", "imu0"))
        _, _, warnings = evaluation.frame_conversion(self.args("--gt-frame", "imu0", "--allow-missing-extrinsics"))
        self.assertIn("cannot generally recover", warnings[0])
        with self.assertRaisesRegex(ValueError, "initial-window"):
            evaluation.frame_conversion(self.args("--gt-frame", "imu0", "--allow-missing-extrinsics", "--alignment", "none"))

    def test_association_overlap_tolerance_ties_and_uniqueness(self):
        estimated = np.array([0, 10, 20, 30, 40])
        gt = np.array([5, 9, 11, 19, 31, 35])
        i, j, counts = evaluation.associate(estimated, gt, 1)
        self.assertEqual(i.tolist(), [1, 2, 3])
        self.assertEqual(j.tolist(), [1, 3, 4])  # 9 wins a tie with 11
        self.assertEqual(counts["estimate_outside_overlap"], 2)
        self.assertEqual(counts["gt_unmatched_in_overlap"], 3)
        self.assertEqual(len(set(j)), len(j))
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            evaluation.associate(estimated, gt, 0)
        with self.assertRaisesRegex(ValueError, "no overlapping"):
            evaluation.associate(np.array([1, 2]), np.array([3, 4]), 10)

    def test_crossing_matches_rejected(self):
        with self.assertRaisesRegex(ValueError, "cross"):
            evaluation.associate(np.array([0, 4, 5, 10]), np.array([0, 3, 4, 10]), 2)

    def test_initial_pose_fit_supports_static_and_straight_motion(self):
        transform = pose(40, (3, 2, -1))
        for gt in (np.array([pose()] * 3), np.array([pose(position=(i, 0, 0)) for i in range(3)])):
            estimated = evaluation.inverse(transform) @ gt
            alignment, _ = evaluation.fit_alignment(estimated, gt, "initial-poses")
            np.testing.assert_allclose(alignment, transform, atol=1e-12)
            with self.assertRaisesRegex(ValueError, "degenerate"):
                evaluation.fit_alignment(estimated, gt, "initial-positions")

    def test_invalid_initial_fits_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            evaluation.fit_alignment(np.array([pose()] * 2), np.array([pose()] * 2), "initial-poses")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            evaluation.fit_alignment(np.array([pose()] * 3), np.array([pose(0), pose(120), pose(240)]), "initial-poses")

    def test_kabsch_no_scale_fitting(self):
        gt = np.array([pose(position=p) for p in [(0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 1, 1)]])
        world = pose(40, (1, 3, 2))
        estimated = evaluation.inverse(world) @ gt
        alignment, _ = evaluation.fit_alignment(estimated, gt, "all-positions")
        np.testing.assert_allclose(alignment, world, atol=1e-12)
        scaled = gt.copy()
        scaled[:, :3, 3] *= 2
        alignment, _ = evaluation.fit_alignment(scaled, gt, "all-positions")
        np.testing.assert_allclose(np.linalg.det(alignment[:3, :3]), 1, atol=1e-12)
        error, _ = evaluation.pose_errors(alignment @ scaled, gt)
        self.assertGreater(evaluation.rmse(error), 0.5)

    def test_metrics_known_translation_and_rotation(self):
        estimated = np.array([pose(30, (3, 4, 0)), pose(180)])
        translation, rotation = evaluation.pose_errors(estimated, np.array([pose()] * 2))
        self.assertAlmostEqual(evaluation.rmse(translation), np.sqrt(12.5))
        np.testing.assert_allclose(rotation, [30, 180], atol=1e-10)

    def test_distance_uses_native_gt_samples_and_zero_distance_is_null(self):
        self.write("estimate.txt", "0 0 0 0 0 0 0 1\n1 2 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n")
        self.write("gt.txt", "0 0 0 0 0 0 0 1\n0.5 1 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n"
                            "1.5 1 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n")
        report, _ = evaluation.evaluate(self.args("--alignment", "none", "--max-time-difference", "0"))
        self.assertEqual(report["metrics"]["gt_traveled_distance_m"], 4)
        self.assertAlmostEqual(report["metrics"]["normalized_ate_rmse_percent"], 100 * np.sqrt(4/3) / 4)
        self.write("gt.txt", "0 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n")
        report, _ = evaluation.evaluate(self.args("--alignment", "none"))
        self.assertIsNone(report["metrics"]["normalized_ate_rmse_percent"])
        self.assertTrue(report["warnings"])

    def test_initial_window_is_fixed_and_late_drift_remains(self):
        self.write("estimate.txt", "0 0 0 0 0 0 0 1\n0.5 1 0 0 0 0 0 1\n1 2 0 0 0 0 0 1\n2 13 0 0 0 0 0 1\n")
        self.write("gt.txt", "0 0 0 0 0 0 0 1\n0.5 1 0 0 0 0 0 1\n1 2 0 0 0 0 0 1\n2 3 0 0 0 0 0 1\n")
        report, errors = evaluation.evaluate(self.args())
        self.assertEqual(report["alignment"]["fit_pose_count"], 3)
        self.assertEqual(report["alignment"]["fit_end_ns"], 10**9)
        self.assertAlmostEqual(report["metrics"]["ate_rmse_m"], 5)
        self.assertEqual(errors[-1][-2], 10)

    def test_time_offset_and_none_world_guard(self):
        self.write("estimate.txt", "0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n2 2 0 0 0 0 0 1\n")
        self.write("gt.txt", "10 0 0 0 0 0 0 1\n11 1 0 0 0 0 0 1\n12 2 0 0 0 0 0 1\n")
        report, _ = evaluation.evaluate(self.args("--time-offset", "10", "--alignment", "none"))
        self.assertEqual(report["metrics"]["ate_rmse_m"], 0)
        self.assertEqual(report["association"]["estimate_time_offset_ns"], 10**10)
        with self.assertRaisesRegex(ValueError, "same declared world"):
            evaluation.evaluate(self.args("--time-offset", "10", "--alignment", "none", "--gt-world", "other"))

    def test_kitti_timestamps_and_se3_validation(self):
        matrix = "1 0 0 0 0 1 0 0 0 0 1 0\n"
        path = self.write("kitti.txt", matrix * 2)
        times = self.write("times.txt", "0\n0.1\n")
        data = evaluation.load_trajectory(path, "kitti", times)
        np.testing.assert_array_equal(data.timestamps, [0, 100000000])
        with self.assertRaisesRegex(ValueError, "requires"):
            evaluation.load_trajectory(path, "kitti")
        self.write("times.txt", "0\n")
        with self.assertRaisesRegex(ValueError, "count"):
            evaluation.load_trajectory(path, "kitti", times)
        for invalid in (np.diag([-1, 1, 1, 1]), np.diag([2, 2, 2, 1])):
            with self.assertRaisesRegex(ValueError, "SE"):
                evaluation.rigid_matrix(invalid)

    def test_cli_writes_reproducible_outputs_and_refuses_overwrite(self):
        rows = "0 0 0 0 0 0 0 1\n0.5 1 0 0 0 0 0 1\n1 2 0 0 0 0 0 1\n"
        estimated = self.write("estimate.txt", rows)
        gt = self.write("gt.txt", rows)
        command = [sys.executable, evaluation.__file__, str(estimated), str(gt),
                   "--estimate-frame", "body", "--gt-frame", "body", "--estimate-world", "world",
                   "--gt-world", "world", "--output-dir", str(self.root / "result")]
        run = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report["metrics"]["ate_rmse_m"], 0)
        self.assertEqual(report["association"]["evaluated_pairs"], 3)
        self.assertEqual(report["inputs"]["estimate"]["sha256"], evaluation.file_record(estimated)["sha256"])
        self.assertTrue(Path(report["outputs"]["errors"]).is_file())
        original_report = Path(report["outputs"]["report"]).read_bytes()
        rerun = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(rerun.returncode, 0)
        self.assertIn("already exists", rerun.stderr)
        self.assertEqual(Path(report["outputs"]["report"]).read_bytes(), original_report)
        self.assertEqual(estimated.read_text(), rows)


if __name__ == "__main__":
    unittest.main()

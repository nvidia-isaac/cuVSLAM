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
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cuvslam_kpi_report as kpi_report  # noqa: E402


def make_kpis(*, ate, are, kabsch, losts, fps):
    return {
        "KITTI_ATE_STEREO_ODOM": ate,
        "KITTI_ARE_STEREO_ODOM": are,
        "KITTI_Kabsch_STEREO_ODOM": kabsch,
        "KITTI_TrackingLosts_STEREO_ODOM": losts,
        "KITTI_FPS_STEREO_ODOM": fps,
    }


def make_report(current, previous=None):
    return kpi_report.build_report("test", current, previous)


class KpiCollectionTest(unittest.TestCase):
    def test_collects_odom_and_slam_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            stats = Path(temporary)
            stats_dir = stats / "kitti-vio_slam_gt" / "2026-08-24" / "stats"
            stats_dir.mkdir(parents=True)
            records = [
                {
                    "odometry_mode": "OdometryMode.Multicamera",
                    "sequence_title": "sequence-ODOM",
                    "gt_av_translation_error": 1.0,
                    "gt_av_rotation_error": 2.0,
                    "gt_simple_error": 3.0,
                    "average_fps": 40.0,
                    "num_tracking_losts": 1,
                },
                {
                    "odometry_mode": "OdometryMode.Multicamera",
                    "sequence_title": "sequence-SLAM",
                    "gt_av_translation_error": 0.5,
                    "gt_av_rotation_error": 1.5,
                    "gt_simple_error": 2.5,
                    "average_fps": 30.0,
                    "num_tracking_losts": 0,
                },
            ]
            (stats_dir / "all_stats.json").write_text(json.dumps(records), encoding="utf-8")

            kpis = kpi_report.collect_kpis(str(stats))

        self.assertEqual(kpis["KITTI_ATE_STEREO_ODOM"], 1.0)
        self.assertEqual(kpis["KITTI_TrackingLosts_STEREO_ODOM"], 1)
        self.assertEqual(kpis["KITTI_FPS_STEREO_SLAM"], 30.0)


class KpiRenderingTest(unittest.TestCase):
    def test_single_config_table_owns_config_column_and_diffs(self):
        current = make_kpis(ate=2.0, are=3.0, kabsch=4.0, losts=2, fps=50.0)
        previous = make_kpis(ate=1.5, are=2.0, kabsch=3.5, losts=1, fps=45.0)

        table = kpi_report.render_report(make_report(current, previous), "x86-test")

        self.assertIn("| Config | Dataset |", table)
        self.assertIn("| x86-test | KITTI-STEREO_ODOM |", table)
        self.assertIn("| 0.5000 | 1.0000 | 0.5000 | 1 | 50.0 |", table)
        self.assertNotIn("diff FPS", table)

    def test_single_config_table_uses_na_without_history(self):
        table = kpi_report.render_report(
            make_report(make_kpis(ate=2.0, are=3.0, kabsch=4.0, losts=2, fps=50.0)),
            "x86-test",
        )

        self.assertIn("| NA | NA | NA | NA | 50.0 |", table)

    def test_aggregate_uses_population_standard_deviation_and_diff_of_means(self):
        first = make_report(
            make_kpis(ate=1.0, are=2.0, kabsch=3.0, losts=0, fps=10.0),
            make_kpis(ate=0.5, are=1.0, kabsch=2.5, losts=0, fps=9.0),
        )
        second = make_report(
            make_kpis(ate=3.0, are=4.0, kabsch=5.0, losts=2, fps=14.0),
            make_kpis(ate=1.5, are=3.0, kabsch=4.5, losts=1, fps=13.0),
        )

        table = kpi_report.render_aggregate_report([("first", first), ("second", second)])

        self.assertIn("Aggregated across 2 configurations", table)
        self.assertIn("mean ± population σ", table)
        self.assertIn("| KITTI-STEREO_ODOM |", table)
        self.assertIn("| 2.0000 ± 1.0000 | 3.0000 ± 1.0000 |", table)
        self.assertIn("| 4.0000 ± 1.0000 | 1.00 ± 1.00 |", table)
        self.assertIn("| 1.0000 | 1.0000 | 0.5000 | 0.50 | 12.0 ± 2.0 |", table)

    def test_aggregate_diff_is_na_if_any_config_lacks_history(self):
        current = make_kpis(ate=1.0, are=2.0, kabsch=3.0, losts=0, fps=10.0)
        reports = [
            ("with-history", make_report(current, current)),
            ("without-history", make_report(current)),
        ]

        table = kpi_report.render_aggregate_report(reports)

        self.assertIn("| NA | NA | NA | NA | 10.0 ± 0.0 |", table)

    def test_aggregate_rejects_mismatched_kpi_sets(self):
        complete = make_report(make_kpis(ate=1.0, are=2.0, kabsch=3.0, losts=0, fps=10.0))
        incomplete_kpis = make_kpis(ate=1.0, are=2.0, kabsch=3.0, losts=0, fps=10.0)
        incomplete_kpis.pop("KITTI_FPS_STEREO_ODOM")

        with self.assertRaisesRegex(ValueError, "KPI keys differ"):
            kpi_report.aggregate_reports(
                [("complete", complete), ("incomplete", make_report(incomplete_kpis))]
            )


class KpiReportDataTest(unittest.TestCase):
    def test_report_json_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            report = make_report(
                make_kpis(ate=1.0, are=2.0, kabsch=3.0, losts=0, fps=10.0)
            )
            kpi_report.write_json(str(path), report)

            loaded = kpi_report.load_report(str(path))

        self.assertEqual(loaded, report)


class KpiCliTest(unittest.TestCase):
    def test_render_and_aggregate_commands_write_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "report.json"
            single_output = root / "single.md"
            aggregate_output = root / "aggregate.md"
            kpi_report.write_json(
                str(report_path),
                make_report(make_kpis(ate=1.0, are=2.0, kabsch=3.0, losts=0, fps=10.0)),
            )
            script = Path(kpi_report.__file__)

            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "render",
                    "--report_json",
                    str(report_path),
                    "--config",
                    "x86-test",
                    "--output",
                    str(single_output),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "aggregate",
                    "--input",
                    f"x86-test={report_path}",
                    "--output",
                    str(aggregate_output),
                ],
                check=True,
            )

            self.assertIn("| x86-test | KITTI-STEREO_ODOM |", single_output.read_text())
            self.assertIn("Aggregated across 1 configuration", aggregate_output.read_text())


if __name__ == "__main__":
    unittest.main()

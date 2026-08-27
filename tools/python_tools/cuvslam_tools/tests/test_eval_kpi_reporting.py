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

"""Tests for scripts/cuvslam_kpi_report.py.

The reporter is a standard-library-only script that runs inside the eval
container, so it is loaded by path rather than imported as a package module.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPORT_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "cuvslam_kpi_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("cuvslam_kpi_report", REPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kpi_report = _load_report_module()


def write_stats(stat_folder, config_name, odometry_mode="multicamera", sequence="SEQ"):
    """Create one cuvslam_app stats tree: <config>/<timestamp>/stats/all_stats.json."""
    stats_dir = Path(stat_folder) / config_name / "2026-08-27_00-00-00" / "stats"
    stats_dir.mkdir(parents=True)
    entries = [
        {
            "odometry_mode": odometry_mode,
            "sequence_title": f"{config_name.upper()}-{sequence}-{mode}",
            "gt_av_translation_error": 1.0,
            "gt_av_rotation_error": 0.5,
            "gt_simple_error": 0.25,
            "average_fps": 30.0,
            "num_tracking_losts": 0,
        }
        for mode in ("ODOM", "SLAM")
    ]
    (stats_dir / "all_stats.json").write_text(json.dumps(entries), encoding="utf-8")


class TestCollectKpis(unittest.TestCase):
    def test_distinct_prefixes_keep_both_reports(self):
        with tempfile.TemporaryDirectory() as stat_folder:
            write_stats(stat_folder, "tartan-osmo-vo_slam")
            write_stats(stat_folder, "tartan_flaky-osmo-vo_slam")

            kpis = kpi_report.collect_kpis(stat_folder)

        self.assertIn("TARTAN_ATE_STEREO_ODOM", kpis)
        self.assertIn("TARTAN_FLAKY_ATE_STEREO_ODOM", kpis)

    def test_colliding_prefixes_fail_and_name_both_configs(self):
        with tempfile.TemporaryDirectory() as stat_folder:
            # Both names reduce to the TARTAN prefix, because the first hyphen
            # terminates it, so the second report would overwrite the first.
            write_stats(stat_folder, "tartan-osmo-vo_slam")
            write_stats(stat_folder, "tartan-flaky-vo_slam")

            with self.assertRaises(ValueError) as context:
                kpi_report.collect_kpis(stat_folder)

        message = str(context.exception)
        self.assertIn("tartan-osmo-vo_slam", message)
        self.assertIn("tartan-flaky-vo_slam", message)
        self.assertIn("TARTAN_ATE_STEREO_ODOM", message)

    def test_different_modalities_do_not_collide(self):
        with tempfile.TemporaryDirectory() as stat_folder:
            write_stats(stat_folder, "kitti-vio_slam_gt", odometry_mode="multicamera")
            write_stats(stat_folder, "euroc-vio_slam", odometry_mode="inertial")
            write_stats(stat_folder, "tum-rgbd", odometry_mode="rgbd")

            kpis = kpi_report.collect_kpis(stat_folder)

        self.assertIn("KITTI_ATE_STEREO_ODOM", kpis)
        self.assertIn("EUROC_ATE_VIO_ODOM", kpis)
        self.assertIn("TUM_ATE_RGBD_ODOM", kpis)

    def test_folder_without_stats_is_skipped(self):
        with tempfile.TemporaryDirectory() as stat_folder:
            write_stats(stat_folder, "kitti-vio_slam_gt")
            (Path(stat_folder) / "empty-config").mkdir()

            kpis = kpi_report.collect_kpis(stat_folder)

        self.assertIn("KITTI_ATE_STEREO_ODOM", kpis)


if __name__ == "__main__":
    unittest.main()

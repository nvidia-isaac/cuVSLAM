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

"""Racing search of the KITTI SLAM defaults, with the reporter runs faked."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cuvslam_tools.reporter import tune_kitti as tune


class TestTuneKitti(unittest.TestCase):
    def test_cells_far_behind_the_leader_are_dropped(self):
        results = {(48, 550): [0.590, 0.594, 0.588, 0.592], (49, 600): [0.592, 0.596, 0.590],
                   (41, 450): [0.620, 0.624, 0.622]}

        self.assertEqual(tune.survivors(results), [(48, 550), (49, 600)])

    def test_race_runs_every_cell_and_ends_on_the_best(self):
        def reporter(command, **kwargs):
            s, tracks = (int(command[command.index(flag) + 1]) for flag in ("--kf_survivor_from_last",
                                                                            "--num_desired_tracks"))
            stats = Path(command[command.index("--output_dir") + 1]) / "stats"
            stats.mkdir(parents=True)
            t_err = 0.59 + 0.001 * abs(s - 48) + 0.0001 * abs(tracks - 550)
            (stats / "all_stats.json").write_text(json.dumps([{
                "gt_n_error_segments": 1, "gt_av_translation_error": t_err, "gt_av_rotation_error": 0.0,
                "gt_simple_error": 0.0, "num_loop_closures": 0}]))

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tune.subprocess, "run", side_effect=reporter):
            racing = tune.Racing(Path(tmp), Path(tmp) / "runs")
            alive = racing.race()
            runs = {(s, n): len(racing.runs((s, n))) for s in tune.S_VALUES for n in tune.TRACKS}

        self.assertEqual(alive, [(48, 550)])
        self.assertEqual(set(runs.values()), {tune.MIN_RUNS})

    def test_run_metrics_pool_sequences_by_segments(self):
        stats = [{"gt_n_error_segments": 100, "gt_av_translation_error": 0.5, "gt_av_rotation_error": 0.001,
                  "gt_simple_error": 1.0, "num_loop_closures": 3},
                 {"gt_n_error_segments": 300, "gt_av_translation_error": 0.7, "gt_av_rotation_error": 0.002,
                  "gt_simple_error": 2.0, "num_loop_closures": 5}]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "stats").mkdir()
            (Path(tmp) / "stats" / "all_stats.json").write_text(json.dumps(stats))
            metrics = tune.run_stats(Path(tmp))

        self.assertAlmostEqual(metrics["t_err"], 0.65)
        self.assertAlmostEqual(metrics["r_err"], 0.175)
        self.assertEqual((metrics["ate"], metrics["loop_closures"]), (1.5, 8))


if __name__ == "__main__":
    unittest.main()

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

"""Racing search over s = kf_survivor_from_last = feature_survivor_from_last and num_desired_tracks (SLAM, 00-10).

Every cell gets MIN_RUNS cuvslam_reporter runs. Each round then drops the cells whose mean pooled t_err exceeds the
leader's by more than Z standard errors of the difference and gives the rest STEP more runs, until STOP_SURVIVORS
cells remain, all have MAX_RUNS runs, or BUDGET runs were made. Runs already in --output_dir count.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

S_VALUES = range(43, 51)
TRACKS = (500, 525, 550, 575, 600)
MIN_RUNS, STEP, MAX_RUNS, BUDGET, Z, STOP_SURVIVORS = 3, 2, 16, 300, 2.0, 3
CONFIG = "kitti_recalibrated/kitti_recalibrated-slam_gt.cfg"
# CI's KITTI flags; three sequences at a time is the fastest on one GPU.
REPORTER_FLAGS = ["--odometry_mode=multicamera", "--rectified_stereo_camera=true", "--async_sba=false",
                  "--multicam_mode=moderate", "--use_segments", "--max_workers", "3"]


def run_stats(run: Path) -> dict:
    """Pooled t_err (%) and r_err (deg/100 m), mean ATE (m) and loop closures of one reporter run."""
    stats = json.loads((run / "stats" / "all_stats.json").read_text())
    weights = np.array([s["gt_n_error_segments"] for s in stats], float)
    return {"t_err": weights @ [s["gt_av_translation_error"] for s in stats] / weights.sum(),
            "r_err": weights @ [s["gt_av_rotation_error"] for s in stats] / weights.sum() * 100,
            "ate": float(np.mean([s["gt_simple_error"] for s in stats])),
            "loop_closures": sum(s["num_loop_closures"] for s in stats)}


def standard_errors(results: dict) -> dict:
    """Cell -> (mean, standard error); the run SD is at least the SD pooled over cells."""
    residuals = [x - np.mean(v) for v in results.values() if len(v) > 1 for x in v]
    pooled = np.sqrt(np.sum(np.square(residuals)) / sum(len(v) - 1 for v in results.values() if len(v) > 1))
    return {c: (np.mean(v), max(pooled, np.std(v, ddof=1) if len(v) > 1 else pooled) / np.sqrt(len(v)))
            for c, v in results.items()}


def survivors(results: dict) -> list:
    """Cells (cell -> t_err of each run) within Z standard errors of the difference from the leader."""
    stats = standard_errors(results)
    best = min(stats, key=lambda c: stats[c][0])
    return [c for c in results if stats[c][0] - stats[best][0] <= Z * np.hypot(stats[c][1], stats[best][1])]


class Racing:
    """Reporter runs under output_dir/s<s>_tracks<n>/run<k>."""

    def __init__(self, datasets_root: Path, output_dir: Path):
        self.datasets_root, self.output_dir = datasets_root, output_dir

    def _cell_dir(self, cell) -> Path:
        return self.output_dir / f"s{cell[0]}_tracks{cell[1]}"

    def runs(self, cell) -> List[Path]:
        return sorted(p.parent.parent for p in self._cell_dir(cell).glob("run*/stats/all_stats.json"))

    def run(self, cell) -> None:
        cell_dir = self._cell_dir(cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        out = cell_dir / f"run{1 + max((int(p.name[3:]) for p in cell_dir.glob('run*') if p.is_dir()), default=0)}"
        command = [sys.executable, "-m", "cuvslam_tools.reporter.cli",
                   "--test_config", str(self.datasets_root / CONFIG),
                   "--datasets_root", str(self.datasets_root), "--output_dir", str(out), *REPORTER_FLAGS,
                   "--kf_survivor_from_last", str(cell[0]), "--feature_survivor_from_last", str(cell[0]),
                   "--num_desired_tracks", str(cell[1])]
        with open(f"{out}.log", "w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"s {cell[0]}, {cell[1]} tracks: {out.name} done", flush=True)

    def race(self) -> list:
        alive = [(s, n) for s in S_VALUES for n in TRACKS]
        todo = {c: max(0, MIN_RUNS - len(self.runs(c))) for c in alive}
        new_runs = 0
        while True:
            # Round-robin, so that load drift spreads over all cells.
            for _, c in sorted((i, c) for c, k in todo.items() for i in range(k)):
                if new_runs < BUDGET:
                    self.run(c)
                    new_runs += 1
            results = {c: [run_stats(r)["t_err"] for r in self.runs(c)] for c in alive}
            alive = survivors(results)
            stats = standard_errors(results)
            print(f"{new_runs} new runs; kept " + ", ".join(
                f"{s}/{n} {stats[(s, n)][0]:.4f} (SE {stats[(s, n)][1]:.4f}, {len(results[(s, n)])} runs)"
                for s, n in sorted(alive, key=lambda c: stats[c][0])), flush=True)
            counts = {c: len(results[c]) for c in alive}
            if len(alive) <= STOP_SURVIVORS or new_runs >= BUDGET or min(counts.values()) >= MAX_RUNS:
                return alive
            todo = {c: STEP for c in alive if counts[c] < MAX_RUNS}


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments, race, and print the survivors' metrics."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--datasets_root", type=Path, required=True,
                        help="Directory with kitti_recalibrated/ from prepare_kitti_recalibrated.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for the reporter runs.")
    args = parser.parse_args(argv)
    racing = Racing(args.datasets_root, args.output_dir)
    alive = racing.race()
    print("| s / tracks | runs | t_err % (SE) | r_err deg/100m | ATE m | loop closures |")
    for cell in alive:
        runs = [run_stats(r) for r in racing.runs(cell)]
        mean = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
        se = np.std([r["t_err"] for r in runs], ddof=1) / np.sqrt(len(runs))
        print(f"| {cell[0]} / {cell[1]} | {len(runs)} | {mean['t_err']:.4f} ({se:.4f}) | {mean['r_err']:.3f} | "
              f"{mean['ate']:.2f} | {mean['loop_closures']:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

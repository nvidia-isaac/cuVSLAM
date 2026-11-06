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

"""Score candidate maps by running cuVSLAM's native relocalizer against frozen queries.

Run ``--prepare`` once to freeze ``queries.json``, then evaluate every candidate map
against that same manifest. Freezing matters: query selection, the random pose priors
and the success thresholds must not vary between candidates, or the comparison means
nothing.

Each query is one stereo frame plus a deliberately perturbed pose prior (0.15 m and
0.5 m offsets with a yaw error), simulating a robot that knows roughly where it is.
``callback_success`` means the relocalizer returned any pose at all;
``strict_success`` additionally requires it to be within 25 cm and 5 degrees.

Ground truth is used here and only here - to place the queries and to score them.
It never reaches a repair policy. Note that native RANSAC is randomized, so repeated
runs on an identical map will not reproduce identical scores exactly.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cuvslam_tools.offline_map_repair.core.workspace import HERE

# Every dataset-specific call comes from one package, so pointing this benchmark at
# another dataset means swapping this import, not editing the scoring below.
from cuvslam_tools.offline_map_repair.sessions.euroc import (
    frame_metadata,
    get_rig,
    ground_truth,
    load_frame,
)
from cuvslam_tools.offline_map_repair.sessions.trajectory import (
    errors,
    interpolate,
    load_trajectory,
    matrix,
    pose,
)


def make_queries(dataset, query_file, start_frame, evenly_spaced=False):
    """Freeze one query manifest to be shared by every candidate map.

    Queries live in the *map* frame, but ground truth lives in the EuRoC world frame,
    so the two are tied together by a single rigid transform anchored at one baseline
    keyframe. Opened with mode "x" so an existing manifest is never silently replaced.
    """
    frames, _ = frame_metadata(dataset)
    gt_t, gt = ground_truth(dataset)
    base_t, base = load_trajectory(HERE / "baseline/optimized_trajectory.json")
    # Fix the gauge at the first frame actually covered by ground truth.
    # EuRoC ground truth starts later than the images: never clamp that missing interval.
    anchor_index = int(np.searchsorted(base_t, gt_t[0]))
    anchor_timestamp = int(base_t[anchor_index])
    first_truth = interpolate(gt_t, gt, [anchor_timestamp])[0]
    map_from_gt = base[anchor_index] @ np.linalg.inv(first_truth)
    rng = np.random.default_rng(20261015)
    queries = []
    # Default selection: every 100th frame, stopping before the final frames whose
    # pose prior would clamp against the end of the ground-truth span.
    indices = range(start_frame, min(1901, len(frames)), 100)
    if evenly_spaced:
        eligible = [i for i, f in enumerate(frames) if i >= start_frame and gt_t[0] <= f["timestamp"] <= gt_t[-1]]
        indices = np.asarray(eligible)[np.linspace(0, len(eligible) - 1, 19, dtype=int)].tolist()
    for i in indices:
        timestamp = frames[i]["timestamp"]
        if not gt_t[0] <= timestamp <= gt_t[-1]:
            continue
        truth = map_from_gt @ interpolate(gt_t, gt, np.array([timestamp]))[0]
        # Two prior qualities per frame: a good guess and a poor one. A map that only
        # survives the tight prior has not really improved its searchable coverage.
        for magnitude in [0.15, 0.5]:
            guess = truth.copy()
            offset = rng.normal(size=3)
            offset *= magnitude / np.linalg.norm(offset)
            guess[:3, 3] += offset
            guess[:3, :3] = truth[:3, :3] @ Rotation.from_rotvec([0, rng.uniform(-0.12, 0.12), 0]).as_matrix()
            queries.append(
                {
                    "id": len(queries),
                    "frame": i,
                    "timestamp_ns": timestamp,
                    "offset_m": magnitude,
                    "guess": pose(guess),
                    "truth": pose(truth),
                }
            )
    manifest = {
        "dataset": str(dataset),
        "seed": 20261015,
        "map_from_gt": map_from_gt.tolist(),
        "frame_selection": "19 evenly spaced eligible frames" if evenly_spaced else "every 100 frames",
        "anchor_timestamp_ns": anchor_timestamp,
        "prior_source": "evaluation-only ground truth with fixed translation/yaw perturbations",
        "settings": {
            "horizontal_search_radius": 0.75,
            "vertical_search_radius": 0.75,
            "horizontal_step": 0.25,
            "vertical_step": 0.25,
            "angular_step_rads": 0.2,
        },
        "strict_success": {"translation_m": 0.25, "rotation_deg": 5},
        "queries": queries,
    }
    path = query_file
    with path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Froze {len(queries)} queries in {path}")


def evaluate(map_name, limit, query_file):
    """Run every frozen query against one map and write a per-query result file."""
    import cuvslam

    manifest = json.loads(query_file.read_text())
    frames, _ = frame_metadata(Path(manifest["dataset"]))
    rig = get_rig(str(Path(manifest["dataset"]) / "mav0"))
    cfg = cuvslam.Slam.Config(sync_mode=True)
    settings = cuvslam.Slam.LocalizationSettings(**manifest["settings"])
    results = []
    for query in manifest["queries"][: limit or None]:
        images = [load_frame(p) for p in frames[query["frame"]]["images_paths"]]
        # New instance for every query prevents query-to-query information leakage.
        slam = cuvslam.Slam(rig, [0], cfg)
        callback = []
        start = time.monotonic()
        slam.localize_in_map(
            str(HERE / map_name),
            query["timestamp_ns"],
            cuvslam.Pose(**query["guess"]),
            images,
            settings,
            lambda: None,
            lambda p, e: callback.append((p, e)),
        )
        if len(callback) != 1:
            raise RuntimeError("Synchronous localization did not finish")
        result, error = callback[0]
        row = {
            "query_id": query["id"],
            "frame": query["frame"],
            "callback_success": result is not None,
            "seconds": time.monotonic() - start,
            "error": error,
            "strict_success": False,
        }
        if result is not None:
            est = {
                "translation": [float(v) for v in result.translation],
                "rotation": [float(v) for v in result.rotation],
            }
            err = errors(matrix(est)[None], matrix(query["truth"])[None])
            row.update(
                {
                    "pose": est,
                    "translation_error_m": err["translation_rmse_m"],
                    "rotation_error_deg": err["rotation_median_deg"],
                }
            )
            row["strict_success"] = row["translation_error_m"] <= 0.25 and row["rotation_error_deg"] <= 5
        results.append(row)
        print(json.dumps(row), flush=True)
        output_name = (
            "localization_pilot.json" if limit else query_file.stem.replace("queries", "localization") + ".json"
        )
        (HERE / Path(map_name).parent / output_name).write_text(json.dumps(results, indent=2))
        del slam


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("map_name", nargs="?")
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--queries", type=Path, default=HERE / "queries.json")
    ap.add_argument("--start-frame", type=int, default=100)
    ap.add_argument("--evenly-spaced", action="store_true")
    args = ap.parse_args(argv)
    if args.prepare:
        make_queries(args.prepare, args.queries, args.start_frame, args.evenly_spaced)
    else:
        if not args.map_name:
            ap.error("map_name is required unless --prepare is given")
        evaluate(args.map_name, args.limit, args.queries)


if __name__ == "__main__":
    main()

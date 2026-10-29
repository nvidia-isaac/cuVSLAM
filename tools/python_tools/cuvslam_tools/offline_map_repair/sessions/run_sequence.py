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

"""Run one EuRoC sequence through cuVSLAM headlessly and save a reusable SLAM map.

Called twice per experiment with the same recording:

``--mode stereo``    produces ``baseline/`` - the map to be repaired.
``--mode inertial``  produces ``replay_inertial/`` - the reference trajectory whose
                     poses the repair policies adopt.

Both modes run ``OdometryWithSlamOffline``; odometry alone never builds the persistent
map. Ground truth is never read here, so a map and its replay stay independent of the
evaluation signal.

Outputs per session: ``map/`` (LMDB), ``tracking.json`` (per-frame odometry and SLAM
poses), ``optimized_trajectory.json`` (the final optimized SLAM poses, the input to
repair), and ``summary.json`` (provenance, including the saved map's SHA256).
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from cuvslam_tools.offline_map_repair.sessions.euroc import (
    calibration_source,
    frame_metadata,
    get_rig,
    load_frame,
)

# Shorthands for the pairing the example uses. They expand to --odometry-mode, and
# exist only so the common case stays short; they are not the available choices.
MODE_SHORTHAND = {"stereo": "Multicamera", "inertial": "Inertial"}

# Defaults, not requirements. Any of these can be overridden from the command line.
ODOMETRY_DEFAULTS = {"async_sba": False, "rectified_stereo_camera": False}
SLAM_DEFAULTS = {"sync_mode": True, "enable_reading_internals": True}


def settings(pairs, defaults):
    """Parse repeated key=value options over a cuVSLAM config, coercing by default type.

    The caller picks the configuration, so unknown keys and uncoercible values are
    errors rather than silent no-ops: a typo must not look like a run that honoured it.
    """
    values = dict(defaults)
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"Expected key=value, got {pair!r}")
        values[key] = raw
    return values


def apply_settings(config, values):
    """Assign parsed values onto a config instance, coercing strings to the field type."""
    for key, raw in values.items():
        if not hasattr(config, key):
            raise ValueError(f"{type(config).__qualname__} has no setting {key!r}")
        if isinstance(raw, str):
            current = getattr(config, key)
            if isinstance(current, bool):
                if raw.lower() not in ("true", "false"):
                    raise ValueError(f"{key} is a flag; expected true or false, got {raw!r}")
                raw = raw.lower() == "true"
            elif isinstance(current, int):
                raw = int(raw)
            elif isinstance(current, float):
                raw = float(raw)
        setattr(config, key, raw)
    return config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mode", choices=sorted(MODE_SHORTHAND), help="Shorthand for --odometry-mode")
    ap.add_argument("--odometry-mode", help="Odometry.OdometryMode name, for example Multicamera, Inertial, RGBD")
    ap.add_argument("--tracker-mode", default="OdometryWithSlamOffline", help="Tracker.Mode name")
    ap.add_argument("--odometry", action="append", metavar="KEY=VALUE", help="Odometry.Config override; repeatable")
    ap.add_argument("--slam", action="append", metavar="KEY=VALUE", help="Slam.Config override; repeatable")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source-commit", help="Commit used to build the installed cuVSLAM binding (optional provenance)")
    args = ap.parse_args(argv)
    if args.mode and args.odometry_mode:
        ap.error("pass either --mode or --odometry-mode, not both")
    odometry_mode = args.odometry_mode or MODE_SHORTHAND.get(args.mode or "stereo")
    os.environ.setdefault("RERUN", "0")
    import cuvslam

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Output directory must be empty")
    for name, enum in (("--odometry-mode", cuvslam.Odometry.OdometryMode), ("--tracker-mode", cuvslam.Tracker.Mode)):
        value = odometry_mode if name == "--odometry-mode" else args.tracker_mode
        if not hasattr(enum, value):
            ap.error(f"{name}: unknown {value!r}; choose from {[n for n in dir(enum) if not n.startswith('_')]}")
    odometry_values = settings(args.odometry, ODOMETRY_DEFAULTS)
    slam_values = settings(args.slam, SLAM_DEFAULTS)
    rig = get_rig(str(args.dataset / "mav0"))
    odom = apply_settings(cuvslam.Odometry.Config(), odometry_values)
    odom.odometry_mode = getattr(cuvslam.Odometry.OdometryMode, odometry_mode)
    slam = apply_settings(cuvslam.Slam.Config(), slam_values)
    # The map is what the repair edits, so the tracker mode has to be one that saves
    # one. Odometry-only modes produce a trajectory and nothing to repair.
    tracker = cuvslam.Tracker(rig, getattr(cuvslam.Tracker.Mode, args.tracker_mode), odom, slam)
    # Only an inertial mode consumes the IMU stream; others would be fed samples the
    # tracker ignores, so read them only when the chosen mode needs them.
    frames, pairing = frame_metadata(args.dataset, "Inertial" in odometry_mode)
    print(json.dumps({"pairing": pairing}), flush=True)
    results = []
    started = time.monotonic()

    def pose(p):
        return {"translation": [float(x) for x in p.translation], "rotation": [float(x) for x in p.rotation]}

    for frame in frames:
        timestamp = int(frame["timestamp"])
        if frame["type"] == "imu":
            m = cuvslam.ImuMeasurement()
            m.timestamp_ns = timestamp
            m.linear_accelerations = np.asarray(frame["accel"])
            m.angular_velocities = np.asarray(frame["gyro"])
            tracker.register_imu_measurement(0, m)
            continue
        images = [load_frame(p) for p in frame["images_paths"]]
        start = time.monotonic()
        estimate, slam_pose = tracker.track(timestamp, images)
        row = {
            "frame": len(results),
            "timestamp_ns": timestamp,
            "seconds": time.monotonic() - start,
            "odom": pose(estimate.world_from_rig.pose) if estimate.world_from_rig else None,
            "slam": pose(slam_pose) if slam_pose else None,
        }
        results.append(row)
        # Checkpoint progress periodically: a long sequence should not lose everything
        # if it is interrupted, and the printed counters make stalls obvious.
        if len(results) % 100 == 0:
            print(
                json.dumps(
                    {
                        "frames": len(results),
                        "failures": sum(x["odom"] is None for x in results),
                        "elapsed_s": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            (args.output / "tracking.json").write_text(json.dumps(results))
        if args.limit and len(results) >= args.limit:
            break
    (args.output / "tracking.json").write_text(json.dumps(results))
    saved = []
    tracker.slam.save_map(str(args.output / "map"), lambda ok: saved.append(ok))
    if saved != [True]:
        raise RuntimeError(f"Map save failed: {saved}")
    # The optimized SLAM poses, not the per-frame odometry: these are post loop
    # closure and pose-graph optimization, and are what a repair policy consumes.
    optimized = [{"timestamp_ns": int(p.timestamp_ns), **pose(p.pose)} for p in tracker.slam.get_all_slam_poses()]
    (args.output / "optimized_trajectory.json").write_text(json.dumps(optimized))
    summary = {
        "dataset": str(args.dataset.resolve()),
        # Record the configuration actually used: the caller chose it, so the session
        # has to carry it for the repair's provenance to mean anything.
        "odometry_mode": odometry_mode,
        "tracker_mode": args.tracker_mode,
        # Read back off the config objects, so this records what was applied rather
        # than what was typed: "false" and False must not look like different runs.
        "odometry_config": {k: str(getattr(odom, k)) for k in odometry_values},
        "slam_config": {k: str(getattr(slam, k)) for k in slam_values},
        "frames": len(results),
        "tracked": sum(x["odom"] is not None for x in results),
        "elapsed_s": time.monotonic() - started,
        "source_commit": args.source_commit,
        "cuvslam_version": str(cuvslam.get_version()),
        # A sequence carrying sensor_recalibrated.yaml is rigged from that file, and
        # this field is the record that mapping and localization used the same rig.
        "calibration": f"dataset {calibration_source(str(args.dataset / 'mav0'))}",
        # Recorded so `verify` can prove afterwards that the baseline was never edited.
        "map_sha256": hashlib.sha256((args.output / "map/data.mdb").read_bytes()).hexdigest(),
        "ground_truth_used": False,
        "pairing": pairing,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

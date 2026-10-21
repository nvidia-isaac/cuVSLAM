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

from cuvslam_tools.offline_map_repair.sessions.euroc import calibration_source, get_rig, load_frame, read_csv_data


def frame_metadata(dataset, inertial=False):
    """Build the ordered replay list of stereo frames and, optionally, IMU samples.

    Stereo frames are paired by *exact* timestamp; EuRoC cameras are hardware
    synchronized, so an unpaired timestamp means a missing file rather than a frame
    needing interpolation, and it is dropped. The returned ``pairing`` counts make
    that loss visible instead of silent.
    """
    root = dataset / "mav0"
    cameras = [
        {int(r["timestamp"]): r["filename"] for r in read_csv_data(str(root / f"cam{i}/data.csv"), "camera")}
        for i in range(2)
    ]
    timestamps = sorted(cameras[0].keys() & cameras[1].keys())
    frames = [
        {
            "type": "stereo",
            "timestamp": t,
            "images_paths": [str(root / f"cam{i}/data" / cameras[i][t]) for i in range(2)],
        }
        for t in timestamps
    ]
    if inertial:
        frames += [
            {"type": "imu", "timestamp": int(r["timestamp"]), "accel": r["accel"], "gyro": r["gyro"]}
            for r in read_csv_data(str(root / "imu0/data.csv"), "imu")
        ]
    # At an identical timestamp, deliver the IMU sample before its camera frame.
    frames.sort(key=lambda r: (r["timestamp"], r["type"] != "imu"))
    return frames, {"cam0": len(cameras[0]), "cam1": len(cameras[1]), "paired": len(timestamps)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mode", choices=["stereo", "inertial"], default="stereo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source-commit", help="Commit used to build the installed cuVSLAM binding (optional provenance)")
    args = ap.parse_args(argv)
    os.environ.setdefault("RERUN", "0")
    import cuvslam

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Output directory must be empty")
    # Multicamera = stereo visual only; Inertial additionally fuses the IMU stream.
    mode = getattr(cuvslam.Odometry.OdometryMode, "Multicamera" if args.mode == "stereo" else "Inertial")
    rig = get_rig(str(args.dataset / "mav0"))
    odom = cuvslam.Odometry.Config(async_sba=False, rectified_stereo_camera=False, odometry_mode=mode)
    slam = cuvslam.Slam.Config(sync_mode=True, enable_reading_internals=True)
    # Both sensor modes run SLAM. Odometry alone does not produce a reusable map.
    tracker = cuvslam.Tracker(rig, cuvslam.Tracker.Mode.OdometryWithSlamOffline, odom, slam)
    frames, pairing = frame_metadata(args.dataset, args.mode == "inertial")
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
        "mode": args.mode,
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

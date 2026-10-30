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


"""Order one EuRoC recording into the stereo frames and IMU samples a run replays."""

from cuvslam_tools.offline_map_repair.sessions.euroc.reader import read_csv_data


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

# Dataset routing

The replay helper accepts raw EuRoC/ASL and KITTI odometry layouts directly. EDEX is the canonical route for other
offline sources, including ROS bags.

## Common invocation

```bash
<python-with-cuvslam>/bin/python \
  <skill-dir>/scripts/replay_dataset.py DATASET \
  --repo CUVSLAM_REPO
```

Useful overrides:

```text
--output PATH                 Absolute, or relative to the dataset directory
--output-format tum|kitti    Default: tum
--mode auto|mono|multicamera|inertial|rgbd
--min-coverage FRACTION      Default: 0.9
--dataset-format FORMAT      Use only when automatic detection is ambiguous
```

## EuRoC / ASL

Accepted layouts are either `DATASET/mav0/{cam0,cam1,imu0}` or a path directly to `mav0`. The default mode is
stereo-inertial. Camera rows are joined by timestamp, so an unmatched trailing camera frame is reported and ignored
without modifying the CSV.

```bash
replay_dataset.py /data/rec_05 --repo /src/cuVSLAM
```

The helper uses `sensor_recalibrated.yaml` only when the dataset already contains a complete recalibrated set;
otherwise it uses each sensor's `sensor.yaml`. Do not copy repository calibration over a recording unless the user
explicitly requests that calibration.

## KITTI odometry

Pass a sequence containing `image_0/`, `image_1/`, `calib.txt`, and `times.txt`:

```bash
replay_dataset.py /data/kitti/sequences/06 --repo /src/cuVSLAM
```

When passing a root that contains several `sequences/NN` directories, select one explicitly:

```bash
replay_dataset.py /data/kitti --sequence 06 --repo /src/cuVSLAM
```

KITTI input does not imply KITTI output. The default remains TUM. Add `--output-format kitti` only when requested.

## EDEX

Pass an EDEX directory containing `stereo.edex`, or pass the `.edex` file itself. If a directory contains multiple
configs, use `--edex-config`.

```bash
replay_dataset.py /data/sequence_edex --repo /src/cuVSLAM
replay_dataset.py /data/sequence_edex --edex-config /data/sequence_edex/custom.edex --repo /src/cuVSLAM
```

Mode inference uses the EDEX rig and stream metadata: depth selects `rgbd`, stereo plus IMU selects `inertial`, two
or more cameras select `multicamera`, and one camera selects `mono`. Override only when the user's requested mode is
compatible with the streams. Legacy configs whose `sequence` contains one filename string per camera are normalized
through a temporary config; the source config is not edited.

## ROS bags

ROS topics and the rig frame cannot be inferred safely from a bag filename. Inspect the bag first, then select or
write an extraction config. Installed examples live under
`tools/python_tools/cuvslam_tools/bag2edex/configs/` (`realsense.yaml`, `realsense_imu.yaml`, `nova_hawk.yaml`, and
`oak6.yaml`). A custom config minimally identifies camera-info topics, image topics, the rig frame, ROS distribution,
and optionally an IMU topic.

```bash
replay_dataset.py /data/my_bag \
  --rosbag-config /src/cuVSLAM/tools/python_tools/cuvslam_tools/bag2edex/configs/realsense_imu.yaml \
  --repo /src/cuVSLAM
```

By default the intermediate conversion is written to `.cuvslam-edex` inside the bag directory and reused on a later
run. Use `--edex-output` for another location. The trajectory itself still defaults to `trajectory.txt` beside the
bag. The helper refuses to overwrite an existing incomplete conversion directory.

If the bag's topics or frames do not match a checked-in config, stop and request or derive the correct mapping from
bag metadata; do not guess extrinsics.

## Unsupported raw formats

Convert other recordings to EDEX with the repository's preparation/conversion tools, then replay the resulting
sequence. Do not add a one-off decoder to the skill unless that source format will be reused and its calibration and
timestamp semantics are known.

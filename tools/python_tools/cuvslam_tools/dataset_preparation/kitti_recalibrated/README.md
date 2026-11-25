# Recalibrated KITTI odometry

The KITTI odometry dataset re-rectified with the stereo calibration of Cvišić et al., "Recalibrating the KITTI Dataset
Camera Setup for Improved Odometry Accuracy", ECMR 2021 ([arXiv:2109.03462](https://arxiv.org/abs/2109.03462)), which
SOFT2 uses on the KITTI benchmark.

| Recording date | Sequences | Calibration |
|---|---|---|
| 2011_10_03 | 00-02, 13-21 | the authors' `lamor_00_02.yaml` (downloaded) |
| 2011_09_30 | 04-12 | the authors' `lamor_04_12.yaml` (downloaded) |
| 2011_09_26 | 03 | `calib_2011_09_26.json` (ours; the authors published none) |

Each new rectified pixel maps to a raw pixel through the new calibration, then into KITTI's rectified image through
KITTI's calibration, which is resampled there with OpenCV cubic interpolation. Images are cropped to where both cameras
have data. The baseline stays KITTI's 0.537 m, which suits cuVSLAM better than the authors' 0.539 m. Ground truth is
rotated into the new camera frame by the per-date rotation A in `gt_rotations.json`, fitted on 00-10 like SOFT2's
`KITTI_TO_CAM1`: `gt.txt` holds A^T G A, and a trajectory T of 11-21 is A T A^T in KITTI's frame.

## Prepare

```bash
prepare_kitti --output-dir ~/datasets/kitti  # if not converted yet
prepare_kitti_recalibrated --kitti-dir ~/datasets/kitti --raw-dir ~/datasets/kitti_recalibrated_raw \
    --output-dir ~/datasets/kitti_recalibrated
```

`--raw-dir` receives KITTI's raw-data calibrations and the authors' files from
[bitbucket.org/unizg-fer-lamor/kittical](https://bitbucket.org/unizg-fer-lamor/kittical). The output (21 GB) has
reporter configs for 00-10: `kitti_recalibrated-vio_gt.cfg` (odometry) and `kitti_recalibrated-slam_gt.cfg` (SLAM).

## Evaluate

```bash
cuvslam_reporter --test_config ~/datasets/kitti_recalibrated/kitti_recalibrated-slam_gt.cfg \
    --datasets_root ~/datasets --output_root ~/reports --pdf --odometry_mode=multicamera \
    --rectified_stereo_camera=true --async_sba=false --multicam_mode=moderate --use_segments \
    --kf_survivor_from_last 48 --feature_survivor_from_last 48 --num_desired_tracks 550
```

The last three flags override `Odometry.Internals` (defaults 41, 41, 450); below, s = kf_survivor_from_last =
feature_survivor_from_last. SLAM on 00-10, means over runs on an RTX 4090 (t_err run SD about 0.006 %); t_err and r_err
pool all KITTI segments, ATE is averaged and loop closures are summed over sequences:

| Dataset | s, tracks | Runs | t_err % | r_err deg/100 m | ATE m | Loop closures |
|---|---|---|---|---|---|---|
| `prepare_kitti` | 41, 450 | 3 | 0.762 | 0.218 | 1.90 | 475 |
| `prepare_kitti_recalibrated` | 41, 450 | 10 | 0.626 | 0.151 | 2.33 | 423 |
| `prepare_kitti_recalibrated` | 48, 550 | 16 | 0.591 | 0.138 | 1.86 | 538 |

At 48, 550 OpenCV cubic resampling beats linear (0.614), Lanczos-4 (0.604) and cubic or quintic B-splines (0.624,
0.599), 16 runs each.

## Reproduce the bundled files and the tuned values

`calib_2011_09_26.json` follows the paper on KITTI's 2011_09_26 checkerboard shot: all boards of one stereo pair, the
authors' fixed left fx, cu, cv and Tx, with fx and cv moved by KITTI's own 2011_09_26 vs 09_30 / 10_03 difference. Two
steps are ours: right fy and stereo pitch/roll fitted to the row alignment of sequence 03 stereo matches, and stereo
yaw set to the authors' disparity at infinity relative to KITTI's rectification (+0.055 px).

```bash
python -m cuvslam_tools.dataset_preparation.kitti_recalibrated.calibrate_09_26 --kitti-dir ~/datasets/kitti \
    --raw-dir ~/datasets/kitti_recalibrated_raw --out calib_2011_09_26.json
```

It reports 11 boards with 0.1145 px RMS reprojection error, row error -0.376 -> +0.004 px and disparity at infinity
+0.346 -> +0.055 px, and matches the bundled file to 1e-10.

`gt_rotations.json` comes from five odometry runs with the default parameters:

```bash
for k in 1 2 3 4 5; do
  cuvslam_reporter --test_config ~/datasets/kitti_recalibrated/kitti_recalibrated-vio_gt.cfg \
      --datasets_root ~/datasets --output_dir runs/odometry$k --odometry_mode=multicamera \
      --rectified_stereo_camera=true --async_sba=false --multicam_mode=moderate --use_segments
done
python -m cuvslam_tools.dataset_preparation.kitti_recalibrated.fit_gt_rotation --kitti-dir ~/datasets/kitti \
    --raw-dir ~/datasets/kitti_recalibrated_raw --runs runs/odometry* --out gt_rotations.json
```

The bundled rotations (deg) are 2011_10_03 (0.431, 0.415, -0.192), 2011_09_30 (0.374, 0.213, -0.171) and 2011_09_26
(0.145, 0.159, -0.483); GPU odometry is not deterministic, so new runs move them slightly.

s = 48 and 550 tracks come from a racing search: every cell of s 43-50 x tracks 500-600 gets three SLAM runs, then each
round drops the cells more than two standard errors behind the leader and gives the rest two more runs.

```bash
python -m cuvslam_tools.dataset_preparation.kitti_recalibrated.tune --datasets_root ~/datasets --output_dir runs/racing
```

It runs up to 300 reports (about 100 s each). The earlier searches that led to this box, 5 runs per cell:

| Search | Result (t_err %) |
|---|---|
| kf_survivor_from_last 20-70 | 41-70 flat (0.626 to 0.615), below 41 worse |
| kf and feature survivor_from_last | best 60 and 60: 0.601; feature 30 or less worse |
| both and num_desired_tracks | best 50, 50 and 600: 0.595; 1000 tracks worse |
| s 20-60 at 600 tracks | flat 44-60, minimum at 50 |
| s 35-48 x tracks 450-550, 10 runs | best 48, 525: 0.590 |
| racing | 48, 550: 0.591 (16 runs); 49/600, 49/550, 48/525, 47/525 within 0.0025 |

The optimum is a plateau, so a new race may end on any of its cells.

# Recalibrated KITTI odometry

`prepare_kitti --recalibrated` re-rectifies the converted images with the stereo calibration of Cvišić et al.,
"Recalibrating the KITTI Dataset Camera Setup for Improved Odometry Accuracy", ECMR 2021
([arXiv:2109.03462](https://arxiv.org/abs/2109.03462)), which SOFT2 uses on the KITTI benchmark.

```bash
prepare_kitti --raw-dir ~/datasets/kitti_raw --output-dir ~/datasets/recalibrated/kitti --recalibrated
```

`--raw-dir` also receives KITTI's raw-data calibrations and the authors' files from
[bitbucket.org/unizg-fer-lamor/kittical](https://bitbucket.org/unizg-fer-lamor/kittical). The layout and reporter
configs are those of `prepare_kitti`.

| Recording date | Sequences | Calibration |
|---|---|---|
| 2011_10_03 | 00-02, 13-21 | the authors' `lamor_00_02.yaml` |
| 2011_09_30 | 04-12 | the authors' `lamor_04_12.yaml` |
| 2011_09_26 | 03 | `calib_2011_09_26.json` (ours; the authors published none) |

Each new rectified pixel maps to a raw pixel through the new calibration, then into KITTI's rectified image through
KITTI's calibration, which is resampled there with OpenCV cubic interpolation (better for cuVSLAM than linear,
Lanczos-4 or B-splines). Images are cropped to where both cameras have data. The baseline stays KITTI's 0.537 m, which
suits cuVSLAM better than the authors' 0.539 m. Ground truth is rotated into the new camera frame by the per-date
rotation A in `gt_rotations.json`, fitted on 00-10 like SOFT2's `KITTI_TO_CAM1`: `gt.txt` holds A^T G A, and a
trajectory T of 11-21 is A T A^T in KITTI's frame.

SLAM on 00-10 (`kitti-slam_gt.cfg`, CI's KITTI flags `--odometry_mode=multicamera --rectified_stereo_camera=true
--async_sba=false --multicam_mode=moderate --use_segments`), means over runs on an RTX 4090; tracking parameters are
kf_survivor_from_last, feature_survivor_from_last and num_desired_tracks:

| Images | Tracking parameters | Runs | t_err % | r_err deg/100 m | ATE m | Loop closures |
|---|---|---|---|---|---|---|
| KITTI's | 41, 41, 450 | 3 | 0.762 | 0.218 | 1.90 | 475 |
| `--recalibrated` | 41, 41, 450 | 10 | 0.626 | 0.151 | 2.33 | 423 |
| `--recalibrated` | 48, 48, 550 | 16 | 0.591 | 0.138 | 1.86 | 538 |

## Reproduce the bundled files

`calib_2011_09_26.json` follows the paper on KITTI's 2011_09_26 checkerboard shot: all boards of one stereo pair, the
authors' fixed left fx, cu, cv and Tx, with fx and cv moved by KITTI's own 2011_09_26 vs 09_30 / 10_03 difference. Two
steps are ours: right fy and stereo pitch/roll fitted to the row alignment of sequence 03 stereo matches, and stereo
yaw set to the authors' disparity at infinity relative to KITTI's rectification (+0.055 px).

```bash
python -m cuvslam_tools.dataset_preparation.kitti.calibrate_09_26 --raw-dir ~/datasets/kitti_raw \
    --out calib_2011_09_26.json
```

It reports 11 boards with 0.1145 px RMS reprojection error, row error -0.376 -> +0.004 px and disparity at infinity
+0.346 -> +0.055 px, and matches the bundled file to 1e-10.

`gt_rotations.json` comes from five odometry runs with tracking parameters 41, 41, 450:

```bash
for k in 1 2 3 4 5; do
  cuvslam_reporter --test_config ~/datasets/recalibrated/kitti/kitti-vio_gt.cfg \
      --datasets_root ~/datasets/recalibrated --output_dir runs/odometry$k --odometry_mode=multicamera \
      --rectified_stereo_camera=true --async_sba=false --multicam_mode=moderate --use_segments
done
python -m cuvslam_tools.dataset_preparation.kitti.fit_gt_rotation --raw-dir ~/datasets/kitti_raw \
    --runs runs/odometry* --out gt_rotations.json
```

The bundled rotations (deg) are 2011_10_03 (0.431, 0.415, -0.192), 2011_09_30 (0.374, 0.213, -0.171) and 2011_09_26
(0.145, 0.159, -0.483); GPU odometry is not deterministic, so new runs move them slightly.

# Offline EDEX tracking

Use `cuvslam_tracker` for rich Python single-sequence diagnostics and `cuvslam_api_launcher` for a clean C++ process.

## Python tracker

```bash
cuvslam_tracker \
  --dataset /path/to/sequence \
  --config_path /path/to/sequence/stereo.edex \
  --odometry_mode multicamera \
  --output_dir /tmp/cuvslam-output
```

Relevant options include `--odometry_mode`, `--use_slam`, `--gt_path`, `--repeat_type`, `--num_loops`,
`--blackout_period`, and `--visualize_rerun`.

## C++ API launcher

```bash
./build/bin/cuvslam_api_launcher \
  --edex=/path/to/sequence/stereo.edex \
  --cfg_odom_mode=0 \
  --print_odom_poses=/tmp/odom.txt
```

Use `--cfg_enable_slam --cfg_enable_export --print_slam_poses=/tmp/slam.txt` for SLAM. Development-only settings
are exposed as launcher gflags such as `--expert_sba_num_frames`.

## Evaluate the C++ launcher with reporter metrics

```bash
cuvslam_reporter \
  --tracker_backend api_launcher \
  --api_launcher_path ./build/bin/cuvslam_api_launcher \
  --test_config /path/to/dataset/report.cfg \
  --datasets_root /path/to/datasets \
  --output_root /tmp/cuvslam-reports \
  --odometry_mode multicamera \
  --use_segments \
  -- \
  --expert_sba_num_frames=9
```

The reporter keeps ground-truth metrics, repeat/shuttle scoring, plots, and benchmark exports while the C++ process
receives development-only flags. Its exact command, stdout/stderr, and JSONL records are saved under `logs/`.

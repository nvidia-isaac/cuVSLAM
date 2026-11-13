# tracker — Offline Image-Sequence Tracking

Standalone CLI tool for running cuVSLAM on a pre-recorded EDEX sequence (images + `stereo.edex`). Runs synchronously (blocking mode), making it deterministic and reproducible — ideal for debugging pose accuracy issues.

Binary: `build/bin/tracker`

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `CUVSLAM_DATASETS` | Yes | Path to the **parent** directory of the sequence folder. **Must end with `/`.** |
| `CUVSLAM_OUTPUT` | Yes | Path for tracker output files. **Must end with `/`.** Create it before running. |

**Common env var errors:**

| Error message | Fix |
|---|---|
| `Missing end slash for Path Environment variable: CUVSLAM_DATASETS` | Add trailing `/` to the path |
| `Missing Environment variable: CUVSLAM_OUTPUT` | Set `CUVSLAM_OUTPUT=/tmp/cuvslam_output/` and `mkdir -p` the directory first |

---

## Basic usage

```bash
CUVSLAM_DATASETS=/path/to/datasets/ \
CUVSLAM_OUTPUT=/tmp/cuvslam_output/ \
  ./build/bin/tracker \
  -edex <sequence_subfolder> \
  -edex_filename stereo.edex \
  -output_edex result.edex \
  -mode multicamera
```

The tracker looks for the EDEX at `CUVSLAM_DATASETS/<edex>/<edex_filename>` and expects images relative to that folder.

---

## Tracking modes

```bash
-mode multicamera   # Stereo VO only (default) — no IMU
-mode inertial      # Stereo + IMU (VIO)
-mode mono          # Single camera, no scale
-mode rgbd          # Monocular + depth
```

---

## Common flag reference

### Input / output

| Flag | Default | Description |
|---|---|---|
| `-edex` | `""` | Sequence subfolder inside `CUVSLAM_DATASETS` |
| `-edex_filename` | `stereo.edex` | EDEX filename inside the sequence folder |
| `-output_edex` | `""` | Path to write result EDEX (trajectory + landmarks) |
| `-output_poses` | `""` | Path to write trajectory in KITTI format |
| `-gt_file` | `""` | Ground truth file in KITTI format — enables ATE/RTE metrics |
| `-start_frame` | `0` | First frame index to process |
| `-end_frame` | `-1` | Last frame index (−1 = all frames) |

### Reproducibility

| Flag | Default | Description |
|---|---|---|
| `-async_sba` | `false` | `false` = blocking SBA (deterministic), and selects an offline `Tracker.Mode`. Keep `false` for debugging, with `-sync_slam=true`. |
| `-slam_reproduce_mode` | `false` | Sync + non-random SLAM for exact reproducibility |

### SLAM

| Flag | Default | Description |
|---|---|---|
| `-use_slam` | `false` | Enable SLAM loop closure (only after odometry is solid) |
| `-throttling_time_ms` | `0` | Min ms between loop closures |

### Masking / image

| Flag | Default | Description |
|---|---|---|
| `-border_top` | `0` | Pixels to ignore at top of image |
| `-border_bottom` | `0` | Pixels to ignore at bottom |
| `-border_left` | `0` | Pixels to ignore at left |
| `-border_right` | `0` | Pixels to ignore at right |

### IMU debug

| Flag | Default | Description |
|---|---|---|
| `-debug_imu_mode` | `false` | Pure IMU integration, no vision — validates IMU alignment |

### Shuttle / repeat

| Flag | Default | Description |
|---|---|---|
| `-repeat_type` | `""` | `Shuttle` = forward then backward (drift check), `Repeat` = loop |
| `-sequence_num_repeats` | `1` | Number of repeats |

---

## Worked examples

### Stereo VO on an EDEX dataset

```bash
CUVSLAM_DATASETS=~/datasets/ \
CUVSLAM_OUTPUT=/tmp/out/ \
  ./build/bin/tracker \
  -edex my_sequence \
  -edex_filename stereo.edex \
  -output_edex ~/datasets/my_sequence/result.edex \
  -mode multicamera
```

### VIO (with IMU)

```bash
CUVSLAM_DATASETS=~/datasets/ \
CUVSLAM_OUTPUT=/tmp/out/ \
  ./build/bin/tracker \
  -edex my_sequence \
  -edex_filename stereo.edex \
  -output_edex ~/datasets/my_sequence/result_inertial.edex \
  -mode inertial
```

### With ground truth — ATE/RTE metrics

```bash
CUVSLAM_DATASETS=~/datasets/ \
CUVSLAM_OUTPUT=/tmp/out/ \
  ./build/bin/tracker \
  -edex my_sequence \
  -edex_filename stereo.edex \
  -output_edex result.edex \
  -gt_file ~/datasets/my_sequence/gt.txt \
  -mode multicamera
```

### Shuttle mode — forward/backward drift check

```bash
CUVSLAM_DATASETS=~/datasets/ \
CUVSLAM_OUTPUT=/tmp/out/ \
  ./build/bin/tracker \
  -edex my_sequence \
  -edex_filename stereo.edex \
  -output_edex result_shuttle.edex \
  -repeat_type Shuttle \
  -mode multicamera
```

A consistent forward/backward trajectory means good odometry. A large discrepancy indicates drift or calibration error.

### IMU alignment check

```bash
CUVSLAM_DATASETS=~/datasets/ \
CUVSLAM_OUTPUT=/tmp/out/ \
  ./build/bin/tracker \
  -edex my_sequence \
  -edex_filename stereo.edex \
  -debug_imu_mode \
  -mode inertial
```

### Run with SLAM enabled

```bash
CUVSLAM_DATASETS=~/datasets/ \
CUVSLAM_OUTPUT=/tmp/out/ \
  ./build/bin/tracker \
  -edex my_sequence \
  -edex_filename stereo.edex \
  -output_edex result_slam.edex \
  -use_slam \
  -mode multicamera
```

---

## Reading the output

**Console summary** (printed at end of run):

```
batch residual = 0.00359       ← mean reprojection error across all frames (lower = better)
average online residual = ...  ← real-time reprojection error
max frame residual = 0.036 in frame 6064  ← worst single frame (spikes indicate jumps)
average visible observations = 291       ← feature track count per frame
fps = 2921                     ← processing speed
```

**Output EDEX** — contains `rig_positions` (trajectory) and optionally landmarks. Visualize with:

```bash
python3 tools/edex/result_visualizer/result_visualizer.py result.edex
```

**Analysing jumps from output EDEX** (Python snippet):

```python
import json, numpy as np

with open('result.edex') as f:
    data = json.load(f)
rp = data[1]['rig_positions']
keys = sorted(rp.keys(), key=int)
pos  = [rp[k]['translation'] for k in keys]
diffs = [np.linalg.norm(np.array(pos[i+1]) - np.array(pos[i])) for i in range(len(pos)-1)]
mean  = np.mean(diffs)
jumps = [(int(keys[i]), diffs[i]) for i in range(len(diffs)) if diffs[i] > 3 * mean]
print(f"mean={mean:.4f}m  jumps(>3×mean): {len(jumps)}")
for fid, d in jumps[:20]:
    print(f"  frame {fid}: {d:.4f}m")
```

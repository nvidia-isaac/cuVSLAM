# Debug Dump EDEX Patching

When you capture a debug dump from `isaac_ros_visual_slam` (via `enable_debug_mode: true`), the resulting EDEX is **not directly compatible** with the `tracker` CLI. Several format issues must be fixed before the tracker can read it.

---

## Known issues

The following bugs have been observed in the debug dump produced by `isaac_ros_visual_slam` (tested with Isaac ROS 4.3):

| # | File | Issue | Symptom |
|---|---|---|---|
| 1 | `stereo.edex` | `frame_end` set to `1` instead of actual last frame index | Tracker processes only 2 frames then exits |
| 2 | `stereo.edex` | `distortion_params` missing from camera intrinsics | `[ERROR] Failed to start camera rig` |
| 3 | `stereo.edex` | `sequence` entries are bare strings instead of `["string"]` lists | `[ERROR] Failed to start camera rig` |
| 4 | `stereo.edex` | `imu` section missing `gyro_noise_density`, `accel_noise_density`, `frequency`, `g` | Tracker crash in inertial mode |
| 5 | `frame_metadata.jsonl` | Malformed JSON: `"depths":[}` instead of `"depths":[]}` | JSON parse error on read |

All five issues are fixed by `scripts/patch_debug_dump_edex.py`.

---

## Quick fix

```bash
python3 scripts/patch_debug_dump_edex.py <dump_dir> [output_dir]
```

- `dump_dir` — directory written by `enable_debug_mode` (contains `stereo.edex`, `frame_metadata.jsonl`, `images/`)
- `output_dir` — where to write patched files (default: `<dump_dir>_patched`)

The script:
1. Writes patched `stereo.edex` and `frame_metadata.jsonl` to `output_dir`
2. Creates a symlink `output_dir/images/ -> dump_dir/images/` (images are not copied)
3. Copies `IMU.jsonl` if present
4. Prints the exact `tracker` command to run on the patched sequence

---

## IMU noise parameters (Fix #4)

The IMU section is missing four noise parameters. The patch script inserts D435i defaults:

| Parameter | Default | Notes |
|---|---|---|
| `gyro_noise_density` | `1.0e-3` | Match your sensor's IMU calibration |
| `accel_noise_density` | `1.0e-2` | Match your sensor's IMU calibration |
| `frequency` | `200.0` Hz | D435i default |
| `g` | `9.81` m/s² | Standard gravity |

**Update these for your sensor.** Wrong values cause VIO drift or jumps. Values should match the IMU noise params you pass to `isaac_ros_visual_slam` (e.g. `gyro.noise_density` in the launch file).

---

## Container and permissions

If the debug dump was written by the Isaac ROS Docker container, the dump directory is **owned by root**. You cannot write patched files back to it directly from the host.

**Option A — Run patch inside the container (recommended):**

```bash
# Copy script into the shared workspace mount first
cp scripts/patch_debug_dump_edex.py ~/workspaces/isaac_ros-dev/

# Run inside the container
docker exec isaac_ros_dev_container \
  python3 /workspaces/isaac_ros-dev/patch_debug_dump_edex.py \
  /workspaces/isaac_ros-dev/path/to/dump \
  /workspaces/isaac_ros-dev/path/to/dump_patched
```

**Option B — Write patched files to a host-owned directory:**

The `output_dir` can be any path the current user can write to. The `images/` symlink still points into the root-owned dump (read access is sufficient).

```bash
python3 scripts/patch_debug_dump_edex.py \
  /root/owned/dump \
  ~/my_patched_dump
```

---

## Verifying the fix

Run the tracker on the patched sequence. It should print frame-processing progress rather than immediately exiting with `[ERROR] Wrong frame metadata log` or `[ERROR] Wrong body`.

See `commands/tracker.md` for the full flag reference and worked examples.

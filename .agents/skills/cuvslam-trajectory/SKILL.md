---
name: cuvslam-trajectory
description: >
  Replay recorded datasets through NVIDIA cuVSLAM and export a verified pose trajectory.
  Use for offline replay of EuRoC/ASL recordings, KITTI odometry sequences, EDEX datasets,
  or ROS bags, especially when the requested deliverable is a TUM or KITTI trajectory file.
  Do not use for live-camera tracking or for diagnosing accuracy regressions in an existing run.
---

# cuVSLAM trajectory replay

Use the bundled scripts instead of writing a one-off tracker. They handle dataset detection, the extra-camera-frame
case, released/source API differences, atomic output, and validation.

## Defaults

Unless the user overrides them:

- output format: TUM, `timestamp tx ty tz qx qy qz qw`;
- filename: `trajectory.txt`;
- output directory: the supplied dataset directory, or the parent of a supplied dataset file;
- tracking mode: infer it from the data (`inertial` for EuRoC with IMU, `multicamera` for KITTI stereo, and from the
  rig/streams for EDEX).

Treat any user-specified format, filename, location, sequence, mode, calibration, or ROS topic mapping as
authoritative.

## Required workflow

1. Resolve the cuVSLAM repository and dataset paths. If the prompt does not identify one dataset and discovery finds
   multiple plausible inputs, ask which one to use.
2. Inspect without running:

   ```bash
   python3 <skill-dir>/scripts/replay_dataset.py <dataset> --repo <cuvslam-repo> --inspect
   ```

   Add `--sequence`, `--edex-config`, or `--dataset-format` when inspection reports ambiguity. Read
   [references/dataset-routing.md](references/dataset-routing.md) for format-specific flags, especially ROS bags.
3. Use a Python interpreter where both `cuvslam` and `cuvslam_tools` import. Prefer an existing compatible
   environment. Otherwise use the released-wheel bootstrap instead of starting a source build:

   ```bash
   python3 <skill-dir>/scripts/bootstrap_runtime.py --repo <cuvslam-repo>
   <cuvslam-repo>/.venv-trajectory/bin/python \
       <skill-dir>/scripts/replay_dataset.py <dataset> --repo <cuvslam-repo> [user overrides]
   ```

   The bootstrap selects the wheel by repository version, CUDA major, Python ABI, glibc, and architecture. It is
   safe to rerun after an interrupted install. If CUDA detection is ambiguous, inspect `/usr/local/cuda` and pass
   `--cuda-major 12|13`; do not select the driver-reported maximum CUDA version unless that runtime is installed. If
   no released wheel matches, explain the mismatch. Build from source only when necessary and compatible with the
   user's time/compute constraints; run it in the foreground and do not leave a polling wrapper as the deliverable.
4. The replay script validates TUM output automatically. For an existing or externally generated TUM file, run:

   ```bash
   python3 <skill-dir>/scripts/validate_tum.py <trajectory-path> [--expected-rows N]
   ```

5. Report success only after the replay command exits zero and validation prints `"valid": true`. State the absolute
   output path, output format, poses written versus frames, tracking losses, and any deliberately excluded unmatched
   inputs.

## Invariants

- Do not claim that a background build or replay "will finish". Stay with the foreground command until it exits.
- Do not fabricate, copy, or substitute ground-truth poses for tracker output.
- Do not delete dataset rows, images, or calibration files to force synchronization. The replay script intersects
  EuRoC camera timestamps in memory and reports unmatched inputs.
- Do not silently invent ROS topics, a rig frame, or extrinsics. A ROS bag needs a matching extraction config.
- Do not launch Rerun for a trajectory-only request. The bundled path is headless.
- Preserve provider-native exit status and completion output when an evaluator may need execution provenance.
- If the user asks for KITTI output, pass `--output-format kitti`; otherwise keep the TUM default even for KITTI
  input.

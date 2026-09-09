# cuVSLAM Agent Skills

Repository skills for NVIDIA cuVSLAM (CUDA-accelerated Visual SLAM). Codex discovers them automatically from this
`.agents/skills/` directory; the same skills can also be installed into Claude Code and OpenClaw.

## Skills

### cuvslam-onboard

Build, install, and run cuVSLAM / PyCuVSLAM. Covers:

- Environment setup and build (source + wheel + Docker)
- Dataset preparation (KITTI, EuRoC, TUM RGB-D, multi-camera)
- Running all tracking modes (stereo, mono, mono-depth, stereo-inertial, multi-camera, Multisensor)
- SLAM workflow (mapping, localization, loop closure)
- Live camera setup (RealSense, ZED, OAK-D, Orbbec)
- C++ tools and ROS 2 integration

### cuvslam-troubleshoot

Diagnose and fix cuVSLAM tracking/pose accuracy issues. Covers:

- Quick triage (build vs tracking vs integration)
- 14-step diagnostic workflow from upstream TROUBLESHOOTING.md
- Debug data dump for all APIs (C++, Python, ROS 2)
- Calibration, synchronization, and image quality checks

### cuvslam-trajectory

Replay recorded datasets through cuVSLAM and export a validated trajectory. Covers:

- Raw EuRoC/ASL and KITTI odometry replay
- EDEX replay and ROS bag conversion through EDEX
- TUM output by default, with dataset-local `trajectory.txt`
- Runtime bootstrapping, camera timestamp pairing, and output validation

### cuvslam-ci

Work on cuVSLAM CI/CD on GitHub Actions. Covers:

- PR-verify and nightly pipelines (build, test, lint, dataset evaluation)
- Dataset provisioning, staging, and KPI reporting
- Build/test matrix, branch rulesets, and repository secrets

## Install

### Codex

No installation is required. Launch Codex anywhere inside the repository and it scans `.agents/skills/`
automatically. Invoke a skill explicitly with `$cuvslam-onboard`, `$cuvslam-troubleshoot`,
`$cuvslam-trajectory`, or `$cuvslam-ci`; matching requests can also trigger skills automatically.

### Claude Code

Skills are loaded by Claude Code from `~/.claude/skills/`. From the repo root, copy the skill folders there:

```bash
cp -r .agents/skills/cuvslam-onboard ~/.claude/skills/
cp -r .agents/skills/cuvslam-troubleshoot ~/.claude/skills/
cp -r .agents/skills/cuvslam-trajectory ~/.claude/skills/
cp -r .agents/skills/cuvslam-ci ~/.claude/skills/
```

Or clone the repo and symlink:

```bash
git clone https://github.com/nvidia-isaac/cuVSLAM.git
cd cuVSLAM
ln -s $(pwd)/.agents/skills/cuvslam-onboard ~/.claude/skills/cuvslam-onboard
ln -s $(pwd)/.agents/skills/cuvslam-troubleshoot ~/.claude/skills/cuvslam-troubleshoot
ln -s $(pwd)/.agents/skills/cuvslam-trajectory ~/.claude/skills/cuvslam-trajectory
ln -s $(pwd)/.agents/skills/cuvslam-ci ~/.claude/skills/cuvslam-ci
```

Once installed, invoke a skill in Claude Code by typing its name as a slash command:

```bash
/cuvslam-onboard
/cuvslam-troubleshoot
/cuvslam-trajectory
/cuvslam-ci
```

Claude Code will also trigger the relevant skill automatically when you describe a related task (e.g. "build cuVSLAM", "pose drift", "cuVSLAM not tracking", "add a dataset to CI").

### OpenClaw

Copy skill folders to your OpenClaw workspace `skills/` directory, or use:

```bash
openclaw skills install .agents/skills/cuvslam-onboard/
openclaw skills install .agents/skills/cuvslam-troubleshoot/
openclaw skills install .agents/skills/cuvslam-trajectory/
openclaw skills install .agents/skills/cuvslam-ci/
```

## Source

Based on [nvidia-isaac/cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) documentation.

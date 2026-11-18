# cuVSLAM Skills

Agent skills for NVIDIA cuVSLAM (CUDA-accelerated Visual SLAM) — for use with [Claude Code](https://claude.ai/code) and [OpenClaw](https://github.com/openclaw/openclaw).

## Skills

### cuvslam-onboard

Build, install, and run cuVSLAM / PyCuVSLAM. Covers:

- Environment setup and build (source + wheel + Docker)
- Dataset preparation (KITTI, EuRoC, TUM RGB-D, multi-camera)
- Running all tracking modes (stereo, mono, mono-depth, stereo-inertial, multi-camera, Multisensor)
- SLAM workflow (mapping, localization, global localization, loop closure)
- Live camera setup (RealSense, ZED, OAK-D, Orbbec)
- C++ tools and ROS 2 integration

### cuvslam-troubleshoot

Diagnose and fix cuVSLAM tracking/pose accuracy issues. Covers:

- Quick triage (build vs tracking vs integration)
- 14-step diagnostic workflow from upstream TROUBLESHOOTING.md
- Debug data dump for all APIs (C++, Python, ROS 2)
- Calibration, synchronization, and image quality checks

### cuvslam-ci

Work on cuVSLAM CI/CD on GitHub Actions. Covers:

- PR-verify and nightly pipelines (build, test, lint, dataset evaluation)
- Dataset provisioning, staging, and KPI reporting
- Build/test matrix, branch rulesets, and repository secrets

## Install

### Claude Code

Skills are loaded by Claude Code from `~/.claude/skills/`. From the repo root, copy the skill folders there:

```bash
cp -r cuvslam-skills/cuvslam-onboard ~/.claude/skills/
cp -r cuvslam-skills/cuvslam-troubleshoot ~/.claude/skills/
cp -r cuvslam-skills/cuvslam-ci ~/.claude/skills/
```

Or clone the repo and symlink:

```bash
git clone https://github.com/nvidia-isaac/cuVSLAM.git
cd cuVSLAM_staging
ln -s $(pwd)/cuvslam-skills/cuvslam-onboard ~/.claude/skills/cuvslam-onboard
ln -s $(pwd)/cuvslam-skills/cuvslam-troubleshoot ~/.claude/skills/cuvslam-troubleshoot
ln -s $(pwd)/cuvslam-skills/cuvslam-ci ~/.claude/skills/cuvslam-ci
```

Once installed, invoke a skill in Claude Code by typing its name as a slash command:

```bash
/cuvslam-onboard
/cuvslam-troubleshoot
/cuvslam-ci
```

Claude Code will also trigger the relevant skill automatically when you describe a related task (e.g. "build cuVSLAM", "pose drift", "cuVSLAM not tracking", "add a dataset to CI").

### OpenClaw

Copy skill folders to your OpenClaw workspace `skills/` directory, or use:

```bash
openclaw skills install cuvslam-onboard/
openclaw skills install cuvslam-troubleshoot/
openclaw skills install cuvslam-ci/
```

## Source

Based on [nvidia-isaac/cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) documentation.

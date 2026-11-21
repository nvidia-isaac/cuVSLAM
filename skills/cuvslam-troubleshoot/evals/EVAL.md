# Troubleshooting evaluation

## Questions

The smoke dataset covers a CUDA wheel mismatch, stereo extrinsic diagnosis, a ROS playback-rate comparison,
and IMU isolation using VO-versus-VIO evidence. Negative cases cover first-time installation and pose export.

## Behaviors

- Use the reported environment and observations to choose the next diagnostic step.
- Prefer a matching runtime/wheel for the import failure rather than unrelated calibration work.
- Prioritize stereo extrinsics when mono works, timestamps match, and image quality is already checked.
- Investigate load and timing when slow ROS playback removes pose jumps.
- Check IMU alignment, extrinsics, gravity, timing, and noise when VO outperforms VIO.
- Distinguish a supported hypothesis from a proven cause, and honor requests for an explanation only.
- Stay inactive for first-time setup and trajectory-only requests.

## Notes

All smoke cases supply observations in the prompt. No bag, source checkout, ROS installation, CUDA runtime,
camera, or download is required. Do not grade on hardware access or a reproduction the prompt did not request.
Negative cases do not require sibling skills; the skill under evaluation must stay inactive.
Use the default SkillEvaluator sandbox and run with and without the skill on both Claude Code and Codex.

For extended evaluation, use team-owned calibration faults, a representative ROS bag, and a debug dump with known
format issues. Preserve inputs, reproduce failures, apply a justified repair to a copy, and measure the result.
Record the dataset and product revision, known fault, diagnosis accuracy, artifact changes, tokens, and wall-clock time.
The evaluation report should distinguish reasoning-only smoke results from measured GPU/ROS outcomes and include
both baseline and with-skill results.

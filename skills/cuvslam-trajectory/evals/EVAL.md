# Trajectory evaluation

## Questions

The smoke dataset covers valid TUM validation, rejection of duplicate timestamps, headless EDEX inspection,
a ROS-bag request missing its mapping, and evaluation of existing poses without replay.
Negative cases distinguish live-camera setup and tracking diagnosis.

## Behaviors

- Use the bundled validator and preserve its exit status and measured output.
- Report three rows, full coverage, two seconds, and two distance units for the valid fixture.
- Reject the duplicate timestamp without repairing or overwriting the supplied file.
- Run `replay_dataset.py --inspect` for metadata-only input and do not claim it produced poses.
- Resolve real ROS topic/frame/configuration information before conversion; never invent extrinsics.
- Avoid runtime bootstrap for inspection and validation; these operations use only Python's standard library.
- Stay inactive for live-camera setup and accuracy-diagnosis requests.
- Evaluate the synthetic offset estimate against the supplied GT in a shared world without alignment: ATE RMSE
  1 meter, ARE RMSE 30 degrees, normalized ATE RMSE 50 percent. Use NumPy, without installing cuVSLAM.
- When frames differ and calibration is absent, label an explicitly requested initial-window comparison
  provisional and explain why a world transform cannot generally recover the missing extrinsic.

## Notes

Each executable case selects its own `files` fixture, staged at `/workspace/input/` by SkillEvaluator.
The TUM fixtures are hand-authored synthetic numbers, not tracker output or ground truth. `stereo.edex` is a minimal
metadata-only fixture with no images or usable calibration; it tests routing and defaults, not tracking.
No GPU, external dataset, product checkout, or custom container is needed for the smoke cases. Use the default
Python sandbox, with NumPy installed for metric evaluation. The incomplete-bag case intentionally stops at identifying missing information.
Negative cases need not have sibling skills installed.

Run with and without this skill on both Claude Code and Codex. In the baseline, remove the skill and all its helpers
from the agent's discovery paths while keeping the selected input fixtures and task prompt identical. An agent may
validate the format itself in the baseline; compare observable correctness, tool use, tokens, and elapsed time.

Extended evaluation requires a compatible CUDA runtime and PyCuVSLAM environment plus real recorded inputs:

- EuRoC with an unmatched camera timestamp: export poses without changing the source CSV or calibration.
- KITTI: verify TUM is the default, and an explicitly requested KITTI file has 12 finite values per row.
- EDEX: replay calibrated images and report frames, poses, coverage, and tracking losses.
- ROS bag: convert with a verified mapping, replay the EDEX, and validate the resulting trajectory.
- Replay-only request with GT present: stop after replay and validation; do not calculate metrics.
- Replay-and-evaluate request: establish the exported rig frame and actual calibration, convert physical frames,
  then associate and align once with fixed scale. Report definitions, units, exclusions, and output paths.

Record dataset/product revisions, GPU/runtime, harness and model versions, transcripts, exit codes, and artifacts.
Report baseline and with-skill results separately for smoke and extended runs. Never substitute synthetic or
ground-truth poses for replay output.

## Calculation regression tests

Tests live here, alongside skill evaluations, rather than under the repository's product or replay scripts.
With NumPy and PyYAML available, run from any directory:

```bash
python3 -m unittest discover -v -s <skill-dir>/evals -p 'test_*.py'
```

These test full sensor/world transforms and their direction, a rotating lever arm, quaternion ordering and signs,
epoch timestamps, one-to-one matching and exclusions, fit degeneracy, fixed initial windows, fixed scale,
metric units and the GT distance denominator, KITTI timestamp requirements, and the independent CLI's artifacts.

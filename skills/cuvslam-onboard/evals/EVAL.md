# Onboarding evaluation

## Questions

The smoke dataset covers wheel selection, RGB-D plus IMU mode selection, source-binding installation,
and offline SLAM configuration. Two negative cases distinguish trajectory export and existing tracking failures.
Cases include explicit invocation, implicit requests, and an existing workspace scenario.

## Behaviors

- Match the installed CUDA toolkit, architecture, Python ABI, and supported wheel matrix.
- Choose Multisensor for one RGB-D camera plus IMU, including its cuNLS and pinhole constraints.
- Reuse provided paths and explain the binding reinstall required after a C++ rebuild.
- Keep the offline Tracker mode and synchronization flags consistent; require a localization pose hint.
- Respect explanation-only prompts and avoid unnecessary downloads, builds, or installations.
- Stay inactive for the two negative cases; sibling skills need not be installed for these cases to pass.

## Notes

Smoke cases require no GPU, camera, product checkout, or network access. Paths in prompts describe a hypothetical
workspace and are not fixtures to inspect. Grade command correctness and task scope, not execution of those commands.
Evaluate both Claude Code and Codex, with and without this skill.
Use the SkillEvaluator default sandbox; no custom environment or provider credentials belong in this dataset.

Extended runs belong on team-controlled hardware: install a supported wheel, build the current product from source,
run a short dataset example, and exercise map save/localization. Verify actual import, tracking, and map results.
Run baseline and with-skill conditions from equivalent clean environments and record product/model/harness versions,
artifacts, success rate, tokens, and elapsed time in the evaluation report. Mark smoke and extended results separately.

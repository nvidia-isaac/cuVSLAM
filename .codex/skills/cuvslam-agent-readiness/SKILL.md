---
name: cuvslam-agent-readiness
description: Validate cuVSLAM agent-readiness scaffolding and route agents to static or GPU validation lanes.
---

# cuVSLAM Agent Readiness

## Description

Use this skill when an agent needs to validate cuVSLAM readiness, emit a
machine-readable validation result, or decide whether the current environment is
only suitable for static checks or full GPU runtime validation.

## Prerequisites

- Python 3.9 or newer for static validation.
- CUDA Toolkit 12 or 13 and NVIDIA GPU hardware for full runtime validation.
- A built or wheel-installed PyCuVSLAM package before running GPU validation.
- No secrets are required for either validation lane.

## Commands

Run the no-secret static lane:

```bash
python3 scripts/agent_readiness_smoke.py --mode static --output reports/validation_result.json
```

Run the root readiness test:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Run the GPU lane after installing or building PyCuVSLAM:

```bash
python3 scripts/agent_readiness_smoke.py --mode gpu --output reports/validation_result.json
```

## Output

The validation command writes `reports/validation_result.json`. The output uses
`.agent/validation_result.schema.json` and contains `status`, `mode`,
`timestamp_utc`, repository commit metadata, and a list of named checks.

## Troubleshooting

- For static failures, read each failed `checks[].name` in
  `reports/validation_result.json` and restore the missing readiness file.
- For cache permission failures, set `PYTHONPYCACHEPREFIX=.cache/pycache`.
- For GPU import failures, install a cuVSLAM wheel matching Python, CUDA, OS,
  and architecture, or build the repository first.

## Safety

The static lane is no-secret and does not download datasets or access cameras.
The GPU lane runs local imports and tests only; do not add live camera,
credential, destructive cleanup, or dataset deletion steps to the default lane.

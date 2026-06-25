# Agent Instructions

## Use This When

Use this repository to build, test, or validate cuVSLAM, the CUDA-accelerated
visual odometry, visual-inertial odometry, mapping, and localization library.

## Prerequisites

- Ubuntu 22.04 or 24.04 on x86_64 or Jetson.
- Python 3.9 or newer.
- CMake 3.19 or newer, gcc/g++, git, and git-lfs.
- CUDA Toolkit 12 or 13 for full build and runtime validation.
- NVIDIA GPU hardware for cuVSLAM tracking, mapping, localization, and Python
  API tests.
- No secrets are required for the no-secret static readiness lane.

## Repo Map

- `README.md`: product overview, installation, build, and first-run guidance.
- `examples/README.md`: dataset and live-camera examples.
- `python/pyproject.toml`: PyCuVSLAM package metadata.
- `python/test/`: Python API tracking, mapping, and binding tests.
- `TROUBLESHOOTING.md`: calibration, synchronization, image quality, and
  debugging guidance.
- `agent-readiness.yaml`: agent-readiness CI contract.
- `scripts/agent_readiness_smoke.py`: emits `reports/validation_result.json`.

## Commands

Run the no-secret static readiness lane:

```bash
python3 scripts/agent_readiness_smoke.py --mode static --output reports/validation_result.json
```

Run the root readiness unit test:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Compile Python sources without running GPU code:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache python3 -m compileall python scripts tests -q
```

Run full Python API validation after building or installing PyCuVSLAM on a
CUDA-capable host:

```bash
python3 scripts/agent_readiness_smoke.py --mode gpu --output reports/validation_result.json
```

## Output

The readiness lanes write `reports/validation_result.json` using the schema in
`.agent/validation_result.schema.json`. A passing static result proves that
agent-readiness scaffolding, docs, and Python syntax are healthy. A passing GPU
result proves that PyCuVSLAM imports and the Python API regression tests pass in
the current environment.

## Troubleshooting

- If static validation fails, inspect the failed check names in
  `reports/validation_result.json`.
- If Python bytecode writes fail on macOS or in a sandbox, set
  `PYTHONPYCACHEPREFIX=.cache/pycache`.
- If GPU validation fails at `cuvslam_import`, install a release wheel matching
  the Python, CUDA, OS, and architecture combination or build from source first.
- If tracking quality is poor, use `TROUBLESHOOTING.md` to check calibration,
  synchronization, frame continuity, image quality, and coordinate frames.

## Safety

- Do not commit datasets, camera recordings, generated maps, `.env`, build
  directories, or generated `reports/` outputs.
- Do not add secret-bearing environment variables to examples or CI.
- Keep destructive cleanup outside the default validation commands.

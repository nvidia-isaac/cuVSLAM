# cuVSLAM wheel packaging

Release wheels use four distributions while preserving the `cuvslam` import:

- `cuvslam` is a file-free metapackage selecting CUDA 13.
- `cuvslam-common` owns the public `cuvslam` package.
- `cuvslam-cu12` owns `cuvslam/cu12/`.
- `cuvslam-cu13` owns `cuvslam/cu13/`.

The CUDA backends deliberately have disjoint file manifests. Python wheel
metadata cannot declare package conflicts, so two differently named
distributions must never install the same file.

`packages.json` is the canonical release metadata and CUDA dependency/RPATH
map. `build_packages.py` renders temporary PEP 517 projects; generated
`pyproject.toml` files are not committed.

The normal source install remains monolithic:

```bash
CUVSLAM_BUILD_DIR=/absolute/path/to/build python -m pip install python/
```

Release builds are produced through:

```bash
CUDA_VERSION=13.2.0 ./scripts/build_pycuvslam_in_docker.sh ./output
./scripts/verify_pycuvslam_wheel_in_docker.sh ./output
```

The verifier checks metadata and installed-file ownership, inspects the repaired
native wheel, and runs the Python tests from the installed wheel rather than
from the source tree.

# PyCuVSLAM

PyCuVSLAM provides Python bindings for NVIDIA cuVSLAM.

## Install a release

CUDA 13 is the default:

```bash
python -m pip install cuvslam
```

Select a CUDA major explicitly when required:

```bash
python -m pip install cuvslam-cu12
python -m pip install cuvslam-cu13
```

The distribution name contains the CUDA major, while the import name remains
`cuvslam`.

Jetson wheels use the CUDA libraries provided by JetPack. On supported x86_64
systems, pip installs the CUDA runtime components required by cuVSLAM.

## Install from source

Build cuVSLAM first, then install the bindings against that build:

```bash
CUVSLAM_BUILD_DIR=/absolute/path/to/build python -m pip install python/
```

See the [cuVSLAM repository](https://github.com/nvidia-isaac/cuVSLAM) for the
supported platform matrix, examples, and API documentation.

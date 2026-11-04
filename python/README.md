# PyCuVSLAM

Python bindings for [cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM), NVIDIA's CUDA-accelerated library for visual
odometry and SLAM.

## Install

Choose the package matching your CUDA Toolkit major version: `cuvslam-cu12` or `cuvslam-cu13`. Both provide the
`cuvslam` module, so install only one of them per environment.

The packages use cuBLAS, cuSOLVER and cuSPARSE from a system [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit)
install. On x86_64 and Jetson Thor, the `cuda` extra installs them from PyPI instead:

```bash
pip install "cuvslam-cu12[cuda]"
```

Visual place recognition (`use_vpr`, AnyLoc) also needs TensorRT 10 (`libnvinfer10`), installed separately. The rest of
cuVSLAM runs without it.

Supported platforms, Jetson notes and source builds are described in the
[cuVSLAM README](https://github.com/nvidia-isaac/cuVSLAM#install-pycuvslam).

## Usage

```python
import cuvslam

print(cuvslam.get_version())
```

See the [Python API reference](https://nvidia-isaac.github.io/cuVSLAM/python/) and the
[examples](https://github.com/nvidia-isaac/cuVSLAM/tree/main/examples).

## License

[NVIDIA Community License](https://github.com/nvidia-isaac/cuVSLAM/blob/main/LICENSE)

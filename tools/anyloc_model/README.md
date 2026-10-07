# AnyLoc model

The AnyLoc place recognition backend (`Slam::VprMode::AnyLoc`, `libs/slam/vpr/vpr_anyloc.cpp`) describes a frame
with DINOv2's patch features, and runs the network as a TensorRT engine on the GPU. This folder makes that engine as
part of the cuVSLAM build, in two steps, from pinned inputs.

## Building it

```bash
cmake -S . -B build -DUSE_TENSORRT=ON
cmake --build build
```

`USE_TENSORRT` needs `USE_CUDA` and TensorRT 8.6 or newer, found next to CUDA or under `TENSORRT_ROOT`. It builds
`anyloc_engine_builder` and turns on `CUVSLAM_BUILD_ANYLOC_ENGINE`, which needs Python 3.10 to 3.12 and network access,
and with which the default build target makes:

| Target | Makes, in the build tree | Takes |
|--------|--------------------------|-------|
| `anyloc_venv` | `anyloc/venv`, a Python venv from `requirements.txt` | about a minute, once |
| `anyloc_onnx` | `anyloc/dinov2_vits14_l9_322.onnx`, and the reference arrays both steps check against | seconds |
| `anyloc_engine` | `bin/dinov2_vits14_l9_322_fp16_trt<TensorRT version>_sm<GPU>.engine` | about 20 seconds |

The engine is what `Slam::Config::vpr_model_path` takes. `cuvslam_vars.sh` exports its path as
`CUVSLAM_ANYLOC_ENGINE`, which `cuvslam_tracker` and `cuvslam_vpr_reporter` use as their default `--vpr_model_path`:

```bash
source build/cuvslam_vars.sh
cuvslam_tracker --edex <kitti>/00 --use_slam true --max_map_size 0 --loop_closure_mode vpr --vpr_mode anyloc ...
```

## What is pinned

- DINOv2's model code, at a fixed commit of facebookresearch/dinov2 (`cmake/ext/dinov2.cmake`, fetched with
  `FetchContent` and checked against its SHA256). Only its Python model code is used.
- The ViT-S/14 checkpoint `dinov2_vits14_pretrain.pth`, downloaded at configure time and checked against its SHA256.
- Every Python package the export runs on, dependencies included (`requirements.txt`). The venv is created from
  scratch whenever that file changes, so it never carries a package an older version of the file pulled in.
- `test_data/sof/left.png`, the reference image.

## Step 1: export only what AnyLoc reads

AnyLoc reads the *value* projection of one intermediate attention block of DINOv2, one row per 14x14 patch.
`cuvslam_export_dinov2` (`tools/python_tools/cuvslam_tools/vpr/export_dinov2_onnx.py`) exports
`AnyLocValueFacet` from `dinov2_value_facet.py`, a module that holds only the weights those rows depend on:

| Tensor | Shape (ViT-S/14, block 9, 322x322) | Taken from |
|--------|------------------------------------|------------|
| patch kernel | [588, 384] | the patch embedding, as a matrix multiply rather than a convolution |
| patch bias and position | [529, 384] | its bias plus the position table, interpolated to 23x23 once, in PyTorch |
| CLS token and position | [1, 384] | the CLS token plus its position |
| blocks 0..8 | 9 x 1,775,232 | complete |
| block 9 `norm1` | 2 x [384] | |
| block 9 value projection | [384, 384] and [384] | the value rows of the fused `qkv` projection only |

That is 16,555,008 parameters, 66 MB as an fp32 ONNX file. The blocks after 9, the final norm, the mask token, the
37x37 position table and block 9's query, key, output projection and MLP are never copied into the module, so they
cannot reach the graph. The export refuses to write a file unless:

1. the module agrees with AnyLoc's own read-out (the full DINOv2 forward, with a hook on block 9) to 1e-5;
2. the graph holds exactly the parameter tensors above and only the ops the module uses, no `Conv` and no `Resize`;
3. onnx's reference evaluator, run on the file, agrees with PyTorch to 1e-4.

It also saves the reference image as the network input (`*_reference_input.npy`) and AnyLoc's fp32 descriptors of it
(`*_reference_output.npy`).

## Step 2: the TensorRT engine

`anyloc_engine_builder` (`build_engine.cpp`) builds the engine for the GPU of the build machine:

- **FP16, with an FP32 head.** Everything after the last matrix multiply, the bias and L2 normalization of the value
  projection, stays in FP32. The layer norms are left alone on purpose: TensorRT computes them in FP32 inside
  already, and pinning them makes it run every transformer block it fuses around them in FP32, which doubled the
  size and the latency of the engine without changing its output.
- **No cuBLAS, cuBLASLt or cuDNN tactics**, so running the engine needs `libnvinfer` and the CUDA driver, nothing else.
  The ONNX parser, which does need cuBLAS and cuDNN, is only linked into the builder.
- **Checked before it is written.** The builder runs the engine on the reference input and requires the cosine
  similarity of every patch descriptor with the fp32 reference to be at least 0.999 on average and 0.99 for the worst
  patch. The engine is written under a temporary name and renamed once that passed.

On an RTX 4090 with TensorRT 8.6.1 that gives a 34.5 MB engine running in 0.48 ms per frame, with a cosine of 0.99993
on average and 0.99865 for the worst patch.

## An engine belongs to one GPU and one TensorRT

A serialized engine only loads into the TensorRT version, and only runs on the GPU architecture, that built it, and
`VprAnyLoc` refuses any other with an error that says to rebuild it. The engine's name records both, so building with
another TensorRT, or for another GPU, makes a new file instead of reusing one that no longer loads, and the builder
refuses to build when CUDA device 0 is not the GPU the name promises. To run AnyLoc on another machine, copy the ONNX
model and the reference arrays there and build the engine on it with the `anyloc_engine_builder` of a `USE_TENSORRT`
build for that machine:

```bash
anyloc_engine_builder --onnx=dinov2_vits14_l9_322.onnx --engine=dinov2_vits14_l9_322.engine \
    --reference_input=dinov2_vits14_l9_322_reference_input.npy \
    --reference_output=dinov2_vits14_l9_322_reference_output.npy
```

## Options

| CMake variable | Default | Purpose |
|----------------|---------|---------|
| `CUVSLAM_BUILD_ANYLOC_ENGINE` | ON with `USE_TENSORRT` | OFF builds the backend without its model |
| `CUVSLAM_ANYLOC_LAYER` | 9 | DINOv2 block whose value facet AnyLoc reads |
| `CUVSLAM_ANYLOC_INPUT_SIZE` | 322 | square input size, a multiple of 14 |
| `CUVSLAM_ANYLOC_GPU_SM` | detected | compute capability to build the engine for, e.g. 89; set it when configuring without a GPU, the build itself still needs one |
| `CUVSLAM_ANYLOC_PYTHON` | empty | a Python that already has `requirements.txt` installed, instead of creating the venv; without it the venv is made from a Python 3.10 to 3.12 found on the system (`Python3_EXECUTABLE`) |
| `CUVSLAM_DINOV2_CHECKPOINT` | empty | a local copy of `dinov2_vits14_pretrain.pth`, instead of downloading it |
| `TENSORRT_ROOT` | empty | TensorRT install prefix, when it is not next to CUDA or in the system paths |

Without network access, point `FETCHCONTENT_SOURCE_DIR_DINOV2` at a copy of the DINOv2 source,
`CUVSLAM_DINOV2_CHECKPOINT` at the checkpoint and `CUVSLAM_ANYLOC_PYTHON` at a prepared interpreter. Without a GPU at
configure time only the ONNX model is built.

## Tests

- `anyloc_export_python_test` (CTest) runs `cuvslam_tools/tests/test_export_dinov2.py` in the venv, on a small random
  DINOv2: the module against the hook read-out, its tensors against the architecture, and the audit against graphs
  that carry a stray tensor, a convolution, or block 9's query and key rows.
- `slam_test` runs the AnyLoc backend on stand-in engines it builds itself, and on this engine against the reference
  arrays.

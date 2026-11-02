# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""The part of DINOv2 that AnyLoc reads, as a module that holds nothing else."""

import copy
import math
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: ImageNet mean and std for [0, 1] pixels; must match libs/slam/vpr/vpr_anyloc.cpp.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

#: Every op an export of `AnyLocValueFacet` may contain; no `Conv` or `Resize`.
ALLOWED_OPS = frozenset({
    "Add",
    "Clip",
    "Concat",
    "Constant",
    "Div",
    "Erf",
    "Gather",
    "LayerNormalization",
    "MatMul",
    "Mul",
    "ReduceL2",
    "Reshape",
    "Slice",
    "Softmax",
    "Transpose",
})

_FLOAT_TYPES = frozenset({1, 10, 11, 16})  # onnx.TensorProto FLOAT, FLOAT16, DOUBLE, BFLOAT16


def load_dinov2(dinov2_src, checkpoint) -> nn.Module:
    """DINOv2 ViT-S/14 from the CMake-pinned source and checkpoint, not torch.hub."""
    # xformers attention does not run on CPU; set this before the first dinov2 import.
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    source = str(Path(dinov2_src).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "xFormers is disabled", twice per layer module
        from dinov2.hub.backbones import dinov2_vits14

        dino = dinov2_vits14(pretrained=False)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    dino.load_state_dict(state, strict=True)
    return dino.eval()


class _StaticBlock(nn.Module):
    """One DINOv2 block in eval mode for a fixed token count, free of run-time branches."""

    def __init__(self, block: nn.Module, tokens: int):
        super().__init__()
        attention = block.attn
        if not all(hasattr(block.mlp, name) for name in ("fc1", "act", "fc2")):
            raise ValueError("only DINOv2 blocks with a plain MLP are supported, not SwiGLU ones")
        self.tokens = tokens
        self.heads = attention.num_heads
        self.head_dim = attention.qkv.in_features // attention.num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = copy.deepcopy(block.norm1)
        self.qkv = copy.deepcopy(attention.qkv)
        self.proj = copy.deepcopy(attention.proj)
        self.ls1 = copy.deepcopy(block.ls1)
        self.norm2 = copy.deepcopy(block.norm2)
        self.fc1 = copy.deepcopy(block.mlp.fc1)
        self.act = copy.deepcopy(block.mlp.act)
        self.fc2 = copy.deepcopy(block.mlp.fc2)
        self.ls2 = copy.deepcopy(block.ls2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(self.norm1(x)).reshape(1, self.tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]  # [1, heads, tokens, head_dim] each
        weights = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(1, self.tokens, self.heads * self.head_dim)
        x = x + self.ls1(self.proj(attended))
        return x + self.ls2(self.fc2(self.act(self.fc1(self.norm2(x)))))


class AnyLocValueFacet(nn.Module):
    """AnyLoc's per-patch descriptors of a square image, from only the weights they need."""

    def __init__(self, dino: nn.Module, layer: int, size: int):
        super().__init__()
        patch = dino.patch_size
        if getattr(dino, "chunked_blocks", False):
            raise ValueError("DINOv2 built with block_chunks > 0 is not supported")
        if getattr(dino, "num_register_tokens", 0) != 0:
            raise ValueError("DINOv2 with register tokens is not supported")
        if not isinstance(dino.patch_embed.norm, nn.Identity) or dino.patch_embed.proj.bias is None:
            # Positions are folded into the patch bias, which needs no norm in between.
            raise ValueError("DINOv2 with a normalized or bias free patch embedding is not supported")
        if size <= 0 or size % patch != 0:
            raise ValueError(f"size {size} is not a positive multiple of the patch size {patch}")
        if not 0 <= layer < len(dino.blocks):
            raise ValueError(f"layer {layer} is outside 0..{len(dino.blocks) - 1}")

        dim = dino.embed_dim
        self.patch = patch
        self.grid = size // patch
        patches = self.grid * self.grid

        weight = dino.patch_embed.proj.weight.detach()  # [dim, 3, patch, patch]
        with torch.no_grad():
            # Interpolated once at the export size; the graph only sees the result.
            position = dino.interpolate_pos_encoding(torch.zeros(1, 1 + patches, dim), size, size)[0]
        self.register_buffer("patch_kernel", weight.reshape(dim, -1).t().contiguous())  # [3 * patch^2, dim]
        self.register_buffer("patch_bias", (dino.patch_embed.proj.bias.detach() + position[1:]).contiguous())
        self.register_buffer("cls_token", (dino.cls_token.detach().reshape(dim) + position[0]).reshape(1, 1, dim))

        self.blocks = nn.ModuleList(_StaticBlock(dino.blocks[index], 1 + patches) for index in range(layer))

        source = dino.blocks[layer]
        qkv = source.attn.qkv
        self.value_norm = copy.deepcopy(source.norm1)
        self.value = nn.Linear(dim, dim, bias=qkv.bias is not None)
        with torch.no_grad():
            # qkv rows are stacked as query, key, value.
            self.value.weight.copy_(qkv.weight[2 * dim:])
            if qkv.bias is not None:
                self.value.bias.copy_(qkv.bias[2 * dim:])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch, grid = self.patch, self.grid
        # [1, 3, H, W] -> [1, N, 3 * P * P], row-major, flattened like the conv kernel.
        patches = image.reshape(1, 3, grid, patch, grid, patch).permute(0, 2, 4, 1, 3, 5).reshape(1, grid * grid, -1)
        tokens = torch.cat((self.cls_token, patches @ self.patch_kernel + self.patch_bias), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        # Drop CLS with narrow(): [:, 1:] traces to an INT64_MAX that TensorRT clamps.
        values = self.value(self.value_norm(tokens.narrow(1, 1, grid * grid)))
        # F.normalize without the Expand and Shape ops it traces to.
        return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def hook_reference(dino: nn.Module, layer: int, image: torch.Tensor) -> torch.Tensor:
    """AnyLoc's own read-out: a full DINOv2 forward with a hook on block `layer`'s `qkv`."""
    captured = []
    handle = dino.blocks[layer].attn.qkv.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        with torch.no_grad():
            dino(image)
    finally:
        handle.remove()
    qkv = captured[0]
    dim = qkv.shape[-1] // 3
    return F.normalize(qkv[:, 1:, 2 * dim:], dim=-1)


def expected_parameter_shapes(embed_dim: int, mlp_dim: int, patch: int, grid: int, layer: int) -> list:
    """Every tensor shape the read-out of block `layer` needs, derived without the wrapper."""
    dim = embed_dim
    shapes = [(3 * patch * patch, dim), (grid * grid, dim), (1, 1, dim)]
    block = [
        (dim,), (dim,),  # norm1
        (3 * dim, dim), (3 * dim,),  # attn.qkv
        (dim, dim), (dim,),  # attn.proj
        (dim,),  # ls1
        (dim,), (dim,),  # norm2
        (mlp_dim, dim), (mlp_dim,),  # mlp.fc1
        (dim, mlp_dim), (dim,),  # mlp.fc2
        (dim,),  # ls2
    ]
    shapes += block * layer
    shapes += [(dim,), (dim,), (dim, dim), (dim,)]  # block `layer`: norm1 and the value rows of qkv
    return shapes


def canonical_shape(shape) -> tuple:
    """`shape` without unit dims, sorted, so transposes and extra unit axes compare equal."""
    return tuple(sorted(dim for dim in shape if dim != 1))


@dataclass
class OnnxAudit:
    """What `audit_onnx` found in a graph that passed."""

    op_counts: Counter
    parameter_tensors: int
    parameter_count: int


def audit_onnx(path, expected_shapes, input_shape, output_shape) -> OnnxAudit:
    """Raise ValueError unless `path` is an ONNX export of the wrapper holding nothing else."""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph
    problems = []

    def shape_of(value_info):
        return tuple(dim.dim_value if dim.HasField("dim_value") else None for dim in value_info.type.tensor_type.shape.dim)

    initializer_names = {tensor.name for tensor in graph.initializer}
    inputs = [(value.name, shape_of(value)) for value in graph.input if value.name not in initializer_names]
    outputs = [(value.name, shape_of(value)) for value in graph.output]
    if inputs != [("image", tuple(input_shape))]:
        problems.append(f"inputs are {inputs}, expected image {tuple(input_shape)}")
    if outputs != [("patch_descriptors", tuple(output_shape))]:
        problems.append(f"outputs are {outputs}, expected patch_descriptors {tuple(output_shape)}")

    op_counts = Counter(node.op_type for node in graph.node)
    foreign = sorted(set(op_counts) - ALLOWED_OPS)
    if foreign:
        problems.append(f"ops the wrapper does not use: {', '.join(foreign)}")

    tensors = list(graph.initializer)
    tensors += [attribute.t for node in graph.node if node.op_type == "Constant" for attribute in node.attribute
                if attribute.name == "value"]
    parameters = [tuple(tensor.dims) for tensor in tensors
                  if tensor.data_type in _FLOAT_TYPES and math.prod(tensor.dims) > 1]
    found = Counter(canonical_shape(shape) for shape in parameters)
    expected = Counter(canonical_shape(shape) for shape in expected_shapes)
    extra = found - expected
    missing = expected - found
    if extra:
        problems.append(f"parameter tensors that are not part of the read-out: {dict(extra)}")
    if missing:
        problems.append(f"parameter tensors of the read-out that are missing: {dict(missing)}")

    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return OnnxAudit(op_counts, len(parameters), sum(math.prod(shape) for shape in parameters))


def make_reference_input(image_path, size: int) -> np.ndarray:
    """Network input from `image_path`, preprocessed roughly as the AnyLoc backend does."""
    from PIL import Image

    gray = Image.open(image_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    planes = [(pixels - mean) / std for mean, std in zip(PIXEL_MEAN, PIXEL_STD)]
    return np.stack(planes)[np.newaxis].astype(np.float32)

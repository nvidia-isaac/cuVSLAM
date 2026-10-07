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

"""The part of DINOv2 that AnyLoc reads, as a module that holds nothing else.

AnyLoc (Keetha et al., 2023) describes an image with the *value* projection of one intermediate
attention block of DINOv2: one row per 14x14 patch, the CLS row dropped, every row L2 normalized.
Its extractor runs the whole network and catches the `qkv` output of block L with a forward hook.
`AnyLocValueFacet` computes the same rows from exactly the weights they depend on:

- the patch embedding, as a matrix multiply over flattened patches rather than a stride 14
  convolution;
- one table holding the patch embedding bias plus the position encoding, interpolated once to the
  export resolution, and the CLS token with its own position folded in the same way;
- blocks 0..L-1, complete;
- `norm1` of block L and the value third of its `qkv` projection, without the query and key rows.

Blocks after L, the final norm, the mask token, the 37x37 position table and block L's query, key,
output projection and MLP are never copied, so no export of the wrapper can carry them, and
`audit_onnx` checks that an export carries exactly the tensors listed above.
"""

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

#: ImageNet statistics the DINOv2 checkpoints were trained with, applied after dividing by 255. The
#: AnyLoc backend in libs/slam/vpr/vpr_anyloc.cpp normalizes its input with the same numbers.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

#: Every op an export of `AnyLocValueFacet` is made of. Anything else in a graph is math the wrapper
#: does not do, so it came from somewhere else; `Conv` and `Resize` in particular would mean the
#: patch embedding or the position interpolation was traced instead of the precomputed tables.
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
    """DINOv2 ViT-S/14 built from a source checkout and loaded from a checkpoint file.

    torch.hub would run whatever the repository's main branch holds at the time and download the
    weights on its own. The build pins both instead: CMake fetches the source at a fixed commit and
    the checkpoint with a fixed SHA256, and this reads them from where CMake put them.
    """
    # Unless told otherwise, dinov2.layers imports xformers when it is installed, and its attention
    # does not run on CPU. Must be set before the first import of dinov2.
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
    """One DINOv2 transformer block as inference runs it, written out for a fixed token count.

    DINOv2's own block chooses between xformers, nested tensors and stochastic depth at run time and
    reads its shapes off the input. None of that is part of AnyLoc's math, and all of it would be
    traced. This holds copies of the same weights and applies them the way DINOv2 does in eval mode:
    x + ls1(proj(attention(norm1(x)))), then x + ls2(fc2(gelu(fc1(norm2(x))))).
    """

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
    """AnyLoc's per-patch descriptors of a `size` x `size` image, holding only the weights they need.

    Input `image`, fp32 [1, 3, size, size], ImageNet normalized; output fp32 [1, (size / 14)^2, D],
    one L2 normalized row per patch in row-major patch order. Equal to `hook_reference` up to fp32
    rounding.
    """

    def __init__(self, dino: nn.Module, layer: int, size: int):
        super().__init__()
        patch = dino.patch_size
        if getattr(dino, "chunked_blocks", False):
            raise ValueError("DINOv2 built with block_chunks > 0 is not supported")
        if getattr(dino, "num_register_tokens", 0) != 0:
            raise ValueError("DINOv2 with register tokens is not supported")
        if not isinstance(dino.patch_embed.norm, nn.Identity) or dino.patch_embed.proj.bias is None:
            # Folding the position table into the patch bias is only exact when nothing sits between
            # the patch projection and the position addition.
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
            # Evaluated once, at the export resolution, through DINOv2's own interpolation; the graph
            # only ever sees the result. Height then width, the order prepare_tokens_with_masks uses.
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
            # The fused projection stacks query, key and value rows in that order.
            self.value.weight.copy_(qkv.weight[2 * dim:])
            if qkv.bias is not None:
                self.value.bias.copy_(qkv.bias[2 * dim:])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch, grid = self.patch, self.grid
        # [1, 3, H, W] -> [1, 3, grid, P, grid, P] -> [1, grid, grid, 3, P, P] -> [1, N, 3 * P * P]: patches in
        # the row-major order the convolution emits them, each flattened the way its kernel is.
        patches = image.reshape(1, 3, grid, patch, grid, patch).permute(0, 2, 4, 1, 3, 5).reshape(1, grid * grid, -1)
        tokens = torch.cat((self.cls_token, patches @ self.patch_kernel + self.patch_bias), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        # Normalization and projection act on every token alone, so dropping CLS first only saves work. narrow()
        # rather than [:, 1:], whose open end traces to an INT64_MAX that TensorRT warns about clamping.
        values = self.value(self.value_norm(tokens.narrow(1, 1, grid * grid)))
        # What F.normalize computes, minus the Expand and Shape its broadcast traces to.
        return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def hook_reference(dino: nn.Module, layer: int, image: torch.Tensor) -> torch.Tensor:
    """AnyLoc's own read-out: the full DINOv2 forward, with block `layer`'s `qkv` output caught by a hook.

    This is how AnyLoc's extractor gets its descriptors, so it is what the wrapper and every model
    built from it are checked against.
    """
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
    """Shape of every tensor the read-out of block `layer` needs, worked out from the architecture alone.

    Independent of the wrapper on purpose, so that comparing the two checks the wrapper rather than
    restating it. Assumes what every DINOv2 ViT-S/B/L checkpoint has: biases on all projections and
    LayerScale on both residual branches.
    """
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
    """`shape` without its unit dimensions, the rest sorted.

    An exporter is free to store a linear layer's weight transposed or to give a tensor a leading
    unit axis; neither changes which tensor it is.
    """
    return tuple(sorted(dim for dim in shape if dim != 1))


@dataclass
class OnnxAudit:
    """What `audit_onnx` found in a graph that passed."""

    op_counts: Counter
    parameter_tensors: int
    parameter_count: int


def audit_onnx(path, expected_shapes, input_shape, output_shape) -> OnnxAudit:
    """Check that the ONNX file at `path` is an export of the wrapper and holds nothing else.

    Checks the graph's single input `image` and single output `patch_descriptors` against the given
    static shapes, every op against `ALLOWED_OPS`, and the multiset of parameter tensors - every
    floating point initializer or constant with more than one element - against `expected_shapes`,
    compared as `canonical_shape`s. Raises ValueError naming every difference.
    """
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
    """A network input made from the image at `image_path` the way the AnyLoc backend makes one.

    Gray, resized to `size` x `size`, replicated into three channels and ImageNet normalized, the
    steps `VprAnyLoc::Describe` takes, though not with its resampling: fp32 [1, 3, size, size].
    Saved next to the model, it is what the TensorRT engine is validated on, so its consumers feed
    it to the network as it is rather than redo the preprocessing.
    """
    from PIL import Image

    gray = Image.open(image_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    planes = [(pixels - mean) / std for mean, std in zip(PIXEL_MEAN, PIXEL_STD)]
    return np.stack(planes)[np.newaxis].astype(np.float32)

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

"""Export the DINOv2 patch descriptor model that the AnyLoc VPR backend runs.

What the exported graph computes
--------------------------------
One L2 normalized descriptor per 14x14 image patch, taken from the *value* projection of an
intermediate transformer block. Concretely: patch embedding, then blocks 0..L-1, then `norm1` and
the fused `qkv` projection of block L; of that projection only the last third is kept, which is the
value half of the attention input, the CLS token is dropped, and each row is L2 normalized. Input is
`image`, fp32 [1, 3, S, S]; output is `patch_descriptors`, fp32 [1, (S/14)^2, D].

This is exactly what AnyLoc (Keetha et al., 2023) reads out with a forward hook on
`blocks[L].attn.qkv`, so the graph is a drop-in replacement for their extractor. The blocks after L
are never evaluated and constant folding prunes their weights out of the file.

Why the patch embedding convolution is rewritten
------------------------------------------------
The patch embedding is a stride-14 kernel-14 convolution, which is a per-patch matrix multiply
written as a convolution. Exported as `Conv` it forces every consumer of the file through a
convolution kernel, and ONNX Runtime's CUDA `Conv` goes through cudnn-frontend, which fails to build
an execution plan for this shape against cuDNN 8.x. Reshaping the input into patches and multiplying
by the flattened kernel instead yields a graph with zero `Conv` nodes that agrees with the
unmodified DINOv2 forward path to fp32 rounding (checked here on every run, and the export is
refused if it does not) and runs on CPU and CUDA without a cuDNN dependency at all.

Why one file per input resolution
---------------------------------
DINOv2's `interpolate_pos_encoding` computes the position grid from the traced input size and bakes
the result in as constants. A model exported with dynamic height and width builds, and advertises a
dynamic input, but fails at run time for every size other than the traced one. So the resolution is
baked in deliberately, and the backend reads it back out of the session instead of assuming it.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "dinov2_vits14"
DEFAULT_LAYER = 9
DEFAULT_SIZE = 322
OPSET = 17


class ValueFacet(nn.Module):
    """DINOv2 truncated to the value facet of block `layer`, with a convolution free patch embed."""

    def __init__(self, dino: nn.Module, layer: int):
        super().__init__()
        self.dino = dino
        self.layer = layer
        self.patch = dino.patch_size
        weight = dino.patch_embed.proj.weight.detach()  # [D, 3, P, P]
        self.register_buffer("kernel", weight.reshape(weight.shape[0], -1).t().contiguous())  # [3PP, D]
        self.register_buffer("bias", dino.patch_embed.proj.bias.detach().clone())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch = self.patch
        batch, _, height, width = image.shape
        rows, cols = height // patch, width // patch
        # [B,3,H,W] -> [B,3,rows,P,cols,P] -> [B,rows,cols,3,P,P] -> [B,N,3PP]
        patches = image.reshape(batch, 3, rows, patch, cols, patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, rows * cols, -1)
        tokens = patches @ self.kernel + self.bias
        tokens = self.dino.patch_embed.norm(tokens)
        tokens = torch.cat((self.dino.cls_token.expand(batch, -1, -1), tokens), dim=1)
        tokens = tokens + self.dino.interpolate_pos_encoding(tokens, width, height)
        for index in range(self.layer):
            tokens = self.dino.blocks[index](tokens)
        block = self.dino.blocks[self.layer]
        qkv = block.attn.qkv(block.norm1(tokens))
        dim = qkv.shape[-1] // 3
        return F.normalize(qkv[:, 1:, 2 * dim :], dim=-1)


class ValueFacetReference(nn.Module):
    """The same read-out through DINOv2's own token preparation, including the `Conv` patch embed.

    Only used to prove that rewriting the patch embedding changed nothing.
    """

    def __init__(self, dino: nn.Module, layer: int):
        super().__init__()
        self.dino = dino
        self.layer = layer

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.dino.prepare_tokens_with_masks(image, None)
        for index in range(self.layer):
            tokens = self.dino.blocks[index](tokens)
        block = self.dino.blocks[self.layer]
        qkv = block.attn.qkv(block.norm1(tokens))
        dim = qkv.shape[-1] // 3
        return F.normalize(qkv[:, 1:, 2 * dim :], dim=-1)


def resolve_torch_home(output: Path) -> str:
    """Point the torch.hub cache next to the output file unless the caller already placed it.

    torch.hub otherwise downloads the checkpoint into `~/.cache/torch`, which is the wrong place for
    a few hundred megabytes of model that belongs with the artifact being built.
    """
    existing = os.environ.get("TORCH_HOME")
    if existing:
        return existing
    cache = output.parent / ".torch_hub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache)
    return str(cache)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="torch.hub DINOv2 entry point, e.g. dinov2_vits14 or dinov2_vitb14 (default: %(default)s)")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help="block index the value facet is read from, 0 based (default: %(default)s)")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="square input size in pixels, must be a multiple of the patch size (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=None,
                        help="destination .onnx file (default: <model>_l<layer>_value_<size>_noconv.onnx)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or Path(f"{args.model}_l{args.layer}_value_{args.size}_noconv.onnx")
    print(f"torch.hub cache: {resolve_torch_home(output)}")

    dino = torch.hub.load("facebookresearch/dinov2", args.model, verbose=False).eval()
    if args.size % dino.patch_size != 0:
        print(f"--size {args.size} is not a multiple of the patch size {dino.patch_size}", file=sys.stderr)
        return 2
    if not 0 <= args.layer < len(dino.blocks):
        print(f"--layer {args.layer} is outside 0..{len(dino.blocks) - 1}", file=sys.stderr)
        return 2

    model = ValueFacet(dino, args.layer).eval()
    reference = ValueFacetReference(dino, args.layer).eval()
    image = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        expected = reference(image)
        actual = model(image)
    drift = (expected - actual).abs().max().item()
    print(f"patch embed rewrite vs DINOv2 forward: max abs diff {drift:.3e}, output {tuple(actual.shape)}")
    if drift > 1e-5:
        print("the rewritten patch embedding no longer matches DINOv2, refusing to export", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (image,), str(output), input_names=["image"],
                          output_names=["patch_descriptors"], opset_version=OPSET, dynamo=False,
                          do_constant_folding=True)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")

    try:
        import onnx
    except ImportError:
        print("install onnx to have the exported graph checked for Conv nodes")
        return 0
    graph = onnx.load(str(output)).graph
    histogram = Counter(node.op_type for node in graph.node)
    print(f"nodes: {len(graph.node)}, Conv: {histogram['Conv']}, top ops: {histogram.most_common(8)}")
    if histogram["Conv"] != 0:
        print("the exported graph still contains Conv nodes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

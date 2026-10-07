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

This is the first of the two build steps behind the AnyLoc engine: the CMake target `anyloc_onnx`
runs it on the DINOv2 source and checkpoint CMake pins, and `anyloc_engine` then turns its output
into the TensorRT engine `Slam::Config::vpr_model_path` names (see tools/anyloc_model). Run by hand,
it exports another layer or input size from the same sources.

What the file holds
-------------------
`AnyLocValueFacet` from dinov2_value_facet.py: the patch embedding, blocks 0..L-1, and `norm1` plus
the value rows of the `qkv` projection of block L - only the weights AnyLoc's descriptors depend on.
Input `image`, fp32 [1, 3, S, S], ImageNet normalized; output `patch_descriptors`, fp32
[1, (S/14)^2, D], one L2 normalized row per image patch.

Checks, before anything is written
----------------------------------
1. The wrapper against AnyLoc's own read-out, the full DINOv2 forward with a hook on block L, on
   the reference image: max abs difference at most 1e-4.
2. The exported graph against the architecture: nothing but the wrapper's ops, and exactly the
   parameter tensors the read-out needs (`audit_onnx`).
3. The exported graph, run by the ONNX reference evaluator, against PyTorch: max abs difference at
   most 1e-4.

Every output goes to a temporary name first and is only renamed into place once all checks passed,
so a failed run never leaves a file behind that a build system would take for up to date.

Why one file per input resolution
---------------------------------
The position encoding is interpolated to the export resolution once, in PyTorch, and stored as a
table of that size, so the graph contains no `Resize` and serves exactly one input size, which is
also all a TensorRT engine built from it can take.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from cuvslam_tools.vpr.dinov2_value_facet import (
    AnyLocValueFacet,
    audit_onnx,
    expected_parameter_shapes,
    hook_reference,
    load_dinov2,
    make_reference_input,
)

DEFAULT_LAYER = 9
DEFAULT_SIZE = 322
OPSET = 17

#: Largest elementwise difference allowed between the wrapper and AnyLoc's hook-based read-out. The two
#: sum in different orders, which leaves about 1e-5 of fp32 noise on unit length rows of 384 values
#: (6.8e-6 measured on ViT-S/14 block 9); a wrong weight or op shows up at 1e-2 and more.
WRAPPER_TOLERANCE = 1e-4
#: Largest elementwise difference allowed between the ONNX reference evaluator and PyTorch.
EVALUATOR_TOLERANCE = 1e-4


def export_onnx(model: torch.nn.Module, image: torch.Tensor, path: Path) -> None:
    """Trace `model` on `image` into the ONNX file `path`, with the input and output names the backend expects."""
    with torch.no_grad():
        torch.onnx.export(model, (image,), str(path), input_names=["image"], output_names=["patch_descriptors"],
                          opset_version=OPSET, dynamo=False, do_constant_folding=True)


def evaluate_onnx(path: Path, image: np.ndarray) -> np.ndarray:
    """Run the ONNX file `path` on `image` with onnx's reference evaluator, which is independent of PyTorch."""
    from onnx.reference import ReferenceEvaluator

    return ReferenceEvaluator(str(path)).run(None, {"image": image})[0]


def _temporary(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dinov2_src", type=Path, required=True,
                        help="DINOv2 source tree, e.g. the one CMake fetches into _deps/dinov2-src")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="DINOv2 ViT-S/14 weights, dinov2_vits14_pretrain.pth")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help="block the value facet is read from, 0 based (default: %(default)s)")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="square input size in pixels, a multiple of the patch size (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=None,
                        help="destination .onnx file (default: dinov2_vits14_l<layer>_<size>.onnx)")
    parser.add_argument("--reference_image", type=Path, default=None,
                        help="image the checks run on, preprocessed as the backend does (default: random input)")
    parser.add_argument("--reference_input", type=Path, default=None,
                        help="also save the network input of the checks here, as .npy")
    parser.add_argument("--reference_output", type=Path, default=None,
                        help="also save AnyLoc's fp32 descriptors of that input here, as .npy")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or Path(f"dinov2_vits14_l{args.layer}_{args.size}.onnx")

    dino = load_dinov2(args.dinov2_src, args.checkpoint)
    if args.size <= 0 or args.size % dino.patch_size != 0:
        print(f"--size {args.size} is not a positive multiple of the patch size {dino.patch_size}", file=sys.stderr)
        return 2
    if not 0 <= args.layer < len(dino.blocks):
        print(f"--layer {args.layer} is outside 0..{len(dino.blocks) - 1}", file=sys.stderr)
        return 2

    model = AnyLocValueFacet(dino, args.layer, args.size).eval()
    if args.reference_image:
        image = torch.from_numpy(make_reference_input(args.reference_image, args.size))
    else:
        image = torch.randn(1, 3, args.size, args.size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = hook_reference(dino, args.layer, image)
        actual = model(image)
    drift = (expected - actual).abs().max().item()
    print(f"wrapper vs DINOv2 hook on block {args.layer}: max abs diff {drift:.3e}, output {tuple(actual.shape)}")
    if not drift <= WRAPPER_TOLERANCE:
        print("the wrapper no longer computes AnyLoc's descriptors, refusing to export", file=sys.stderr)
        return 1

    staged = []  # (temporary, final) pairs, renamed only once everything passed
    done = False
    try:
        onnx_path = _temporary(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        staged.append((onnx_path, output))
        export_onnx(model, image, onnx_path)

        grid = args.size // dino.patch_size
        shapes = expected_parameter_shapes(dino.embed_dim, dino.blocks[0].mlp.fc1.out_features, dino.patch_size,
                                           grid, args.layer)
        audit = audit_onnx(onnx_path, shapes, tuple(image.shape), tuple(actual.shape))
        print(f"audit: {audit.parameter_count:,} parameters in {audit.parameter_tensors} tensors, "
              f"{audit.op_counts['Conv']} Conv, {audit.op_counts['Resize']} Resize, "
              f"ops {dict(sorted(audit.op_counts.items()))}")

        evaluated = evaluate_onnx(onnx_path, image.numpy())
        evaluator_drift = float(np.abs(evaluated - actual.numpy()).max())
        print(f"ONNX reference evaluator vs PyTorch: max abs diff {evaluator_drift:.3e}")
        if not evaluator_drift <= EVALUATOR_TOLERANCE:
            raise ValueError("the exported graph does not compute what the wrapper does")

        for path, array in ((args.reference_input, image.numpy()), (args.reference_output, expected.numpy())):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                staged.append((_temporary(path), path))
                with open(_temporary(path), "wb") as stream:
                    np.save(stream, np.ascontiguousarray(array, dtype=np.float32))
        for temporary, final in staged:
            os.replace(temporary, final)
        done = True
    except ValueError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 1
    finally:
        if not done:
            # Whatever stopped the export, onnx's checker or an interrupt included, nothing half
            # written may stay behind.
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

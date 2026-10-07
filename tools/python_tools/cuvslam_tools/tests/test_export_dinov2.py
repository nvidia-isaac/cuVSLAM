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

"""Tests of the AnyLoc DINOv2 export: cuvslam_tools.vpr.dinov2_value_facet and export_dinov2_onnx.

They need torch, onnx and the DINOv2 source, which only the environment of the AnyLoc build step has. CMake runs
them there as the `anyloc_export_python_test` CTest, with CUVSLAM_ANYLOC_EXPORT_TESTS=required turning a missing
dependency into a failure; anywhere else, such as the tools test suite, they skip. A small randomly initialized
DINOv2 stands in for the checkpoint, so they take seconds and no network.
"""

import copy
import importlib.util
import math
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

_MISSING = [name for name in ("torch", "onnx", "dinov2", "PIL") if importlib.util.find_spec(name) is None]
if _MISSING and os.environ.get("CUVSLAM_ANYLOC_EXPORT_TESTS") == "required":
    raise ImportError(f"CUVSLAM_ANYLOC_EXPORT_TESTS=required, but {', '.join(_MISSING)} cannot be imported")

if not _MISSING:
    import numpy as np
    import onnx
    import torch

    from cuvslam_tools.vpr import dinov2_value_facet as facet
    from cuvslam_tools.vpr import export_dinov2_onnx as exporter

_REFERENCE_IMAGE = Path(__file__).resolve().parents[4] / "test_data" / "sof" / "left.png"

PATCH = 14
DIM = 48
HEADS = 4
DEPTH = 4
LAYER = 2
# 4 x 4 patches, while the model's position table is 5 x 5: the wrapper has to interpolate it.
SIZE = 56
PATCHES = (SIZE // PATCH) ** 2


def make_small_dino():
    """A DINOv2 built like the ViT-S/14 checkpoint, only smaller, with every weight random."""
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from functools import partial

        from dinov2.layers import MemEffAttention, NestedTensorBlock
        from dinov2.models.vision_transformer import DinoVisionTransformer

    torch.manual_seed(0)
    dino = DinoVisionTransformer(img_size=70, patch_size=PATCH, embed_dim=DIM, depth=DEPTH, num_heads=HEADS,
                                 mlp_ratio=4, block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
                                 init_values=1.0, block_chunks=0)
    # DINOv2 initializes LayerScale to one and biases to zero; random values keep any term from hiding behind those.
    with torch.no_grad():
        for parameter in dino.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    return dino.eval()


def random_image():
    return torch.randn(1, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(1))


def expected_shapes():
    return facet.expected_parameter_shapes(DIM, 4 * DIM, PATCH, SIZE // PATCH, LAYER)


@unittest.skipIf(_MISSING, f"needs {', '.join(_MISSING)}")
class TestAnyLocValueFacet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dino = make_small_dino()
        cls.model = facet.AnyLocValueFacet(cls.dino, LAYER, SIZE).eval()
        cls.image = random_image()

    def test_computes_anylocs_hook_read_out(self):
        with torch.no_grad():
            expected = facet.hook_reference(self.dino, LAYER, self.image)
            actual = self.model(self.image)
        self.assertEqual(tuple(actual.shape), (1, PATCHES, DIM))
        self.assertLessEqual((expected - actual).abs().max().item(), exporter.WRAPPER_TOLERANCE)

    def test_holds_exactly_the_tensors_the_read_out_needs(self):
        shapes = sorted(tuple(tensor.shape) for tensor in self.model.state_dict().values())
        self.assertEqual(shapes, sorted(expected_shapes()))

    def test_ignores_every_weight_the_read_out_does_not_need(self):
        # Scramble the query and key rows and the rest of the read-out block, every later block and the final norm:
        # a wrapper built from that model has to compute exactly what the original one does.
        scrambled = copy.deepcopy(self.dino)
        with torch.no_grad():
            block = scrambled.blocks[LAYER]
            block.attn.qkv.weight[:2 * DIM].normal_()
            block.attn.qkv.bias[:2 * DIM].normal_()
            for module in (block.attn.proj, block.ls1, block.norm2, block.mlp, block.ls2, scrambled.norm,
                           *scrambled.blocks[LAYER + 1:]):
                for parameter in module.parameters():
                    parameter.normal_()
            scrambled.mask_token.normal_()
            rebuilt = facet.AnyLocValueFacet(scrambled, LAYER, SIZE).eval()
            self.assertTrue(torch.equal(rebuilt(self.image), self.model(self.image)))

    def test_rejects_a_size_or_layer_it_cannot_serve(self):
        with self.assertRaisesRegex(ValueError, "multiple of the patch size"):
            facet.AnyLocValueFacet(self.dino, LAYER, SIZE + 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            facet.AnyLocValueFacet(self.dino, DEPTH, SIZE)

    @unittest.skipUnless(_REFERENCE_IMAGE.exists(), f"needs {_REFERENCE_IMAGE}")
    def test_reference_input_is_one_gray_image_imagenet_normalized(self):
        array = facet.make_reference_input(_REFERENCE_IMAGE, SIZE)
        self.assertEqual(array.shape, (1, 3, SIZE, SIZE))
        self.assertEqual(array.dtype, np.float32)
        # Every channel is the same gray image under its own ImageNet normalization.
        gray = [array[0, c] * std + mean for c, (mean, std) in enumerate(zip(facet.PIXEL_MEAN, facet.PIXEL_STD))]
        np.testing.assert_allclose(gray[1], gray[0], atol=1e-6)
        np.testing.assert_allclose(gray[2], gray[0], atol=1e-6)
        self.assertGreaterEqual(gray[0].min(), -1e-6)
        self.assertLessEqual(gray[0].max(), 1 + 1e-6)
        self.assertGreater(gray[0].std(), 0.01)


@unittest.skipIf(_MISSING, f"needs {', '.join(_MISSING)}")
class TestOnnxExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dino = make_small_dino()
        cls.model = facet.AnyLocValueFacet(cls.dino, LAYER, SIZE).eval()
        cls.image = random_image()
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "value_facet.onnx"
        exporter.export_onnx(cls.model, cls.image, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def audit(self, path):
        return facet.audit_onnx(path, expected_shapes(), (1, 3, SIZE, SIZE), (1, PATCHES, DIM))

    def modified(self, name, edit):
        model = onnx.load(str(self.path))
        edit(model.graph)
        path = Path(self.directory.name) / name
        onnx.save(model, str(path))
        return path

    def test_export_holds_only_the_read_out(self):
        audit = self.audit(self.path)
        self.assertEqual(audit.op_counts["Conv"], 0)
        self.assertEqual(audit.op_counts["Resize"], 0)
        self.assertEqual(audit.parameter_count, sum(math.prod(shape) for shape in expected_shapes()))

    def test_export_computes_what_the_wrapper_does(self):
        evaluated = exporter.evaluate_onnx(self.path, self.image.numpy())
        with torch.no_grad():
            expected = self.model(self.image).numpy()
        self.assertLessEqual(float(np.abs(evaluated - expected).max()), exporter.EVALUATOR_TOLERANCE)

    def test_audit_rejects_a_tensor_the_read_out_does_not_need(self):
        def add_initializer(graph):
            graph.initializer.append(onnx.numpy_helper.from_array(np.ones((7, 5), dtype=np.float32), "stray"))

        with self.assertRaisesRegex(ValueError, "not part of the read-out"):
            self.audit(self.modified("stray.onnx", add_initializer))

    def test_audit_rejects_a_convolution(self):
        def add_convolution(graph):
            graph.initializer.append(
                onnx.numpy_helper.from_array(np.ones((DIM, 3, PATCH, PATCH), dtype=np.float32), "kernel"))
            graph.node.append(onnx.helper.make_node("Conv", ["image", "kernel"], ["unused"], strides=[PATCH, PATCH]))

        with self.assertRaisesRegex(ValueError, "Conv"):
            self.audit(self.modified("conv.onnx", add_convolution))

    def test_audit_rejects_the_query_and_key_rows_of_the_read_out_block(self):
        class WholeQkv(facet.AnyLocValueFacet):
            """The read-out done the obvious way: the whole fused projection, sliced afterwards."""

            def __init__(self, dino):
                super().__init__(dino, LAYER, SIZE)
                self.value = copy.deepcopy(dino.blocks[LAYER].attn.qkv)

            def forward(self, image):
                return super().forward(image)[..., 2 * DIM:]

        path = Path(self.directory.name) / "whole_qkv.onnx"
        exporter.export_onnx(WholeQkv(self.dino).eval(), self.image, path)
        with self.assertRaisesRegex(ValueError, "not part of the read-out"):
            self.audit(path)

    def test_audit_rejects_another_input_size(self):
        with self.assertRaisesRegex(ValueError, "inputs are"):
            facet.audit_onnx(self.path, expected_shapes(), (1, 3, SIZE + PATCH, SIZE + PATCH), (1, PATCHES, DIM))


@unittest.skipIf(_MISSING, f"needs {', '.join(_MISSING)}")
@unittest.skipUnless(_REFERENCE_IMAGE.exists(), f"needs {_REFERENCE_IMAGE}")
class TestExporterCli(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = Path(self.directory.name)
        self.output = folder / "model.onnx"
        self.reference_input = folder / "reference_input.npy"
        self.reference_output = folder / "reference_output.npy"
        self.dino = make_small_dino()

    def run_main(self):
        argv = ["--dinov2_src", "unused", "--checkpoint", "unused", "--layer", str(LAYER), "--size", str(SIZE),
                "--output", str(self.output), "--reference_image", str(_REFERENCE_IMAGE),
                "--reference_input", str(self.reference_input), "--reference_output", str(self.reference_output)]
        with mock.patch.object(exporter, "load_dinov2", return_value=self.dino):
            return exporter.main(argv)

    def test_writes_the_model_and_its_reference(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(sorted(path.name for path in Path(self.directory.name).iterdir()),
                         ["model.onnx", "reference_input.npy", "reference_output.npy"])
        reference_input = np.load(self.reference_input)
        reference_output = np.load(self.reference_output)
        self.assertEqual(reference_input.shape, (1, 3, SIZE, SIZE))
        np.testing.assert_array_equal(reference_input, facet.make_reference_input(_REFERENCE_IMAGE, SIZE))
        expected = facet.hook_reference(self.dino, LAYER, torch.from_numpy(reference_input)).numpy()
        np.testing.assert_array_equal(reference_output, expected)
        evaluated = exporter.evaluate_onnx(self.output, reference_input)
        self.assertLessEqual(float(np.abs(evaluated - reference_output).max()), exporter.EVALUATOR_TOLERANCE)

    def test_writes_nothing_when_a_check_fails(self):
        with mock.patch.object(exporter, "audit_onnx", side_effect=ValueError("stray tensor")):
            self.assertEqual(self.run_main(), 1)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

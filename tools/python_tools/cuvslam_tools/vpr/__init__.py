"""Visual place recognition tooling.

The AnyLoc backend in `libs/slam/vpr/vpr_anyloc.cpp` runs a TensorRT engine of DINOv2 that the build
makes in two steps (tools/anyloc_model): `export_dinov2_onnx` exports the part of DINOv2 AnyLoc reads
to ONNX, using the module `dinov2_value_facet` defines, and anyloc_engine_builder turns that into the
engine.
"""

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

import os
from pathlib import Path
import tempfile
import unittest
import json

from cuvslam_tools.common.edex import EDEXMetadata, DistortionModel


DATA_DIR = Path(os.path.abspath(__file__)).parent / "data"


class TestEdex(unittest.TestCase):
    def test_read_edex(self):
        edex = EDEXMetadata.read(DATA_DIR / "edex")
        self.assertEqual(edex.header.version, "0.9")
        self.assertEqual(edex.header.frame_start, 0)
        self.assertEqual(edex.header.frame_end, 1600)
        self.assertEqual(len(edex.header.cameras), 2)
        self.assertEqual(
            edex.header.cameras[0].intrinsics.distortion_model,
            DistortionModel.POLYNOMIAL,
        )
        self.assertEqual(
            edex.header.cameras[0].intrinsics.distortion_params.shape, (8,)
        )
        self.assertEqual(edex.header.cameras[0].intrinsics.focal.shape, (2,))
        self.assertEqual(edex.header.cameras[0].intrinsics.principal.shape, (2,))
        self.assertEqual(edex.header.cameras[0].intrinsics.resolution.shape, (2,))
        self.assertEqual(edex.header.cameras[0].transform.shape, (3, 4))

    def test_write_edex_uses_size_key(self):
        # Round tripping through EDEXMetadata accepts either spelling, so check the raw json:
        # the C++ reader and dataset_reader.py only know "size".
        edex = EDEXMetadata.read(DATA_DIR / "edex")
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=True, encoding="utf-8"
        ) as temp_file:
            edex.write(Path(temp_file.name))
            with open(temp_file.name) as f:
                header = json.load(f)[0]

        for camera in header["cameras"]:
            self.assertIn("size", camera["intrinsics"])
            self.assertNotIn("resolution", camera["intrinsics"])

    def test_read_edex_copy_transform(self):
        # Read the intrinsics-only EDEX file
        edex_intrinsics = EDEXMetadata.read(DATA_DIR / "edex_intrinsics")
        self.assertIsNone(edex_intrinsics.header.cameras[0].transform)

        # Written output should be the same as the input
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=True, encoding="utf-8"
        ) as temp_file:
            edex_intrinsics.write(Path(temp_file.name))
            edex_intrinsics_2 = EDEXMetadata.read(Path(temp_file.name))
            self.assertEqual(edex_intrinsics, edex_intrinsics_2)

        # Read the original EDEX file
        edex = EDEXMetadata.read(DATA_DIR / "edex")

        edex_intrinsics.header.cameras[0].transform = edex.header.cameras[0].transform
        edex_intrinsics.header.cameras[1].transform = edex.header.cameras[1].transform

        # Written output should be the same as the input with the transform added
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=True, encoding="utf-8"
        ) as temp_file:
            edex_intrinsics.write(Path(temp_file.name))
            edex_intrinsics_2 = EDEXMetadata.read(Path(temp_file.name))
            self.assertEqual(edex, edex_intrinsics_2)

    def test_read_edex_nested_sequence(self):
        edex_data = json.loads((DATA_DIR / "edex").read_text(encoding="utf-8"))
        edex_data[1]["sequence"] = [
            ["images/front_stereo_camera/left/000000.png"],
            ["images/front_stereo_camera/right/000000.png"],
        ]

        with tempfile.NamedTemporaryFile(
            mode="w+", delete=True, encoding="utf-8"
        ) as temp_file:
            json.dump(edex_data, temp_file)
            temp_file.flush()

            edex = EDEXMetadata.read(Path(temp_file.name))
            self.assertEqual(
                edex.body.sequence,
                [
                    [Path("images/front_stereo_camera/left/000000.png")],
                    [Path("images/front_stereo_camera/right/000000.png")],
                ],
            )

            edex.write(Path(temp_file.name))
            round_trip = json.loads(Path(temp_file.name).read_text(encoding="utf-8"))

        self.assertEqual(
            round_trip[1]["sequence"],
            [
                ["images/front_stereo_camera/left/000000.png"],
                ["images/front_stereo_camera/right/000000.png"],
            ],
        )

    def test_read_edex_bad(self):
        with self.assertRaises(Exception):
            EDEXMetadata.read(DATA_DIR / "edex_bad")

    def test_metadata_requires_body(self):
        edex = EDEXMetadata.read(DATA_DIR / "edex")
        with self.assertRaises(TypeError):
            EDEXMetadata(edex.header)


if __name__ == "__main__":
    unittest.main()

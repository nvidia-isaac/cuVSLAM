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

"""Archive extraction safety and sequence discovery for the TUM RGB-D preparation.

Extraction is hand-rolled on Python tarfile instead of tar, which validates
member names and link targets separately. One case covers each branch.
"""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from cuvslam_tools.dataset_preparation.common import PreparationError
from cuvslam_tools.dataset_preparation.tum import prepare as tum_prepare


class TestTumSafeExtraction(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.destination = self.root / "out"
        self.destination.mkdir()

    def tearDown(self):
        self._temporary.cleanup()

    def _archive(self, build) -> Path:
        archive = self.root / "archive.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            build(tar)
        return archive

    def test_traversing_member_is_rejected(self):
        def build(tar):
            info = tarfile.TarInfo("../escaped.txt")
            info.size = len(b"payload")
            tar.addfile(info, io.BytesIO(b"payload"))

        with self.assertRaisesRegex(PreparationError, "unsafe member path"):
            tum_prepare.extract_archive(self._archive(build), self.destination)
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_traversing_link_target_is_rejected(self):
        def build(tar):
            info = tarfile.TarInfo("rgbd_dataset_freiburg3_cabinet/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
            tar.addfile(info)

        with self.assertRaisesRegex(PreparationError, "unsafe link target"):
            tum_prepare.extract_archive(self._archive(build), self.destination)


class TestTumSequenceDiscovery(unittest.TestCase):
    """extract_sequence locates the sequence directory rather than assuming its name."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.destination = self.root / "out"
        self.destination.mkdir()

    def tearDown(self):
        self._temporary.cleanup()

    def _archive(self, directories) -> Path:
        archive = self.root / "archive.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            for directory in directories:
                info = tarfile.TarInfo(f"{directory}/rgb.txt")
                payload = b"# timestamp filename\n"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        return archive

    def test_single_directory_is_returned(self):
        archive = self._archive(["rgbd_dataset_freiburg3_teddy"])
        extracted = tum_prepare.extract_sequence(archive, self.destination)
        self.assertEqual(extracted.name, "rgbd_dataset_freiburg3_teddy")
        self.assertTrue((extracted / "rgb.txt").is_file())

    def test_unexpected_archive_shape_is_reported(self):
        archive = self._archive(["first", "second"])
        with self.assertRaisesRegex(PreparationError, "expected one top-level directory"):
            tum_prepare.extract_sequence(archive, self.destination)


if __name__ == "__main__":
    unittest.main()

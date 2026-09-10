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

"""Where prepare_m3ed_spot reads its source from.

M3ED is the only corpus with no download step, so it has to reconcile that with
a provisioning contract that hands every dataset a directory to download into.
"""

import tempfile
import unittest
from pathlib import Path

from cuvslam_tools.dataset_preparation.common import PreparationError
from cuvslam_tools.dataset_preparation.m3ed_spot import convert_m3ed_spot, prepare


class TestSourceSelection(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.raw = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def _download(self, sequence, kind="data"):
        path = prepare.local_path(self.raw, convert_m3ed_spot.source_name(sequence), kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not really HDF5")
        return path

    def test_no_raw_dir_reads_the_bucket(self):
        opener, source = prepare._select_source(None)
        self.assertIs(opener, prepare._open_remote)
        self.assertIn(prepare.BUCKET_URL, source)

    def test_an_empty_raw_dir_reads_the_bucket(self):
        # provision_dataset.sh creates this directory and passes it for every
        # dataset. Reading it as a source made provisioning fail on the first
        # sequence, looking for files that this converter never downloads.
        opener, source = prepare._select_source(self.raw)
        self.assertIs(opener, prepare._open_remote)
        self.assertIn("no source files", source)

    def test_downloaded_files_are_read_when_they_are_there(self):
        self._download("skatepark_2")
        opener, source = prepare._select_source(self.raw)
        self.assertIsNot(opener, prepare._open_remote)
        self.assertEqual(source, str(self.raw))

    def test_a_raw_dir_that_is_not_a_directory_is_rejected(self):
        # Not the provisioning contract but a typo, and streaming the whole
        # corpus instead would take hours to notice.
        with self.assertRaisesRegex(PreparationError, "not a directory"):
            prepare._select_source(self.raw / "absent")


if __name__ == "__main__":
    unittest.main()

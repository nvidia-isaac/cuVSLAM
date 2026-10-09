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

"""The one-time IMU patch must reproduce a fresh v2 conversion byte for byte."""

import contextlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py

from cuvslam_tools.dataset_preparation.m3ed_spot import convert_m3ed_spot, patch_imu
from cuvslam_tools.tests.test_m3ed_spot_conversion import write_data_file, write_pose_file


def _drop_imu_counts(entry):
    del entry["source_counts"]["imu_samples"]
    del entry["converted_counts"]["imu_samples"]
    del entry["imu_dropped_samples"]
    del entry["imu_max_gap_ms"]


def _downgrade_to_v1(root: Path) -> None:
    """Strip a v2 conversion back to what converter v1 wrote."""
    metadata_path = root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["converter_version"] = 1
    del metadata["imu"]
    for entry in metadata["sequences"]:
        _drop_imu_counts(entry)
        sequence_dir = root / entry["sequence"]
        (sequence_dir / "IMU.jsonl").unlink()
        edex = sequence_dir / "stereo.edex"
        document = json.loads(edex.read_text())
        del document[0]["imu"]
        edex.write_text(json.dumps(document, indent=4) + "\n")
        state = convert_m3ed_spot.read_state(sequence_dir)
        _drop_imu_counts(state["metadata"])
        convert_m3ed_spot.write_state(sequence_dir, state["frame_limit"], state["metadata"])
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _tree(root: Path):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()
    }


class TestPatchImu(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.sequence = convert_m3ed_spot.ALL_SEQS[0]
        write_data_file(self.root / "data.h5")
        write_pose_file(self.root / "pose.h5")
        self.fresh = self.root / "fresh"
        convert_m3ed_spot.convert(self.fresh, [self.sequence], open_sequence=self._open, frame_limit=4)
        self.patched = self.root / "patched"
        shutil.copytree(self.fresh, self.patched)
        _downgrade_to_v1(self.patched)

    def tearDown(self):
        self._temporary.cleanup()

    @contextlib.contextmanager
    def _open(self, published, kind):
        name = "data.h5" if kind == "data" else "pose.h5"
        with h5py.File(self.root / name, "r") as handle:
            yield handle, {"path": name}

    def test_patching_v1_output_reproduces_a_fresh_v2_conversion(self):
        report = patch_imu.patch(self.patched, self._open)
        self.assertIn(f"{self.sequence}/IMU.jsonl", report["changed_files"])
        self.assertEqual(_tree(self.patched), _tree(self.fresh))

    def test_patching_twice_changes_nothing(self):
        patch_imu.patch(self.patched, self._open)
        report = patch_imu.patch(self.patched, self._open)
        self.assertEqual(report["changed_files"], [])

    def test_a_dataset_already_at_v2_changes_nothing(self):
        report = patch_imu.patch(self.fresh, self._open)
        self.assertEqual(report["changed_files"], [])

    def test_cameras_that_no_longer_match_the_source_are_refused(self):
        edex = self.patched / self.sequence / "stereo.edex"
        document = json.loads(edex.read_text())
        document[0]["cameras"][1]["intrinsics"]["focal"][0] += 1.0
        edex.write_text(json.dumps(document, indent=4) + "\n")
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "re-provision"):
            patch_imu.patch(self.patched, self._open)
        self.assertFalse((self.patched / self.sequence / "IMU.jsonl").exists())

    def test_a_frame_count_that_disagrees_with_the_edex_is_refused(self):
        metadata = self.patched / self.sequence / "frame_metadata.jsonl"
        lines = metadata.read_text().splitlines()
        metadata.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "frame_end"):
            patch_imu.patch(self.patched, self._open)

    def test_imu_counts_land_in_the_dataset_metadata(self):
        patch_imu.patch(self.patched, self._open)
        metadata = json.loads((self.patched / "dataset_metadata.json").read_text())
        self.assertEqual(metadata["converter_version"], 2)
        entry = metadata["sequences"][0]
        self.assertEqual(entry["converted_counts"]["imu_samples"], 49)
        self.assertGreater(entry["source_counts"]["imu_samples"], 49)
        self.assertEqual(entry["imu_dropped_samples"], 0)


if __name__ == "__main__":
    unittest.main()

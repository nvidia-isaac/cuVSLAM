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

"""Offline map repair packaging, raw EuRoC pairing, and transactional storage."""

import base64
import builtins
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from unittest import mock

from cuvslam_tools.offline_map_repair import cli
from cuvslam_tools.offline_map_repair.run_sequence import frame_metadata


class OfflineMapRepairCliTests(unittest.TestCase):
    def test_help_does_not_import_cuvslam_or_nemo(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "cuvslam" or name == "nat" or name.startswith("nat."):
                raise AssertionError(f"Help must not load {name}")
            return original_import(name, *args, **kwargs)

        for args in [["--help"], ["serve", "--help"], ["run-sequence", "--help"], ["evaluate", "--help"]]:
            with self.subTest(args=args), mock.patch.object(builtins, "__import__", side_effect=guarded_import):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                    cli.main(args)
                self.assertEqual(result.exception.code, 0)

    def test_dispatch_preserves_workflow_arguments(self):
        with mock.patch.object(cli, "import_module") as load:
            cli.main(["verify", "baseline", "repaired", "--extractor", "/opt/bin/map_extractor"])
        load.assert_called_once_with("cuvslam_tools.offline_map_repair.verify_maps")
        load.return_value.main.assert_called_once_with(
            ["baseline", "repaired", "--extractor", "/opt/bin/map_extractor"]
        )

    def test_server_configuration_is_package_data(self):
        package = resources.files("cuvslam_tools.offline_map_repair.mcp_server")
        config = package.joinpath("configs/map_records.yaml").read_text()
        self.assertIn("cuvslam_tools.offline_map_repair.mcp_server/write_records", config)
        self.assertIn("${CUVSLAM_MAP_ROOT}", config)
        connection = json.loads(package.joinpath("configs/mcp-server.json").read_text())
        self.assertEqual(connection["mcpServers"]["cuvslam-map-records"]["url"], "http://127.0.0.1:9906/mcp")

    def test_raw_euroc_pairs_exact_timestamps_and_delivers_imu_first(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder)
            for sensor, rows in {
                "cam0": "#timestamp,filename\n10,a.png\n20,b.png\n30,c.png\n",
                "cam1": "#timestamp,filename\n10,d.png\n30,e.png\n",
                "imu0": "#timestamp,gx,gy,gz,ax,ay,az\n10,1,2,3,4,5,6\n25,1,2,3,4,5,6\n",
            }.items():
                target = dataset / "mav0" / sensor
                target.mkdir(parents=True)
                (target / "data.csv").write_text(rows)
            frames, pairing = frame_metadata(dataset, inertial=True)
        self.assertEqual(pairing, {"cam0": 3, "cam1": 2, "paired": 2})
        self.assertEqual(
            [(row["timestamp"], row["type"]) for row in frames],
            [(10, "imu"), (10, "stereo"), (25, "imu"), (30, "stereo")],
        )
        self.assertEqual(frames[0]["gyro"], [1.0, 2.0, 3.0])
        self.assertEqual(frames[0]["accel"], [4.0, 5.0, 6.0])


@unittest.skipUnless(importlib.util.find_spec("lmdb"), "Install cuvslam-tools[map-repair] for LMDB tests")
class OfflineMapStoreTests(unittest.TestCase):
    def setUp(self):
        import lmdb

        from cuvslam_tools.offline_map_repair.mcp_server.store import TABLES, MapStore

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        baseline = self.root / "baseline"
        baseline.mkdir()
        self.key = (7).to_bytes(8, "little")
        self.value = b"\x00\xfforiginal\x00"
        with lmdb.open(str(baseline), max_dbs=10) as env:
            handles = {name: env.open_db(name.encode()) for name in TABLES}
            with env.begin(write=True) as txn:
                txn.put(self.key, self.value, db=handles["descriptors"])
        self.store = MapStore(self.root)

    def change(self, value):
        from cuvslam_tools.offline_map_repair.mcp_server.store import digest

        return {
            "table": "descriptors",
            "key_hex": self.key.hex(),
            "expected_sha256": digest(self.value),
            "value_base64": base64.b64encode(value).decode() if value is not None else None,
        }

    def test_original_map_and_outside_root_are_protected(self):
        with self.assertRaisesRegex(ValueError, "copies"):
            self.store.write_records("baseline", [self.change(b"new")], "Original must be read-only")
        with self.assertRaisesRegex(ValueError, "below the configured root"):
            self.store.describe_map(str(self.root.parent))
        self.store.clone_map("baseline", "copy")
        with self.assertRaises(FileExistsError):
            self.store.clone_map("baseline", "copy")

    def test_stale_batch_rolls_back_data_and_audit(self):
        self.store.clone_map("baseline", "copy")
        before = self.store.read_records("copy", "descriptors")
        stale = {**self.change(b"stale"), "key_hex": (8).to_bytes(8, "little").hex()}
        with self.assertRaisesRegex(ValueError, "Stale record"):
            self.store.write_records("copy", [self.change(b"new"), stale], "Atomicity test")
        self.assertEqual(self.store.read_records("copy", "descriptors"), before)
        with self.store.open_map("copy") as env:
            audit = env.open_db(b"agent_audit", create=False)
            with env.begin(db=audit) as txn:
                self.assertEqual(txn.stat()["entries"], 0)

    def test_delete_preserves_undo_bytes_and_original(self):
        self.store.clone_map("baseline", "copy")
        result = self.store.write_records("copy", [self.change(None)], "Delete test")
        self.assertEqual(self.store.read_records("copy", "descriptors")["records"], [])
        self.assertEqual(self.store.read_records("baseline", "descriptors")["total"], 1)
        with self.store.open_map("copy") as env:
            audit = env.open_db(b"agent_audit", create=False)
            with env.begin(db=audit) as txn:
                record = json.loads(txn.get(result["audit_id"].encode()))
        self.assertEqual(base64.b64decode(record["changes"][0]["before_base64"]), self.value)


if __name__ == "__main__":
    unittest.main()

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

import asyncio
import base64
import builtins
import contextlib
import importlib.util
import io
import json
import pickle
import sys
import tempfile
import threading
import unittest
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from cuvslam_tools.offline_map_repair import cli
from cuvslam_tools.offline_map_repair.sessions.euroc import frame_metadata


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
        load.assert_called_once_with("cuvslam_tools.offline_map_repair.validation.verify_maps")
        load.return_value.main.assert_called_once_with(
            ["baseline", "repaired", "--extractor", "/opt/bin/map_extractor"]
        )

    def test_every_command_resolves_to_an_existing_module(self):
        # The command table carries dotted paths into the functional subpackages, so a
        # moved or renamed workflow module would otherwise only fail when invoked.
        for command, (module, _) in cli.COMMANDS.items():
            with self.subTest(command=command):
                name = f"cuvslam_tools.offline_map_repair.{module}"
                self.assertIsNotNone(importlib.util.find_spec(name), name)

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


class EurocCalibrationProvenanceTests(unittest.TestCase):
    def sequence(self, recalibrated):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        for sensor in ("cam0", "cam1", "imu0"):
            (root / sensor).mkdir()
            (root / sensor / "sensor.yaml").touch()
            if recalibrated:
                (root / sensor / "sensor_recalibrated.yaml").touch()
        return str(root)

    def test_reports_the_calibration_file_get_rig_would_load(self):
        from cuvslam_tools.offline_map_repair.sessions.euroc import calibration_source

        self.assertEqual(calibration_source(self.sequence(False)), "sensor.yaml")
        self.assertEqual(calibration_source(self.sequence(True)), "sensor_recalibrated.yaml")


class EvaluateArgumentTests(unittest.TestCase):
    def test_map_name_is_required_unless_preparing(self):
        from cuvslam_tools.offline_map_repair.validation import evaluate_localization

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            evaluate_localization.main([])
        self.assertEqual(result.exception.code, 2)


class TrajectoryCoverageTests(unittest.TestCase):
    def test_strict_interpolation_preserves_endpoints_and_interpolates_inside(self):
        from cuvslam_tools.offline_map_repair.sessions.trajectory import interpolate

        timestamps = np.array([100, 200], dtype=np.int64)
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, 0, 3] = 2
        actual = interpolate(timestamps, poses, [100, 150, 200], bounds_error=True)
        np.testing.assert_allclose(actual[:, 0, 3], [0, 1, 2])
        for query in ([99], [201]):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, "does not cover"):
                interpolate(timestamps, poses, query, bounds_error=True)
        np.testing.assert_allclose(interpolate(timestamps, poses, [99, 201])[:, 0, 3], [0, 2])

    def test_rejects_empty_single_duplicate_or_unordered_reference(self):
        from cuvslam_tools.offline_map_repair.sessions.trajectory import interpolate

        for timestamps in ([], [1], [1, 1], [2, 1]):
            with self.subTest(timestamps=timestamps), self.assertRaisesRegex(ValueError, "strictly increasing"):
                interpolate(np.array(timestamps), np.tile(np.eye(4), (len(timestamps), 1, 1)), [1])


class SequenceRunnerTests(unittest.TestCase):
    def test_waits_for_save_callback_before_returning(self):
        from cuvslam_tools.offline_map_repair.sessions.run_sequence import save_map

        for delayed in (False, True):
            with self.subTest(delayed=delayed):
                callbacks = []

                def native_save(destination, callback):
                    def finish():
                        callbacks.append(destination)
                        callback(True)

                    if delayed:
                        timer = threading.Timer(0.01, finish)
                        self.addCleanup(timer.join)
                        timer.start()
                    else:
                        finish()

                save_map(SimpleNamespace(save_map=native_save), Path("saved-map"), 1)
                self.assertEqual(callbacks, ["saved-map"])

    def test_reports_save_failure_and_timeout(self):
        from cuvslam_tools.offline_map_repair.sessions.run_sequence import save_map

        with self.assertRaisesRegex(RuntimeError, "Map save failed"):
            save_map(SimpleNamespace(save_map=lambda path, callback: callback(False)), Path("map"), 1)
        with self.assertRaisesRegex(TimeoutError, "did not complete"):
            save_map(SimpleNamespace(save_map=lambda path, callback: None), Path("map"), 0.01)

    def test_rejects_invalid_save_timeout_before_loading_native_binding(self):
        from cuvslam_tools.offline_map_repair.sessions import run_sequence

        for timeout in ("0", "-1", "inf", "nan"):
            with self.subTest(timeout=timeout), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    run_sequence.main(["dataset", "output", "--save-timeout", timeout])
                self.assertEqual(result.exception.code, 2)

    def test_delivers_and_records_imu_for_modes_that_consume_it(self):
        from cuvslam_tools.offline_map_repair.sessions import run_sequence

        for mode, imus, expected in (
            ("Multisensor", [object()], 1),
            ("Multisensor", [], 0),
            ("Inertial", [object()], 1),
            ("Multicamera", [object()], 0),
        ):
            with self.subTest(mode=mode, imus=len(imus)), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "session"

                def frames(dataset, use_imu):
                    events = [{"type": "imu", "timestamp": 10, "accel": [0, 0, 0], "gyro": [0, 0, 0]}]
                    return (events if use_imu else []) + [
                        {"type": "stereo", "timestamp": 10, "images_paths": []}
                    ], {}

                def native_save(destination, callback):
                    Path(destination).mkdir()
                    (Path(destination) / "data.mdb").write_bytes(b"map")
                    callback(True)

                tracker = mock.Mock()
                tracker.track.return_value = (SimpleNamespace(world_from_rig=None), None)
                tracker.slam.save_map.side_effect = native_save
                tracker.slam.get_all_slam_poses.return_value = []
                tracker_factory = mock.Mock(return_value=tracker)
                tracker_factory.Mode = SimpleNamespace(OdometryWithSlamOffline=1)
                binding = SimpleNamespace(
                    Odometry=SimpleNamespace(
                        Config=lambda: SimpleNamespace(async_sba=False, rectified_stereo_camera=False),
                        OdometryMode=SimpleNamespace(Multisensor=1, Inertial=2, Multicamera=3),
                    ),
                    Slam=SimpleNamespace(Config=lambda: SimpleNamespace(sync_mode=True, enable_reading_internals=True)),
                    Tracker=tracker_factory,
                    ImuMeasurement=SimpleNamespace,
                    get_version=lambda: "test",
                )
                with (
                    mock.patch.dict(sys.modules, {"cuvslam": binding}),
                    mock.patch.object(run_sequence, "get_rig", return_value=SimpleNamespace(imus=imus)),
                    mock.patch.object(run_sequence, "frame_metadata", side_effect=frames),
                    mock.patch.object(run_sequence, "calibration_source", return_value="sensor.yaml"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    run_sequence.main(["dataset", str(output), "--odometry-mode", mode])
                self.assertEqual(tracker.register_imu_measurement.call_count, expected)
                self.assertEqual(json.loads((output / "summary.json").read_text())["imu_samples"], expected)


# core.commit reaches the server through core.mcp_client, which imports the mcp client
# and, for the shared table names, mcp_server.store - which imports lmdb. Both ship in
# the map-repair extra, and the tools test step installs base dependencies only.
@unittest.skipUnless(
    importlib.util.find_spec("mcp") and importlib.util.find_spec("lmdb"),
    "Install cuvslam-tools[map-repair] for MCP client tests",
)
class CommitGuardTests(unittest.TestCase):
    def test_incomplete_reference_is_rejected_before_edit_or_commit(self):
        from cuvslam_tools.offline_map_repair.policies import full_pose_repair

        for span in ([10, 20], [0, 10]):
            with self.subTest(span=span), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "baseline").mkdir()
                (root / "baseline/records.pkl").write_bytes(pickle.dumps({}))
                map_data = mock.Mock()
                map_data.keyframes = {1: {"timestamp_ns": 0}, 2: {"timestamp_ns": 20}}
                map_data.poses = {1: np.eye(4), 2: np.eye(4)}
                with (
                    mock.patch.object(full_pose_repair, "HERE", root),
                    mock.patch.object(full_pose_repair, "MapData", return_value=map_data),
                    mock.patch.object(full_pose_repair, "load_trajectory", return_value=(
                        np.array(span), np.tile(np.eye(4), (2, 1, 1))
                    )),
                    mock.patch.object(full_pose_repair, "commit") as commit,
                    self.assertRaisesRegex(ValueError, "does not cover"),
                ):
                    asyncio.run(full_pose_repair.run(None))
                commit.assert_not_called()
                map_data.encode.assert_not_called()
                for pose in map_data.poses.values():
                    np.testing.assert_array_equal(pose, np.eye(4))

    def test_empty_change_set_is_refused_before_anything_is_cloned(self):
        # Pruning zero landmarks re-encodes to identical bytes. Cloning first would
        # leave an orphaned destination that makes every retry fail on mkdir.
        from cuvslam_tools.offline_map_repair.core import commit as commit_module

        records = {table: {} for table in ("main", "landmarks", "descriptors", "spatial_cells")}
        with mock.patch.object(commit_module, "MapData") as decoded, mock.patch.object(commit_module, "call") as tool:
            decoded.return_value.validate.return_value = []
            self.assertRaisesRegex(
                ValueError,
                "selected nothing to edit",
                asyncio.run,
                commit_module.commit(None, "baseline/map", "copy/map", records, records, "No-op"),
            )
            tool.assert_not_called()


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
        # A missing database reports separately, so confinement can be asserted on.
        with self.assertRaisesRegex(ValueError, "existing database"):
            self.store.describe_map("no_such_map")
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

    def test_failed_clone_leaves_no_directory_behind(self):
        with (
            mock.patch.object(type(self.store), "open_map", side_effect=RuntimeError("copy failed")),
            self.assertRaisesRegex(RuntimeError, "copy failed"),
        ):
            self.store.clone_map("baseline", "copy")
        self.assertFalse((self.root / "copy").exists())
        # The destination is reusable, so a retry is not blocked by a partial clone.
        self.store.clone_map("baseline", "copy")
        self.assertTrue((self.root / "copy" / "agent-copy.json").is_file())

    def test_pagination_returns_every_record_once_for_any_page_size(self):
        import lmdb

        from cuvslam_tools.offline_map_repair.mcp_server.store import TABLES

        many = self.root / "many"
        many.mkdir()
        expected = {(i).to_bytes(8, "little"): bytes([i]) * (i + 1) for i in range(7)}
        with lmdb.open(str(many), max_dbs=10) as env:
            handles = {name: env.open_db(name.encode()) for name in TABLES}
            with env.begin(write=True) as txn:
                for key, value in expected.items():
                    txn.put(key, value, db=handles["landmarks"])
        for limit in (1, 2, 3, 7, 10):
            with self.subTest(limit=limit):
                rows, offset = [], 0
                while True:
                    page = self.store.read_records("many", "landmarks", offset=offset, limit=limit)
                    self.assertEqual(page["total"], len(expected))
                    rows.extend(page["records"])
                    offset = page["next_offset"]
                    if offset >= page["total"]:
                        break
                self.assertEqual(
                    {bytes.fromhex(r["key_hex"]): base64.b64decode(r["value_base64"]) for r in rows}, expected
                )
                self.assertEqual(len(rows), len(expected))

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

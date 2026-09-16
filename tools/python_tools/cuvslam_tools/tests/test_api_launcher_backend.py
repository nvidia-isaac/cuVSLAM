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

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cuvslam_tools.reporter import api_launcher_backend as backend


def _args(**overrides):
    values = {
        "odometry_mode": "multicamera",
        "multicam_mode": "moderate",
        "repeat_type": "none",
        "num_loops": 0,
        "blackout_period": 0,
        "blackout_duration": 10,
        "use_gpu": True,
        "async_sba": False,
        "use_motion_model": True,
        "use_denoising": False,
        "debug_imu_mode": False,
        "rectified_stereo_camera": False,
        "max_frame_delta_s": 0.1,
        "use_slam": False,
        "sync_slam": True,
        "loop_closure_mode": "lsi",
        "use_vpr": False,
        "max_map_size": None,
        "enable_observations_export": False,
        "enable_landmarks_export": False,
        "enable_final_landmarks_export": False,
        "cache_uncompressed": False,
        "camera_ids": None,
        "debug_dump_directory": "",
        "enable_depth_stereo_tracking": False,
        "depth_scale_factor": 1.0,
        "api_launcher_args": [],
        "api_launcher_path": "",
        "config_path": "/data/stereo.edex",
        "dataset": "/data",
        "gt_path": None,
        "gt_from_shuttle": False,
        "visualize_rerun": False,
        "visualize_plot": False,
        "save_output_tracker_data": False,
        "use_segments": False,
        "segment_lengths": [],
        "sequence_title": "sequence",
        "output_dir": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _pose(x=0.0):
    return {"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [x, 0.0, 0.0]}


class TestLauncherCommand(unittest.TestCase):
    def test_maps_reporter_options_and_forwards_internal_flags(self):
        args = _args(
            repeat_type="shuttle",
            num_loops=2,
            use_slam=True,
            camera_ids=[1, 0],
            api_launcher_args=["--expert_sba_num_frames", "9"],
        )
        command = backend.build_launcher_command(args, "/bin/launcher", "/tmp/result.jsonl")

        self.assertIn("--repeat=2", command)
        self.assertIn("--shuttle=true", command)
        self.assertIn("--cfg_enable_slam=true", command)
        self.assertIn("--cfg_enable_export=true", command)
        self.assertIn("--cameras=1,0", command)
        self.assertEqual(command[-2:], args.api_launcher_args)

    def test_rejects_reporter_owned_passthrough_flag(self):
        with self.assertRaisesRegex(ValueError, "managed by cuvslam_reporter"):
            backend.build_launcher_command(
                _args(api_launcher_args=["--cfg_odom_mode=3"]),
                "/bin/launcher",
                "/tmp/result.jsonl",
            )

    def test_maps_max_map_size_only_when_set(self):
        default = backend.build_launcher_command(_args(), "/bin/launcher", "/tmp/result.jsonl")
        unlimited = backend.build_launcher_command(_args(max_map_size=0), "/bin/launcher", "/tmp/result.jsonl")

        self.assertFalse(any(flag.startswith("--cfg_slam_max_map_size") for flag in default))
        self.assertIn("--cfg_slam_max_map_size=0", unlimited)

    @staticmethod
    def _rgbd_command(cameras, camera_ids=None):
        with tempfile.TemporaryDirectory() as directory:
            edex = Path(directory) / "stereo.edex"
            edex.write_text(json.dumps([{"cameras": cameras}, {"fps": 30}]), encoding="utf-8")
            return backend.build_launcher_command(
                _args(odometry_mode="rgbd", config_path=str(edex), camera_ids=camera_ids),
                "/bin/launcher",
                "/tmp/result.jsonl",
            )

    def test_rgbd_uses_edex_divisor_and_api_scale_one(self):
        command = self._rgbd_command([{"depth_id": 0, "depth_scale_factor": 5000.0}])

        self.assertIn("--cfg_depth_camera=0", command)
        self.assertIn("--depth_scale_factor=5000.0", command)
        self.assertIn("--cfg_depth_scale_factor=1.0", command)

    def test_rgbd_depth_camera_follows_the_camera_subset(self):
        command = self._rgbd_command([{}, {"depth_id": 1, "depth_scale_factor": 1000.0}], camera_ids=[1])

        self.assertIn("--cameras=1", command)
        self.assertIn("--cfg_depth_camera=0", command)

    def test_rgbd_rejects_several_depth_cameras(self):
        with self.assertRaisesRegex(ValueError, "multisensor"):
            self._rgbd_command([{"depth_id": 0}, {"depth_id": 1}])

    def test_rejects_vpr_loop_closure(self):
        for overrides in ({"use_vpr": True}, {"loop_closure_mode": "vpr", "use_vpr": True}):
            with self.subTest(**overrides):
                with self.assertRaisesRegex(ValueError, "VPR"):
                    backend.run_api_launcher(_args(use_slam=True, **overrides))

    def test_rejects_multisensor_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported C\\+\\+ odometry mode"):
            backend.build_launcher_command(_args(odometry_mode="multisensor"), "/bin/launcher", "/tmp/r.jsonl")


class TestLauncherRecords(unittest.TestCase):
    def test_shuttle_ground_truth_uses_first_pass_and_scores_last_pass(self):
        frames = [
            backend.FrameRecord(index, source, index + 1, True, np.eye(4), None)
            for index, source in enumerate((0, 1, 1, 0))
        ]
        frames[1].odom_pose = np.array(
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        records = backend.LauncherRecords(frames, {}, {}, {"attempted_frames": 4})
        poses, _ = backend._frame_poses(records, use_slam=False)

        gt, mapping = backend._ground_truth(
            _args(gt_from_shuttle=True, repeat_type="shuttle", num_loops=1),
            records,
            poses,
        )

        self.assertEqual(len(gt), 2)
        np.testing.assert_allclose(gt[1][:3, 3], [1, 0, 0])
        self.assertEqual(mapping, {3: 0, 2: 1})

    def test_parses_loss_final_slam_and_loop_closure(self):
        records = [
            {
                "type": "frame",
                "frame_id": 0,
                "source_frame_number": 0,
                "timestamp_ns": 10,
                "track_seconds": 0.01,
                "tracking_valid": True,
                "odom_pose": _pose(1),
                "slam_pose": _pose(2),
            },
            {
                "type": "frame",
                "frame_id": 1,
                "source_frame_number": 1,
                "timestamp_ns": 20,
                "track_seconds": 0.02,
                "tracking_valid": False,
                "odom_pose": None,
                "slam_pose": None,
            },
            {"type": "final_slam_pose", "timestamp_ns": 10, "pose": _pose(3)},
            {"type": "loop_closure", "timestamp_ns": 10, "pose": _pose(4)},
            {
                "type": "summary",
                "attempted_frames": 2,
                "successful_frames": 1,
                "lost_frames": 1,
                "total_track_seconds": 0.03,
                "wall_seconds": 0.04,
                "success": True,
            },
        ]
        with tempfile.NamedTemporaryFile("w", delete=False) as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
            path = stream.name
        try:
            parsed = backend.parse_launcher_output(path)
        finally:
            os.unlink(path)

        poses, loop_closures = backend._frame_poses(parsed, use_slam=True)
        np.testing.assert_allclose(poses[0][:3, 3], [3, 0, 0])
        np.testing.assert_allclose(poses[1], poses[0])
        np.testing.assert_allclose(loop_closures[10][:3, 3], [4, 0, 0])

    def test_rejects_non_finite_pose(self):
        records = [
            {
                "type": "frame",
                "frame_id": 0,
                "source_frame_number": 0,
                "timestamp_ns": 10,
                "tracking_valid": True,
                "odom_pose": _pose(float("nan")),
                "slam_pose": None,
            },
            {"type": "summary", "attempted_frames": 1},
        ]
        with tempfile.NamedTemporaryFile("w", delete=False) as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
            path = stream.name
        try:
            with self.assertRaisesRegex(ValueError, "non-finite"):
                backend.parse_launcher_output(path)
        finally:
            os.unlink(path)


_FAKE_LAUNCHER = (
    "#!/usr/bin/env python3\n"
    "import json, sys\n"
    "path = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--report_output='))\n"
    "pose = {'rotation': [0, 0, 0, 1], 'translation': [0, 0, 0]}\n"
    "records = [\n"
    " {'type': 'frame', 'frame_id': 0, 'source_frame_number': 0, 'timestamp_ns': 1,\n"
    "  'track_seconds': 0.25, 'tracking_valid': True, 'odom_pose': pose, 'slam_pose': None},\n"
    "]\n"
    "if '--cfg_enable_slam=true' in sys.argv:\n"
    "    records.append({'type': 'loop_closure', 'timestamp_ns': 1, 'pose': pose})\n"
    "records.append({'type': 'summary', 'attempted_frames': 1, 'successful_frames': 1, 'lost_frames': 0,\n"
    "                'total_track_seconds': 0.25, 'wall_seconds': 0.3, 'success': True})\n"
    "open(path, 'w').write('\\n'.join(json.dumps(r) for r in records) + '\\n')\n"
)


class TestLauncherSubprocess(unittest.TestCase):
    def _run(self, root: Path, **overrides):
        launcher = root / "fake_launcher.py"
        launcher.write_text(_FAKE_LAUNCHER, encoding="utf-8")
        launcher.chmod(0o755)
        return backend.run_api_launcher(
            _args(
                api_launcher_path=str(launcher),
                output_dir=str(root / "output"),
                config_path=str(root / "stereo.edex"),
                dataset=str(root),
                **overrides,
            )
        )

    def test_runs_fake_launcher_and_writes_report_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stat = self._run(root)

            self.assertEqual(stat.n_frames, 1)
            self.assertEqual(stat.average_fps, 4.0)
            self.assertEqual(stat.loop_closure_mode, "")
            self.assertTrue((root / "output" / "logs" / "sequence.api_launcher.log").is_file())
            self.assertTrue((root / "output" / "sequence.txt").is_file())

    def test_slam_run_reports_its_loop_closures(self):
        with tempfile.TemporaryDirectory() as directory:
            stat = self._run(Path(directory), use_slam=True)

        self.assertEqual(stat.loop_closure_mode, "lsi")
        self.assertEqual(stat.num_loop_closures, 1)


if __name__ == "__main__":
    unittest.main()

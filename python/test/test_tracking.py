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
import unittest
import numpy as np

# Unit tests must not require the external Rerun Viewer, even in USE_RERUN builds.
os.environ.setdefault("RERUN", "0")

import cuvslam as vslam
import data_gen as data

TrackerMode = vslam.Tracker.Mode

class TestTracking(unittest.TestCase):
    def setUp(self):
        cameras = data.generate_stereo_camera(640, 480, baseline=0.25)
        self.num_cameras = len(cameras)
        imu = vslam.ImuCalibration()
        self.rig = vslam.Rig(cameras, [imu])

    def test_init_arguments(self):
        # accept partial, positional & keyword arguments
        _ = vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime, vslam.Odometry.Config())
        _ = vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime)
        _ = vslam.Tracker(rig=self.rig, mode=TrackerMode.OdometryOnlyRealtime,
                          odom_config=vslam.Odometry.Config())
        _ = vslam.Tracker(rig=self.rig, mode=TrackerMode.OdometryOnlyRealtime)
        _ = vslam.Tracker(rig=self.rig, mode=TrackerMode.OdometryWithSlamRealtime,
                          slam_config=vslam.Slam.Config())

        with self.assertRaises(TypeError):
            vslam.Tracker()  # type: ignore # missing required argument "rig"

        with self.assertRaises(TypeError):
            vslam.Tracker(self.rig)  # type: ignore # missing required argument "mode"

    def test_component_accessors(self):
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime)
        self.assertIsInstance(tracker.odometry, vslam.Odometry)
        self.assertFalse(tracker.is_slam_enabled())
        with self.assertRaisesRegex(RuntimeError, "SLAM is not enabled"):
            _ = tracker.slam

        tracker_with_slam = vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamRealtime,
                                          slam_config=vslam.Slam.Config())
        self.assertTrue(tracker_with_slam.is_slam_enabled())
        self.assertIsInstance(tracker_with_slam.slam, vslam.Slam)

    def test_init_mode_must_match_configs(self):
        # The mode and the configurations have to say the same thing. The tracker checks them
        # instead of rewriting them, so a disagreement is an error rather than a silent override.
        offline = vslam.Odometry.Config(async_sba=False)
        blocking_slam = vslam.Slam.Config(sync_mode=True)

        with self.assertRaisesRegex(ValueError, "async_sba"):
            vslam.Tracker(self.rig, TrackerMode.OdometryOnlyOffline, vslam.Odometry.Config())
        with self.assertRaisesRegex(ValueError, "async_sba"):
            vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime, offline)

        with self.assertRaisesRegex(ValueError, "sync_mode"):
            vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamOffline, offline, vslam.Slam.Config())
        with self.assertRaisesRegex(ValueError, "sync_mode"):
            vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamRealtime,
                          vslam.Odometry.Config(), blocking_slam)

        # A SLAM configuration is meaningless without a SLAM mode.
        with self.assertRaisesRegex(ValueError, "slam_config must be null"):
            vslam.Tracker(self.rig, TrackerMode.OdometryOnlyOffline, offline, blocking_slam)

        # The combinations that agree are accepted.
        _ = vslam.Tracker(self.rig, TrackerMode.OdometryOnlyOffline, offline)
        _ = vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamOffline, offline, blocking_slam)

    @unittest.skip("TODO: add a check that cameras don't have the same pose")
    def test_init_same_cameras(self):
        # don't accept rig with duplicate cameras
        bad_rig = vslam.Rig(cameras=[self.rig.cameras[0], self.rig.cameras[0]])
        with self.assertRaises(ValueError):
            vslam.Tracker(bad_rig, TrackerMode.OdometryOnlyRealtime)

    def test_init_no_cameras(self):
        # don't accept empty rig with no cameras
        with self.assertRaises(ValueError):
            vslam.Tracker(vslam.Rig([]), TrackerMode.OdometryOnlyRealtime)

    def test_init_different_sizes(self):
        # don't accept cameras with different sizes
        self.rig.cameras[1].size[0] = 480
        with self.assertRaises(ValueError):
            vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime)

    def test_init_inertial_no_imus(self):
        # don't accept rig with no IMUs in inertial mode
        self.rig.imus = []
        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Inertial
        with self.assertRaises(ValueError):
            vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime, cfg)

    def test_init_multiple_imus(self):
        # don't accept rig with multiple IMUs
        self.rig.imus = [vslam.ImuCalibration(), vslam.ImuCalibration()]
        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Inertial
        with self.assertRaises(ValueError):
            vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime)

    def test_init_negative_image_size(self):
        # image sizes should be positive
        self.rig.cameras[0].size[0] = -1
        self.rig.cameras[1].size[0] = -1
        with self.assertRaises(ValueError):
            vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime)

    def test_tracking(self):
        modes = [vslam.Odometry.OdometryMode.Multicamera, vslam.Odometry.OdometryMode.Inertial,
                 vslam.Odometry.OdometryMode.Mono, vslam.Odometry.OdometryMode.RGBD]
        synthetic_images = [np.zeros((480, 640), dtype=np.uint8) for _ in range(self.num_cameras)]
        synthetic_masks = [np.ones((480, 640), dtype=np.uint8) for _ in range(self.num_cameras)]
        synthetic_depths = [np.random.randint(0, 1024, size=(480, 640), dtype=np.uint16)]
        print("")  # to insert a newline after python unittest output
        for mode in modes:
            for with_mask in [False, True]:
                with self.subTest(mode=mode, with_mask=with_mask):
                    print(f"Testing mode={mode}, with_mask={with_mask}")
                    cfg = vslam.Odometry.Config()
                    cfg.odometry_mode = mode
                    cfg.rgbd_settings.depth_scale_factor = 1000.0
                    cfg.rgbd_settings.depth_camera_id = 0
                    cfg.rgbd_settings.enable_depth_stereo_tracking = False
                    tracker = vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime, cfg)
                    for i in range(60):
                        # Add 10 IMU measurements between frames
                        if mode == vslam.Odometry.OdometryMode.Inertial:
                            for j in range(10):
                                imu = vslam.ImuMeasurement()
                                imu.timestamp_ns = i * 1000 + j * 100
                                # Gravity points down (+Y in OpenCV)
                                imu.linear_accelerations = [0.0, 9.81, 0.0]
                                # No rotation
                                imu.angular_velocities = [0.0, 0.0, 0.0]
                                tracker.register_imu_measurement(0, imu)
                        odom_pose, slam_pose = tracker.track(
                            (i + 1) * 1000, synthetic_images, synthetic_masks if with_mask else None,
                            synthetic_depths if mode == vslam.Odometry.OdometryMode.RGBD else None)
                        self.assertIs(slam_pose, None)
                        if odom_pose.world_from_rig:
                            np.testing.assert_array_almost_equal(
                                odom_pose.world_from_rig.pose.rotation, [0.0, 0.0, 0.0, 1.0],
                                err_msg=f"iteration {i}")
                            np.testing.assert_array_almost_equal(
                                odom_pose.world_from_rig.pose.translation, [0.0, 0.0, 0.0],
                                err_msg=f"iteration {i}")
                            # TODO: fix gravity
                            # if mode == vslam.Odometry.OdometryMode.Inertial:
                            #     gravity = tracker.odometry.get_last_gravity()
                            #     if gravity is not None:
                            #         self.assertAlmostEqual(gravity[1], 9.81, msg=f"iteration {i}")

    def test_multisensor_tracking_variants(self):
        image = np.zeros((480, 640), dtype=np.uint8)
        depth = np.full((480, 640), 1000, dtype=np.uint16)

        with self.subTest(variant="single RGB-D"):
            config = vslam.Odometry.Config(
                odometry_mode=vslam.Odometry.OdometryMode.Multisensor,
                multisensor_settings=vslam.Odometry.MultisensorSettings(
                    depth_camera_ids=[0], depth_scale_factor=1000.0))
            rig = vslam.Rig([self.rig.cameras[0]])
            tracker = vslam.Tracker(rig, TrackerMode.OdometryOnlyRealtime, config)
            tracker.track(1000, [image], depths=[depth])
            tracker.track(2000, [image], depths=[depth])

        with self.subTest(variant="stereo without depth"):
            config = vslam.Odometry.Config(
                odometry_mode=vslam.Odometry.OdometryMode.Multisensor)
            rig = vslam.Rig(self.rig.cameras)
            tracker = vslam.Tracker(rig, TrackerMode.OdometryOnlyRealtime, config)
            stereo_images = [image.copy(), image.copy()]
            tracker.track(1000, stereo_images)
            tracker.track(2000, stereo_images)

        with self.subTest(variant="single RGB-D with IMU"):
            config = vslam.Odometry.Config(
                odometry_mode=vslam.Odometry.OdometryMode.Multisensor,
                multisensor_settings=vslam.Odometry.MultisensorSettings(
                    depth_camera_ids=[0], depth_scale_factor=1000.0))
            rig = vslam.Rig([self.rig.cameras[0]], [vslam.ImuCalibration()])
            tracker = vslam.Tracker(rig, TrackerMode.OdometryOnlyRealtime, config)
            for frame in range(2):
                frame_start_ns = frame * 1100
                for sample in range(10):
                    imu = vslam.ImuMeasurement()
                    imu.timestamp_ns = frame_start_ns + sample * 100
                    imu.linear_accelerations = [0.0, 9.81, 0.0]
                    imu.angular_velocities = [0.0, 0.0, 0.0]
                    tracker.register_imu_measurement(0, imu)
                tracker.track(frame_start_ns + 1000, [image], depths=[depth])

    def test_multisensor_requires_depth_or_overlap(self):
        config = vslam.Odometry.Config(
            odometry_mode=vslam.Odometry.OdometryMode.Multisensor)
        rig = vslam.Rig([self.rig.cameras[0]])
        with self.assertRaisesRegex(ValueError, "at least one RGB-D camera.*or one camera pair"):
            vslam.Tracker(rig, TrackerMode.OdometryOnlyRealtime, config)

    def test_multisensor_rejects_duplicate_depth_camera_ids(self):
        config = vslam.Odometry.Config(
            odometry_mode=vslam.Odometry.OdometryMode.Multisensor,
            multisensor_settings=vslam.Odometry.MultisensorSettings(
                depth_camera_ids=[0, 0]))
        rig = vslam.Rig([self.rig.cameras[0]])
        with self.assertRaisesRegex(ValueError, "duplicate camera id"):
            vslam.Tracker(rig, TrackerMode.OdometryOnlyRealtime, config)

    def _run_keyframe_overrides(self, mode, overrides):
        """Track a fixed feature-rich frame repeatedly, applying a per-frame keyframe override.

        ``overrides`` is one entry per frame: None for automatic selection, True/False to force the
        keyframe decision. Returns the list of per-frame ``keyframe`` flags reported by odometry.
        """
        img = data.ImageGenerator(self.rig.cameras, 10)
        # Reuse one feature-rich frame so the scene is near-static: the automatic selector sees a
        # high survivor ratio and would naturally pick non-keyframes after the first frame.
        static_images, _ = img.generate_zoomed_images(0)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = mode
        cfg.enable_observations_export = True  # required so get_state() (and its keyframe flag) is available
        odometry = vslam.Odometry(self.rig, cfg)

        keyframes = []
        for i, override in enumerate(overrides):
            internals = None
            if override is not None:
                internals = vslam.Odometry.Internals()
                internals.kf_override_frame_selection = override
            # timestamps stay well under kf_max_timedelta_between_kfs_s (60 s) so the time-based
            # keyframe rule never fires on its own.
            odometry.track((i + 1) * 1_000_000, static_images, internals=internals)
            keyframes.append(odometry.get_state().keyframe)
        return keyframes

    def test_keyframe_override_forces_decision(self):
        # Multicamera mode routes keyframe selection through KFSelector, which honors
        # kf_override_frame_selection. Forcing the decision must override automatic selection.
        num_frames = 6
        mode = vslam.Odometry.OdometryMode.Multicamera

        forced_kf = self._run_keyframe_overrides(mode, [True] * num_frames)
        forced_non_kf = self._run_keyframe_overrides(mode, [None] + [False] * (num_frames - 1))
        automatic = self._run_keyframe_overrides(mode, [None] * num_frames)

        # Forcing True makes every frame a keyframe.
        self.assertTrue(all(forced_kf), f"override=True should force keyframes, got {forced_kf}")
        # Forcing False makes every frame after the first a non-keyframe.
        self.assertTrue(forced_non_kf[0], "first frame is a keyframe under automatic selection")
        self.assertFalse(any(forced_non_kf[1:]), f"override=False should force non-keyframes, got {forced_non_kf}")
        # Same input frames, opposite results: the override is what flips the decision.
        for i in range(1, num_frames):
            self.assertNotEqual(forced_kf[i], forced_non_kf[i],
                                f"override should flip the keyframe decision at frame {i}")
        # The override also changes the automatic decision: the near-static scene yields at least one
        # automatic non-keyframe, yet forcing True keyframes every frame.
        self.assertIn(False, automatic, "near-static scene should produce at least one automatic non-keyframe")

    def test_keyframe_override_forces_decision_in_mono(self):
        # Mono mode uses SelectorMono internally, and kf_override_frame_selection must still force
        # the reported keyframe decision.
        num_frames = 6
        mode = vslam.Odometry.OdometryMode.Mono

        forced_kf = self._run_keyframe_overrides(mode, [True] * num_frames)
        forced_non_kf = self._run_keyframe_overrides(mode, [None] + [False] * (num_frames - 1))
        automatic = self._run_keyframe_overrides(mode, [None] * num_frames)

        self.assertTrue(all(forced_kf), f"override=True should force Mono keyframes, got {forced_kf}")
        self.assertTrue(forced_non_kf[0], "first frame is a keyframe under automatic selection")
        self.assertFalse(any(forced_non_kf[1:]),
                         f"override=False should force Mono non-keyframes, got {forced_non_kf}")
        for i in range(1, num_frames):
            self.assertNotEqual(forced_kf[i], forced_non_kf[i],
                                f"override should flip the Mono keyframe decision at frame {i}")
        self.assertIn(False, automatic, "near-static Mono scene should produce at least one automatic non-keyframe")

    def test_get_observations_and_landmarks(self):
        img = data.ImageGenerator(self.rig.cameras, 10)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        cfg.enable_observations_export = True
        cfg.enable_landmarks_export = True
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime, cfg)

        # Track a few frames to build up features
        for i in range(5):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)

        # get_last_observations returns a list for each camera
        for cam_idx in range(self.num_cameras):
            obs = tracker.odometry.get_last_observations(cam_idx)
            self.assertIsInstance(obs, list)
            if obs:
                self.assertIsInstance(obs[0], vslam.Observation)
                self.assertIsInstance(obs[0].id, int)
                self.assertIsInstance(obs[0].u, float)
                self.assertIsInstance(obs[0].v, float)

        # get_last_landmarks returns a list
        landmarks = tracker.odometry.get_last_landmarks()
        self.assertIsInstance(landmarks, list)
        if landmarks:
            self.assertIsInstance(landmarks[0], vslam.Landmark)

    def test_get_final_landmarks(self):
        img = data.ImageGenerator(self.rig.cameras, 10)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        cfg.enable_final_landmarks_export = True
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime, cfg)

        for i in range(5):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)

        final = tracker.odometry.get_final_landmarks()
        self.assertIsInstance(final, dict)

    def test_get_gravity_non_inertial(self):
        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryOnlyRealtime, cfg)

        images = [np.zeros((480, 640), dtype=np.uint8) for _ in range(self.num_cameras)]
        tracker.track(1000, images)

        # get_last_gravity raises ValueError when IMU fusion is disabled
        with self.assertRaises(ValueError):
            tracker.odometry.get_last_gravity()

    def test_slam_metrics(self):
        img = data.ImageGenerator(self.rig.cameras, 10)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        cfg.async_sba = False
        s_cfg = vslam.Slam.Config()
        s_cfg.sync_mode = True
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamOffline, cfg, s_cfg)

        for i in range(3):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)

        metrics = tracker.slam.get_slam_metrics()
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics.lc_status, bool)
        self.assertIsInstance(metrics.pgo_status, bool)

    def test_slam_loop_closure_poses(self):
        img = data.ImageGenerator(self.rig.cameras, 10)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        cfg.async_sba = False
        s_cfg = vslam.Slam.Config()
        s_cfg.sync_mode = True
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamOffline, cfg, s_cfg)

        for i in range(3):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)

        poses = tracker.slam.get_loop_closure_poses()
        self.assertIsNotNone(poses)
        self.assertIsInstance(poses, list)

    def test_slam_get_all_poses(self):
        img = data.ImageGenerator(self.rig.cameras, 10)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        cfg.async_sba = False
        s_cfg = vslam.Slam.Config()
        s_cfg.sync_mode = True
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamOffline, cfg, s_cfg)

        for i in range(5):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)

        poses = tracker.slam.get_all_slam_poses()
        self.assertIsInstance(poses, list)
        self.assertGreater(len(poses), 0)
        self.assertIsInstance(poses[0], vslam.PoseStamped)
        self.assertIsInstance(poses[0].timestamp_ns, int)

    def test_slam_pose_graph(self):
        img = data.ImageGenerator(self.rig.cameras, 10)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        cfg.async_sba = False
        s_cfg = vslam.Slam.Config()
        s_cfg.sync_mode = True
        s_cfg.enable_reading_internals = True
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamOffline, cfg, s_cfg)

        for i in range(5):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)

        pg = tracker.slam.get_pose_graph()
        self.assertIsNotNone(pg)
        self.assertIsInstance(pg.nodes, list)
        self.assertIsInstance(pg.edges, list)

    def test_slam_landmarks(self):
        img = data.ImageGenerator(self.rig.cameras, 10)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        cfg.async_sba = False
        s_cfg = vslam.Slam.Config()
        s_cfg.sync_mode = True
        s_cfg.enable_reading_internals = True
        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamOffline, cfg, s_cfg)

        for i in range(5):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)

        landmarks = tracker.slam.get_landmarks(vslam.Slam.DataLayer.Landmarks)
        self.assertIsNotNone(landmarks)
        self.assertIsInstance(landmarks.timestamp_ns, int)
        self.assertIsInstance(landmarks.landmarks, list)

    def test_slam_pose(self):
        img = data.ImageGenerator(self.rig.cameras, 10)
        synthetic_images, _ = img.generate_zoomed_images(0)

        cfg = vslam.Odometry.Config()
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera

        cfg.async_sba = False
        s_cfg = vslam.Slam.Config()
        s_cfg.sync_mode = True

        tracker = vslam.Tracker(self.rig, TrackerMode.OdometryWithSlamOffline, cfg, s_cfg)

        identity = vslam.Pose(translation=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0])
        odom_pose, slam_pose = tracker.track(1000, synthetic_images)

        np.testing.assert_array_almost_equal(slam_pose.translation, identity.translation)
        np.testing.assert_array_almost_equal(slam_pose.rotation, identity.rotation)
        np.testing.assert_array_almost_equal(odom_pose.world_from_rig.pose.translation, identity.translation)
        np.testing.assert_array_almost_equal(odom_pose.world_from_rig.pose.rotation, identity.rotation)

        odom_pose, slam_pose = tracker.track(2000, synthetic_images)

        np.testing.assert_array_almost_equal(slam_pose.translation, identity.translation)
        np.testing.assert_array_almost_equal(slam_pose.rotation, identity.rotation)
        np.testing.assert_array_almost_equal(odom_pose.world_from_rig.pose.translation, identity.translation)
        np.testing.assert_array_almost_equal(odom_pose.world_from_rig.pose.rotation, identity.rotation)

if __name__ == "__main__":
    unittest.main()

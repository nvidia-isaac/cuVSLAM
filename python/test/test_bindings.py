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

import unittest
import numpy as np
import cuvslam as vslam


class TestBindings(unittest.TestCase):
    def test_version(self):
        ver, major, minor, patch = vslam.get_version()
        self.assertIsInstance(ver, str)
        self.assertIsInstance(major, int)
        self.assertIsInstance(minor, int)
        self.assertIsInstance(patch, int)
        expected_prefix = f"{major}.{minor}.{patch}+"
        self.assertTrue(ver.startswith(expected_prefix),
                        f"Expected version to start with '{expected_prefix}', got '{ver}'")

    def test_distortion_default(self):
        d = vslam.Distortion()
        self.assertEqual(d.model, vslam.Distortion.Model.Pinhole)
        self.assertEqual(d.parameters, [])

    def test_distortion_constructor(self):
        d1 = vslam.Distortion(vslam.Distortion.Model.Polynomial,
                              [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(d1.model, vslam.Distortion.Model.Polynomial)
        self.assertEqual(d1.parameters, [1, 2, 3, 4, 5, 6, 7, 8])

        d2 = vslam.Distortion(parameters=[1, 2, 3, 4, 5],
                              model=vslam.Distortion.Model.Brown)
        self.assertEqual(d2.model, vslam.Distortion.Model.Brown)
        self.assertEqual(d2.parameters, [1, 2, 3, 4, 5])

    def test_distortion_assignment(self):
        d = vslam.Distortion()
        d.model = vslam.Distortion.Model.Fisheye
        self.assertEqual(d.model, vslam.Distortion.Model.Fisheye)
        self.assertEqual(d.parameters, [])
        d.parameters = [4, 3, 2, 1]
        self.assertEqual(d.parameters, [4, 3, 2, 1])

        # Distortion parameters are not individually mutable, they are supposed to be obtained
        # as a whole from calibration but this leads to an unexpected behavior in python
        # (would be better if exception was raised by nanobind)
        d.parameters[0] = 1000000
        self.assertEqual(d.parameters, [4, 3, 2, 1])

    def test_pose_default(self):
        p = vslam.Pose()
        np.testing.assert_array_equal(p.translation, [0, 0, 0])
        np.testing.assert_array_equal(p.rotation, [0, 0, 0, 1])

    def test_pose_assignment(self):
        p = vslam.Pose()
        p.translation = [1, 2, 3]
        p.rotation = [0, 0, 0.7071, 0.7071]
        np.testing.assert_array_equal(p.translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(p.rotation, [0, 0, 0.7071, 0.7071])

        with self.assertRaisesRegex(ValueError, 'Sequence must have exactly 3 elements'):
            p.translation = []
        with self.assertRaisesRegex(ValueError, 'Sequence must have exactly 3 elements'):
            p.translation = [0, 0]
        with self.assertRaisesRegex(ValueError, 'Sequence must have exactly 3 elements'):
            p.translation = [0, 0, 0, 0]

        with self.assertRaises(TypeError):
            p.translation = None  # type: ignore
        with self.assertRaises(TypeError):
            p.translation = {"x": 1, "y": 2, "z": 3}  # type: ignore

    def test_pose_constructor(self):
        p1 = vslam.Pose([0, 0, 0.7071, 0.7071], [1, 2, 3])
        np.testing.assert_array_equal(p1.translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(p1.rotation, [0, 0, 0.7071, 0.7071])

        p2 = vslam.Pose(translation=[1, 2, 3], rotation=[0, 0, 0, 1])
        np.testing.assert_array_equal(p2.translation, [1, 2, 3])
        np.testing.assert_array_equal(p2.rotation, [0, 0, 0, 1])

    def test_pose_element_access(self):
        p = vslam.Pose()
        p.translation = [1, 2, 3]
        p.rotation = [0, 0, 0, 1]

        p.translation[0] = 9
        p.translation[1] *= 3
        p.rotation[2] = 0.5
        np.testing.assert_array_equal(p.translation, [9, 6, 3])
        np.testing.assert_array_equal(p.rotation, [0, 0, 0.5, 1])

        with self.assertRaisesRegex(IndexError, 'index 3 is out of bounds'):
            p.translation[3] = 1

        with self.assertRaisesRegex(IndexError, 'index 4 is out of bounds'):
            p.rotation[4] = 1

        # Cannot prevent assignment of some non-numeric types convertible to float
        p.translation[0] = None  # type: ignore # NaN
        p.translation[0] = "42"  # type: ignore # float("42")==42.0

        with self.assertRaises(ValueError):
            p.translation[0] = "nonsence"  # type: ignore

    def test_camera_default(self):
        c = vslam.Camera()
        np.testing.assert_array_equal(c.size, (0, 0))
        np.testing.assert_array_equal(c.focal, (0.0, 0.0))
        np.testing.assert_array_equal(c.principal, (0.0, 0.0))
        # TODO: add __eq__ to distortion?
        # self.assertEqual(c.distortion, vslam.Distortion())
        self.assertEqual(c.distortion.model, vslam.Distortion.Model.Pinhole)
        self.assertEqual(c.distortion.parameters, [])
        self.assertEqual(c.border_top, 0)
        self.assertEqual(c.border_bottom, 0)
        self.assertEqual(c.border_left, 0)
        self.assertEqual(c.border_right, 0)

    def test_camera_constructor(self):
        c1 = vslam.Camera(
            size=[640, 480], principal=[320, 240], focal=[320.0, 320.0],
            distortion=vslam.Distortion(vslam.Distortion.Model.Polynomial, [1, 2, 3, 4, 5, 6, 7, 8]),
            border_top=10, border_bottom=20, border_left=30, border_right=40)
        np.testing.assert_array_equal(c1.size, (640, 480))
        np.testing.assert_array_equal(c1.focal, (320.0, 320.0))
        np.testing.assert_array_equal(c1.principal, (320.0, 240.0))
        self.assertEqual(c1.distortion.model, vslam.Distortion.Model.Polynomial)
        self.assertEqual(c1.distortion.parameters, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(c1.border_top, 10)
        self.assertEqual(c1.border_bottom, 20)
        self.assertEqual(c1.border_left, 30)
        self.assertEqual(c1.border_right, 40)

        # verify optional params default correctly when omitted
        c2 = vslam.Camera(size=[640, 480], principal=[320, 240], focal=[320.0, 320.0])
        np.testing.assert_array_equal(c2.rig_from_camera.translation, [0, 0, 0])
        np.testing.assert_array_equal(c2.rig_from_camera.rotation, [0, 0, 0, 1])
        self.assertEqual(c2.distortion.model, vslam.Distortion.Model.Pinhole)
        self.assertEqual(c2.border_top, 0)
        self.assertEqual(c2.border_bottom, 0)
        self.assertEqual(c2.border_left, 0)
        self.assertEqual(c2.border_right, 0)

        # unnamed parameters are not supported
        with self.assertRaises(TypeError):
            vslam.Camera((640, 480), (320, 240), (320.0, 320.0))  # type: ignore

    def test_camera_assignment(self):
        c = vslam.Camera(size=[640, 480], principal=[320, 240], focal=[320.0, 320.0])
        c.rig_from_camera = vslam.Pose([0, 0, 0.7071, 0.7071], [1, 2, 3])
        np.testing.assert_array_almost_equal(c.rig_from_camera.rotation, [0, 0, 0.7071, 0.7071])
        np.testing.assert_array_equal(c.rig_from_camera.translation, [1, 2, 3])

        c.distortion = vslam.Distortion(vslam.Distortion.Model.Brown, [1, 2, 3, 4, 5])
        self.assertEqual(c.distortion.model, vslam.Distortion.Model.Brown)
        self.assertEqual(c.distortion.parameters, [1, 2, 3, 4, 5])

        c.border_top = 5
        c.border_bottom = 10
        c.border_left = 15
        c.border_right = 20
        self.assertEqual(c.border_top, 5)
        self.assertEqual(c.border_bottom, 10)
        self.assertEqual(c.border_left, 15)
        self.assertEqual(c.border_right, 20)

    def test_rig_constructor(self):
        r = vslam.Rig()
        self.assertEqual(len(r.cameras), 0)
        self.assertEqual(len(r.imus), 0)

        # accept partial, positional & keyword arguments
        r1 = vslam.Rig([], [])
        self.assertEqual(len(r1.cameras), 0)
        self.assertEqual(len(r1.imus), 0)

        r2 = vslam.Rig(cameras=[vslam.Camera()])
        self.assertEqual(len(r2.cameras), 1)
        self.assertEqual(len(r2.imus), 0)

        r3 = vslam.Rig([vslam.Camera()])
        self.assertEqual(len(r3.cameras), 1)
        self.assertEqual(len(r3.imus), 0)

        r4 = vslam.Rig(cameras=[vslam.Camera()], imus=[vslam.ImuCalibration()])
        self.assertEqual(len(r4.cameras), 1)
        self.assertEqual(len(r4.imus), 1)

        with self.assertRaises(TypeError):
            vslam.Rig([vslam.ImuCalibration()], [vslam.Camera()])  # type: ignore

    def test_rig_modifiers(self):
        r = vslam.Rig()
        r.cameras = [vslam.Camera()]
        self.assertEqual(len(r.cameras), 1)
        self.assertEqual(len(r.imus), 0)
        # TODO: support appending
        # rig.cameras.append(vslam.Camera())
        # self.assertEqual(len(rig.cameras), 2)

    def test_imu_calibration_default(self):
        imu = vslam.ImuCalibration()
        np.testing.assert_array_equal(imu.rig_from_imu.translation, [0, 0, 0])
        np.testing.assert_array_equal(imu.rig_from_imu.rotation, [0, 0, 0, 1])

    def test_imu_calibration_constructor(self):
        # Use distinct values to catch any parameter ordering bugs in the binding
        imu = vslam.ImuCalibration(
            gyroscope_noise_density=1.0,
            gyroscope_random_walk=2.0,
            accelerometer_noise_density=3.0,
            accelerometer_random_walk=4.0,
            frequency=200.0,
        )
        self.assertEqual(imu.gyroscope_noise_density, 1.0)
        self.assertEqual(imu.gyroscope_random_walk, 2.0)
        self.assertEqual(imu.accelerometer_noise_density, 3.0)
        self.assertEqual(imu.accelerometer_random_walk, 4.0)
        self.assertEqual(imu.frequency, 200.0)

        # unnamed parameters are not supported (kw_only)
        with self.assertRaises(TypeError):
            vslam.ImuCalibration(vslam.Pose(), 1.0, 2.0, 3.0, 4.0, 200.0)  # type: ignore

    def test_imu_calibration_assignment(self):
        imu = vslam.ImuCalibration()
        imu.rig_from_imu = vslam.Pose([0, 0, 0.7071, 0.7071], [1, 2, 3])
        np.testing.assert_array_almost_equal(imu.rig_from_imu.rotation, [0, 0, 0.7071, 0.7071])
        np.testing.assert_array_equal(imu.rig_from_imu.translation, [1, 2, 3])

        imu.gyroscope_noise_density = 0.01
        imu.gyroscope_random_walk = 0.02
        imu.accelerometer_noise_density = 0.03
        imu.accelerometer_random_walk = 0.04
        imu.frequency = 400.0
        self.assertAlmostEqual(imu.gyroscope_noise_density, 0.01, places=5)
        self.assertAlmostEqual(imu.gyroscope_random_walk, 0.02, places=5)
        self.assertAlmostEqual(imu.accelerometer_noise_density, 0.03, places=5)
        self.assertAlmostEqual(imu.accelerometer_random_walk, 0.04, places=5)
        self.assertEqual(imu.frequency, 400.0)

    def test_imu_measurement_constructor(self):
        # Use distinct values to catch parameter ordering bugs between the two Array<3> params
        imu = vslam.ImuMeasurement(
            timestamp_ns=1234567890,
            linear_accelerations=[1, 2, 3],
            angular_velocities=[4, 5, 6],
        )
        self.assertEqual(imu.timestamp_ns, 1234567890)
        np.testing.assert_array_equal(imu.linear_accelerations, [1, 2, 3])
        np.testing.assert_array_equal(imu.angular_velocities, [4, 5, 6])

        # unnamed parameters are not supported (kw_only)
        with self.assertRaises(TypeError):
            vslam.ImuMeasurement(1234567890, [1, 2, 3], [4, 5, 6])  # type: ignore

    def test_imu_measurement_assignment(self):
        imu_measurement = vslam.ImuMeasurement()
        imu_measurement.timestamp_ns = 1234567890
        imu_measurement.linear_accelerations = np.array([1, 2, 3])  # type: ignore
        imu_measurement.angular_velocities = (4, 5, 6)
        self.assertEqual(imu_measurement.timestamp_ns, 1234567890)
        np.testing.assert_array_equal(imu_measurement.linear_accelerations, [1, 2, 3])
        np.testing.assert_array_equal(imu_measurement.angular_velocities, np.array([4, 5, 6]))

    def test_pose_stamped(self):
        ps = vslam.PoseStamped()
        np.testing.assert_array_equal(ps.pose.translation, [0, 0, 0])
        np.testing.assert_array_equal(ps.pose.rotation, [0, 0, 0, 1])

        ps.timestamp_ns = 999
        ps.pose = vslam.Pose([0, 0, 0.7071, 0.7071], [1, 2, 3])
        self.assertEqual(ps.timestamp_ns, 999)
        np.testing.assert_array_equal(ps.pose.translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(ps.pose.rotation, [0, 0, 0.7071, 0.7071])

    def test_pose_with_covariance(self):
        pwc = vslam.PoseWithCovariance()
        np.testing.assert_array_equal(pwc.pose.translation, [0, 0, 0])
        np.testing.assert_array_equal(pwc.pose.rotation, [0, 0, 0, 1])

        self.assertFalse(hasattr(pwc, "covariance"))

        # pose and covariance_xyz_rpy are read-only
        with self.assertRaises(AttributeError):
            pwc.pose = vslam.Pose()  # type: ignore
        with self.assertRaises(AttributeError):
            pwc.covariance_xyz_rpy = [0.0] * 36  # type: ignore

    def test_pose_estimate(self):
        pe = vslam.PoseEstimate()

        # timestamp_ns and world_from_rig are read-only
        with self.assertRaises(AttributeError):
            pe.timestamp_ns = 123  # type: ignore
        with self.assertRaises(AttributeError):
            pe.world_from_rig = None  # type: ignore

    def test_observation(self):
        obs = vslam.Observation()

        obs.id = 42
        obs.u = 100.5
        obs.v = 200.5
        obs.camera_index = 1
        self.assertEqual(obs.id, 42)
        self.assertAlmostEqual(obs.u, 100.5)
        self.assertAlmostEqual(obs.v, 200.5)
        self.assertEqual(obs.camera_index, 1)

    def test_landmark(self):
        lm = vslam.Landmark()

        lm.id = 7
        lm.coords = [1.0, 2.0, 3.0]
        self.assertEqual(lm.id, 7)
        np.testing.assert_array_equal(lm.coords, [1.0, 2.0, 3.0])

    def test_rgbd_settings_default(self):
        s = vslam.Odometry.RGBDSettings()
        self.assertEqual(s.depth_scale_factor, 1.0)
        self.assertEqual(s.depth_camera_id, -1)
        self.assertFalse(s.enable_depth_stereo_tracking)

    def test_rgbd_settings_constructor(self):
        s = vslam.Odometry.RGBDSettings(
            depth_scale_factor=5000.0,
            depth_camera_id=0,
            enable_depth_stereo_tracking=True,
        )
        self.assertEqual(s.depth_scale_factor, 5000.0)
        self.assertEqual(s.depth_camera_id, 0)
        self.assertTrue(s.enable_depth_stereo_tracking)

        # unnamed parameters are not supported (kw_only)
        with self.assertRaises(TypeError):
            vslam.Odometry.RGBDSettings(5000.0, 0, True)  # type: ignore

    def test_rgbd_settings_assignment(self):
        s = vslam.Odometry.RGBDSettings()
        s.depth_scale_factor = 5000.0
        s.depth_camera_id = 2
        s.enable_depth_stereo_tracking = True
        self.assertEqual(s.depth_scale_factor, 5000.0)
        self.assertEqual(s.depth_camera_id, 2)
        self.assertTrue(s.enable_depth_stereo_tracking)

    def test_odometry_config_constructor(self):
        # Test default constructor
        cfg = vslam.Odometry.Config()
        self.assertEqual(cfg.multicam_mode, vslam.Odometry.MulticameraMode.Precision)
        self.assertEqual(cfg.odometry_mode, vslam.Odometry.OdometryMode.Multicamera)
        self.assertTrue(cfg.use_gpu)
        self.assertTrue(cfg.async_sba)
        self.assertTrue(cfg.use_motion_model)
        self.assertFalse(cfg.use_denoising)
        self.assertFalse(cfg.rectified_stereo_camera)
        self.assertTrue(cfg.enable_observations_export)
        self.assertTrue(cfg.enable_landmarks_export)
        self.assertFalse(cfg.enable_final_landmarks_export)
        self.assertEqual(cfg.max_frame_delta_s, 1.0)
        self.assertEqual(cfg.debug_dump_directory, "")
        self.assertFalse(cfg.debug_imu_mode)

        # Test constructor with keyword arguments
        # Adjacent bool params must alternate True/False to catch any ordering bugs
        cfg = vslam.Odometry.Config(
            multicam_mode=vslam.Odometry.MulticameraMode.Performance,
            odometry_mode=vslam.Odometry.OdometryMode.Inertial,
            use_gpu=True,
            async_sba=False,
            use_motion_model=True,
            use_denoising=False,
            rectified_stereo_camera=True,
            enable_observations_export=False,
            enable_landmarks_export=True,
            enable_final_landmarks_export=False,
            max_frame_delta_s=0.5,
            debug_dump_directory="/tmp/debug",
            debug_imu_mode=True,
        )
        self.assertEqual(cfg.multicam_mode, vslam.Odometry.MulticameraMode.Performance)
        self.assertEqual(cfg.odometry_mode, vslam.Odometry.OdometryMode.Inertial)
        self.assertTrue(cfg.use_gpu)
        self.assertFalse(cfg.async_sba)
        self.assertTrue(cfg.use_motion_model)
        self.assertFalse(cfg.use_denoising)
        self.assertTrue(cfg.rectified_stereo_camera)
        self.assertFalse(cfg.enable_observations_export)
        self.assertTrue(cfg.enable_landmarks_export)
        self.assertFalse(cfg.enable_final_landmarks_export)
        self.assertEqual(cfg.max_frame_delta_s, 0.5)
        self.assertEqual(cfg.debug_dump_directory, "/tmp/debug")
        self.assertTrue(cfg.debug_imu_mode)

        # test rgbd_settings passthrough
        cfg2 = vslam.Odometry.Config(
            rgbd_settings=vslam.Odometry.RGBDSettings(
                depth_scale_factor=5000.0,
                depth_camera_id=0,
                enable_depth_stereo_tracking=True,
            ))
        self.assertEqual(cfg2.rgbd_settings.depth_scale_factor, 5000.0)
        self.assertEqual(cfg2.rgbd_settings.depth_camera_id, 0)
        self.assertTrue(cfg2.rgbd_settings.enable_depth_stereo_tracking)

        # unnamed parameters are not supported (kw_only)
        with self.assertRaises(TypeError):
            vslam.Odometry.Config(
                vslam.Odometry.MulticameraMode.Performance,  # type: ignore
                vslam.Odometry.OdometryMode.Inertial)

    def test_odometry_config_modifiers(self):
        cfg = vslam.Odometry.Config()
        cfg.multicam_mode = vslam.Odometry.MulticameraMode.Moderate
        cfg.odometry_mode = vslam.Odometry.OdometryMode.Mono
        cfg.use_gpu = False
        cfg.async_sba = False
        cfg.use_motion_model = False
        cfg.use_denoising = True
        cfg.rectified_stereo_camera = True
        cfg.enable_observations_export = True
        cfg.enable_landmarks_export = True
        cfg.enable_final_landmarks_export = True
        cfg.max_frame_delta_s = 2.0
        cfg.debug_dump_directory = "/tmp/test"
        cfg.debug_imu_mode = True
        cfg.rgbd_settings = vslam.Odometry.RGBDSettings(
            depth_scale_factor=5000.0, depth_camera_id=0, enable_depth_stereo_tracking=True)

        self.assertEqual(cfg.multicam_mode, vslam.Odometry.MulticameraMode.Moderate)
        self.assertEqual(cfg.odometry_mode, vslam.Odometry.OdometryMode.Mono)
        self.assertFalse(cfg.use_gpu)
        self.assertFalse(cfg.async_sba)
        self.assertFalse(cfg.use_motion_model)
        self.assertTrue(cfg.use_denoising)
        self.assertTrue(cfg.rectified_stereo_camera)
        self.assertTrue(cfg.enable_observations_export)
        self.assertTrue(cfg.enable_landmarks_export)
        self.assertTrue(cfg.enable_final_landmarks_export)
        self.assertEqual(cfg.max_frame_delta_s, 2.0)
        self.assertEqual(cfg.debug_dump_directory, "/tmp/test")
        self.assertTrue(cfg.debug_imu_mode)
        self.assertEqual(cfg.rgbd_settings.depth_scale_factor, 5000.0)
        self.assertEqual(cfg.rgbd_settings.depth_camera_id, 0)
        self.assertTrue(cfg.rgbd_settings.enable_depth_stereo_tracking)

    def test_multisensor_settings_default(self):
        s = vslam.Odometry.MultisensorSettings()
        self.assertEqual(list(s.depth_camera_ids), [])
        self.assertEqual(s.depth_scale_factor, 1.0)
        # Multisensor defaults enable_depth_stereo_tracking=True (multisensor mode benefits
        # from cross-camera 2D tracks; matches the launcher's auto-on behavior).
        self.assertTrue(s.enable_depth_stereo_tracking)

    def test_multisensor_settings_constructor(self):
        s = vslam.Odometry.MultisensorSettings(
            depth_camera_ids=[0, 2],
            depth_scale_factor=5000.0,
            enable_depth_stereo_tracking=False,
        )
        self.assertEqual(list(s.depth_camera_ids), [0, 2])
        self.assertEqual(s.depth_scale_factor, 5000.0)
        self.assertFalse(s.enable_depth_stereo_tracking)

        # kw_only
        with self.assertRaises(TypeError):
            vslam.Odometry.MultisensorSettings([0], 1.0, False)  # type: ignore

    def test_multisensor_settings_assignment(self):
        s = vslam.Odometry.MultisensorSettings()
        s.depth_camera_ids = [1, 3]
        s.depth_scale_factor = 1000.0
        s.enable_depth_stereo_tracking = False
        self.assertEqual(list(s.depth_camera_ids), [1, 3])
        self.assertEqual(s.depth_scale_factor, 1000.0)
        self.assertFalse(s.enable_depth_stereo_tracking)

    def test_multisensor_mode_in_enum(self):
        # The Multisensor odometry mode must be exposed and selectable.
        self.assertTrue(hasattr(vslam.Odometry.OdometryMode, 'Multisensor'))
        cfg = vslam.Odometry.Config(
            odometry_mode=vslam.Odometry.OdometryMode.Multisensor,
            multisensor_settings=vslam.Odometry.MultisensorSettings(
                depth_camera_ids=[0, 1], depth_scale_factor=2.0,
                enable_depth_stereo_tracking=True))
        self.assertEqual(cfg.odometry_mode, vslam.Odometry.OdometryMode.Multisensor)
        self.assertEqual(list(cfg.multisensor_settings.depth_camera_ids), [0, 1])
        self.assertEqual(cfg.multisensor_settings.depth_scale_factor, 2.0)
        self.assertTrue(cfg.multisensor_settings.enable_depth_stereo_tracking)

    def test_set_verbosity(self):
        # Smoke test — should not raise
        vslam.set_verbosity(0)
        vslam.set_verbosity(3)
        vslam.set_verbosity(0)

    def test_warm_up_gpu(self):
        # Smoke test — should not raise
        vslam.warm_up_gpu()

    def test_slam_config_default(self):
        cfg = vslam.Slam.Config()
        self.assertEqual(cfg.map_cache_path, "")
        self.assertTrue(cfg.use_gpu)
        self.assertFalse(cfg.sync_mode)
        self.assertTrue(cfg.enable_reading_internals)
        self.assertFalse(cfg.planar_constraints)
        self.assertFalse(cfg.gt_align_mode)
        self.assertEqual(cfg.map_cell_size, 0.0)
        self.assertEqual(cfg.max_landmarks_distance, 100.0)
        self.assertEqual(cfg.max_map_size, 300)
        self.assertEqual(cfg.throttling_time_ms, 0)

    def test_slam_config_constructor(self):
        # Adjacent bools alternate to catch ordering bugs
        cfg = vslam.Slam.Config(
            map_cache_path="/tmp/cache",
            use_gpu=False,
            sync_mode=True,
            enable_reading_internals=False,
            planar_constraints=True,
            gt_align_mode=False,
            map_cell_size=1.5,
            max_landmarks_distance=50.0,
            max_map_size=500,
            throttling_time_ms=100,
        )
        self.assertEqual(cfg.map_cache_path, "/tmp/cache")
        self.assertFalse(cfg.use_gpu)
        self.assertTrue(cfg.sync_mode)
        self.assertFalse(cfg.enable_reading_internals)
        self.assertTrue(cfg.planar_constraints)
        self.assertFalse(cfg.gt_align_mode)
        self.assertEqual(cfg.map_cell_size, 1.5)
        self.assertEqual(cfg.max_landmarks_distance, 50.0)
        self.assertEqual(cfg.max_map_size, 500)
        self.assertEqual(cfg.throttling_time_ms, 100)

        # unnamed parameters are not supported (kw_only)
        with self.assertRaises(TypeError):
            vslam.Slam.Config("/tmp/cache", False, True)  # type: ignore

    def test_slam_config_modifiers(self):
        cfg = vslam.Slam.Config()
        cfg.map_cache_path = "/tmp/test"
        cfg.use_gpu = False
        cfg.sync_mode = True
        cfg.enable_reading_internals = False
        cfg.planar_constraints = True
        cfg.gt_align_mode = True
        cfg.map_cell_size = 2.0
        cfg.max_landmarks_distance = 25.0
        cfg.max_map_size = 1000
        cfg.throttling_time_ms = 500

        self.assertEqual(cfg.map_cache_path, "/tmp/test")
        self.assertFalse(cfg.use_gpu)
        self.assertTrue(cfg.sync_mode)
        self.assertFalse(cfg.enable_reading_internals)
        self.assertTrue(cfg.planar_constraints)
        self.assertTrue(cfg.gt_align_mode)
        self.assertEqual(cfg.map_cell_size, 2.0)
        self.assertEqual(cfg.max_landmarks_distance, 25.0)
        self.assertEqual(cfg.max_map_size, 1000)
        self.assertEqual(cfg.throttling_time_ms, 500)

    def test_slam_localization_settings(self):
        s = vslam.Slam.LocalizationSettings(
            horizontal_search_radius=1.0,
            vertical_search_radius=0.5,
            horizontal_step=0.25,
            vertical_step=0.125,
            angular_step_rads=0.1,
        )
        self.assertEqual(s.horizontal_search_radius, 1.0)
        self.assertEqual(s.vertical_search_radius, 0.5)
        self.assertEqual(s.horizontal_step, 0.25)
        self.assertEqual(s.vertical_step, 0.125)
        self.assertAlmostEqual(s.angular_step_rads, 0.1, places=5)

        # unnamed parameters are not supported (kw_only)
        with self.assertRaises(TypeError):
            vslam.Slam.LocalizationSettings(1.0, 0.5, 0.25, 0.125, 0.1)  # type: ignore

    def test_slam_data_layer_enum(self):
        # Verify all exposed enum values exist
        self.assertIsNotNone(vslam.Slam.DataLayer.Landmarks)
        self.assertIsNotNone(vslam.Slam.DataLayer.Map)
        self.assertIsNotNone(vslam.Slam.DataLayer.LoopClosure)

        # Values should be distinct
        values = [
            vslam.Slam.DataLayer.Landmarks,
            vslam.Slam.DataLayer.Map,
            vslam.Slam.DataLayer.LoopClosure,
        ]
        self.assertEqual(len(values), len(set(values)))

    def test_tracker_mode_enum(self):
        # Verify all exposed enum values exist
        self.assertIsNotNone(vslam.Tracker.Mode.OdometryOnlyRealtime)
        self.assertIsNotNone(vslam.Tracker.Mode.OdometryOnlyOffline)
        self.assertIsNotNone(vslam.Tracker.Mode.OdometryWithSlamRealtime)
        self.assertIsNotNone(vslam.Tracker.Mode.OdometryWithSlamOffline)

        # Values should be distinct
        values = [
            vslam.Tracker.Mode.OdometryOnlyRealtime,
            vslam.Tracker.Mode.OdometryOnlyOffline,
            vslam.Tracker.Mode.OdometryWithSlamRealtime,
            vslam.Tracker.Mode.OdometryWithSlamOffline,
        ]
        self.assertEqual(len(values), len(set(values)))


if __name__ == "__main__":
    unittest.main()

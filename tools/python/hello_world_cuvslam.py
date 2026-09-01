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

import numpy as np

import cuvslam as vslam

# The Rig defines the sensors used by cuvslam, consisting of two arrays, one for cameras and one
# for IMUs.
cameras = []
# Create a synthetic stereo rig with two cameras, each with a 320x320 focal length and 640x480 image
# size. The principal point is at (320, 240) and the baseline is 0.25m.
for i in range(2):
    cam = vslam.Camera(size=(640, 480), focal=(320.0, 320.0), principal=(320.0, 240.0))
    # Right camera is 0.25m to the right
    if i == 1:
        cam.rig_from_camera.translation[0] = 0.25
    cameras.append(cam)
rig = vslam.Rig(cameras=cameras, imus=[])
# Create a tracker with horizontal (rectified) stereo camera; SLAM is disabled.
odom_cfg = vslam.Odometry.Config(rectified_stereo_camera=True)
tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryOnlyRealtime, odom_cfg)

print("cuVSLAM version:", vslam.get_version())
print("Odometry.Config:")
for attr in dir(cfg):
    if not callable(getattr(cfg, attr)) and not attr.startswith("__"):
        print(f"  {attr}: {getattr(cfg, attr)}")

# Create a synthetic stereo pair of RGB images as numpy arrays.
# cuvslam will also work with pytorch tensors or other frameworks that support dlpack.
synthetic_images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(2)]
for i in range(10):
    pose_estimate, _ = tracker.track(int(i), synthetic_images) # ignore slam pose
    if pose_estimate.world_from_rig:
        rotation_quat = pose_estimate.world_from_rig.pose.rotation
        translation_vector = pose_estimate.world_from_rig.pose.translation
        print(f"{i}: T={translation_vector}, R={rotation_quat}")
    else:
        print(f"{i}: Tracking lost")

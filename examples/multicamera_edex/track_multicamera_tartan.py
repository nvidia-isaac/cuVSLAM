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

import cuvslam as vslam
import os
import json
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dataset_utils import read_stereo_edex
from PIL import Image

# generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier): return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]

CAMERA_LIST = ['lcam_front', 'rcam_front', 'lcam_back', 'rcam_back', 'lcam_left', 'rcam_left', 'lcam_right',  'rcam_right', 'lcam_top', 'rcam_top', 'lcam_bottom', 'rcam_bottom']

data_path = 'dataset/tartan_ground/OldTownFall/Data_anymal/P2000'
num_frames = len(os.listdir(os.path.join(data_path, f'image_{CAMERA_LIST[0]}')))


### setup rerun visualizer
rr.init('tartan_ground', strict=True, spawn=True)  # launch re-run instance
# setup rerun views
rr.send_blueprint(rrb.Blueprint(rrb.TimePanel(state="collapsed"),
                                rrb.Vertical(
                                    row_shares=[0.25, 0.5, 0.25],
                                    contents=[
                                        rrb.Horizontal(
                                            contents=[rrb.Spatial2DView(origin='car/cam0', name='front-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam1', name='front-stereo_right'),
                                                      rrb.Spatial2DView(origin='car/cam2', name='back-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam3',  name='back-stereo_right'),
                                                      rrb.Spatial2DView(origin='car/cam8', name='top-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam9', name='top-stereo_right'),]),
                                        rrb.Spatial3DView(
                                            name="3D",
                                            contents=[
                                                '+ /**',
                                                '- /car/cam1/**',
                                                '- /car/cam3/**',
                                                '- /car/cam5/**',
                                                '- /car/cam7/**',
                                                '- /car/cam9/**',
                                                '- /car/cam11/**',
                                            ],
                                            defaults=[rr.Pinhole.from_fields(image_plane_distance=0.5)]
                                        ),
                                        rrb.Horizontal(
                                            contents=[rrb.Spatial2DView(origin='car/cam4', name='left-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam5', name='left-stereo_right'),
                                                      rrb.Spatial2DView(origin='car/cam6', name='right-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam7',  name='right-stereo_right'),
                                                      rrb.Spatial2DView(origin='car/cam10',  name='bottom-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam11',  name='bottom-stereo_right')])
                                        ]
                                    ),
                                ),
                            make_active=True)
# setup coordinate basis for root, cuvslam uses right-hand system with  X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Load camera configuration from EDEX file
cameras = read_stereo_edex('tartan_ground.edex')

for i, camera in enumerate(cameras):
    rr.log('car/cam%s' % i,
           rr.Transform3D(translation=camera.rig_from_camera.translation,
                          rotation=rr.Quaternion(xyzw=camera.rig_from_camera.rotation),
                          relation=rr.TransformRelation.ParentFromChild),
           rr.Pinhole(image_plane_distance=1.,
                      image_from_camera=np.array([[camera.focal[0], 0, camera.principal[0]],
                                                  [0, camera.focal[1], camera.principal[1]],
                                                  [0, 0, 1]]),
                      width=camera.size[0], height=camera.size[1]),
           static=True)

# Set up VSLAM rig and tracker
rig = vslam.Rig()
rig.cameras = cameras

odom_cfg = vslam.Odometry.Config(enable_final_landmarks_export = True, rectified_stereo_camera=True)

tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryOnlyRealtime, odom_cfg)

trajectory = []

# Process each frame
for frame_id in range(num_frames):
    timestamp = frame_id
    try:
        images = [np.asarray(Image.open(os.path.join(data_path, f'image_{cam}', f'{frame_id:06d}_{cam}.png')))
                  for cam in CAMERA_LIST]
    except FileNotFoundError as e:
        print(f"Error: Missing image file for frame {frame_id}: {e}")
        continue
    # do multicamera visual tracking
    odom_pose_estimate, _ = tracker.track(timestamp, images)

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame_id}")
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odom_pose_estimate.world_from_rig.pose

    # get visualization data
    observations = [tracker.odometry.get_last_observations(i) for i in range(len(CAMERA_LIST))]
    landmarks = tracker.odometry.get_last_landmarks()
    final_landmarks = tracker.odometry.get_final_landmarks()
    # prepare visualization data
    observations_uv = [[[o.u, o.v] for o in obs_instance] for obs_instance in observations]
    observations_colors = [[color_from_id(o.id) for o in obs_instance] for obs_instance in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]
    trajectory.append(odom_pose.translation)
    # send results to rerun for visualization
    rr.set_time('frame', sequence=frame_id)
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('final_landmarks', rr.Points3D(list(final_landmarks.values()), radii=0.01))
    rr.log('car', rr.Transform3D(translation=odom_pose.translation, quaternion=odom_pose.rotation))
    rr.log('car/body', rr.Boxes3D(centers=[0, 0.3 / 2, 0], sizes=[[0.35, 0.3, 0.66]]))
    rr.log('car/landmarks_center', rr.Points3D(landmark_xyz, radii=0.02, colors=landmarks_colors))

    for i in range(len(cameras)):
        rr.log('car/cam%s/image' % i, rr.Image(images[i]).compress(jpeg_quality=80))
        rr.log('car/cam%s/observations' % i, rr.Points2D(observations_uv[i], radii=5, colors=observations_colors[i]))

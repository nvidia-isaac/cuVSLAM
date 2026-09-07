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

import json
import os.path


def fix_distortion(version, intrinsics):
    # Always leaves an edex model name here, the ROS name is applied when the CameraInfo is built.
    if version == "0.8":
        num_params = len(intrinsics['distortion'])
        intrinsics['distortion_model'] = 'fisheye' if num_params == 4 else 'brown5k'
        if num_params == 2:
            intrinsics['distortion_params'] = [0, 0, 0, 0, 0]
        else:
            intrinsics['distortion_params'] = intrinsics['distortion']
        return intrinsics
    if version == "0.9":
        return intrinsics
    raise ValueError('Unexpected version (%s)' % version)


def load_imu_measurements(imu_file):
    measurements = []
    with open(imu_file, 'r') as f:
        json_lines = f.readlines()
        for json_line in json_lines:
            measurements.append(json.loads(json_line))
    return measurements


class EdexReader:
    def __init__(self, filename):
        array = json.load(open(filename, 'r'))
        header = array[0]
        body = array[1]
        sequences = body['sequence']
        cameras = header['cameras']
        version = header['version']
        self.image_masks = [sequences[0][0], sequences[1][0]]
        self.fps = body['fps']
        self.frame_start = header['frame_start']
        self.frame_end = header['frame_end']
        self.current_frame = self.frame_start
        self.transforms = [cameras[0]['transform'], cameras[1]['transform']]
        self.intrinsics = [fix_distortion(version, cameras[0]['intrinsics']),
                           fix_distortion(version, cameras[1]['intrinsics'])]
        self.edex_dir = os.path.dirname(filename)
        if 'imu' in header:
            self.imu = header['imu']
            imu_file = self.imu['measurements']
            self.imu_measurements = load_imu_measurements(os.path.join(self.edex_dir, imu_file))
        else:
            self.imu = None

    def read_frame(self):
        delta_in_ns = int(1000000000 / self.fps)
        timestamp = delta_in_ns * self.current_frame
        frame_l = self._image_filename(self.image_masks[0])
        frame_r = self._image_filename(self.image_masks[1])
        self.current_frame = self.current_frame + 1
        return timestamp, [frame_l, frame_r]

    def end(self):
        return self.current_frame > self.frame_end

    def _image_filename(self, mask):
        padding = 4
        ext = '.png'
        prefix = mask[:-(len(ext) + padding)]
        filename = prefix + str(self.current_frame).zfill(padding) + ext
        return os.path.join(self.edex_dir, filename)

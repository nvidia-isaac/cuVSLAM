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

"""The depth configuration an EDEX declares, and the rules for reading it.

Deliberately free of ``import cuvslam``: the tools test suite installs this
package without the binding, so keeping these rules here is what makes them
testable. ``edex_reader`` turns a ``DepthDescription`` into the binding's
``RGBDSettings`` or ``MultisensorSettings``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

# Depth stored as .npy holds meters, which the reader converts to millimeters on
# load, so the scale the EDEX declares does not describe what cuVSLAM receives.
NPY_DEPTH_SCALE_FACTOR = 1000.0


@dataclass(frozen=True)
class DepthDescription:
    """Which rig cameras supply depth images, and how to interpret them."""

    camera_ids: tuple[int, ...]
    scale_factor: float
    enable_depth_stereo_tracking: bool


def parse_depth_description(config_data) -> DepthDescription:
    """Collect every depth-capable camera and the shared depth scale from stereo.edex.

    Args:
        config_data: Configuration data loaded from stereo.edex

    Returns:
        DepthDescription, whose camera_ids is empty when no camera supplies depth

    Raises:
        ValueError: If config_data structure is invalid, or if the depth cameras
            disagree on depth_scale_factor. cuVSLAM applies one scale to every
            depth camera, so picking one of several would misread the others.
    """
    if not isinstance(config_data, list):
        raise ValueError("config_data must be a list")

    if len(config_data) < 2:
        raise ValueError("config_data must have at least 2 elements (rig config and metadata)")

    if not isinstance(config_data[0], dict):
        raise ValueError("config_data[0] (rig configuration) must be a dictionary")

    if not isinstance(config_data[1], dict):
        raise ValueError("config_data[1] (metadata) must be a dictionary")

    if 'cameras' not in config_data[0]:
        raise ValueError("'cameras' key not found in rig configuration (config_data[0])")

    cameras = config_data[0]['cameras']
    if not isinstance(cameras, list):
        raise ValueError("'cameras' must be a list")

    if len(cameras) == 0:
        raise ValueError("'cameras' list is empty")

    camera_ids = []
    scale_by_camera = {}
    for cam in cameras:
        if not isinstance(cam, dict) or 'depth_id' not in cam:
            continue  # Skip invalid or depth-less camera entries

        try:
            camera_id = int(cam['depth_id'])
        except (ValueError, TypeError):
            print(f"Warning: Invalid depth_id value: {cam['depth_id']}")
            continue

        camera_ids.append(camera_id)
        try:
            scale_by_camera[camera_id] = float(cam.get('depth_scale_factor', 1.0))
        except (ValueError, TypeError):
            print("Warning: Invalid depth_scale_factor value, using default 1.0")
            scale_by_camera[camera_id] = 1.0

    distinct_scales = set(scale_by_camera.values())
    if len(distinct_scales) > 1:
        raise ValueError(
            f"depth cameras declare different depth_scale_factor values ({scale_by_camera}); "
            "cuVSLAM applies a single scale to every depth camera"
        )
    scale_factor = next(iter(distinct_scales), 1.0)

    depth_sequence = config_data[1].get('depth_sequence')
    if isinstance(depth_sequence, list) and any(
        isinstance(paths, list) and paths
        and isinstance(paths[0], str) and paths[0].endswith('.npy')
        for paths in depth_sequence
    ):
        scale_factor = NPY_DEPTH_SCALE_FACTOR

    return DepthDescription(
        camera_ids=tuple(camera_ids),
        scale_factor=scale_factor,
        enable_depth_stereo_tracking=config_data[0].get('enable_depth_stereo_tracking', False),
    )


def single_depth_camera_id(description: DepthDescription) -> int:
    """Return the one depth camera RGBD mode tracks.

    Raises:
        ValueError: If the rig declares no depth camera, or more than one.
    """
    if not description.camera_ids:
        raise ValueError(
            "RGBD mode is enabled but 'depth_id' not found in camera config in stereo.edex. "
            "Please add 'depth_id' field to one of the cameras in the configuration."
        )

    if len(description.camera_ids) > 1:
        raise ValueError(
            f"RGBD mode tracks a single RGB-D camera, but stereo.edex declares depth on cameras "
            f"{list(description.camera_ids)}. Use --odometry_mode=multisensor for a rig with several."
        )

    return description.camera_ids[0]


def remap_depth_camera_ids(
    camera_ids: Sequence[int],
    camera_id_map: Optional[Mapping[int, int]],
    requested_camera_ids: Optional[Sequence[int]] = None,
) -> list[int]:
    """Move depth camera ids into a filtered rig's contiguous camera-id space.

    Raises:
        ValueError: If a depth camera was filtered out of the rig, which would
            otherwise leave the settings pointing at an unrelated camera.
    """
    if camera_id_map is None:
        return list(camera_ids)

    remapped = []
    for camera_id in camera_ids:
        if camera_id not in camera_id_map:
            raise ValueError(
                f"depth camera id {camera_id} is not included in camera_ids {requested_camera_ids}."
            )
        remapped.append(camera_id_map[camera_id])
    return remapped

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

"""
Python helper functions for loading cuVSLAM configuration from YAML files.

These are intentionally separate from the core C++ bindings (pycuvslam) to keep
file I/O out of the libcuvslam binary. Use these helpers in tools and scripts;
core tracking code should receive fully-constructed Config objects directly.
"""

import warnings
from typing import Optional
import yaml

from ._backend import load_backend

_core = load_backend()
Odometry = _core.Odometry
Slam = _core.Slam


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).lower()
    if s == "true":
        return True
    if s == "false":
        return False
    raise ValueError(f"expected bool, got: {value!r}")


def _apply_odometry_section(config: Odometry.Config, section: dict) -> None:
    """Apply a flat key/value dict (from the YAML 'odometry' section) to an Odometry.Config."""
    for key, value in section.items():
        if key == "multicam_mode":
            mapping = {
                "performance": Odometry.MulticameraMode.Performance,
                "precision":   Odometry.MulticameraMode.Precision,
                "moderate":    Odometry.MulticameraMode.Moderate,
            }
            normalized = str(value).lower()
            resolved = mapping.get(normalized)
            if resolved is None:
                raise ValueError(
                    f"unknown multicam_mode: {value!r}; allowed values: {list(mapping.keys())}")
            config.multicam_mode = resolved
        elif key == "odometry_mode":
            mapping = {
                "multicamera": Odometry.OdometryMode.Multicamera,
                "inertial":    Odometry.OdometryMode.Inertial,
                "rgbd":        Odometry.OdometryMode.RGBD,
                "mono":        Odometry.OdometryMode.Mono,
                "multisensor": Odometry.OdometryMode.Multisensor,
            }
            normalized = str(value).lower()
            resolved = mapping.get(normalized)
            if resolved is None:
                raise ValueError(
                    f"unknown odometry_mode: {value!r}; allowed values: {list(mapping.keys())}")
            config.odometry_mode = resolved
        elif key == "use_gpu":
            config.use_gpu = _parse_bool(value)
        elif key == "async_sba":
            config.async_sba = _parse_bool(value)
        elif key == "use_motion_model":
            config.use_motion_model = _parse_bool(value)
        elif key == "use_denoising":
            config.use_denoising = _parse_bool(value)
        elif key == "rectified_stereo_camera":
            config.rectified_stereo_camera = _parse_bool(value)
        elif key == "enable_observations_export":
            config.enable_observations_export = _parse_bool(value)
        elif key == "enable_landmarks_export":
            config.enable_landmarks_export = _parse_bool(value)
        elif key == "enable_final_landmarks_export":
            config.enable_final_landmarks_export = _parse_bool(value)
        elif key == "max_frame_delta_s":
            config.max_frame_delta_s = float(value)
        elif key == "debug_imu_mode":
            config.debug_imu_mode = _parse_bool(value)
        elif key == "rgbd_settings":
            if isinstance(value, dict):
                if "depth_scale_factor" in value:
                    config.rgbd_settings.depth_scale_factor = float(value["depth_scale_factor"])
                if "depth_camera_id" in value:
                    config.rgbd_settings.depth_camera_id = int(value["depth_camera_id"])
                if "enable_depth_stereo_tracking" in value:
                    config.rgbd_settings.enable_depth_stereo_tracking = _parse_bool(
                        value["enable_depth_stereo_tracking"])
        elif key == "multisensor_settings":
            if isinstance(value, dict):
                if "depth_camera_ids" in value:
                    depth_camera_ids = value["depth_camera_ids"]
                    if not isinstance(depth_camera_ids, list):
                        raise ValueError("multisensor_settings.depth_camera_ids must be a list")
                    config.multisensor_settings.depth_camera_ids = [
                        int(camera_id) for camera_id in depth_camera_ids
                    ]
                if "depth_scale_factor" in value:
                    config.multisensor_settings.depth_scale_factor = float(value["depth_scale_factor"])
                if "enable_depth_stereo_tracking" in value:
                    config.multisensor_settings.enable_depth_stereo_tracking = _parse_bool(
                        value["enable_depth_stereo_tracking"])
        elif key == "debug_dump_directory":
            warnings.warn("'debug_dump_directory' cannot be loaded from YAML (string does not persist); "
                          "set it directly on the config object after loading.", stacklevel=2)
        else:
            warnings.warn(f"try_load_odometry_config_from_file: unknown key {key!r} (ignored)", stacklevel=2)


def _apply_slam_section(config: Slam.Config, section: dict) -> None:
    """Apply a flat key/value dict (from the YAML 'slam' section) to a Slam.Config."""
    for key, value in section.items():
        if key == "use_gpu":
            config.use_gpu = _parse_bool(value)
        elif key == "sync_mode":
            config.sync_mode = _parse_bool(value)
        elif key == "enable_reading_internals":
            config.enable_reading_internals = _parse_bool(value)
        elif key == "planar_constraints":
            config.planar_constraints = _parse_bool(value)
        elif key == "gt_align_mode":
            config.gt_align_mode = _parse_bool(value)
        elif key == "map_cell_size":
            config.map_cell_size = float(value)
        elif key == "max_landmarks_distance":
            config.max_landmarks_distance = float(value)
        elif key == "max_map_size":
            config.max_map_size = int(value)
        elif key == "throttling_time_ms":
            config.throttling_time_ms = int(value)
        elif key == "map_cache_path":
            warnings.warn("'map_cache_path' cannot be loaded from YAML (string does not persist); "
                          "set it directly on the config object after loading.", stacklevel=2)
        else:
            warnings.warn(f"try_load_slam_config_from_file: unknown key {key!r} (ignored)", stacklevel=2)


def try_load_odometry_config_from_file(filepath: str) -> Optional[Odometry.Config]:
    """Load an Odometry.Config from the ``odometry:`` section of a YAML file.

    Returns:
        Configured ``Odometry.Config``, or ``None`` if the file has no ``odometry`` section.
    Raises:
        RuntimeError: If the file cannot be opened or parsed.
    """
    try:
        with open(filepath) as f:
            root = yaml.safe_load(f)
    except OSError as e:
        raise RuntimeError(f"cannot open file '{filepath}': {e}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(f"failed to parse YAML file '{filepath}': {e}") from e

    if not isinstance(root, dict) or "odometry" not in root:
        return None

    config = Odometry.Config()
    _apply_odometry_section(config, root["odometry"])
    return config


def load_odometry_config_from_file(filepath: str) -> Odometry.Config:
    """Load an Odometry.Config from the ``odometry:`` section of a YAML file.

    Raises:
        RuntimeError: If the file cannot be opened, parsed, or has no ``odometry`` section.
    """
    config = try_load_odometry_config_from_file(filepath)
    if config is None:
        raise RuntimeError(f"No 'odometry' section found in config file: {filepath}")
    return config


def try_load_slam_config_from_file(filepath: str) -> Optional[Slam.Config]:
    """Load a Slam.Config from the ``slam:`` section of a YAML file.

    Returns:
        Configured ``Slam.Config``, or ``None`` if the file has no ``slam`` section.
    Raises:
        RuntimeError: If the file cannot be opened or parsed.
    """
    try:
        with open(filepath) as f:
            root = yaml.safe_load(f)
    except OSError as e:
        raise RuntimeError(f"cannot open file '{filepath}': {e}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(f"failed to parse YAML file '{filepath}': {e}") from e

    if not isinstance(root, dict) or "slam" not in root:
        return None

    config = Slam.Config()
    _apply_slam_section(config, root["slam"])
    return config


def load_slam_config_from_file(filepath: str) -> Slam.Config:
    """Load a Slam.Config from the ``slam:`` section of a YAML file.

    Raises:
        RuntimeError: If the file cannot be opened, parsed, or has no ``slam`` section.
    """
    config = try_load_slam_config_from_file(filepath)
    if config is None:
        raise RuntimeError(f"No 'slam' section found in config file: {filepath}")
    return config


def _apply_internals_section(options: Odometry.Internals, section: dict) -> None:
    """Apply a key/value dict (from the YAML 'internals' section) to a Odometry.Internals."""
    for key, value in section.items():
        if key == "num_desired_tracks":
            options.num_desired_tracks = int(value)
        elif key == "border_top":
            options.border_top = int(value)
        elif key == "border_bottom":
            options.border_bottom = int(value)
        elif key == "border_left":
            options.border_left = int(value)
        elif key == "border_right":
            options.border_right = int(value)
        elif key == "box3_prefilter":
            options.box3_prefilter = _parse_bool(value)
        elif key == "ransac_filter":
            options.ransac_filter = _parse_bool(value)
        elif key == "kf_survivor_from_last":
            options.kf_survivor_from_last = float(value)
        elif key in ("kf_max_timedelta_between_kfs_s", "kf_max_timedelta_s"):
            options.kf_max_timedelta_between_kfs_s = round(float(value))
        elif key == "kf_override_frame_selection":
            options.kf_override_frame_selection = None if value is None else _parse_bool(value)
        elif key == "vo_pnp_lambda":
            options.vo_pnp_lambda = float(value)
        elif key == "vo_pnp_huber":
            options.vo_pnp_huber = float(value)
        elif key == "vo_pnp_max_iteration":
            options.vo_pnp_max_iteration = int(value)
        elif key == "vo_pnp_recalculate_cov":
            options.vo_pnp_recalculate_cov = _parse_bool(value)
        elif key == "vo_pnp_filter_new_observations":
            options.vo_pnp_filter_new_observations = _parse_bool(value)
        elif key == "vo_pnp_max_obs_per_camera":
            options.vo_pnp_max_obs_per_camera = int(value)
        elif key == "vo_pnp_point_z_thresh":
            options.vo_pnp_point_z_thresh = float(value)
        elif key == "vo_pnp_min_observations":
            options.vo_pnp_min_observations = int(value)
        elif key == "vo_pnp_cost_thresh":
            options.vo_pnp_cost_thresh = float(value)
        elif key == "inertial_stereo_pnp_lambda":
            options.inertial_stereo_pnp_lambda = float(value)
        elif key == "inertial_stereo_pnp_huber":
            options.inertial_stereo_pnp_huber = float(value)
        elif key == "inertial_stereo_pnp_max_iteration":
            options.inertial_stereo_pnp_max_iteration = int(value)
        elif key == "inertial_stereo_pnp_recalculate_cov":
            options.inertial_stereo_pnp_recalculate_cov = _parse_bool(value)
        elif key == "inertial_stereo_pnp_filter_new_observations":
            options.inertial_stereo_pnp_filter_new_observations = _parse_bool(value)
        elif key == "inertial_stereo_pnp_max_obs_per_camera":
            options.inertial_stereo_pnp_max_obs_per_camera = int(value)
        elif key == "inertial_stereo_pnp_point_z_thresh":
            options.inertial_stereo_pnp_point_z_thresh = float(value)
        elif key == "inertial_stereo_pnp_min_observations":
            options.inertial_stereo_pnp_min_observations = int(value)
        elif key == "inertial_stereo_pnp_cost_thresh":
            options.inertial_stereo_pnp_cost_thresh = float(value)
        else:
            warnings.warn(f"try_load_internals_from_file: unknown key {key!r} (ignored)", stacklevel=2)


def try_load_internals_from_file(filepath: str) -> Optional[Odometry.Internals]:
    """Load a Internals from the ``internals:`` section of a YAML file.

    Returns:
        Configured ``Odometry.Internals``, or ``None`` if the file has no ``internals`` section.
    Raises:
        RuntimeError: If the file cannot be opened or parsed.
    """
    try:
        with open(filepath) as f:
            root = yaml.safe_load(f)
    except OSError as e:
        raise RuntimeError(f"cannot open file '{filepath}': {e}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(f"failed to parse YAML file '{filepath}': {e}") from e

    if not isinstance(root, dict):
        return None
    section = root.get("internals")
    if section is None:
        return None

    options = Odometry.Internals()
    _apply_internals_section(options, section)
    return options


def load_internals_from_file(filepath: str) -> Odometry.Internals:
    """Load a Internals from the ``internals:`` section of a YAML file.

    Raises:
        RuntimeError: If the file cannot be opened, parsed, or has no ``internals`` section.
    """
    options = try_load_internals_from_file(filepath)
    if options is None:
        raise RuntimeError(f"No 'internals' section found in config file: {filepath}")
    return options

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

"""Pydantic models and JSON helpers for reading and writing EDEX metadata."""

from enum import Enum
from functools import partial
import json
from pathlib import Path
from typing import Annotated, Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)


def to_np_array(value: list[Union[float, int]], dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert JSON numeric lists to numpy arrays with the requested dtype."""
    return np.array(value, dtype=dtype)


ArrayFloat = Annotated[
    np.ndarray, BeforeValidator(partial(to_np_array, dtype=np.float32))
]
ArrayInt = Annotated[np.ndarray, BeforeValidator(partial(to_np_array, dtype=np.int32))]


def all_close_or_none(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    """Return true when both arrays are close, or both values are None."""
    return (a is None and b is None) or (
        a is not None and b is not None and np.allclose(a, b)
    )


class EDEXEncoder(json.JSONEncoder):
    """Custom JSON encoder for EDEX files that handles numpy arrays and Path objects."""

    def default(self, obj):
        """Serialize numpy arrays and paths before falling back to JSON defaults."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class DistortionModel(str, Enum):
    """
    Supported camera distortion models in EDEX format.

    - PINHOLE: No distortion (0 parameters)
    - FISHEYE: Fisheye distortion model (k1, k2, k3, k4)
    - BROWN5K: Brown-Conrady distortion (k1, k2, k3, p1, p2)
    - POLYNOMIAL: Rational polynomial distortion (k1, k2, p1, p2, k3, k4, k5, k6)

    Only BROWN5K departs from the OpenCV coefficient order, so it is the one that needs
    reordering when reading from or writing to OpenCV and ROS calibrations.
    """

    PINHOLE = "pinhole"
    FISHEYE = "fisheye"
    BROWN5K = "brown5k"
    POLYNOMIAL = "polynomial"


class Intrinsics(BaseModel):
    """
    Camera intrinsic parameters including pinhole model and distortion.

    Attributes:
        distortion_model: Type of distortion model used
        distortion_params: Distortion coefficients (length depends on model)
        focal: Focal lengths [fx, fy] in pixels
        principal: Principal point [cx, cy] in pixels
        resolution: Image resolution [width, height] in pixels (aliased as 'size')
        projection: Optional 3x4 projection matrix [K'|t] for a stereo camera pair
        rectification: Optional 3x3 rectification matrix [R] for a stereo camera pair

    See also: https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    distortion_model: DistortionModel
    distortion_params: ArrayFloat
    focal: ArrayFloat
    principal: ArrayFloat
    resolution: ArrayInt = Field(alias="size")
    projection: Optional[ArrayFloat] = None
    rectification: Optional[ArrayFloat] = None

    @model_validator(mode="after")
    def check_fields(self):
        """Validate intrinsic array shapes for the selected distortion model."""
        # Check distortion model
        if self.distortion_model == DistortionModel.PINHOLE:
            expected_distortion_shape = (0,)
        elif self.distortion_model == DistortionModel.FISHEYE:
            expected_distortion_shape = (4,)
        elif self.distortion_model == DistortionModel.BROWN5K:
            expected_distortion_shape = (5,)
        elif self.distortion_model == DistortionModel.POLYNOMIAL:
            expected_distortion_shape = (8,)
        else:
            raise ValueError(f"Invalid distortion model: {self.distortion_model}")

        if self.distortion_params.shape != expected_distortion_shape:
            raise ValueError(f"Invalid distortion params: {self.distortion_params}")

        # Check pinhole intrinsics
        if self.focal.shape != (2,):
            raise ValueError(f"Invalid focal: {self.focal}")
        if self.principal.shape != (2,):
            raise ValueError(f"Invalid principal: {self.principal}")
        if (
            self.resolution.shape != (2,)
            or self.resolution[0] <= 0
            or self.resolution[1] <= 0
        ):
            raise ValueError(f"Invalid resolution: {self.resolution}")

        # Check stereo rectification matrices
        if self.projection is not None and self.projection.shape != (3, 4):
            raise ValueError(f"Invalid projection: {self.projection}")
        if self.rectification is not None and self.rectification.shape != (3, 3):
            raise ValueError(f"Invalid rectification: {self.rectification}")

        return self

    def __eq__(self, other: "Intrinsics") -> bool:
        """Compare intrinsics using exact enum/shape checks and tolerant floats."""
        return (
            isinstance(other, Intrinsics)
            and self.distortion_model == other.distortion_model
            and np.allclose(self.distortion_params, other.distortion_params)
            and np.allclose(self.focal, other.focal)
            and np.allclose(self.principal, other.principal)
            and np.array_equal(self.resolution, other.resolution)
            and all_close_or_none(self.projection, other.projection)
            and all_close_or_none(self.rectification, other.rectification)
        )


class Camera(BaseModel):
    """
    Camera specification including intrinsics and extrinsic transform.

    Attributes:
        intrinsics: Camera intrinsic parameters
        transform: Optional 3x4 extrinsic transformation matrix [R|t] from rig to camera frame
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    intrinsics: Intrinsics
    transform: Optional[ArrayFloat] = None

    @model_validator(mode="after")
    def check_fields(self):
        """Validate the optional rig-to-camera transform shape."""
        if self.transform is not None and self.transform.shape != (3, 4):
            raise ValueError(f"Invalid transform: {self.transform}")
        return self

    def __eq__(self, other: "Camera") -> bool:
        """Compare camera intrinsics and optional transform values."""
        return (
            isinstance(other, Camera)
            and self.intrinsics == other.intrinsics
            and all_close_or_none(self.transform, other.transform)
        )


class IMU(BaseModel):
    """
    IMU (Inertial Measurement Unit) specification and data location.

    Attributes:
        g: Gravity vector [gx, gy, gz] in m/s^2
        measurements: Path to IMU measurements file (JSONL format)
        transform: 3x4 extrinsic transform matrix [R|t] from rig to IMU frame
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    g: ArrayFloat
    measurements: Path
    transform: ArrayFloat

    @model_validator(mode="after")
    def check_fields(self):
        """Validate gravity and rig-to-IMU transform shapes."""
        if self.g.shape != (3,):
            raise ValueError(f"Invalid g: {self.g}")
        if self.transform.shape != (3, 4):
            raise ValueError(f"Invalid transform: {self.transform}")
        return self

    def __eq__(self, other: "IMU") -> bool:
        """Compare IMU calibration, measurement path, and transform values."""
        return (
            isinstance(other, IMU)
            and np.allclose(self.g, other.g)
            and self.measurements == other.measurements
            and np.allclose(self.transform, other.transform)
        )


class EDEXHeader(BaseModel):
    """
    Header section of EDEX file containing dataset metadata.

    Attributes:
        version: EDEX format version
        frame_start: Starting frame index (inclusive)
        frame_end: Ending frame index (exclusive)
        cameras: List of camera specifications in the rig
        imu: Optional IMU specification
    """

    version: str = "0.9"
    frame_start: int
    frame_end: int
    cameras: list[Camera]
    imu: Optional[IMU] = None


class EDEXBody(BaseModel):
    """
    Body section of EDEX file containing data file references.

    Attributes:
        frame_metadata: Optional path to per-frame metadata file (JSONL format)
        sequence: List of paths, or per-camera lists of paths, to frame images.
    """

    frame_metadata: Optional[Path] = None
    sequence: list[Union[Path, list[Path]]]


class EDEXMetadata:
    """
    EDEX metadata file reader/writer.

    EDEX is a format for storing multi-camera dataset metadata including
    camera intrinsics, extrinsics, IMU data, and frame sequences.

    Attributes:
        header: Dataset metadata (cameras, IMU, frame range)
        body: Data file references (images, frame metadata)
    """

    def __init__(self, header: EDEXHeader, body: EDEXBody):
        """Create metadata from parsed EDEX header and body sections."""
        self.header = header
        self.body = body

    @classmethod
    def read(cls, filename: Path) -> "EDEXMetadata":
        """Read and validate an EDEX metadata file from disk."""
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            header = EDEXHeader.model_validate(data[0])
            body = EDEXBody.model_validate(data[1])
            return cls(header, body)
        except Exception as e:
            print(f"Error reading EDEX file {filename}")
            raise e

    def write(self, filename: Path):
        """Validate and write EDEX metadata to disk."""
        try:
            # Validate the header and body before writing
            new_header = EDEXHeader.model_validate(self.header.model_dump())
            new_body = EDEXBody.model_validate(self.body.model_dump())
            # by_alias is what puts "size" in the file. Without it pydantic writes the field
            # name "resolution", which neither the C++ nor the python reader accepts.
            data = [
                new_header.model_dump(exclude_none=True, by_alias=True),
                new_body.model_dump(exclude_none=True, by_alias=True),
            ]
            with open(filename, "w") as f:
                json.dump(data, f, indent=2, cls=EDEXEncoder)
        except Exception as e:
            print(f"Error writing EDEX file {filename}")
            raise e

    def __str__(self) -> str:
        """Return a debug string containing header and body data."""
        return f"EDEXMetadata(header={self.header.model_dump()}, body={self.body.model_dump()})"

    def __eq__(self, other: "EDEXMetadata") -> bool:
        """Compare EDEX header and body sections."""
        return (
            isinstance(other, EDEXMetadata)
            and self.header == other.header
            and self.body == other.body
        )

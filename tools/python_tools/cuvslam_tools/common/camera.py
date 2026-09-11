#!/usr/bin/env python3
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

"""Python implementation of camera models for distortion and undistortion.
This is equivalent to camera.h and camera.cpp from the C++ codebase.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import sys


def epsilon() -> float:
    """Returns machine epsilon for float32."""
    return np.finfo(np.float32).eps


class CameraModel(ABC):
    """
    Base camera model class.

    There are three coordinate spaces:
    1) pixel coordinates (top left is zero)
    2) uv - float image coordinates in pixels.
            left-top pixel corner coordinates are (0.f, 0.f), (0.f, 1.f), (1.f, 1.f), (1.f, 0.f)
            right-bottom pixel has coordinates
                (resolution.x - 1.f, resolution.y - 1.f),
                (resolution.x - 1.f, resolution.y),
                (resolution.x, resolution.y),
                (resolution.x, resolution.y - 1.f)
    3) normalized_uv = inv_calibration * uv
       translated with principal point and scaled with focal.
       Focal is measured in pixels unit. For example for 1024x1024 focal can be 512.f
       (0.f, 0.f) can be considered as lens center.
       no more resolutions/pixels.
       Min/max coordinates can be positive/negative.
       [-1.f, 1.f] is an example of good mapping.
       Maximum radius is limited by max_normalized_uv_radius2.
    4) xy - like normalized_uv but without lens distortion (AKA undistorted coordinates) and double-flipped
       (0.f, 0.f) also can be considered as lens center.
       maximum values depends on lens distortion, can be big values for example in case of fisheye.
       maximum radius is limited with max_xy_radius
       xy is double flipped (flip x and flip y) from uv. Consider xy as image from camera obscura.
       Since Z is pointing behind the camera, when we move forward Z is negative. At the time of projection,
       X/Z and Y/Z will be negative.
    """

    def __init__(
        self,
        resolution: np.ndarray,
        focal: np.ndarray,
        principal: np.ndarray,
        max_normalized_uv_radius: float = 10000.0,
        max_xy_radius: float = 10000.0,
    ):
        """
        Initialize camera model.

        Args:
            resolution: Image resolution [width, height] (not used in calculations, see focal)
            focal: Focal length in pixels [fx, fy]
            principal: Principal point in pixels [cx, cy]
            max_normalized_uv_radius: Maximum radius for normalized UV coordinates
            max_xy_radius: Maximum radius for undistorted XY coordinates
        """
        self.resolution_ = np.array(resolution, dtype=np.float32)
        self.focal_ = np.array(focal, dtype=np.float32)
        self.principal_ = np.array(principal, dtype=np.float32)
        self.max_normalized_uv_radius2_ = (
            max_normalized_uv_radius * max_normalized_uv_radius
        )
        self.max_xy_radius2_ = max_xy_radius * max_xy_radius

        # Assertions
        assert np.all(self.principal_ >= 0.0), "Principal point must be non-negative"
        assert np.all(self.focal_ > 0.0), "Focal length must be positive"
        assert np.all(self.resolution_ > 0), "Resolution must be positive"

        # Build calibration matrix (3x3 affine transformation in homogeneous coordinates)
        self.calibration_ = np.zeros((3, 3), dtype=np.float32)
        self.calibration_[0, 0] = self.focal_[0]
        self.calibration_[0, 2] = self.principal_[0]
        self.calibration_[1, 1] = self.focal_[1]
        self.calibration_[1, 2] = self.principal_[1]
        self.calibration_[2, 2] = 1.0

        # Check if invertible
        self.is_invertible_ = (
            abs(self.focal_[0]) > epsilon() and abs(self.focal_[1]) > epsilon()
        )

        if self.is_invertible_:
            self.inv_calibration_ = np.linalg.inv(self.calibration_)
        else:
            self.inv_calibration_ = np.zeros((3, 3), dtype=np.float32)

        assert self.is_invertible_, "Calibration matrix must be invertible"

    def normalize_point(self, uv: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Convert pixel coordinates to undistorted normalized coordinates.

        Args:
            uv: Pixel coordinates [u, v]

        Returns:
            (success, xy): Success flag and undistorted normalized coordinates [x, y] or None
        """
        if not self.is_invertible_:
            return False, None

        # Convert to homogeneous coordinates
        uv_homog = np.array([uv[0], uv[1], 1.0], dtype=np.float32)
        normalized_uv_homog = self.inv_calibration_ @ uv_homog
        normalized_uv = normalized_uv_homog[:2]

        if np.dot(normalized_uv, normalized_uv) > self.max_normalized_uv_radius2_:
            return False, None

        success, xy = self._undistort(normalized_uv)
        if not success:
            return False, None

        # Flip x for camera coordinate system
        xy[0] *= -1.0

        if np.dot(xy, xy) > self.max_xy_radius2_:
            return False, None

        return True, xy

    def denormalize_point(self, xy: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Convert undistorted normalized coordinates to pixel coordinates.

        Args:
            xy: Undistorted normalized coordinates [x, y]

        Returns:
            (success, uv): Success flag and pixel coordinates [u, v] or None
        """
        if not self.is_invertible_:
            return False, None

        if np.dot(xy, xy) > self.max_xy_radius2_:
            return False, None

        # Flip x back (see normalize_point for explanation)
        xy_flipped = np.array([-xy[0], xy[1]], dtype=np.float32)

        success, normalized_uv = self._distort(xy_flipped)
        if not success:
            return False, None

        if np.dot(normalized_uv, normalized_uv) > self.max_normalized_uv_radius2_:
            return False, None

        # Convert to homogeneous coordinates
        normalized_uv_homog = np.array(
            [normalized_uv[0], normalized_uv[1], 1.0], dtype=np.float32
        )
        uv_homog = self.calibration_ @ normalized_uv_homog
        uv = uv_homog[:2]

        return True, uv

    def get_principal(self) -> np.ndarray:
        """Get principal point."""
        return self.principal_.copy()

    def get_focal(self) -> np.ndarray:
        """Get focal length."""
        return self.focal_.copy()

    def get_resolution(self) -> np.ndarray:
        """Get image resolution."""
        return self.resolution_.copy()

    @abstractmethod
    def _distort(self, xy: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Apply lens distortion. Must be implemented by derived classes."""
        pass

    @abstractmethod
    def _undistort(
        self, normalized_uv: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Remove lens distortion. Must be implemented by derived classes."""
        pass


class PinholeCameraModel(CameraModel):
    """Pinhole camera model (no distortion)."""

    def __init__(
        self,
        resolution: np.ndarray,
        focal: np.ndarray,
        principal: np.ndarray,
        max_normalized_uv_radius: float = 10000.0,
        max_xy_radius: float = 10000.0,
    ):
        super().__init__(
            resolution, focal, principal, max_normalized_uv_radius, max_xy_radius
        )

    def _distort(self, xy: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """No distortion for pinhole model."""
        return True, xy.copy()

    def _undistort(
        self, normalized_uv: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """No distortion for pinhole model."""
        return True, normalized_uv.copy()


class FisheyeCameraModel(CameraModel):
    """Fisheye camera model with 4 distortion coefficients (K1, K2, K3, K4)."""

    def __init__(
        self,
        resolution: np.ndarray,
        focal: np.ndarray,
        principal: np.ndarray,
        K1: float,
        K2: float,
        K3: float,
        K4: float,
        max_normalized_uv_radius: float = 10000.0,
        max_xy_radius: float = 10000.0,
    ):
        super().__init__(
            resolution, focal, principal, max_normalized_uv_radius, max_xy_radius
        )
        self.K1_ = K1
        self.K2_ = K2
        self.K3_ = K3
        self.K4_ = K4

    def _compute_distorted_radius(self, undistorted_radius: float) -> float:
        """Compute distorted radius from undistorted radius."""
        if undistorted_radius <= np.finfo(np.float32).eps:
            return undistorted_radius

        theta = np.arctan(undistorted_radius)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta6 * theta2

        return theta * (
            1.0
            + theta2 * self.K1_
            + theta4 * self.K2_
            + theta6 * self.K3_
            + theta8 * self.K4_
        )

    def _compute_distorted_radius_derivative(
        self, undistorted_radius: float
    ) -> Tuple[bool, float]:
        """
        Compute derivative of distorted radius with respect to undistorted radius.

        f(r) = atan(r) * (1 + k1 * atan(r)^2 + k2 * atan(r)^4 + k3 * atan(r)^6 + k4 * atan(r)^8)
        f'(r) = (1 + 3 * k1 * atan(r)^2 + 5 * k2 * atan(r)^4 + 7 * k3 * atan(r)^6 + 9 * k4 * atan(r)^8) / (r^2 + 1)
        """
        if abs(undistorted_radius) <= np.finfo(np.float32).eps:
            return False, 0.0

        r = undistorted_radius
        r2 = r * r
        theta = np.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta6 * theta2

        derivative = (
            1.0
            + 3.0 * self.K1_ * theta2
            + 5.0 * self.K2_ * theta4
            + 7.0 * self.K3_ * theta6
            + 9.0 * self.K4_ * theta8
        ) / (1.0 + r2)
        return True, derivative

    def _distort(self, xy: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Apply fisheye distortion."""
        undistorted_radius = np.linalg.norm(xy)

        if undistorted_radius > np.finfo(np.float32).eps:
            distorted_radius = self._compute_distorted_radius(undistorted_radius)
            normalized_uv = distorted_radius * xy / undistorted_radius
        else:
            normalized_uv = xy.copy()

        return True, normalized_uv

    def _undistort(
        self, normalized_uv: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Remove fisheye distortion using Newton-Raphson method."""
        distorted_radius = np.linalg.norm(normalized_uv)

        # Newton-Raphson method for finding successively better approximations
        max_n_iters = 10
        precision = (1.0 / 1000.0) / 100.0  # about 1/100 of the pixel for F = 1000px

        r = distorted_radius  # initial guess
        n_iters = 0
        previous_step_size = float("inf")
        eps = np.finfo(np.float32).eps

        while previous_step_size > precision and n_iters < max_n_iters:
            # STEP 1: check current solution
            f = self._compute_distorted_radius(r) - distorted_radius
            if abs(f) <= eps:
                break

            # STEP 2: compute derivatives
            success, df = self._compute_distorted_radius_derivative(r)
            if not success:
                return False, None
            if abs(df) < eps:
                return False, None

            # STEP 3: make new step
            prev_r = r
            r -= f / df
            previous_step_size = abs(r - prev_r)
            n_iters += 1

        if n_iters >= max_n_iters:
            return False, None

        if distorted_radius > eps:
            xy = r * normalized_uv / distorted_radius
        else:
            xy = normalized_uv.copy()

        return True, xy


class CameraModelWithJacobianBasedUndistortion(CameraModel):
    """
    Base class for camera models that use Jacobian-based undistortion.
    Implements iterative undistort algorithm using Jacobian.
    Jacobian computation must be provided by a derived class.
    """

    def __init__(
        self,
        resolution: np.ndarray,
        focal: np.ndarray,
        principal: np.ndarray,
        max_normalized_uv_radius: float = 10000.0,
        max_xy_radius: float = 10000.0,
    ):
        super().__init__(
            resolution, focal, principal, max_normalized_uv_radius, max_xy_radius
        )

    def _undistort(
        self, normalized_uv: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Remove distortion using Newton-Raphson method with Jacobian."""
        max_n_iters = 10
        precision = (1.0 / 1000.0) / 100.0  # about 1/100 of the pixel for F = 1000px

        xy = normalized_uv.copy()  # initial guess
        n_iters = 0
        previous_step_size = float("inf")
        eps = np.finfo(np.float32).eps

        while previous_step_size > precision and n_iters < max_n_iters:
            # STEP 1: check current solution
            success, estimate_normalized_uv = self._distort(xy)
            if not success:
                return False, None

            f = estimate_normalized_uv - normalized_uv
            if np.linalg.norm(f) <= eps:
                return True, xy

            # STEP 2: compute derivatives
            df = self._compute_distort_jacobian(xy)

            # Check if df is invertible
            try:
                df_inverse = np.linalg.inv(df)
            except np.linalg.LinAlgError:
                return False, None

            # STEP 3: make new step
            prev_xy = xy.copy()
            xy -= df_inverse @ f
            previous_step_size = np.linalg.norm(xy - prev_xy)
            n_iters += 1

        if n_iters < max_n_iters:
            return True, xy
        return False, None

    @abstractmethod
    def _compute_distort_jacobian(self, xy: np.ndarray) -> np.ndarray:
        """Compute Jacobian of distortion function. Must be implemented by derived classes."""
        pass


class Brown5KCameraModel(CameraModelWithJacobianBasedUndistortion):
    """
    Brown (aka Brown-Conrady) distortion model with 5 coefficients:
    3 for radial distortion (K1, K2, K3) and 2 for tangential (P1, P2).

    edex stores them as (k1, k2, k3, p1, p2), which is not the OpenCV order.

    References:
    - https://www.control.isy.liu.se/student/graduate/DynVis/Lectures/le2.pdf
    - http://robots.stanford.edu/cs223b04/JeanYvesCalib/htmls/parameters.html (Caltech's calibration toolbox)

    This model is a special case of rational function that OpenCV uses.
    """

    def __init__(
        self,
        resolution: np.ndarray,
        focal: np.ndarray,
        principal: np.ndarray,
        K1: float,
        K2: float,
        K3: float,
        P1: float,
        P2: float,
        max_normalized_uv_radius: float = 10000.0,
        max_xy_radius: float = 10000.0,
    ):
        super().__init__(
            resolution, focal, principal, max_normalized_uv_radius, max_xy_radius
        )
        self.K1_ = K1
        self.K2_ = K2
        self.K3_ = K3
        self.P1_ = P1
        self.P2_ = P2

    def _distort(self, xy: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Apply Brown-Conrady distortion."""
        x = xy[0]
        y = xy[1]
        x2 = x * x
        y2 = y * y
        x_y = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2

        radial = (1.0 + self.K1_ * r2 + self.K2_ * r4 + self.K3_ * r6) * xy
        tangential = np.array(
            [
                2.0 * self.P1_ * x_y + self.P2_ * (r2 + 2.0 * x2),
                2.0 * self.P2_ * x_y + self.P1_ * (r2 + 2.0 * y2),
            ],
            dtype=np.float32,
        )

        normalized_uv = radial + tangential
        return True, normalized_uv

    def _compute_distort_jacobian(self, xy: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of Brown-Conrady distortion.

        You can use expression below on https://www.wolframalpha.com/
        jacobian matrix of (x*R+2p1xy+p2*(y^2+3x^2),y*R+p1*(x^2+3y^2)+2p2xy) where
        R=(1+k1*(x^2+y^2)+k2*(x^2+y^2)^2+k3*(x^2+y^2)^3)
        """
        x = xy[0]
        y = xy[1]
        x2 = x * x
        y2 = y * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2

        t1 = 1.0 + self.K3_ * r6 + self.K2_ * r4 + self.K1_ * r2
        t3 = 6.0 * self.K3_ * r4 + 4.0 * self.K2_ * r2 + 2.0 * self.K1_

        jacobian = np.zeros((2, 2), dtype=np.float32)
        jacobian[0, 0] = t1 + 6.0 * self.P2_ * x + 2.0 * self.P1_ * y + t3 * x2
        jacobian[1, 1] = t1 + 2.0 * self.P2_ * x + 6.0 * self.P1_ * y + t3 * y2

        t2 = 2.0 * (self.P1_ * x + self.P2_ * y) + x * y * t3
        jacobian[0, 1] = t2
        jacobian[1, 0] = t2

        return jacobian


class PolynomialCameraModel(CameraModelWithJacobianBasedUndistortion):
    """
    Rational polynomial model with 6 radial distortion coefficients and 2 tangential distortion coefficients.

    Reference:
    https://docs.nvidia.com/vpi/group__VPI__LDC.html#structVPIPolynomialLensDistortionModel
    """

    def __init__(
        self,
        resolution: np.ndarray,
        focal: np.ndarray,
        principal: np.ndarray,
        K1: float,
        K2: float,
        K3: float,
        K4: float,
        K5: float,
        K6: float,
        P1: float,
        P2: float,
        max_normalized_uv_radius: float = 10000.0,
        max_xy_radius: float = 10000.0,
    ):
        super().__init__(
            resolution, focal, principal, max_normalized_uv_radius, max_xy_radius
        )
        self.K1_ = K1
        self.K2_ = K2
        self.K3_ = K3
        self.K4_ = K4
        self.K5_ = K5
        self.K6_ = K6
        self.P1_ = P1
        self.P2_ = P2

    def _distort(self, xy: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Apply polynomial distortion."""
        x2 = xy[0] * xy[0]
        y2 = xy[1] * xy[1]
        r2 = x2 + y2

        kr = (1.0 + ((self.K3_ * r2 + self.K2_) * r2 + self.K1_) * r2) / (
            1.0 + ((self.K6_ * r2 + self.K5_) * r2 + self.K4_) * r2
        )

        normalized_uv = np.array(
            [
                xy[0] * kr
                + self.P1_ * (2.0 * xy[0] * xy[1])
                + self.P2_ * (r2 + 2.0 * x2),
                xy[1] * kr
                + self.P1_ * (r2 + 2.0 * y2)
                + self.P2_ * (2.0 * xy[0] * xy[1]),
            ],
            dtype=np.float32,
        )

        return True, normalized_uv

    def _compute_distort_jacobian(self, xy: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of polynomial distortion.

        You can use expression below on https://www.wolframalpha.com/
        jacobian matrix of (x*k+p1*(2*x*y)+p2*(r2+2*x*x), y*k+p1*(r2+2*y*y)+p2*(2*x*y))
        where k = (1+((k3*r2+k2)*r2+k1)*r2) / (1+((k6*r2+k5)*r2+k4)*r2) and r2 = x*x+y*y
        """
        x = xy[0]
        y = xy[1]
        x2 = x * x
        y2 = y * y
        r2 = x2 + y2

        s = 1.0 + ((self.K3_ * r2 + self.K2_) * r2 + self.K1_) * r2
        t = 1.0 + ((self.K6_ * r2 + self.K5_) * r2 + self.K4_) * r2
        t2 = t * t

        a = (3.0 * self.K3_ * r2 + 2.0 * self.K2_) * r2 + self.K1_
        b = (3.0 * self.K6_ * r2 + 2.0 * self.K5_) * r2 + self.K4_

        c = 2.0 * (a * t - s * b) / t2

        jacobian = np.zeros((2, 2), dtype=np.float32)
        jacobian[0, 0] = x2 * c + s / t + 2.0 * self.P1_ * y + 6.0 * self.P2_ * x
        jacobian[1, 1] = y2 * c + s / t + 6.0 * self.P1_ * y + 2.0 * self.P2_ * x

        j01 = x * y * c + 2.0 * self.P1_ * x + 2.0 * self.P2_ * y
        jacobian[0, 1] = j01
        jacobian[1, 0] = j01

        return jacobian


def create_camera_model(
    resolution: np.ndarray,
    focal: np.ndarray,
    principal: np.ndarray,
    distortion_model: str,
    parameters: np.ndarray,
) -> Optional[CameraModel]:
    """
    Factory function for camera models.

    Instead of accepting named distortion parameters it accepts parameters array.
    Order of parameters in array is the same as in appropriate constructors except for PolynomialCameraModel,
    where parameters order in array is K1, K2, P1, P2, K3, K4, K5, K6 as in cv::initUndistortRectifyMap
    and in ROS messages.

    Args:
        resolution: Image resolution [width, height]
        focal: Focal length [fx, fy]
        principal: Principal point [cx, cy]
        distortion_model: Name of distortion model ("pinhole", "fisheye", "fisheye4", "brown5k", "polynomial")
        parameters: Array of distortion parameters

    Returns:
        Camera model instance or None if distortion model is unsupported or number of parameters is wrong
    """
    dm_name = distortion_model.lower()
    num_parameters = len(parameters)

    if dm_name == "polynomial":
        if num_parameters != 8:
            return None
        # Note: Parameter order is K1, K2, P1, P2, K3, K4, K5, K6
        return PolynomialCameraModel(
            resolution,
            focal,
            principal,
            parameters[0],
            parameters[1],
            parameters[4],
            parameters[5],
            parameters[6],
            parameters[7],
            parameters[2],
            parameters[3],
        )
    elif dm_name == "brown5k":
        if num_parameters != 5:
            return None
        # Note: Parameter order is K1, K2, K3, P1, P2 - radial first, unlike polynomial above
        return Brown5KCameraModel(
            resolution,
            focal,
            principal,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
        )
    elif dm_name in ["fisheye4", "fisheye"]:
        if num_parameters != 4:
            return None
        return FisheyeCameraModel(
            resolution,
            focal,
            principal,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
        )
    elif dm_name == "pinhole":
        if num_parameters != 0:
            return None
        return PinholeCameraModel(resolution, focal, principal)

    return None

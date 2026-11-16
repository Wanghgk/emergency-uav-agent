"""Project aligned planar-depth pixels into the ESAR map frame."""

from math import radians, tan

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Point = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Quaternion = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
# UE's half-float depth target saturates at this value for sky/far clipping.
DEPTH_SATURATION_M = 65504


class Pose(BaseModel):
    position: Point
    orientation_xyzw: Quaternion

    @model_validator(mode="after")
    def check_rotation(self):
        if not np.isclose(np.linalg.norm(self.orientation_xyzw), 1, atol=1e-3):
            raise ValueError("camera or vehicle orientation must be a unit quaternion")
        return self


class Intrinsics(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float

    @classmethod
    def from_horizontal_fov(cls, width: int, height: int, fov: float):
        if not 0 < fov < 180:
            raise ValueError(
                "horizontal field of view must be between 0 and 180 degrees"
            )
        focal = width / (2 * tan(radians(fov / 2)))
        return cls(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=(width - 1) / 2,
            cy=(height - 1) / 2,
        )


def ned_to_ue(point: Point, origin_ue_cm: Point) -> Point:
    x, y, z = point
    a, b, c = origin_ue_cm
    return a + 100 * x, b + 100 * y, c - 100 * z


def rotate(vector: np.ndarray, orientation: Quaternion) -> np.ndarray:
    quaternion = np.array(orientation) / np.linalg.norm(orientation)
    axis, scalar = quaternion[:3], quaternion[3]
    cross = 2 * np.cross(axis, vector)
    return vector + scalar * cross + np.cross(axis, cross)


def pixel_to_ned(
    u: float, v: float, depth_m: float, camera: Intrinsics, pose: Pose
) -> Point:
    if not (-0.5 <= u < camera.width - 0.5 and -0.5 <= v < camera.height - 0.5):
        raise ValueError("pixel lies outside the capture")
    if not 0 < depth_m < DEPTH_SATURATION_M:
        raise ValueError("pixel has no usable positive planar depth")
    # Optical right-down-forward becomes AirSim camera forward-right-down.
    right = (u - camera.cx) * depth_m / camera.fx
    down = (v - camera.cy) * depth_m / camera.fy
    world = np.array(pose.position) + rotate(
        np.array([depth_m, right, down]), pose.orientation_xyzw
    )
    return tuple(world.tolist())


def ned_to_pixel(point: Point, camera: Intrinsics, pose: Pose) -> tuple[float, float]:
    forward, right, down = ned_to_camera(point, pose)
    if forward <= 0:
        raise ValueError("point is behind the camera")
    return float(camera.cx + camera.fx * right / forward), float(
        camera.cy + camera.fy * down / forward
    )


def ned_to_camera(point: Point, pose: Pose) -> np.ndarray:
    x, y, z, w = pose.orientation_xyzw
    return rotate(np.array(point) - pose.position, (-x, -y, -z, w))

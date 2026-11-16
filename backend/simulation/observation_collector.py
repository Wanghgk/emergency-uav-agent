"""Capture and replay registered RGB and planar depth with simulator poses."""

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import AwareDatetime, BaseModel, Field

from .coordinate_transform import (
    Intrinsics,
    Point,
    Pose,
    Quaternion,
    ned_to_camera,
    ned_to_pixel,
    ned_to_ue,
    pixel_to_ned,
)


class Observation(BaseModel):
    schema_version: Literal[1] = 1
    source: Literal["ESARBench/AirSim"] = "ESARBench/AirSim"
    map_id: str
    task_id: int
    frame_id: str
    run_id: str | None = None
    timestamp: AwareDatetime
    rgb_timestamp_ns: int = Field(gt=0)
    depth_timestamp_ns: int = Field(gt=0)
    drone_state_timestamp_ns: int = Field(gt=0)
    rgb_image: Literal["rgb.png"] = "rgb.png"
    depth_image: Literal["depth.npy"] = "depth.npy"
    depth_type: Literal["DepthPlanar_m"] = "DepthPlanar_m"
    drone_position: Point
    drone_orientation: Quaternion
    camera_name: str
    camera_pose: Pose
    depth_camera_pose: Pose
    camera_intrinsics: Intrinsics
    depth_camera_intrinsics: Intrinsics
    intrinsics_source: Literal["AirSim_fov_pinhole"] = "AirSim_fov_pinhole"
    coordinate_frame: Literal["AirSim_NED_m"] = "AirSim_NED_m"
    map_origin_ue_cm: Point


def vector(value) -> Point:
    return value.x_val, value.y_val, value.z_val


def pose(position, orientation) -> Pose:
    return Pose(
        position=vector(position),
        orientation_xyzw=(*vector(orientation), orientation.w_val),
    )


def capture(
    client, api, task, origin: Point
) -> tuple[Observation, Image.Image, np.ndarray]:
    camera = "front_camera"
    vehicle = "Drone0"
    info = client.simGetCameraInfo(camera, vehicle_name=vehicle)
    rgb, depth = client.simGetImages(
        [
            api.ImageRequest(camera, api.ImageType.Scene, False, True),
            api.ImageRequest(camera, api.ImageType.DepthPlanar, True, False),
        ],
        vehicle_name=vehicle,
    )
    state = client.getMultirotorState(vehicle_name=vehicle)
    kinematics = state.kinematics_estimated
    drone = pose(kinematics.position, kinematics.orientation)
    image = Image.open(BytesIO(bytes(rgb.image_data_uint8))).convert("RGB")
    if image.size != (rgb.width, rgb.height):
        raise ValueError("RGB dimensions differ from the AirSim response")
    values = np.asarray(depth.image_data_float, dtype=np.float32).reshape(
        depth.height, depth.width
    )
    observation = Observation(
        map_id=task.map_name,
        task_id=task.task_id,
        frame_id=uuid4().hex,
        timestamp=datetime.now(UTC),
        rgb_timestamp_ns=rgb.time_stamp,
        depth_timestamp_ns=depth.time_stamp,
        drone_state_timestamp_ns=state.timestamp,
        drone_position=drone.position,
        drone_orientation=drone.orientation_xyzw,
        camera_name=camera,
        camera_pose=pose(rgb.camera_position, rgb.camera_orientation),
        depth_camera_pose=pose(depth.camera_position, depth.camera_orientation),
        camera_intrinsics=Intrinsics.from_horizontal_fov(
            rgb.width, rgb.height, info.fov
        ),
        depth_camera_intrinsics=Intrinsics.from_horizontal_fov(
            depth.width, depth.height, info.fov
        ),
        map_origin_ue_cm=origin,
    )
    return observation, image, values


def check_registration(observation: Observation) -> None:
    rgb, depth = observation.camera_pose, observation.depth_camera_pose
    distance = np.linalg.norm(np.subtract(rgb.position, depth.position))
    orientation_match = abs(np.dot(rgb.orientation_xyzw, depth.orientation_xyzw))
    if distance > 0.01 or orientation_match < 0.99999:
        raise ValueError("RGB and depth camera poses differ")
    if abs(observation.rgb_timestamp_ns - observation.depth_timestamp_ns) > 50_000_000:
        raise ValueError("RGB and depth timestamps differ by more than 50 ms")


def map_point(observation: Observation, depth: np.ndarray, u: int, v: int) -> dict:
    check_registration(observation)
    camera = observation.depth_camera_intrinsics
    if depth.shape != (camera.height, camera.width):
        raise ValueError("depth dimensions differ from recorded intrinsics")
    if not (0 <= u < camera.width and 0 <= v < camera.height):
        raise ValueError("pixel lies outside the depth frame")
    point = pixel_to_ned(
        u, v, float(depth[v, u]), camera, observation.depth_camera_pose
    )
    return {
        "depth_pixel": [u, v],
        "rgb_pixel": ned_to_pixel(
            point, observation.camera_intrinsics, observation.camera_pose
        ),
        "ned_m": point,
        "ue_cm": ned_to_ue(point, observation.map_origin_ue_cm),
    }


def compare_static_depth(
    first: Observation,
    first_depth: np.ndarray,
    second: Observation,
    second_depth: np.ndarray,
) -> dict:
    """Check that points seen from one pose agree with depth at another pose."""
    camera = second.depth_camera_intrinsics
    residuals = []
    for v in range(0, first_depth.shape[0], 4):
        for u in range(0, first_depth.shape[1], 4):
            depth = float(first_depth[v, u])
            if not 1 <= depth <= 200:
                continue
            point = pixel_to_ned(
                u, v, depth, first.depth_camera_intrinsics, first.depth_camera_pose
            )
            forward, right, down = ned_to_camera(point, second.depth_camera_pose)
            if forward <= 0:
                continue
            col = round(camera.cx + camera.fx * right / forward)
            row = round(camera.cy + camera.fy * down / forward)
            if not (0 <= row < camera.height and 0 <= col < camera.width):
                continue
            measured = float(second_depth[row, col])
            if np.isfinite(measured) and 1 <= measured <= 200:
                residuals.append(abs(measured - forward))
    if len(residuals) < 20:
        raise ValueError(
            f"only {len(residuals)} overlapping depth points in the 1–200 m range"
        )
    median = float(np.median(residuals))
    if median > 1:
        raise ValueError(f"cross-view median depth residual {median:.3f} m exceeds 1 m")
    return {
        "cross_view_sample_count": len(residuals),
        "median_depth_residual_m": median,
    }


def save_observation(
    root: Path, observation: Observation, rgb: Image.Image, depth: np.ndarray
) -> Path:
    directory = root / observation.frame_id
    directory.mkdir(parents=True)
    rgb.save(directory / observation.rgb_image)
    np.save(directory / observation.depth_image, depth, allow_pickle=False)
    manifest = directory / "observation.json"
    temporary = directory / "observation.tmp"
    temporary.write_text(observation.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(manifest)
    return manifest


def load_observation(manifest: Path) -> tuple[Observation, Image.Image, np.ndarray]:
    record = Observation.model_validate_json(manifest.read_bytes())
    with Image.open(manifest.parent / record.rgb_image) as image:
        rgb = image.convert("RGB")
    depth = np.load(manifest.parent / record.depth_image, allow_pickle=False)
    if rgb.size != (record.camera_intrinsics.width, record.camera_intrinsics.height):
        raise ValueError("saved RGB dimensions differ from recorded intrinsics")
    if depth.dtype != np.float32 or depth.shape != (
        record.depth_camera_intrinsics.height,
        record.depth_camera_intrinsics.width,
    ):
        raise ValueError("invalid recorded float32 depth")
    return record, rgb, depth

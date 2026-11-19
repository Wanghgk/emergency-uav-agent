"""Capture an orthographic ESAR terrain raster with NED bounds in metres."""

import json
import math
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .coordinate_transform import Pose, rotate
from .observation_collector import pose

CAMERA = "terrain_overview"
WIDTH_M = 1200
RESOLUTION = 1536
CAMERA_HEIGHT_M = 400


class Basemap(BaseModel):
    map_id: str
    task_id: int
    camera_pose: Pose
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    width_m: float = Field(gt=0)
    render_source: Literal["rgb", "rgb_enhanced", "depth_relief"] = "rgb"

    def bounds(self):
        north, east, _ = self.camera_pose.position
        half_x = self.width_m / 2
        half_y = half_x * self.height_px / self.width_px
        return {
            "north": north + half_y,
            "south": north - half_y,
            "west": east - half_x,
            "east": east + half_x,
        }


def map_settings(
    source: Path,
    destination: Path,
    origin,
    *,
    search_camera=False,
    camera_size=(768, 384),
):
    settings = json.loads(source.read_text(encoding="utf-8"))
    if search_camera:
        front = settings["Vehicles"]["Drone0"]["Cameras"]["front_camera"][
            "CaptureSettings"
        ]
        for entry in front:
            resolution = dict(enumerate(camera_size)).get(entry["ImageType"])
            if resolution:
                entry.update(Width=resolution, Height=resolution)
    north, east, down = origin
    settings.setdefault("ExternalCameras", {})[CAMERA] = {
        "X": north,
        "Y": east,
        "Z": down - CAMERA_HEIGHT_M,
        "Pitch": -90,
        "Roll": 0,
        "Yaw": 0,
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": RESOLUTION,
                "Height": RESOLUTION,
                "ProjectionMode": "orthographic",
                "OrthoWidth": WIDTH_M,
            },
            {
                "ImageType": 1,
                "Width": 512,
                "Height": 512,
                "ProjectionMode": "orthographic",
                "OrthoWidth": WIDTH_M,
            },
        ],
    }
    destination.write_text(json.dumps(settings), encoding="utf-8")


def read_terrain_down(client, api, basemap: Basemap) -> np.ndarray:
    [response] = client.simGetImages(
        [api.ImageRequest(CAMERA, api.ImageType.DepthPlanar, True, False)],
        external=True,
    )
    camera = pose(response.camera_position, response.camera_orientation)
    if math.dist(camera.position, basemap.camera_pose.position) > 0.1 or not np.isclose(
        abs(np.dot(camera.orientation_xyzw, basemap.camera_pose.orientation_xyzw)),
        1,
        atol=1e-4,
    ):
        raise ValueError("terrain RGB and depth poses disagree")
    if response.width * basemap.height_px != response.height * basemap.width_px:
        raise ValueError("terrain RGB and depth aspect ratios disagree")
    depth = np.asarray(response.image_data_float).reshape(
        response.height, response.width
    )
    valid = np.isfinite(depth) & (depth > 1) & (depth < 3000)
    return np.where(valid, depth + basemap.camera_pose.position[2], np.nan)


def visible_rgb(image: Image.Image):
    light = np.asarray(image.convert("L"))
    shadows, median, white = np.percentile(light, [10, 50, 98])
    if white < 16 or light.std() < 3:
        return None, "depth_relief"
    if median < 100 or shadows < 30:
        levels = [
            round(70 + 185 * min(1, value / max(white, 48)) ** 0.55)
            for value in range(256)
        ]
        return image.point(levels * 3), "rgb_enhanced"
    return image, "rgb"


def terrain_relief(down: np.ndarray, basemap: Basemap) -> Image.Image:
    if min(down.shape) < 2 or not np.isfinite(down).any():
        raise ValueError("no terrain depth available for the basemap")
    pixel_m = basemap.width_m / down.shape[1]
    dy, dx = np.gradient(down, pixel_m)
    shade = np.nan_to_num((dx + dy) / np.sqrt(2 * (1 + dx**2 + dy**2)))
    light = (193 + 53 * shade).astype(np.uint8)
    return (
        Image.fromarray(light)
        .resize((basemap.width_px, basemap.height_px), Image.Resampling.BILINEAR)
        .convert("RGB")
    )


def capture_basemap(client, api, task, output: Path):
    info = client.simGetCameraInfo(CAMERA, external=True)
    [response] = client.simGetImages(
        [api.ImageRequest(CAMERA, api.ImageType.Scene, False, True)], external=True
    )
    try:
        with Image.open(BytesIO(bytes(response.image_data_uint8))) as scene:
            image, source = visible_rgb(scene.convert("RGB"))
            if scene.size != (RESOLUTION, RESOLUTION):
                image, source = None, "depth_relief"
    except OSError:
        image, source = None, "depth_relief"
    camera = pose(response.camera_position, response.camera_orientation)
    # A nadir camera with yaw zero puts North at the top and East at the right.
    forward = rotate(np.array([1, 0, 0]), camera.orientation_xyzw)
    right = rotate(np.array([0, 1, 0]), camera.orientation_xyzw)
    if not np.allclose(forward, [0, 0, 1], atol=1e-3) or not np.allclose(
        right, [0, 1, 0], atol=1e-3
    ):
        raise ValueError("terrain camera is not north-up and nadir")
    matrix = np.asarray(info.proj_mat.matrix).reshape(4, 4)
    horizontal_scale = np.linalg.norm(matrix[0, :3])
    vertical_scale = np.linalg.norm(matrix[1, :3])
    if not np.isclose(matrix[3, 3], 1) or not np.allclose(
        [horizontal_scale, vertical_scale],
        2 / (100 * WIDTH_M),
        rtol=0.02,
        atol=0,
    ):
        raise ValueError(
            f"AirSim did not report the expected orthographic width: {matrix.tolist()}"
        )
    meta = Basemap(
        map_id=task.map_name,
        task_id=task.task_id,
        camera_pose=camera,
        width_px=RESOLUTION,
        height_px=RESOLUTION,
        width_m=WIDTH_M,
        render_source=source,
    )
    if image is None:
        image = terrain_relief(read_terrain_down(client, api, meta), meta)
    folder = output / f"{task.map_name}-{task.task_id}"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "terrain.tmp"
    image.save(temporary, format="WEBP", quality=91)
    temporary.replace(folder / "terrain.webp")
    (folder / "basemap.json").write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )
    return meta

"""Capture an orthographic ESAR terrain raster with NED bounds in metres."""

import json
from io import BytesIO
from pathlib import Path

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


def map_settings(source: Path, destination: Path, origin, *, search_camera=False):
    settings = json.loads(source.read_text(encoding="utf-8"))
    if search_camera:
        front = settings["Vehicles"]["Drone0"]["Cameras"]["front_camera"][
            "CaptureSettings"
        ]
        for entry in front:
            resolution = {0: 768, 1: 384}.get(entry["ImageType"])
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


def capture_basemap(client, api, task, output: Path):
    info = client.simGetCameraInfo(CAMERA, external=True)
    [response] = client.simGetImages(
        [api.ImageRequest(CAMERA, api.ImageType.Scene, False, True)], external=True
    )
    image = Image.open(BytesIO(bytes(response.image_data_uint8))).convert("RGB")
    if image.size != (RESOLUTION, RESOLUTION) or np.asarray(image).std() < 3:
        raise ValueError("AirSim did not render the configured terrain camera")
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
        width_px=image.width,
        height_px=image.height,
        width_m=WIDTH_M,
    )
    folder = output / f"{task.map_name}-{task.task_id}"
    folder.mkdir(parents=True, exist_ok=True)
    image.save(folder / "terrain.webp", quality=91)
    (folder / "basemap.json").write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )
    return meta

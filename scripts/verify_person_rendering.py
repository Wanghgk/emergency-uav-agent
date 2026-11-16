"""Capture diagnostic views at official victim coordinates, outside mission data."""

import argparse
import json
import math
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from backend.simulation.esar_adapter import load_task
from backend.simulation.observation_collector import capture, save_observation, vector
from backend.simulation.session import VEHICLE, simulator


def target_views(targets, origin):
    for item in targets:
        ue = item["location"]
        target = (
            (ue["x"] - origin[0]) / 100,
            (ue["y"] - origin[1]) / 100,
            (origin[2] - ue["z"]) / 100,
        )
        for suffix, north_offset, east_offset, yaw in (
            ("south", -5, 0, 0),
            ("west", 0, -5, 90),
        ):
            name = f"victim_{item['target_id']}_{suffix}"
            position = (
                target[0] + north_offset,
                target[1] + east_offset,
                target[2] - 3,
            )
            yield item["target_id"], target, name, position, yaw


def diagnostic(task_id, esar, maps, root):
    task, _, origin = load_task(esar, "task_data.json", task_id)
    upstream = json.loads((esar / "data/task_data.json").read_text(encoding="utf-8"))
    targets = next(item for item in upstream if item["task_id"] == task_id)["targets"]
    if not targets:
        raise ValueError(f"Task {task_id} has no official people coordinates")
    output = root / f"{task.map_name}-{task_id}"
    output.mkdir(parents=True, exist_ok=True)
    settings_base = json.loads(
        (esar / "settings/settings.json").read_text(encoding="utf-8")
    )
    cameras = list(target_views(targets, origin))
    settings_base["ExternalCameras"] = {
        name: {
            "X": x,
            "Y": y,
            "Z": z,
            "Pitch": -28,
            "Roll": 0,
            "Yaw": yaw,
            "CaptureSettings": [
                {"ImageType": 0, "Width": 640, "Height": 640, "FOV_Degrees": 65},
                {"ImageType": 1, "Width": 320, "Height": 320, "FOV_Degrees": 65},
            ],
        }
        for _, _, name, (x, y, z), yaw in cameras
    }
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        settings.write_text(json.dumps(settings_base), encoding="utf-8")
        with simulator(maps, settings, task) as (client, api):
            time.sleep(2)
            names = sorted(client.simListSceneObjects())
            (output / "scene-objects.json").write_text(
                json.dumps(names, indent=2), encoding="utf-8"
            )
            # Actor names are corroborating evidence. Hidden actors may still have a pose.
            candidates = client.simListSceneObjects(
                ".*([Tt]arget|[Hh]uman|[Pp]erson|[Vv]ictim|[Ss]urvivor|[Cc]haracter|[Ss]keletal|[Mm]annequin|[Qq]uinn|[Mm]anny).*"
            )
            actors = []
            for name in sorted(candidates):
                if name.startswith("external_"):
                    continue
                location = vector(client.simGetObjectPose(name).position)
                if all(map(math.isfinite, location)):
                    actors.append({"name": name, "position_ned_m": location})
            rows = []
            for target_id, target, name, expected_position, _ in cameras:
                rgb, depth = client.simGetImages(
                    [
                        api.ImageRequest(name, api.ImageType.Scene, False, True),
                        api.ImageRequest(name, api.ImageType.DepthPlanar, True, False),
                    ],
                    external=True,
                )
                image = Image.open(BytesIO(bytes(rgb.image_data_uint8))).convert("RGB")
                camera_position = vector(rgb.camera_position)
                if (
                    math.dist(camera_position, expected_position) > 0.15
                    or np.asarray(image).std() < 2
                ):
                    raise RuntimeError(
                        f"Camera {name} pose or image failed verification"
                    )
                image.save(output / f"{name}.png")
                values = np.asarray(depth.image_data_float, dtype=np.float32).reshape(
                    depth.height, depth.width
                )
                np.save(output / f"{name}.npy", values, allow_pickle=False)
                nearby = [
                    {
                        **actor,
                        "distance_m": round(
                            math.dist(target, actor["position_ned_m"]), 3
                        ),
                    }
                    for actor in actors
                    if math.dist(target, actor["position_ned_m"]) <= 5
                ]
                rows.append(
                    {
                        "target_id": target_id,
                        "official_ned_m": target,
                        "camera": name,
                        "camera_ned_m": camera_position,
                        "actors_within_5m": nearby,
                    }
                )
            # Exercise the same onboard RGB-D capture path used by run_mission.
            target_id, _, _, (x, y, z), _ = cameras[0]
            client.simSetVehiclePose(
                api.Pose(api.Vector3r(x - 0.5, y, z + 0.1), api.to_quaternion(0, 0, 0)),
                True,
                vehicle_name=VEHICLE,
            )
            client.simSetCameraPose(
                "front_camera",
                api.Pose(
                    api.Vector3r(0.5, 0, -0.1),
                    api.to_quaternion(math.radians(-28), 0, 0),
                ),
                vehicle_name=VEHICLE,
            )
            time.sleep(0.35)
            client.simPause(True)
            try:
                observation, image, depth = capture(client, api, task, origin)
            finally:
                client.simPause(False)
            pose_error = math.dist(observation.camera_pose.position, (x, y, z))
            manifest = save_observation(output / "onboard", observation, image, depth)
    report = {
        "task_id": task_id,
        "snapshot_id": task.snapshot_id,
        "ue_task_num": task.snapshot_id % 15,
        "map": task.map_name,
        "actor_count": len(names),
        "candidate_actor_count": len(actors),
        "onboard_check": {
            "target_id": target_id,
            "observation": manifest.relative_to(output).as_posix(),
            "pose_error_m": pose_error,
            "pose_verified": pose_error <= 0.75,
        },
        "views": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    thumb = 280
    sheet = Image.new("RGB", (thumb * 2, (thumb + 22) * len(targets)), "#171b23")
    labels = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x, y = index % 2 * thumb, index // 2 * (thumb + 22)
        with Image.open(output / f"{row['camera']}.png") as image:
            sheet.paste(image.resize((thumb, thumb)), (x, y + 22))
        labels.text((x + 6, y + 4), row["camera"], fill="white")
    sheet.save(output / "contact-sheet.jpg", quality=92)
    print(
        f"{output.resolve()} | {len(targets)} targets, {len(names)} actors, {len(actors)} candidate actors",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=290)
    parser.add_argument("--esar", type=Path, default=Path(".external/ESAR-pinned"))
    parser.add_argument("--maps", type=Path, default=Path(".external/ESARMaps_win"))
    parser.add_argument(
        "--output", type=Path, default=Path(".external/diagnostics/person-rendering")
    )
    args = parser.parse_args()
    diagnostic(args.task_id, args.esar, args.maps, args.output)


if __name__ == "__main__":
    main()

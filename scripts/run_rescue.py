"""Run a simulated dispatch, terrain-aware search, and independent ESAR scoring."""

import argparse
import json
import math
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
from PIL import Image

from backend.brain.planning import DeepSeekProvider, MockLLMProvider, select_plan
from backend.cerebellum.person_detector import PersonDetector
from backend.simulation.basemap import capture_basemap, map_settings
from backend.simulation.esar_adapter import load_task
from backend.simulation.rescue_search import (
    Alarm,
    Terrain,
    match_people,
    person_results,
    review_search,
    sweep,
    track_detections,
)
from backend.simulation.session import VEHICLE, simulator
from scripts.run_mission import fly, observe, sample_legs


class TiledPersonDetector(PersonDetector):
    orientations = ()

    def rotated_hits(self, rgb):
        width, height = rgb.size
        pixels = np.asarray(rgb)
        result = []
        for angle in self.orientations:
            transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            rotated = cv2.warpAffine(
                pixels, transform, (width, height), borderValue=(114, 114, 114)
            )
            hits, _ = super().predict(Image.fromarray(rotated))
            inverse = cv2.invertAffineTransform(transform)
            for hit in hits:
                lft, top, rgt, btm = hit.bbox
                if rgt - lft > 0.7 * width or btm - top > 0.7 * height:
                    continue
                points = np.array([[(lft, top), (rgt, top), (lft, btm), (rgt, btm)]])
                x, y = cv2.transform(points, inverse)[0].T
                if not (0 <= x.mean() < width and 0 <= y.mean() < height):
                    continue
                bbox = (
                    max(0, x.min()),
                    max(0, y.min()),
                    min(width, x.max()),
                    min(height, y.max()),
                )
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    result.append(
                        hit.model_copy(update={"bbox": tuple(map(float, bbox))})
                    )
        return result

    def predict(self, rgb):
        start = time.perf_counter()
        proposed, _ = super().predict(rgb)
        proposed.extend(self.rotated_hits(rgb))
        width, height = rgb.size
        tile_w, tile_h = int(width * 0.65), int(height * 0.65)
        for x in (0, width - tile_w):
            for y in (0, height - tile_h):
                tile = rgb.crop((x, y, x + tile_w, y + tile_h))
                hits, _ = super().predict(tile)
                hits.extend(self.rotated_hits(tile))
                for hit in hits:
                    left, top, right, bottom = hit.bbox
                    proposed.append(
                        hit.model_copy(
                            update={"bbox": (left + x, top + y, right + x, bottom + y)}
                        )
                    )
        boxes = [
            (left, top, right - left, bottom - top)
            for left, top, right, bottom in (hit.bbox for hit in proposed)
        ]
        keep = cv2.dnn.NMSBoxes(
            boxes, [hit.confidence for hit in proposed], self.threshold, 0.4
        )
        return [proposed[index] for index in keep], round(
            (time.perf_counter() - start) * 1000, 2
        )


def evaluate(esar, task, origin, tracks):
    # Only called after the simulator and the search have ended.
    official = json.loads((esar / "data/task_data.json").read_text(encoding="utf-8"))
    truth = next(
        item["targets"] for item in official if item["task_id"] == task.task_id
    )
    locations = [
        ((t["location"]["x"] - origin[0]) / 100, (t["location"]["y"] - origin[1]) / 100)
        for t in truth
    ]
    candidates = [track.point for track in tracks]
    confirmed = [track.point for track in tracks if len(track.observations) >= 2]
    matches = match_people(confirmed, locations)
    return {
        "official_people": len(truth),
        "verified_confirmed": len(matches),
        "verified_candidates": len(match_people(candidates, locations)),
        "matched_target_ids": [truth[index]["target_id"] for index in sorted(matches)],
        "unmatched_target_ids": [
            t["target_id"] for index, t in enumerate(truth) if index not in matches
        ],
        "match_radius_m": 4,
    }


def prior_candidates(path, task_id, alarm):
    run = json.loads(path.read_text(encoding="utf-8"))
    if run["task"]["task_id"] != task_id or run["profile"] != "dispatch":
        raise ValueError(
            "The follow-up requires observations from the same official search task"
        )
    tracks = []
    for item in run["events"]:
        if item["type"] == "observation" and item["sector"] != "START":
            track_detections(
                tracks,
                item["frame_id"],
                item["detections"],
                alarm,
                minimum_confidence=0.40,
            )
    return [track.point for track in tracks]


def run(args):
    task, start, origin = load_task(args.esar, "task_data.json", args.task_id)
    alarm = Alarm(
        **json.loads(args.alarms.read_text(encoding="utf-8"))[str(task.task_id)]
    )
    provider = DeepSeekProvider() if args.deepseek else MockLLMProvider()
    detector = TiledPersonDetector(args.weights, threshold=0.33)
    run_id = uuid4().hex
    output = Path("data/missions")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"{run_id}.json"
    pending = (
        prior_candidates(output / f"{args.revisit_run.hex}.json", task.task_id, alarm)
        if args.revisit_run
        else []
    )
    report = {
        "run_id": run_id,
        "task": task.model_dump(),
        "instruction": task.prompt,
        "status": "running",
        "profile": "dispatch",
        "planner": provider.name,
        "detector": "YOLOX-Tiny / tiled",
        "start_ned_m": start,
        "alarm": {**vars(alarm), "source": "simulated_call_coarse_area"},
        "events": [],
    }
    if args.revisit_run:
        report["followup_of"] = args.revisit_run.hex
        if not pending:
            raise ValueError(
                "The previous flight has no localized candidates to revisit"
            )
    tracks, observations = [], []

    def event(kind, **details):
        item = {"type": kind, "timestamp": datetime.now(UTC).isoformat(), **details}
        report["events"].append(item)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(manifest)
        print(json.dumps(item, ensure_ascii=True), flush=True)

    def plan(choices, feedback):
        return select_plan(
            provider,
            f"{task.prompt} An alarm reports approximately {alarm.expected_people} people within an 80 m "
            "square. Sector codes are entry corners of the search square, not victim positions. "
            "Search all rows, and prioritize a different entry for a follow-up pass.",
            choices,
            feedback,
        )

    reason = "search_budget"
    reviewed_count = None

    def planner_finished():
        nonlocal reviewed_count
        count = sum(len(track.observations) >= 2 for track in tracks)
        if count < alarm.expected_people or count == reviewed_count:
            return False
        reviewed_count = count
        decision = review_search(
            provider, tracks, alarm.expected_people, can_continue=True
        )
        event("search_review", **decision)
        return decision["action"] == "finish"

    try:
        entries = sorted(
            ["SW", "NE", "SE", "NW"],
            key=lambda corner: math.dist(start[:2], next(sweep(alarm, corner, 20))),
        )
        first = (
            None
            if pending
            else plan(entries, {"closest_entry_from_launch": entries[0]})
        )
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / "settings.json"
            map_settings(
                args.esar / "settings/settings.json",
                settings,
                start,
                search_camera=True,
            )
            with simulator(args.maps, settings, task) as (client, api):
                basemap = capture_basemap(client, api, task, Path("data/maps"))
                terrain = Terrain.read(client, api, basemap)
                collision = client.simGetCollisionInfo(vehicle_name=VEHICLE).time_stamp
                client.simSetVehiclePose(
                    api.Pose(api.Vector3r(*start), api.to_quaternion(0, 0, 0)),
                    True,
                    vehicle_name=VEHICLE,
                )
                client.enableApiControl(True, vehicle_name=VEHICLE)
                if not client.armDisarm(True, vehicle_name=VEHICLE):
                    raise RuntimeError("AirSim refused to arm")
                position = start

                def move(xy, clearance, face=None):
                    nonlocal position
                    z = terrain.highest(position[:2], xy) - clearance
                    yaw = (
                        face
                        if face is not None
                        else math.degrees(
                            math.atan2(xy[1] - position[1], xy[0] - position[0])
                        )
                    )
                    if position[2] > z + 0.5:
                        lift = (*position[:2], z)
                        for chunk in sample_legs(position, [lift], 65):
                            fly(
                                client,
                                api,
                                chunk,
                                yaw,
                                collision,
                                accept_position_tolerance=True,
                            )
                        position = lift
                    goal = (*xy, z)
                    for chunk in sample_legs(position, [goal], 65):
                        if math.dist(position, chunk) >= 1:
                            fly(
                                client,
                                api,
                                chunk,
                                yaw,
                                collision,
                                accept_position_tolerance=True,
                                speed=8 if math.dist(position, goal) > 45 else 5,
                            )
                    position = goal

                def collect(sector):
                    detector.orientations = (
                        (45, 135, 225, 315) if sector in {"PASS 2", "RECHECK"} else ()
                    )
                    item = observe(
                        client,
                        api,
                        task,
                        origin,
                        run_id,
                        Path("data/observations"),
                        detector,
                    )
                    observations.append(item["drone_ned_m"])
                    if sector != "START":
                        track_detections(
                            tracks, item["frame_id"], item["detections"], alarm
                        )
                    event("observation", sector=sector, **item)

                move(start[:2], 21, 0)
                client.simSetCameraPose(
                    "front_camera",
                    api.Pose(
                        api.Vector3r(0.5, 0, -0.1),
                        api.to_quaternion(math.radians(-65), 0, 0),
                    ),
                    vehicle_name=VEHICLE,
                )
                collect("START")
                center = (alarm.north, alarm.east)
                event(
                    "dispatch",
                    waypoint_ned_m=(*center, terrain.highest(center, center)),
                )
                move(center, 16)
                collect("ALARM")
                if pending:
                    event(
                        "replan",
                        sector="RECHECK",
                        rationale="Revisit the prior flight's localized candidates.",
                        waypoint_ned_m=pending[0],
                        objective="Obtain independent close views.",
                    )
                    while pending and len(observations) + 4 <= args.max_frames:
                        point = min(
                            pending,
                            key=lambda candidate: math.dist(
                                position[:2], candidate[:2]
                            ),
                        )
                        pending.remove(point)
                        north, east, _ = point
                        for dx, dy, yaw in [
                            (-6, 0, 0),
                            (0, -6, 90),
                            (6, 0, 180),
                            (0, 6, -90),
                        ]:
                            xy = north + dx, east + dy
                            move(xy, 10, yaw)
                            move(xy, 10, yaw)
                            collect("RECHECK")
                        if planner_finished():
                            reason = "onboard_count_met"
                            break
                    event(
                        "sector_complete",
                        sector="RECHECK",
                        person_detections=len(tracks),
                    )
                for index in range(2) if first is not None else ():
                    if index == 0:
                        search_plan = first
                    else:
                        search_plan = plan(
                            [c for c in ["SW", "NE", "SE", "NW"] if c != first.sector],
                            {
                                "completed_pass": 1,
                                "distinct_confirmed": sum(
                                    len(t.observations) >= 2 for t in tracks
                                ),
                                "candidate_clusters": len(tracks),
                            },
                        )
                    route = list(
                        sweep(
                            alarm,
                            search_plan.sector,
                            20 if index == 0 else 16,
                            cross=bool(index),
                        )
                    )
                    event(
                        "replan" if index else "plan",
                        **search_plan.model_dump(),
                        waypoint_ned_m=(*route[0], terrain.highest(route[0], route[0])),
                    )
                    for point in route:
                        if len(observations) >= args.max_frames:
                            break
                        move(point, 16 if index == 0 else 12)
                        collect(f"PASS {index + 1}")
                        if planner_finished():
                            reason = "onboard_count_met"
                            break
                    event(
                        "sector_complete",
                        sector=f"PASS {index + 1}",
                        person_detections=len(tracks),
                    )
                    if (
                        reason == "onboard_count_met"
                        or len(observations) >= args.max_frames
                    ):
                        break
                    for track in [t for t in tracks if len(t.observations) == 1][:4]:
                        if len(observations) >= args.max_frames:
                            break
                        north, east, _ = track.point
                        offset = (
                            -5 if north - alarm.north > -alarm.half_width + 5 else 5
                        )
                        move((north + offset, east), 11, 0 if offset < 0 else 180)
                        collect("RECHECK")
                        if planner_finished():
                            reason = "onboard_count_met"
                            break
                    if reason == "onboard_count_met":
                        break
                client.enableApiControl(False, vehicle_name=VEHICLE)
        if reason == "search_budget":
            event(
                "search_review",
                **review_search(
                    provider, tracks, alarm.expected_people, can_continue=False
                ),
            )
        report["status"] = "finished"
    except Exception as exc:
        report["status"] = "interrupted"
        reason = "execution_error"
        event("error", error=str(exc))
    report["people_results"] = person_results(tracks, origin)
    distance = sum(
        math.dist(a[:2], b[:2])
        for a, b in zip(observations, observations[1:], strict=False)
    )
    event(
        "summary",
        reason=reason,
        frames=len(observations),
        horizontal_distance_m=round(distance, 1),
        **evaluate(args.esar, task, origin, tracks),
    )
    print(manifest, flush=True)
    return report["status"] == "finished"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument(
        "--revisit-run",
        type=UUID,
        help="Recheck localized candidates from a previous run.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=62, choices=range(2, 101), metavar="2..100"
    )
    parser.add_argument(
        "--alarms", type=Path, default=Path("configs/rescue_alarms.json")
    )
    parser.add_argument("--esar", type=Path, default=Path(".external/ESAR-pinned"))
    parser.add_argument("--maps", type=Path, default=Path(".external/ESARMaps_win"))
    parser.add_argument(
        "--weights", type=Path, default=Path(".external/models/yolox_tiny.onnx")
    )
    raise SystemExit(0 if run(parser.parse_args()) else 1)


if __name__ == "__main__":
    main()

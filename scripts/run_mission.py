"""Run a two-leg, feedback-driven search in the local ESAR simulator."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from backend.brain.planning import (
    DeepSeekProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    select_plan,
)
from backend.cerebellum.person_detector import PersonDetector
from backend.simulation.esar_adapter import load_task
from backend.simulation.observation_collector import capture, save_observation
from backend.simulation.session import VEHICLE, simulator

# Candidate waypoints are offsets from the public takeoff position. The LLM
# chooses a sector, while the executor alone sets altitude, speed and timeout.
SECTORS = {"N": (45, 0), "E": (0, 45), "S": (-45, 0), "W": (0, -45)}


def sector_route(start, sector, extended):
    north, east = (value / 45 for value in SECTORS[sector])
    offsets = [(30, -25), (115, -25), (115, 25), (45, 25)] if extended else [(45, 0)]
    altitude = start[2] - (42 if extended else 8)
    return [
        (
            start[0] + north * forward - east * side,
            start[1] + east * forward + north * side,
            altitude,
        )
        for forward, side in offsets
    ]


def sample_legs(start, waypoints, spacing):
    for end in waypoints:
        subdivisions = max(1, math.ceil(math.dist(start, end) / spacing))
        for step in range(1, subdivisions + 1):
            yield tuple(
                a + (b - a) * step / subdivisions
                for a, b in zip(start, end, strict=True)
            )
        start = end


def check_collision(client, initial_collision):
    collision = client.simGetCollisionInfo(vehicle_name=VEHICLE)
    if collision.has_collided and collision.time_stamp > initial_collision:
        raise RuntimeError(
            f"AirSim reported a new collision with {collision.object_name or 'the scene'}"
        )


def fly(
    client,
    api,
    target,
    yaw,
    initial_collision,
    *,
    accept_position_tolerance=False,
    speed=5,
):
    check_collision(client, initial_collision)
    arrived = client.moveToPositionAsync(
        *target,
        speed,
        timeout_sec=24,
        yaw_mode=api.YawMode(False, yaw),
        vehicle_name=VEHICLE,
    ).get()
    state = client.getMultirotorState(vehicle_name=VEHICLE)
    position = state.kinematics_estimated.position
    error = math.dist((position.x_val, position.y_val, position.z_val), target)
    check_collision(client, initial_collision)
    if error > 4 or (not arrived and not accept_position_tolerance):
        raise RuntimeError(f"AirSim waypoint failed: arrival error {error:.2f} m")


def observe(client, api, task, origin, run_id, output, detector):
    time.sleep(0.35)
    client.simPause(True)
    try:
        frame, rgb, depth = capture(client, api, task, origin)
    finally:
        client.simPause(False)
    if math.dist(frame.camera_pose.position, frame.drone_position) > 1:
        raise RuntimeError("Camera pose is out of sync with the drone")
    frame.run_id = run_id
    detections, latency = detector.detect(frame, rgb, depth)
    save_observation(output, frame, rgb, depth)
    return {
        "frame_id": frame.frame_id,
        "drone_ned_m": frame.drone_position,
        "detections": [d.model_dump() for d in detections],
        "inference_ms": latency,
    }


def run(args):
    task, start, origin = load_task(args.esar, "task_data.json", args.task_id)
    provider = args.provider()
    detector = PersonDetector(args.weights)
    run_id = uuid4().hex
    args.runs.mkdir(parents=True, exist_ok=True)
    manifest = args.runs / f"{run_id}.json"
    report = {
        "run_id": run_id,
        "task": task.model_dump(),
        "instruction": args.instruction or task.prompt,
        "planner": provider.name,
        "detector": detector.model_name,
        "weights_version": detector.model_version,
        "weights_sha256": sha256(args.weights.read_bytes()).hexdigest(),
        "profile": "extended" if args.extended else "short",
        "start_ned_m": start,
        "status": "running",
        "events": [],
    }

    def event(kind, **details):
        line = {"type": kind, "timestamp": datetime.now(UTC).isoformat(), **details}
        report["events"].append(line)
        temp = manifest.with_suffix(".tmp")
        temp.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temp.replace(manifest)
        print(json.dumps(line, ensure_ascii=True), flush=True)

    routes = {name: sector_route(start, name, args.extended) for name in SECTORS}
    targets = {name: route[-1] for name, route in routes.items()}
    cruise = (*start[:2], targets["N"][2])
    position = cruise
    try:
        first = select_plan(provider, report["instruction"], targets, None)
        event("plan", **first.model_dump(), waypoint_ned_m=targets[first.sector])
        with simulator(args.maps, args.esar / "settings/settings.json", task) as (
            client,
            api,
        ):
            # AirSim can retain the pre-mission spawn contact after teleporting.
            initial_collision = client.simGetCollisionInfo(
                vehicle_name=VEHICLE
            ).time_stamp
            client.enableApiControl(True, vehicle_name=VEHICLE)
            if not client.armDisarm(True, vehicle_name=VEHICLE):
                raise RuntimeError("AirSim refused to arm Drone0")
            client.simSetVehiclePose(
                api.Pose(api.Vector3r(*start), api.to_quaternion(0, 0, 0)),
                True,
                vehicle_name=VEHICLE,
            )
            fly(client, api, cruise, 0, initial_collision)
            if args.extended:
                client.simSetCameraPose(
                    "front_camera",
                    api.Pose(
                        api.Vector3r(0.5, 0, -0.1), api.to_quaternion(-0.44, 0, 0)
                    ),
                    vehicle_name=VEHICLE,
                )
            event(
                "observation",
                sector="START",
                **observe(
                    client, api, task, origin, run_id, args.observations, detector
                ),
            )
            plan = first
            for leg in range(2):
                sampled = (
                    sample_legs(position, routes[plan.sector], 22)
                    if args.extended
                    else routes[plan.sector]
                )
                detections = []
                for waypoint in sampled:
                    north, east = (waypoint[axis] - position[axis] for axis in range(2))
                    fly(
                        client,
                        api,
                        waypoint,
                        math.degrees(math.atan2(east, north)),
                        initial_collision,
                    )
                    observed = observe(
                        client, api, task, origin, run_id, args.observations, detector
                    )
                    event("observation", sector=plan.sector, **observed)
                    detections.extend(observed["detections"])
                    position = waypoint
                event(
                    "sector_complete",
                    sector=plan.sector,
                    person_detections=len(detections),
                )
                if leg == 0:
                    next_sectors = {
                        name: pos
                        for name, pos in targets.items()
                        if name != plan.sector
                    }
                    feedback = {
                        "searched_sector": plan.sector,
                        "person_detection_count": len(detections),
                        "localized_detection_count": sum(
                            item["ned_m"] is not None for item in detections
                        ),
                    }
                    plan = select_plan(
                        provider, report["instruction"], next_sectors, feedback
                    )
                    event(
                        "replan",
                        **plan.model_dump(),
                        waypoint_ned_m=targets[plan.sector],
                        trigger="search_sector_observed",
                    )
            client.enableApiControl(False, vehicle_name=VEHICLE)
        report["status"] = "finished"
        observations = [
            item for item in report["events"] if item["type"] == "observation"
        ]
        count = sum(len(item["detections"]) for item in observations)
        positions = [item["drone_ned_m"] for item in observations]
        distance = sum(
            math.dist(a[:2], b[:2])
            for a, b in zip(positions, positions[1:], strict=False)
        )
        elapsed = datetime.fromisoformat(
            observations[-1]["timestamp"]
        ) - datetime.fromisoformat(observations[0]["timestamp"])
        event(
            "summary",
            detected_people=count,
            searched_sectors=2,
            frames=len(observations),
            horizontal_distance_m=round(distance, 2),
            elapsed_s=round(elapsed.total_seconds(), 2),
        )
    except Exception as exc:
        report["status"] = "failed"
        event("error", error=str(exc))
        raise
    print(f"Run report: {manifest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=290)
    parser.add_argument("--instruction")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Fly wider switchback routes and capture along each leg.",
    )
    providers = parser.add_mutually_exclusive_group()
    providers.add_argument(
        "--mock", dest="provider", action="store_const", const=MockLLMProvider
    )
    providers.add_argument(
        "--deepseek", dest="provider", action="store_const", const=DeepSeekProvider
    )
    parser.set_defaults(provider=OpenAICompatibleProvider)
    parser.add_argument(
        "--weights", type=Path, default=Path(".external/models/yolox_tiny.onnx")
    )
    parser.add_argument("--esar", type=Path, default=Path(".external/ESAR-pinned"))
    parser.add_argument("--maps", type=Path, default=Path(".external/ESARMaps_win"))
    parser.add_argument("--observations", type=Path, default=Path("data/observations"))
    parser.add_argument("--runs", type=Path, default=Path("data/missions"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()

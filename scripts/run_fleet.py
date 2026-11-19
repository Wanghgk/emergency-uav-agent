"""Fly a configurable search and communications fleet in ESARBench."""

import argparse
import json
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from uuid import UUID, uuid4

import numpy as np

from backend.brain.ground_rescue import configured_entry, ground_rescue
from backend.brain.planning import DeepSeekProvider, MockLLMProvider, select_plan
from backend.cerebellum.person_detector import PersonDetector
from backend.cerebellum.pose_person import PosePersonDetector
from backend.cerebellum.rtdetr import RTDETRPersonDetector
from backend.simulation.basemap import WIDTH_M, capture_basemap, map_settings
from backend.simulation.esar_adapter import load_task
from backend.simulation.fleet import (
    fleet_roles,
    network_links,
    relay_choices,
    relay_options,
    scout_route,
)
from backend.simulation.ground_terrain import SurfaceGrid
from backend.simulation.mission_manifest import write_manifest
from backend.simulation.observation_collector import (
    capture,
    check_registration,
    save_observation,
    vector,
)
from backend.simulation.rescue_search import (
    Alarm,
    Terrain,
    person_results,
    review_search,
    track_detections,
)
from backend.simulation.session import HOST, PORT, simulator


def configure(source, path, start, map_center, fleet):
    map_settings(source, path, map_center, search_camera=True, camera_size=(640, 192))
    config = json.loads(path.read_text(encoding="utf-8"))
    template = config["Vehicles"]["Drone0"]
    spawns = {}
    # Spawn outside the orthographic footprint so vehicles cannot become DSM peaks.
    staging_north = map_center[0] + WIDTH_M / 2 + 80
    for index, (name, role) in enumerate(fleet.items()):
        spawn = (staging_north, map_center[1] + 8 * index, start[2] - 65 - 13 * index)
        spawns[name] = spawn
        vehicle = deepcopy(template)
        vehicle.update(dict(zip(("X", "Y", "Z"), spawn, strict=True)), Yaw=0)
        if role == "relay":
            vehicle.pop("Cameras", None)
        config["Vehicles"][name] = vehicle
    path.write_text(json.dumps(config), encoding="utf-8")
    return spawns


def position(client, name, offset):
    local = vector(client.simGetVehiclePose(vehicle_name=name).position)
    return tuple(value + shift for value, shift in zip(local, offset, strict=True))


def fly(api, name, origin, target, collision_stamp, *, client=None, speed=7):
    client = client or api.MultirotorClient(ip=HOST, port=PORT, timeout_value=85)
    current = position(client, name, origin)
    if math.dist(current[:2], target[:2]) < 1.5:
        # Keep each vertical RPC within its deadline even on steep alpine terrain.
        max_climb = min(speed, 3) * 40
        segments = math.ceil(abs(target[2] - current[2]) / max_climb)
        for down in np.linspace(current[2], target[2], segments + 1)[1:-1]:
            reach_waypoint(
                client,
                api,
                name,
                origin,
                (*target[:2], float(down)),
                collision_stamp,
                speed,
            )
    reach_waypoint(client, api, name, origin, target, collision_stamp, speed)


def reach_waypoint(client, api, name, origin, target, collision_stamp, speed):
    local = [value - shift for value, shift in zip(target, origin, strict=True)]
    for attempt in range(3):
        current = position(client, name, origin)
        if math.dist(current, target) < 1:
            return
        pace = speed if attempt == 0 else 3
        yaw = api.YawMode(False, 0)
        if math.dist(current[:2], target[:2]) < 1.5:
            vertical_speed = min(pace, 3)
            timeout = max(35, abs(target[2] - current[2]) / vertical_speed + 12)
            client.moveToZAsync(
                local[2],
                vertical_speed,
                timeout_sec=timeout,
                yaw_mode=yaw,
                vehicle_name=name,
            ).get()
        else:
            client.moveToPositionAsync(
                *local,
                pace,
                timeout_sec=70 if attempt == 0 else 25,
                yaw_mode=yaw,
                vehicle_name=name,
            ).get()
        collision = client.simGetCollisionInfo(vehicle_name=name)
        if collision.has_collided and collision.time_stamp > collision_stamp:
            raise RuntimeError(
                f"{name} contacted {collision.object_name or 'the scene'}"
            )
        # SimpleFlight may report completion before the aircraft finishes braking.
        time.sleep(0.75)
        actual = position(client, name, origin)
        error = math.dist(actual, target)
        if error <= 3:
            return
    raise RuntimeError(f"{name} missed {target}, actual {actual}, error {error:.1f} m")


def observe(client, api, task, ue_origin, run_id, detector, name, offset):
    from msgpackrpc.error import RPCError

    try:
        frame, rgb, depth = capture(
            client, api, task, ue_origin, vehicle=name, offset=offset
        )
    except (RPCError, OSError, ValueError) as exc:
        return {
            "vehicle_name": name,
            "reason": f"camera capture failed ({type(exc).__name__})",
        }
    localization_available = True
    try:
        check_registration(frame)
    except ValueError:
        localization_available = False
    localization_available &= (
        math.dist(frame.camera_pose.position, frame.drone_position) <= 6
    )
    frame.run_id = run_id
    if localization_available:
        detections, latency = detector.detect(frame, rgb, depth)
    else:
        detections, latency = detector.predict(rgb)
    save_observation(Path("data/observations"), frame, rgb, depth)
    return {
        "frame_id": frame.frame_id,
        "captured_at": frame.timestamp.isoformat(),
        "localization_available": localization_available,
        "vehicle_name": name,
        "drone_ned_m": frame.drone_position,
        "detections": [item.model_dump() for item in detections],
        "inference_ms": latency,
    }


def select(provider, task, choices, feedback):
    try:
        return select_plan(provider, task, choices, feedback).model_dump() | {
            "source": provider.name
        }
    except (RuntimeError, ValueError) as exc:
        return {
            "sector": next(iter(choices)),
            "source": "fallback",
            "rationale": str(exc),
        }


def run(args):
    task, official_start, ue_origin = load_task(
        args.esar, "task_data.json", args.task_id
    )
    entry = configured_entry(task.map_name)
    start = (
        entry["north"],
        entry["east"],
        min(official_start[2], entry["reference_down_m"]),
    )
    alarm = Alarm(
        **json.loads(args.alarms.read_text(encoding="utf-8"))[str(task.task_id)]
    )
    provider = DeepSeekProvider() if args.deepseek else MockLLMProvider()
    if args.detector == "pose":
        detector = PosePersonDetector(args.weights, args.proposal_weights)
    elif args.detector == "rtdetr":
        detector = RTDETRPersonDetector(args.weights, threshold=0.40)
    else:
        detector = PersonDetector(args.weights, threshold=0.33)
    run_id = (args.run_id or uuid4()).hex
    manifest = Path("data/missions") / f"{run_id}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "task": task.model_dump(),
        "instruction": task.prompt,
        "profile": "fleet",
        "status": "running",
        "planner": provider.name,
        "detector": detector.model_name,
        "detector_version": detector.model_version,
        "frame_interval_s": args.frame_interval,
        "weights_sha256": sha256(args.weights.read_bytes()).hexdigest(),
        "start_ned_m": start,
        "alarm": {**vars(alarm), "source": "simulated_call_coarse_area"},
        "fleet": args.fleet,
        "radio_range_m": args.radio_range,
        "events": [],
    }
    if args.detector == "pose":
        report["proposal_weights_sha256"] = sha256(
            args.proposal_weights.read_bytes()
        ).hexdigest()

    def event(kind, **details):
        line = {"type": kind, "timestamp": datetime.now(UTC).isoformat(), **details}
        report["events"].append(line)
        write_manifest(manifest, report)
        if kind != "fleet_telemetry":
            print(json.dumps(line, ensure_ascii=True), flush=True)

    tracks = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / "settings.json"
            spawns = configure(
                args.esar / "settings/settings.json",
                settings,
                start,
                official_start,
                args.fleet,
            )
            with simulator(args.maps, settings, task) as (client, api):
                tracks, grid, measured_start, aircraft_errors = fly_mission(
                    args,
                    client,
                    api,
                    task,
                    start,
                    ue_origin,
                    spawns,
                    alarm,
                    detector,
                    provider,
                    run_id,
                    event,
                    entry_down=entry["reference_down_m"],
                )
        report["people_results"] = person_results(tracks, ue_origin)
        report["start_ned_m"] = measured_start
        report["aircraft_errors"] = aircraft_errors
        if all(
            name in aircraft_errors
            for name, role in args.fleet.items()
            if role == "scout"
        ):
            raise RuntimeError("No perception aircraft can continue searching")
        try:
            report["ground_rescue"] = ground_rescue(
                grid, task.map_name, entry, report["people_results"], provider
            )
        except (OSError, ValueError) as exc:
            report["ground_rescue"] = {
                "status": "unavailable",
                "reason": str(exc),
                "routes": [],
            }
        event("ground_rescue", status=report["ground_rescue"]["status"])
        report["status"] = "partial" if aircraft_errors else "completed"
        event(
            "summary",
            frames=sum(item["type"] == "observation" for item in report["events"]),
            confirmed=sum(len(track.observations) >= 2 for track in tracks),
            expected=alarm.expected_people,
        )
    except Exception as exc:
        report["status"] = "failed"
        event("error", error=str(exc))
        raise


def fly_mission(
    args,
    client,
    api,
    task,
    start,
    ue_origin,
    offsets,
    alarm,
    detector,
    provider,
    run_id,
    event,
    *,
    entry_down,
):
    vehicles = args.fleet
    if missing := set(vehicles).difference(client.listVehicles()):
        raise RuntimeError(
            f"AirSim did not expose configured vehicles: {sorted(missing)}"
        )
    basemap = capture_basemap(client, api, task, Path("data/maps"))
    terrain = Terrain.read(client, api, basemap)
    grid = SurfaceGrid.from_terrain(terrain)
    start = grid.entry_point(*start[:2], entry_down)
    terrain.save(Path("data/ground-terrain") / f"{run_id}.npz", task)
    if args.terrain_sample:
        terrain.save(args.terrain_sample, task)
    half = alarm.half_width
    north_limits = (
        min(start[0], alarm.north - half) - 10,
        max(start[0], alarm.north + half) + 10,
    )
    east_limits = (
        min(start[1], alarm.east - half) - 10,
        max(start[1] + 8 * (len(vehicles) - 1), alarm.east + half) + 10,
    )
    top = min(
        terrain.highest((north, east_limits[0]), (north, east_limits[1]))
        for north in np.linspace(
            *north_limits, math.ceil((north_limits[1] - north_limits[0]) / 2) + 1
        )
    )
    base = (*start[:2], terrain.highest(start[:2], start[:2]) - 8)
    cruise = {name: top - 20 - 13 * index for index, name in enumerate(vehicles)}
    collisions = {}
    client.simPause(True)
    try:
        for index, name in enumerate(vehicles):
            shift = anchor(client, name, offsets[name])
            offsets[name] = shift
            spot = (start[0], start[1] + 8 * index, cruise[name])
            local = [value - origin for value, origin in zip(spot, shift, strict=True)]
            client.simSetVehiclePose(
                api.Pose(api.Vector3r(*local), api.to_quaternion(0, 0, 0)),
                True,
                vehicle_name=name,
            )
            collisions[name] = client.simGetCollisionInfo(vehicle_name=name).time_stamp
        for name in vehicles:
            client.enableApiControl(True, vehicle_name=name)
            if not client.armDisarm(True, vehicle_name=name):
                raise RuntimeError(f"{name} failed to arm")
            if vehicles[name] == "scout":
                client.simSetCameraPose(
                    "front_camera",
                    api.Pose(
                        api.Vector3r(0.5, 0, -0.1),
                        api.to_quaternion(math.radians(-74), 0, 0),
                    ),
                    vehicle_name=name,
                )
    finally:
        client.simPause(False)
    # Unreal applies the paused pose reset on its first running ticks.
    time.sleep(0.3)

    # Set a controller target for every aircraft before planning can leave one
    # waiting in place. Arming after a pose reset alone does not hold altitude.
    def hold_departure(name):
        control = api.MultirotorClient(ip=HOST, port=PORT, timeout_value=20)
        control.moveToZAsync(
            cruise[name] - offsets[name][2], 3, timeout_sec=15, vehicle_name=name
        ).get()

    with ThreadPoolExecutor(max_workers=len(vehicles)) as departure:
        list(departure.map(hold_departure, vehicles))
    ready = {name: position(client, name, offsets[name]) for name in vehicles}
    for index, (name, point) in enumerate(ready.items()):
        actor = vector(client.simGetObjectPose(name).position)
        expected_xy = start[0], start[1] + 8 * index
        if (
            not all(map(math.isfinite, actor))
            or math.dist(actor, point) > 1
            or math.dist(point[:2], expected_xy) > 2
            or abs(point[2] - cruise[name]) > 4
        ):
            raise RuntimeError(
                f"{name} departure pose {point} (actor {actor}) did not reach {(*expected_xy, cruise[name])}"
            )
    event("fleet_ready", vehicles=ready, ned_offsets=offsets, base_ned_m=base)
    event(
        "dispatch",
        waypoint_ned_m=(alarm.north, alarm.east, cruise[next(iter(vehicles))]),
    )
    tracks, aircraft_errors = coordinate(
        args,
        client,
        api,
        task,
        ue_origin,
        alarm,
        detector,
        provider,
        run_id,
        event,
        terrain,
        base,
        cruise,
        offsets,
        collisions,
        ready,
    )
    return tracks, grid, start, aircraft_errors


def anchor(client, name, spawn):
    """Measure the translation from vehicle-local NED to PlayerStart NED."""
    local = vector(client.simGetVehiclePose(vehicle_name=name).position)
    world = vector(client.simGetObjectPose(name).position)
    if not all(map(math.isfinite, world)) or math.dist(world[:2], spawn[:2]) > 2:
        raise RuntimeError(
            f"{name} actor does not match its configured spawn in map NED"
        )
    return tuple(value - relative for value, relative in zip(world, local, strict=True))


def search_route(terrain, route, locate, move, stopped, clearance, departure_altitude):
    if departure_altitude is not None and not stopped():
        move((*locate()[:2], departure_altitude), speed=8)
        if not stopped():
            move((*route[0], departure_altitude), speed=8)
    for index, xy in enumerate(route):
        if stopped():
            return False
        current = locate()
        surface = terrain.highest(current[:2], xy)
        if index == 0 and len(route) > 1 and math.dist(current[:2], xy) < 3:
            surface = min(surface, terrain.highest(xy, route[1]))
        down = surface - clearance
        if current[2] > down + 1:
            move((*current[:2], down), speed=5)
            if stopped():
                return False
        move((*xy, down), speed=5)
    return True


def coordinate(
    args,
    client,
    api,
    task,
    ue_origin,
    alarm,
    detector,
    provider,
    run_id,
    event,
    terrain,
    base,
    cruise,
    offsets,
    collisions,
    ready,
):
    frames, tracks, pending = Queue(), [], []
    stop = Event()
    scout_names = [name for name, role in args.fleet.items() if role == "scout"]
    relay_names = [name for name, role in args.fleet.items() if role == "relay"]
    scout_stop = {name: Event() for name in scout_names}
    aircraft_errors = {}
    clearance = {name: 14 + 5 * index for index, name in enumerate(scout_names)}
    assignments = {
        name: scout_names[index :: len(relay_names)]
        or [scout_names[index % len(scout_names)]]
        for index, name in enumerate(relay_names)
    }
    lead = {name: (*ready[name][:2], cruise[name]) for name in scout_names}
    rounds = dict.fromkeys(scout_names, 0)
    orientations = {}
    linked_counts = dict.fromkeys(scout_names, 0)
    scout_plans, scouts = {}, {}
    relay_moves, relay_plans = {}, {}
    review = None
    last_relay_plan = dict.fromkeys(relay_names, -math.inf)
    last_motion = last_relay_plan.copy()
    link_state = None
    same_count = -1
    ending = "search_budget"

    def record_aircraft_error(name, stage, exc):
        if name in aircraft_errors:
            return
        aircraft_errors[name] = {"stage": stage, "error": str(exc)}
        if name in scout_stop:
            scout_stop[name].set()
        event("fleet_aircraft_error", vehicle_name=name, **aircraft_errors[name])

    def sample_scout(name, delay):
        stopped = scout_stop[name]
        if stopped.wait(delay):
            return
        sensor = api.MultirotorClient(ip=HOST, port=PORT, timeout_value=8)
        next_sample = time.monotonic()
        while not stopped.is_set():
            frames.put(
                observe(
                    sensor, api, task, ue_origin, run_id, detector, name, offsets[name]
                )
            )
            next_sample = max(next_sample + args.frame_interval, time.monotonic())
            stopped.wait(max(0.0, next_sample - time.monotonic()))

    def forward_frames(connected):
        nonlocal pending
        while True:
            try:
                item = frames.get_nowait()
            except Empty:
                break
            if "frame_id" not in item:
                event("fleet_capture_error", **item)
                continue
            event("observation", **item)
            pending.append(item)
        waiting = []
        for item in pending:
            name = item["vehicle_name"]
            if not connected[name]:
                waiting.append(item)
                continue
            linked_counts[name] += 1
            track_detections(tracks, item["frame_id"], item["detections"], alarm)
            event("fleet_feedback", vehicle_name=name, observation_id=item["frame_id"])
        pending = waiting

    def search(name, route, round_number):
        vehicle_client = api.MultirotorClient(ip=HOST, port=PORT, timeout_value=85)

        def go(point, speed=8):
            fly(
                api,
                name,
                offsets[name],
                point,
                collisions[name],
                client=vehicle_client,
                speed=speed,
            )

        return search_route(
            terrain,
            route,
            lambda: position(vehicle_client, name, offsets[name]),
            go,
            lambda: stop.is_set() or scout_stop[name].is_set(),
            clearance[name],
            cruise[name] if round_number == 1 else None,
        )

    def submit_scout(pool, name, current):
        possible = ("EW", "NS")
        routes = {
            axis: list(
                scout_route(
                    alarm, scout_names.index(name), len(scout_names), axis, args.spacing
                )
            )
            for axis in possible
        }
        for route in routes.values():
            if math.dist(current[:2], route[-1]) < math.dist(current[:2], route[0]):
                route.reverse()
        choices = sorted(
            possible,
            key=lambda axis: (
                axis == orientations.get(name),
                math.dist(current[:2], routes[axis][0]),
            ),
        )
        scout_plans[name] = (
            pool.submit(
                select,
                provider,
                f"{task.prompt} Plan the {name} sector of the alarm search. EW and NS "
                "are validated perpendicular sweep paths. Prefer complementary coverage on follow-up.",
                choices,
                {
                    "round": rounds[name] + 1,
                    "previous_axis": orientations.get(name),
                    "alarm_people": alarm.expected_people,
                    "linked_frames": linked_counts[name],
                    "estimated_candidates": len(tracks),
                    "route_options": {
                        axis: {
                            "waypoint_count": len(routes[axis]),
                            "entry_distance_m": round(
                                math.dist(current[:2], routes[axis][0])
                            ),
                        }
                        for axis in choices
                    },
                },
            ),
            routes,
        )

    with (
        ThreadPoolExecutor(max_workers=len(args.fleet)) as flight,
        ThreadPoolExecutor(max_workers=len(args.fleet) + 1) as plans,
        ThreadPoolExecutor(max_workers=len(scout_names)) as cameras,
    ):
        camera_jobs = {
            name: cameras.submit(
                sample_scout, name, args.frame_interval * i / len(scout_names)
            )
            for i, name in enumerate(scout_names)
        }
        next_tick = time.monotonic()
        try:
            while True:
                for name, job in list(camera_jobs.items()):
                    if job.done():
                        camera_jobs.pop(name)
                        try:
                            job.result()
                        except Exception as exc:
                            record_aircraft_error(name, "camera", exc)
                now = time.monotonic()
                positions = {
                    name: position(client, name, offsets[name]) for name in args.fleet
                }
                links, connected = network_links(
                    terrain,
                    base,
                    {name: positions[name] for name in scout_names},
                    {
                        name: positions[name]
                        for name in relay_names
                        if name not in aircraft_errors
                    },
                    args.radio_range,
                )
                event(
                    "fleet_telemetry",
                    positions=positions,
                    radio_links=[
                        {
                            "from": origin,
                            "to": target,
                            "distance_m": round(result[0], 1),
                            "clearance_m": round(result[1], 1),
                            "margin": round(result[2], 1),
                            "ok": result[2] >= 0,
                        }
                        for (origin, target), result in links.items()
                    ],
                )
                changed = link_state is not None and connected != link_state
                link_state = connected
                forward_frames(connected)

                if review and review.done():
                    decision = review.result()
                    event("search_review", **decision)
                    if decision["action"] == "finish":
                        ending = "planner_finish"
                        stop.set()
                    review = None
                confirmed = sum(len(track.observations) >= 2 for track in tracks)
                if (
                    not stop.is_set()
                    and review is None
                    and confirmed >= alarm.expected_people
                    and confirmed != same_count
                ):
                    same_count = confirmed
                    review = plans.submit(
                        review_search,
                        provider,
                        deepcopy(tracks),
                        alarm.expected_people,
                        True,
                    )

                for name in scout_names:
                    if name in scouts and scouts[name].done():
                        try:
                            if scouts.pop(name).result():
                                event(
                                    "fleet_round_complete",
                                    vehicle_name=name,
                                    round=rounds[name],
                                )
                                if rounds[name] == args.rounds:
                                    scout_stop[name].set()
                        except Exception as exc:
                            record_aircraft_error(name, "search", exc)
                        lead[name] = positions[name]
                    if name in scout_plans and scout_plans[name][0].done():
                        future, routes = scout_plans.pop(name)
                        if stop.is_set() or name in aircraft_errors:
                            continue
                        try:
                            plan = future.result()
                        except Exception as exc:
                            record_aircraft_error(name, "planning", exc)
                            continue
                        axis = plan["sector"]
                        route = routes[axis][: args.points_per_scout]
                        rounds[name] += 1
                        orientations[name] = axis
                        lead[name] = (*route[-1], positions[name][2])
                        event(
                            "fleet_plan",
                            vehicle_name=name,
                            round=rounds[name],
                            waypoints_ned_xy=route,
                            **plan,
                        )
                        scouts[name] = flight.submit(search, name, route, rounds[name])
                    if (
                        not stop.is_set()
                        and name not in aircraft_errors
                        and name not in scouts
                        and name not in scout_plans
                        and rounds[name] < args.rounds
                    ):
                        submit_scout(plans, name, positions[name])

                for relay, future in list(relay_moves.items()):
                    if future.done():
                        relay_moves.pop(relay)
                        try:
                            future.result()
                        except Exception as exc:
                            record_aircraft_error(relay, "relay", exc)
                if (
                    not scouts
                    and not scout_plans
                    and review is None
                    and not relay_moves
                ):
                    break

                for relay, assigned in assignments.items():
                    assigned = [
                        name for name in assigned if name not in aircraft_errors
                    ]
                    if (
                        not assigned
                        or relay in aircraft_errors
                        or (not scouts and not scout_plans)
                    ):
                        continue
                    assigned_links = {name: connected[name] for name in assigned}
                    options, margins = relay_options(
                        terrain,
                        base,
                        positions[relay],
                        {name: positions[name] for name in assigned},
                        {name: lead[name] for name in assigned},
                        cruise[relay],
                        args.radio_range,
                    )
                    choices = relay_choices(margins)
                    best = max(choices, key=margins.__getitem__)
                    picked, source = best, "local_link_policy"
                    if relay in relay_plans and relay_plans[relay].done():
                        plan = relay_plans.pop(relay).result()
                        if plan["sector"] in choices and plan["source"] != "fallback":
                            picked, source = plan["sector"], plan["source"]
                        event(
                            "fleet_relay_plan",
                            vehicle_name=relay,
                            recommendation=plan["sector"],
                            accepted=source == plan["source"],
                            rationale=plan["rationale"],
                            source=plan["source"],
                        )
                    moving_needed = math.dist(positions[relay], options[best]) >= 10
                    if (
                        relay not in relay_plans
                        and (moving_needed or changed)
                        and now - last_relay_plan[relay] >= args.relay_interval
                    ):
                        relay_plans[relay] = plans.submit(
                            select,
                            provider,
                            "Choose a communications relay waypoint for a simulated SAR fleet. "
                            "Maximize the worst current and projected link margin while avoiding unnecessary movement. "
                            "HOLD stays in place, BALANCE uses current aircraft and base, LEAD uses intended destinations, "
                            "SCOUT_MID follows scouts; MIDPOINT is between the base and the scout centroid. "
                            "Margin is a metre-equivalent simulation score, not dB. "
                            "A negative margin predicts an outage.",
                            sorted(choices, key=margins.__getitem__, reverse=True),
                            {
                                "worst_margin_by_option": {
                                    name: round(margins[name], 1) for name in choices
                                },
                                "connected_scouts": assigned_links,
                                "displacement_m": {
                                    name: round(
                                        math.dist(positions[relay], options[name])
                                    )
                                    for name in choices
                                },
                            },
                        )
                        last_relay_plan[relay] = now
                    if (
                        relay not in relay_moves
                        and now - last_motion[relay] >= 3
                        and (
                            relay not in relay_plans
                            or margins["HOLD"] < 8
                            or now - last_relay_plan[relay] > 4
                        )
                        and math.dist(positions[relay], options[picked]) >= 8
                        and (
                            margins[picked] > margins["HOLD"] + 3
                            or not all(assigned_links.values())
                        )
                    ):
                        event(
                            "fleet_relay_move",
                            vehicle_name=relay,
                            sector=picked,
                            source=source,
                            waypoint_ned_m=options[picked],
                            projected_margin=round(margins[picked], 1),
                        )
                        relay_moves[relay] = flight.submit(
                            fly,
                            api,
                            relay,
                            offsets[relay],
                            options[picked],
                            collisions[relay],
                            speed=9,
                        )
                        last_motion[relay] = now
                next_tick += 1
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.monotonic()
        finally:
            stop.set()
            for event_flag in scout_stop.values():
                event_flag.set()
        for name, job in camera_jobs.items():
            try:
                job.result()
            except Exception as exc:
                record_aircraft_error(name, "camera", exc)
        forward_frames(connected)
    event(
        "fleet_end",
        reason="scout_failures"
        if all(name in aircraft_errors for name in scout_names)
        else ending,
        linked_observations=linked_counts,
        unforwarded_frames=len(pending),
    )
    return tracks, aircraft_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=UUID)
    parser.add_argument("--esar", type=Path, default=Path(".external/ESAR-pinned"))
    parser.add_argument("--maps", type=Path, default=Path(".external/ESARMaps_win"))
    parser.add_argument(
        "--detector", choices=("pose", "rtdetr", "yolox"), default="pose"
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=3,
        help="Seconds between scout inspections",
    )
    parser.add_argument(
        "--alarms", type=Path, default=Path("configs/fleet_alarms.json")
    )
    parser.add_argument("--task-id", type=int, default=290)
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=12)
    parser.add_argument(
        "--points-per-scout",
        type=int,
        help="Optional per-scout, per-round diagnostic limit",
    )
    parser.add_argument("--radio-range", type=float, default=320)
    parser.add_argument(
        "--scouts", type=int, default=2, help="Number of perception aircraft"
    )
    parser.add_argument(
        "--relays",
        type=int,
        default=1,
        help="Number of relay aircraft (0 uses direct base links)",
    )
    parser.add_argument("--relay-interval", type=float, default=12)
    parser.add_argument(
        "--terrain-sample",
        type=Path,
        help="Optionally retain measured terrain locally as NPZ",
    )
    args = parser.parse_args()
    if not math.isfinite(args.frame_interval) or not 1 <= args.frame_interval <= 5:
        parser.error("--frame-interval must be between 1 and 5 seconds")
    if args.weights is None:
        model = {
            "pose": "yolo11s-pose.onnx",
            "rtdetr": "rtdetr_v2_r18vd.onnx",
            "yolox": "yolox_tiny.onnx",
        }[args.detector]
        args.weights = Path(".external/models") / model
    args.proposal_weights = Path(".external/models/rtdetr_v2_r18vd.onnx")
    if args.detector == "pose" and not args.proposal_weights.is_file():
        parser.error(
            f"missing candidate proposal weights: {args.proposal_weights}; see docs/deployment-windows.md"
        )
    if not args.weights.is_file():
        parser.error(
            f"missing {args.detector} weights: {args.weights}; see docs/deployment-windows.md"
        )
    values = [
        args.rounds,
        args.spacing,
        args.radio_range,
        args.relay_interval,
        args.points_per_scout or 0,
    ]
    if not all(math.isfinite(value) and value > 0 for value in values[:-1]) or (
        args.points_per_scout is not None and args.points_per_scout < 1
    ):
        parser.error("flight limits must be finite and positive")
    try:
        args.fleet = fleet_roles(args.scouts, args.relays)
    except ValueError as error:
        parser.error(str(error))
    run(args)


if __name__ == "__main__":
    main()

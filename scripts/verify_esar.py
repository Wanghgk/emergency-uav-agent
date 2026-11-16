"""Verify one real Windows ESAR scenario and replay its RGB-D captures."""

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from backend.simulation.esar_adapter import launch_args, load_task
from backend.simulation.observation_collector import (
    capture,
    compare_static_depth,
    load_observation,
    map_point,
    save_observation,
)

HOST = "127.0.0.1"
PORT = 41451
VEHICLE = "Drone0"


def connect(api, process):
    from msgpackrpc.error import RPCError

    deadline = time.monotonic() + 120
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"ESAR exited during startup with code {process.returncode}"
            )
        try:
            with socket.create_connection((HOST, PORT), timeout=1):
                pass
            client = api.MultirotorClient(ip=HOST, port=PORT, timeout_value=5)
            if client.ping() and VEHICLE in client.listVehicles():
                return client
        except (OSError, RuntimeError, RPCError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(f"AirSim did not expose {VEHICLE} within 120 s") from last_error


def capture_at_hover(client, api, task, origin, output):
    # A vehicle teleport can update kinematics before the camera component ticks.
    for _ in range(5):
        time.sleep(0.2)
        client.simPause(True)
        try:
            observation, rgb, depth = capture(client, api, task, origin)
        finally:
            client.simPause(False)
        separation = np.linalg.norm(
            np.subtract(observation.camera_pose.position, observation.drone_position)
        )
        if separation < 1:
            break
    else:
        raise RuntimeError("front camera pose did not catch up to the drone pose")
    finite = np.argwhere((depth >= 1) & (depth <= 200))
    if not len(finite):
        raise ValueError("AirSim returned no usable depth pixels")
    center = (np.array(depth.shape) - 1) / 2
    v, u = finite[np.argmin(np.sum((finite - center) ** 2, axis=1))]
    reference = map_point(observation, depth, int(u), int(v))
    manifest = save_observation(output, observation, rgb, depth)
    restored, _, restored_depth = load_observation(manifest)
    if reference != map_point(restored, restored_depth, int(u), int(v)):
        raise RuntimeError(f"map point differs after replay of {manifest}")
    print(json.dumps({"manifest": str(manifest), **reference}), flush=True)
    return restored, restored_depth


def verify(args):
    task, start, origin = load_task(args.esar, args.tasks, args.task_id)
    print(json.dumps({"task": task.model_dump(), "start_ned_m": start}), flush=True)
    if args.task_only:
        return
    command = launch_args(args.maps, args.esar / "settings" / "settings.json", task)
    import airsim as api

    with socket.socket() as probe:
        if probe.connect_ex((HOST, PORT)) == 0:
            raise RuntimeError(f"AirSim port {PORT} is already in use")
    process = subprocess.Popen(
        command,
        cwd=Path(command[0]).parent,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    client = None
    try:
        client = connect(api, process)
        client.enableApiControl(True, vehicle_name=VEHICLE)
        if not client.armDisarm(True, vehicle_name=VEHICLE):
            raise RuntimeError("simulator refused to arm Drone0")
        client.simSetVehiclePose(
            api.Pose(api.Vector3r(*start), api.to_quaternion(0, 0, 0)),
            True,
            vehicle_name=VEHICLE,
        )
        initial, initial_depth = capture_at_hover(
            client, api, task, origin, args.output
        )
        if np.linalg.norm(np.subtract(initial.drone_position, start)) > 2:
            raise RuntimeError("reported drone pose differs from configured task start")
        target_z = start[2] - 4
        client.moveToZAsync(target_z, 2, timeout_sec=8, vehicle_name=VEHICLE).join()
        moved, moved_depth = capture_at_hover(client, api, task, origin, args.output)
        if abs(moved.drone_position[2] - target_z) > 1.5:
            raise RuntimeError(
                f"drone ended at NED Z={moved.drone_position[2]:.3f}, requested {target_z:.3f}"
            )
        camera_delta = np.subtract(
            moved.camera_pose.position, initial.camera_pose.position
        )
        drone_delta = np.subtract(moved.drone_position, initial.drone_position)
        if drone_delta[2] > -2:
            raise RuntimeError(
                f"measured upward displacement was only {-drone_delta[2]:.3f} m"
            )
        print(
            json.dumps({"drone_displacement_ned_m": drone_delta.tolist()}), flush=True
        )
        if np.linalg.norm(camera_delta - drone_delta) > 0.5:
            raise RuntimeError("camera and drone movement disagree")
        print(
            json.dumps(
                compare_static_depth(initial, initial_depth, moved, moved_depth)
            ),
            flush=True,
        )
        print(
            "Live ESAR RGB-D, simulator motion, camera pose and recording replay passed."
        )
    finally:
        try:
            if client is not None:
                client.enableApiControl(False, vehicle_name=VEHICLE)
        finally:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esar", type=Path, default=Path(".external/ESAR-pinned"))
    parser.add_argument("--maps", type=Path, default=Path(".external/ESARMaps_win"))
    parser.add_argument("--tasks", default="task_data.json")
    parser.add_argument("--task-id", default=290, type=int)
    parser.add_argument("--output", type=Path, default=Path("data/observations"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--task-only", action="store_true")
    mode.add_argument("--replay", type=Path, metavar="OBSERVATION_JSON")
    parser.add_argument("--pixel", type=int, nargs=2, metavar=("U", "V"))
    args = parser.parse_args()
    if args.replay:
        if args.pixel is None:
            parser.error("--replay requires --pixel U V in the depth image")
        record, _, depth = load_observation(args.replay)
        print(json.dumps(map_point(record, depth, *args.pixel)))
        return
    if args.pixel is not None:
        parser.error("--pixel requires --replay")
    verify(args)


if __name__ == "__main__":
    main()

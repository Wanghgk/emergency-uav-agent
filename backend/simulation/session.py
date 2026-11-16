"""Own one local packaged ESARBench simulator process."""

import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from .esar_adapter import launch_args

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
            client = api.MultirotorClient(ip=HOST, port=PORT, timeout_value=30)
            if client.ping() and VEHICLE in client.listVehicles():
                return client
        except (OSError, RuntimeError, RPCError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(f"AirSim did not expose {VEHICLE} within 120 s") from last_error


@contextmanager
def simulator(maps, settings, task):
    import airsim as api

    command = launch_args(maps, settings, task)
    with socket.socket() as probe:
        if probe.connect_ex((HOST, PORT)) == 0:
            raise RuntimeError(f"AirSim port {PORT} is already in use")
    process = subprocess.Popen(
        command,
        cwd=Path(command[0]).parent,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        yield connect(api, process), api
    finally:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
        process.wait(timeout=10)

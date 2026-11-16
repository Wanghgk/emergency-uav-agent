"""Render and save the ESARBench task terrain basemap."""

import argparse
import tempfile
from pathlib import Path

from backend.simulation.basemap import capture_basemap, map_settings
from backend.simulation.esar_adapter import load_task
from backend.simulation.session import simulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esar", type=Path, default=Path(".external/ESAR-pinned"))
    parser.add_argument("--maps", type=Path, default=Path(".external/ESARMaps_win"))
    parser.add_argument("--task-id", type=int, default=290)
    parser.add_argument("--output", type=Path, default=Path("data/maps"))
    args = parser.parse_args()
    task, start, _ = load_task(args.esar, "task_data.json", args.task_id)
    with tempfile.TemporaryDirectory() as directory:
        settings = Path(directory) / "settings.json"
        map_settings(args.esar / "settings/settings.json", settings, start)
        with simulator(args.maps, settings, task) as (client, api):
            basemap = capture_basemap(client, api, task, args.output)
    print(basemap.bounds())


if __name__ == "__main__":
    main()

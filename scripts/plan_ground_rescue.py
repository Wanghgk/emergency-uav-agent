"""Replay ground-route planning against an archived mission's observed people."""

import argparse
import json
from pathlib import Path
from uuid import UUID

from backend.brain.ground_rescue import configured_entry, ground_rescue
from backend.brain.planning import DeepSeekProvider, MockLLMProvider
from backend.simulation.ground_terrain import SurfaceGrid
from backend.simulation.rescue_search import archived_people


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=UUID, required=True)
    parser.add_argument("--terrain", type=Path)
    parser.add_argument("--deepseek", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id.hex
    mission = json.loads(
        (Path("data/missions") / f"{run_id}.json").read_text(encoding="utf-8")
    )
    if mission["status"] not in {"finished", "completed", "interrupted"}:
        parser.error("only completed or interrupted missions can be replayed")
    task = mission["task"]
    native = Path("data/ground-terrain") / f"{run_id}.npz"
    terrain_path = args.terrain or (
        native
        if native.is_file()
        else Path("data/terrain-comparison") / f"{task['task_id']}.npz"
    )
    if not terrain_path.is_file():
        parser.error(f"missing terrain raster for this task: {terrain_path}")
    observed_frames = {
        event["frame_id"]
        for event in mission["events"]
        if event["type"] == "observation"
    }
    people = [
        person
        for person in archived_people(mission, Path("data/observations"))
        if person.get("evidence", {}).get("frame_id") in observed_frames
    ]
    provider = DeepSeekProvider() if args.deepseek else MockLLMProvider()
    map_id = task["map_name"]
    result = ground_rescue(
        SurfaceGrid.from_file(terrain_path, task=task),
        map_id,
        configured_entry(map_id),
        people,
        provider,
    )
    result["source_run_id"] = run_id
    target = Path("data/ground-routes") / f"{run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(target)
    print(f"{result['status']} {target}")


if __name__ == "__main__":
    main()

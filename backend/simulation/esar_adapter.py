"""ESAR task setup for the Windows packaged simulator."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pydantic import BaseModel, Field

from .coordinate_transform import Point


class PublicTask(BaseModel):
    task_id: int
    map_name: str
    snapshot_id: int
    prompt: str
    weather_id: int = Field(ge=0, le=12)
    time_of_day: int = Field(ge=0, le=2400)
    original_time_of_day: int | None = Field(default=None, ge=0, le=2400)


def load_task(esar: Path, task_file: str, task_id: int):
    tasks = json.loads((esar / "data" / task_file).read_text(encoding="utf-8"))
    matches = [task for task in tasks if task["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one official task with id {task_id}, got {len(matches)}"
        )
    task = matches[0]
    public = PublicTask.model_validate(task)
    if not 700 <= public.time_of_day < 1800:
        public.original_time_of_day = public.time_of_day
        public.time_of_day = 1200

    path = esar / "source" / "coordinate_utils.py"
    spec = spec_from_file_location("official_esar_coordinate_utils", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    converter = module.CoordinateConverter
    if public.map_name not in converter.PLAYER_STARTS:
        raise ValueError(f"no official UE origin for map {public.map_name}")
    start = converter.to_airsim(public.map_name, task["start_location"])
    start_ned: Point = tuple(start[axis] for axis in "xyz")
    return public, start_ned, converter.PLAYER_STARTS[public.map_name]


def launch_args(maps: Path, settings: Path, task: PublicTask) -> list[str]:
    map_dir = maps / task.map_name
    name = f"{task.map_name}.exe"
    candidates = (map_dir / name, map_dir / "Windows" / name)
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        raise FileNotFoundError(
            f"ESAR map executable not found under {map_dir.resolve()}"
        )
    if not settings.is_file():
        raise FileNotFoundError(settings)
    return [
        str(executable.resolve()),
        f"-TaskNum={task.snapshot_id % 15}",
        f"-Weather={task.weather_id}",
        f"-Time={task.time_of_day}",
        f"-settings={settings.resolve()}",
        "-RenderOffScreen",
        "-windowed",
        "-ResX=640",
        "-ResY=480",
    ]

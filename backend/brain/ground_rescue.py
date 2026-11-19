"""Plan a simulated rescue approach from terrain and observed people positions."""

import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.simulation.ground_terrain import (
    OBSTACLE_CLEARANCE_M,
    check_route,
    find_route,
)

ENTRY_CONFIG = Path("configs/rescue_entries.json")
MAX_SLOPE_DEG = 30
APPROACH_M = 12


class TerrainInspection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_ids: list[str] = Field(min_length=1, max_length=2)


class GroundDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_id: str | None
    rationale: str = Field(min_length=1, max_length=300)


def configured_entry(map_id, path=ENTRY_CONFIG):
    try:
        return json.loads(path.read_text(encoding="utf-8"))[map_id]
    except KeyError:
        raise ValueError(f"no configured rescue entry for {map_id}") from None


def ground_rescue(grid, map_id, entry, people, provider):
    start = grid.entry_point(
        entry["north"], entry["east"], entry.get("reference_down_m")
    )
    results = [
        plan_person(grid, map_id, start, person, provider)
        for person in people
        if person["status"] == "confirmed"
    ]
    return {
        "status": "reviewed" if results else "waiting_for_confirmation",
        "entry": {"label": entry["label"], "ned_m": start},
        "coordinate_frame": "AirSim NED m",
        "surface_source": "ESARBench UE5 task terrain · AirSim depth-derived TIN",
        "max_slope_deg": MAX_SLOPE_DEG,
        "obstacle_clearance_m": OBSTACLE_CLEARANCE_M,
        "max_approach_distance_m": APPROACH_M,
        "routes": results,
    }


def plan_person(grid, map_id, start_ned, person, provider):
    target = tuple(person["ned_m"][:2])
    result = {
        "person_id": person["id"],
        "estimated_person_ned_m": person["ned_m"],
        "horizontal_observation_spread_m": person.get(
            "horizontal_observation_spread_m"
        ),
    }
    start = grid.cell(start_ned)
    try:
        goal = grid.cell(target)
    except ValueError:
        return {
            **result,
            "status": "no_route",
            "reason": "estimated position is outside the terrain raster",
        }
    paths = {}
    for route_id, penalty in (("short", 0.0), ("gentler", 2.7)):
        path = find_route(
            grid,
            start,
            goal,
            target,
            max_slope=MAX_SLOPE_DEG,
            approach_m=APPROACH_M,
            penalty=penalty,
        )
        if path and path not in paths.values():
            paths[route_id] = path
    if not paths:
        return {
            **result,
            "status": "no_route",
            "reason": "no connected approach met obstacle clearance, DSM coverage, and slope limits",
        }
    options = {
        route_id: {
            **check_route(grid, path, start, target, MAX_SLOPE_DEG, APPROACH_M),
            "route_preview_ne_m": [
                [round(value, 1) for value in grid.point(path[index])[:2]]
                for index in sorted(
                    {
                        0,
                        len(path) // 4,
                        len(path) // 2,
                        len(path) * 3 // 4,
                        len(path) - 1,
                    }
                )
            ],
        }
        for route_id, path in paths.items()
    }
    windows = {"approach": grid.window((goal,), 4)}
    for route_id, path in paths.items():
        steep = max(path, key=lambda cell: grid.slope[cell])
        windows[f"{route_id}_steepest"] = grid.window((steep,), 4)
        windows[f"{route_id}_midway"] = grid.window((path[len(path) // 2],), 4)
    window = grid.window(
        tuple(cell for path in paths.values() for cell in path) + (goal,), 10
    )
    stride = max(1, math.ceil(max(window[1] - window[0], window[3] - window[2]) / 24))
    context = {
        "task": "Select a simulated ground rescue approach. A DSM may represent vegetation or rocks.",
        "map_id": map_id,
        "coordinate_frame": "AirSim NED metres",
        "obstacle_clearance_m": OBSTACLE_CLEARANCE_M,
        "terrain_note": "contains_blocked_cells marks raised obstacles, steep breaks or missing terrain; "
        "at coarse resolution it means at least one original cell in the block is excluded.",
        "entry_ned_m": start_ned,
        **result,
        "draft_routes": options,
        "terrain_source": "UE5 terrain captured by AirSim in the current ESARBench task",
        "terrain_overview": grid.overview_packet(),
        "terrain_summary": grid.packet(window, stride=stride),
        "inspection_windows": {
            name: {
                "center_ne_m": grid.point(((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2))[:2],
                "native_cell_count": (r1 - r0) * (c1 - c0),
            }
            for name, (r0, r1, c0, c1) in windows.items()
        },
    }
    inspected = []
    try:
        first = provider.inspect_ground(context)
        inspected = list(dict.fromkeys(first.window_ids))
        if not set(inspected).issubset(windows):
            raise ValueError("planner requested an unavailable terrain window")
        final = provider.choose_ground(
            {
                **context,
                "terrain_details": {
                    name: grid.packet(windows[name]) for name in inspected
                },
            }
        )
        if final.route_id is None:
            return {
                **result,
                "status": "manual_review",
                "model": provider.name,
                "inspected_windows": inspected,
                "reason": final.rationale,
            }
        if final.route_id not in paths:
            raise ValueError("planner selected an unavailable route")
        selected, rationale, source = final.route_id, final.rationale, provider.name
    except (RuntimeError, ValueError) as exc:
        selected = min(
            options,
            key=lambda name: (
                options[name]["max_slope_deg"],
                options[name]["distance_m"],
            ),
        )
        rationale, source = str(exc), "local_fallback"
    path = paths[selected]
    metrics = check_route(grid, path, start, target, MAX_SLOPE_DEG, APPROACH_M)
    return {
        **result,
        "status": "surface_checked",
        "model": source,
        "route_id": selected,
        "reason": rationale,
        "inspected_windows": inspected,
        "metrics": metrics,
        "waypoints_ned_m": [
            [round(value, 3) for value in grid.point(cell)] for cell in path
        ],
    }

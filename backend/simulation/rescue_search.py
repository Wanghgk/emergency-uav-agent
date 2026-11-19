"""Local SAR coverage and scoring primitives for ESAR simulation."""

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .basemap import read_terrain_down
from .coordinate_transform import ned_to_ue
from .observation_collector import Observation


@dataclass(frozen=True)
class Alarm:
    north: float
    east: float
    expected_people: int
    half_width: float = 40

    def contains(self, point):
        return (
            max(abs(point[0] - self.north), abs(point[1] - self.east))
            <= self.half_width
        )


def sweep(alarm, start_corner, step, cross=False):
    """Cover the alarm rectangle with a selectable serpentine ordering."""
    count = math.ceil(2 * alarm.half_width / step) + 1
    offsets = np.linspace(-alarm.half_width, alarm.half_width, count).tolist()
    north_offsets = offsets if start_corner.startswith("S") else offsets[::-1]
    east_offsets = offsets if start_corner.endswith("W") else offsets[::-1]
    rows, cols = (
        (east_offsets, north_offsets) if cross else (north_offsets, east_offsets)
    )
    for row_id, row in enumerate(rows):
        for col in cols if row_id % 2 == 0 else cols[::-1]:
            n, e = (col, row) if cross else (row, col)
            yield alarm.north + n, alarm.east + e


@dataclass
class Track:
    observations: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    @property
    def point(self):
        return tuple(np.median(list(self.observations.values()), axis=0).tolist())


def track_detections(tracks, frame_id, detections, alarm, *, minimum_confidence=0.44):
    for item in detections:
        point = item["ned_m"]
        if (
            point is None
            or not alarm.contains(point)
            or item["confidence"] < minimum_confidence
        ):
            continue
        neighbours = [t for t in tracks if math.dist(t.point[:2], point[:2]) <= 1.8]
        if neighbours:
            closest = min(neighbours, key=lambda t: math.dist(t.point[:2], point[:2]))
        else:
            closest = Track()
            tracks.append(closest)
        if frame_id not in closest.observations:
            closest.observations[frame_id] = point
            closest.evidence[frame_id] = {
                "frame_id": frame_id,
                "bbox": item.get("bbox"),
                "confidence": item["confidence"],
            }


def person_results(tracks, origin):
    return [
        {
            "id": index,
            "status": "confirmed" if len(track.observations) >= 2 else "candidate",
            "ned_m": track.point,
            "ue_cm": ned_to_ue(track.point, origin),
            "observations": len(track.observations),
            "horizontal_observation_spread_m": (
                round(
                    max(
                        math.dist(track.point[:2], point[:2])
                        for point in track.observations.values()
                    ),
                    2,
                )
                if len(track.observations) >= 2
                else None
            ),
            "evidence": max(
                track.evidence.values(), key=lambda item: item["confidence"]
            ),
        }
        for index, track in enumerate(tracks, 1)
    ]


def archived_people(run, observations):
    if "people_results" in run:
        return run["people_results"]
    failed_fleet = run.get("profile") == "fleet" and run.get("status") == "failed"
    if not run.get("alarm") or (
        not failed_fleet and run.get("status") not in {"finished", "interrupted"}
    ):
        return []
    delivered = (
        {
            event["observation_id"]
            for event in run["events"]
            if event["type"] == "fleet_feedback"
        }
        if failed_fleet
        else None
    )
    alarm = Alarm(
        **{key: value for key, value in run["alarm"].items() if key != "source"}
    )
    tracks = []
    origin = None
    for event in run["events"]:
        if (
            event["type"] != "observation"
            or event.get("sector") == "START"
            or not event.get("detections")
        ):
            continue
        frame_id = event["frame_id"]
        if delivered is not None and frame_id not in delivered:
            continue
        if not re.fullmatch(r"[0-9a-f]{32}", frame_id):
            continue
        try:
            frame = Observation.model_validate_json(
                (observations / frame_id / "observation.json").read_bytes()
            )
        except (OSError, ValueError):
            continue
        if frame.run_id != run["run_id"] or frame.task_id != run["task"]["task_id"]:
            continue
        origin = frame.map_origin_ue_cm
        track_detections(tracks, frame_id, event["detections"], alarm)
    return person_results(tracks, origin) if origin is not None else []


def review_search(provider, tracks, expected, can_continue):
    confirmed = sum(len(track.observations) >= 2 for track in tracks)
    context = {
        "confirmed_count": confirmed,
        "candidate_count": len(tracks) - confirmed,
        "expected_count": expected,
        "can_continue": can_continue,
    }
    decision = None
    try:
        decision = provider.review_search(context)
    except (RuntimeError, ValueError):
        pass
    finish = not can_continue or (
        decision is not None and decision.action == "finish" and confirmed >= expected
    )
    return {
        **context,
        "provider": provider.name,
        "recommendation": decision.action if decision else None,
        "rationale": decision.rationale if decision else None,
        "action": "finish" if finish else "continue",
        "source": "search_limit"
        if not can_continue
        else ("planner" if decision else "fallback"),
    }


def match_people(points, truths, tolerance_m=4):
    """Assign at most one observed position per distinct reference victim."""
    neighbours = [
        [
            index
            for index, _ in sorted(
                enumerate(truths), key=lambda row: math.dist(point[:2], row[1][:2])
            )
            if math.dist(point[:2], truths[index][:2]) <= tolerance_m
        ]
        for point in points
    ]
    matched = {}

    def assign(index, visited):
        for target in neighbours[index]:
            if target in visited:
                continue
            visited.add(target)
            if target not in matched or assign(matched[target], visited):
                matched[target] = index
                return True
        return False

    for index in range(len(points)):
        assign(index, set())
    return matched


class Terrain:
    def __init__(self, basemap, image):
        self.bounds = basemap.bounds()
        self.image = image

    def save(self, path, task):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            np.savez_compressed(
                output,
                down=self.image,
                task_id=task.task_id,
                map_id=task.map_name,
                **self.bounds,
            )

    @classmethod
    def read(cls, client, api, basemap):
        return cls(basemap, read_terrain_down(client, api, basemap))

    def profile(self, a, b):
        distance = math.dist(a, b)
        north, east = np.linspace(a, b, max(2, math.ceil(distance / 2) + 1)).T
        height, width = self.image.shape
        bounds = self.bounds
        rows = (
            (bounds["north"] - north) / (bounds["north"] - bounds["south"]) * height
        ).astype(int)
        cols = (
            (east - bounds["west"]) / (bounds["east"] - bounds["west"]) * width
        ).astype(int)
        if (
            np.any(rows < 1)
            or np.any(cols < 1)
            or np.any(rows >= height - 1)
            or np.any(cols >= width - 1)
        ):
            raise ValueError("flight corridor is outside terrain coverage")
        surfaces = np.stack(
            [self.image[rows + dr, cols + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        )
        if np.any(~np.isfinite(surfaces)):
            raise ValueError("flight corridor contains missing terrain depth")
        return np.min(surfaces, axis=0)

    def highest(self, a, b):
        return float(np.min(self.profile(a, b)))

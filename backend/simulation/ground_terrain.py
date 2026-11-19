"""Use registered surface samples for simulated ground-route checks."""

import math
from dataclasses import dataclass
from heapq import heappop, heappush

import cv2
import numpy as np

Cell = tuple[int, int]
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
OBSTACLE_CLEARANCE_M = 3.5


def footprint(radius_m, row_m, col_m):
    rows, cols = math.ceil(radius_m / row_m), math.ceil(radius_m / col_m)
    y, x = np.ogrid[-rows : rows + 1, -cols : cols + 1]
    return ((y * row_m) ** 2 + (x * col_m) ** 2 <= radius_m**2).astype(np.uint8)


@dataclass
class SurfaceGrid:
    down: np.ndarray
    north: float
    west: float
    row_m: float
    col_m: float

    def __post_init__(self):
        if (
            self.down.ndim != 2
            or min(self.down.shape) < 3
            or min(self.row_m, self.col_m) <= 0
        ):
            raise ValueError("invalid terrain grid")
        gradients = np.gradient(self.down, self.row_m, self.col_m)
        self.slope = np.degrees(np.arctan(np.hypot(*gradients)))
        valid = np.isfinite(self.down)
        height = np.where(valid, -self.down, -1e6).astype(np.float32)
        neighbourhood = footprint(12, self.row_m, self.col_m)
        # Opening estimates the surrounding ground underneath raised objects.
        # A smooth surface on top of a rock can still be locally prominent.
        background = cv2.morphologyEx(height, cv2.MORPH_OPEN, neighbourhood)
        raised = (height - background > 1.25) & valid
        hazards = raised | (self.slope > 35) | ~valid | ~np.isfinite(self.slope)
        clearance = footprint(OBSTACLE_CLEARANCE_M, self.row_m, self.col_m)
        self.blocked = cv2.dilate(hazards.astype(np.uint8), clearance).astype(bool)

    @classmethod
    def from_terrain(cls, terrain):
        bounds = terrain.bounds
        rows, cols = terrain.image.shape
        return cls(
            terrain.image,
            bounds["north"],
            bounds["west"],
            (bounds["north"] - bounds["south"]) / rows,
            (bounds["east"] - bounds["west"]) / cols,
        )

    @classmethod
    def from_file(cls, path, *, task=None):
        with np.load(path, allow_pickle=False) as data:
            if task is not None:
                if "task_id" in data and "map_id" in data:
                    matches = (
                        int(data["task_id"]) == task["task_id"]
                        and str(data["map_id"]) == task["map_name"]
                    )
                else:
                    matches = path.stem == str(task["task_id"])
                if not matches:
                    raise ValueError("terrain raster belongs to a different ESAR task")
            north, west = float(data["north"]), float(data["west"])
            down = data["down"]
            return cls(
                down,
                north,
                west,
                (north - float(data["south"])) / down.shape[0],
                (float(data["east"]) - west) / down.shape[1],
            )

    def cell(self, point):
        north, east = point[:2]
        cell = (
            math.floor((self.north - north) / self.row_m),
            math.floor((east - self.west) / self.col_m),
        )
        if not self.contains(cell):
            raise ValueError("point is outside the terrain raster")
        return cell

    def contains(self, cell):
        row, col = cell
        return 0 <= row < self.down.shape[0] and 0 <= col < self.down.shape[1]

    def point(self, cell):
        row, col = cell
        return (
            self.north - (row + 0.5) * self.row_m,
            self.west + (col + 0.5) * self.col_m,
            float(self.down[cell]),
        )

    def entry_point(self, north, east, expected_down=None):
        cell = self.cell((north, east))
        slope = self.slope[cell]
        if not np.isfinite(slope) or slope > 25:
            raise ValueError(
                "entry is missing depth or exceeds the 25 degree entry threshold"
            )
        down = float(self.down[cell])
        if expected_down is not None and abs(down - expected_down) > 5:
            raise ValueError(
                "entry surface differs from the surveyed elevation by more than 5 m"
            )
        return north, east, down

    def window(self, cells, padding):
        rows, cols = zip(*cells, strict=True)
        return (
            max(0, min(rows) - padding),
            min(self.down.shape[0], max(rows) + padding + 1),
            max(0, min(cols) - padding),
            min(self.down.shape[1], max(cols) + padding + 1),
        )

    def packet(self, window, *, stride=1):
        r0, r1, c0, c1 = window
        top_left = self.point((r0, c0))
        blocked = self.blocked[r0:r1, c0:c1]
        # Include the last source row and column even when stride is uneven.
        rows = np.unique(
            np.r_[np.arange(0, blocked.shape[0], stride), blocked.shape[0] - 1]
        )
        columns = np.unique(
            np.r_[np.arange(0, blocked.shape[1], stride), blocked.shape[1] - 1]
        )
        indices = np.ix_(r0 + rows, c0 + columns)
        excluded = np.logical_or.reduceat(
            np.logical_or.reduceat(blocked, rows, axis=0), columns, axis=1
        )
        return {
            "geometry": "regularly connected triangle mesh (OBJ-compatible TIN)",
            "surface_function": "piecewise affine interpolation on triangles",
            "cell_diagonal": "northwest_to_southeast",
            "triangle_rule": "split each adjacent four vertices NW-NE-SE and NW-SE-SW; skip triangles with missing heights",
            "northwest_cell_center_ne_m": [round(value, 3) for value in top_left[:2]],
            "row_step_m_south": round(self.row_m * stride, 6),
            "column_step_m_east": round(self.col_m * stride, 6),
            "vertex_north_m": np.round(top_left[0] - rows * self.row_m, 3).tolist(),
            "vertex_east_m": np.round(top_left[1] + columns * self.col_m, 3).tolist(),
            "height_reference": "negative NED Down",
            "surface_height_m": finite_matrix(-self.down[indices], 2),
            "surface_slope_deg": finite_matrix(self.slope[indices], 1),
            "contains_blocked_cells": excluded.tolist(),
        }

    def overview_packet(self):
        rows, cols = self.down.shape
        stride = max(1, math.ceil((max(rows, cols) - 1) / 16))
        return self.packet((0, rows, 0, cols), stride=stride)

    def passable(self, max_slope):
        return ~self.blocked & (self.slope <= max_slope)


def finite_matrix(values, decimals):
    return [
        [round(float(value), decimals) if np.isfinite(value) else None for value in row]
        for row in values
    ]


def edge(grid, allowed, first, last, max_slope):
    if not allowed[first] or not allowed[last]:
        return None
    dr, dc = last[0] - first[0], last[1] - first[1]
    if dr and dc and not (allowed[first[0], last[1]] and allowed[last[0], first[1]]):
        return None
    distance = math.hypot(dr * grid.row_m, dc * grid.col_m)
    grade = math.degrees(math.atan2(abs(grid.down[last] - grid.down[first]), distance))
    return (distance, grade) if grade <= max_slope else None


def find_route(grid, start, goal, target_ne, *, max_slope, approach_m, penalty):
    allowed = grid.passable(max_slope)
    if not allowed[start]:
        return []
    padding = math.ceil(95 / min(grid.row_m, grid.col_m))
    r0, r1, c0, c1 = grid.window((start, goal), padding)

    def remaining(cell):
        return max(0, math.dist(grid.point(cell)[:2], target_ne) - approach_m)

    queue = [(remaining(start), 0.0, start)]
    cost = {start: 0.0}
    previous = {}
    while queue:
        _, distance, cell = heappop(queue)
        if distance != cost[cell]:
            continue
        if remaining(cell) == 0:
            path = [cell]
            while cell in previous:
                cell = previous[cell]
                path.append(cell)
            return list(reversed(path))
        for dr, dc in NEIGHBOURS:
            next_cell = cell[0] + dr, cell[1] + dc
            if not (r0 <= next_cell[0] < r1 and c0 <= next_cell[1] < c1):
                continue
            segment = edge(grid, allowed, cell, next_cell, max_slope)
            if segment is None:
                continue
            step, grade = segment
            value = distance + step * (
                1 + penalty * (max(grade, grid.slope[next_cell]) / max_slope) ** 2
            )
            if value < cost.get(next_cell, math.inf):
                cost[next_cell] = value
                previous[next_cell] = cell
                heappush(queue, (value + remaining(next_cell), value, next_cell))
    return []


def check_route(grid, path, start, target_ne, max_slope, approach_m):
    if not path or path[0] != start or not all(grid.contains(cell) for cell in path):
        raise ValueError("route must start at the entry and stay in raster coverage")
    allowed = grid.passable(max_slope)
    if not all(allowed[cell] for cell in path):
        raise ValueError("route crosses a blocked, steep, or unobserved surface cell")
    end_distance = math.dist(grid.point(path[-1])[:2], target_ne)
    if end_distance > approach_m:
        raise ValueError("route stops too far from the observed person")
    distance, gain, loss, max_segment_grade = 0.0, 0.0, 0.0, 0.0
    for first, last in zip(path, path[1:]):
        if (last[0] - first[0], last[1] - first[1]) not in NEIGHBOURS:
            raise ValueError("route skips original cells")
        segment = edge(grid, allowed, first, last, max_slope)
        if segment is None:
            raise ValueError("route crosses a blocked cell boundary")
        step, grade = segment
        distance += step
        max_segment_grade = max(max_segment_grade, grade)
        delta = grid.down[first] - grid.down[last]
        gain += max(0, delta)
        loss += max(0, -delta)
    return {
        "distance_m": round(distance, 1),
        "ascent_m": round(gain, 1),
        "descent_m": round(loss, 1),
        "max_slope_deg": round(
            max(max_segment_grade, *(float(grid.slope[cell]) for cell in path)), 1
        ),
        "approach_distance_m": round(end_distance, 1),
    }

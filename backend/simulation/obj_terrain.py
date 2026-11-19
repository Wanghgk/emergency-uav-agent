"""Read structured terrain OBJ files and evaluate their original triangles."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjSurface:
    east: np.ndarray
    north: np.ndarray
    height: np.ndarray
    anti_diagonal: np.ndarray

    @classmethod
    def read(cls, path):
        with path.open(encoding="utf-8") as stream:
            vertices = np.loadtxt(
                (line[2:] for line in stream if line.startswith("v ")), ndmin=2
            )
        if (
            not vertices.size
            or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("OBJ needs finite XYZ vertices")
        east, c = np.unique(vertices[:, 0], return_inverse=True)
        north, r = np.unique(vertices[:, 1], return_inverse=True)
        cells = r * len(east) + c
        if (
            min(len(east), len(north)) < 2
            or len(np.unique(cells)) != len(vertices)
            or len(vertices) != len(east) * len(north)
        ):
            raise ValueError("OBJ is not a single-valued rectangular height grid")
        height = np.empty((len(north), len(east)))
        height[r, c] = vertices[:, 2]
        with path.open(encoding="utf-8") as stream:
            faces = np.loadtxt(
                (
                    " ".join(part.split("/")[0] for part in line[2:].split())
                    for line in stream
                    if line.startswith("f ")
                ),
                dtype=np.int64,
                ndmin=2,
            )
        if (
            faces.shape[1] != 3
            or len(faces) != 2 * (len(north) - 1) * (len(east) - 1)
            or (np.any(faces < 1) or np.any(faces > len(vertices)))
        ):
            raise ValueError("OBJ needs exactly two indexed triangles per terrain cell")
        rr, cc = r[faces - 1], c[faces - 1]
        row, col = rr.min(axis=1), cc.min(axis=1)
        corner = (rr - row[:, None]) * 2 + cc - col[:, None]
        if np.any(rr.max(axis=1) - row != 1) or np.any(cc.max(axis=1) - col != 1):
            raise ValueError("OBJ faces do not follow adjacent grid cells")
        mask = (1 << corner).sum(axis=1)
        if (
            np.any(np.diff(np.sort(corner, axis=1), axis=1) == 0)
            or not np.isin(mask, (7, 11, 13, 14)).all()
        ):
            raise ValueError("OBJ contains a malformed terrain triangle")
        shape = (len(north) - 1, len(east) - 1)
        pairs = np.zeros(shape, dtype=np.uint8)
        count = np.zeros(shape, dtype=np.uint8)
        np.bitwise_or.at(pairs, (row, col), 1 << (mask - 7).clip(0, 7))
        np.add.at(count, (row, col), 1)
        anti, main = (1 << 0) | (1 << 7), (1 << 4) | (1 << 6)
        if not np.all(count == 2) or not np.isin(pairs, (anti, main)).all():
            raise ValueError("OBJ terrain cells overlap or are missing a triangle")
        return cls(east, north, height, pairs == anti)

    def evaluate(self, east, north):
        i = np.clip(
            np.searchsorted(self.north, north, side="right") - 1, 0, len(self.north) - 2
        )
        j = np.clip(
            np.searchsorted(self.east, east, side="right") - 1, 0, len(self.east) - 2
        )
        u = (east - self.east[j]) / (self.east[j + 1] - self.east[j])
        v = (north - self.north[i]) / (self.north[i + 1] - self.north[i])
        a, b, c, d = (
            self.height[i, j],
            self.height[i, j + 1],
            self.height[i + 1, j],
            self.height[i + 1, j + 1],
        )
        anti = np.where(
            u + v <= 1,
            a + u * (b - a) + v * (c - a),
            d + (1 - u) * (c - d) + (1 - v) * (b - d),
        )
        main = np.where(
            v <= u, a + u * (b - a) + v * (d - b), a + u * (d - c) + v * (c - a)
        )
        return np.where(
            (0 <= u) & (u <= 1) & (0 <= v) & (v <= 1),
            np.where(self.anti_diagonal[i, j], anti, main),
            np.nan,
        )

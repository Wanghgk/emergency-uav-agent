"""Validate a structured OBJ terrain and describe its coordinate extent."""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.simulation.obj_terrain import ObjSurface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj", type=Path)
    args = parser.parse_args()
    surface = ObjSurface.read(args.obj)
    print(
        json.dumps(
            {
                "vertices": surface.height.size,
                "triangles": surface.anti_diagonal.size * 2,
                "x_bounds": [float(surface.east[0]), float(surface.east[-1])],
                "y_bounds": [float(surface.north[0]), float(surface.north[-1])],
                "z_bounds": [float(surface.height.min()), float(surface.height.max())],
                "median_xy_spacing_obj_units": [
                    float(np.median(np.diff(surface.east))),
                    float(np.median(np.diff(surface.north))),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

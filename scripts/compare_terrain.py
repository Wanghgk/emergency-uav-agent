"""Compare compact terrain representations on held-out ESARBench depth pixels."""

import argparse
import csv
import json
import socket
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from backend.simulation.basemap import capture_basemap, map_settings
from backend.simulation.esar_adapter import load_task
from backend.simulation.rescue_search import Terrain
from backend.simulation.session import HOST, PORT, simulator

SCENES = {290: "Lop Nur", 28: "Aotai", 370: "K2"}
INTERVALS = 192
STRIDE = 8


def capture(args, task_id, path):
    task, start, _ = load_task(args.esar, "task_data.json", task_id)
    with tempfile.TemporaryDirectory() as directory:
        settings = Path(directory) / "settings.json"
        map_settings(args.esar / "settings/settings.json", settings, start)
        with simulator(args.maps, settings, task) as (client, api):
            basemap = capture_basemap(client, api, task, Path("data/maps"))
            terrain = Terrain.read(client, api, basemap)
            terrain.save(path, task)
    # The packaged UE child may still be releasing the AirSim socket.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.25)
            if probe.connect_ex((HOST, PORT)) != 0:
                return
        time.sleep(0.5)
    raise TimeoutError("ESARBench did not release the AirSim port after capture")


def patch(sample, alarm, start):
    raw = sample["down"]
    dy = float((sample["north"] - sample["south"]) / raw.shape[0])
    dx = float((sample["east"] - sample["west"]) / raw.shape[1])
    north, east = ((start[0] + alarm["north"]) / 2, (start[1] + alarm["east"]) / 2)
    row = int((sample["north"] - north) / dy) - INTERVALS // 2
    col = int((east - sample["west"]) / dx) - INTERVALS // 2
    size = INTERVALS + 1
    data = -raw[row : row + size, col : col + size]
    if data.shape != (size, size) or not np.all(np.isfinite(data)):
        raise ValueError("the launch-to-alarm comparison window is not fully covered")
    return (
        data,
        dy,
        dx,
        float(sample["north"] - (row + 0.5) * dy),
        float(sample["west"] + (col + 0.5) * dx),
    )


def interpolate(nodes, y, x):
    iy = np.minimum(y.astype(int), nodes.shape[0] - 2)
    ix = np.minimum(x.astype(int), nodes.shape[1] - 2)
    u, v = x - ix, y - iy
    a, b = nodes[iy, ix], nodes[iy, ix + 1]
    c, d = nodes[iy + 1, ix], nodes[iy + 1, ix + 1]
    bilinear = (1 - v) * (a + u * (b - a)) + v * (c + u * (d - c))
    triangle = np.where(
        u >= v, a + u * (b - a) + v * (d - b), a + v * (c - a) + u * (d - c)
    )

    def cubic(p, q, r, s, t):
        return q + 0.5 * t * (
            (r - p) + t * (2 * p - 5 * q + 4 * r - s + t * (3 * (q - r) + s - p))
        )

    rows = []
    for delta in (-1, 0, 1, 2):
        row = (iy + delta).clip(0, nodes.shape[0] - 1)
        values = [
            nodes[row, (ix + offset).clip(0, nodes.shape[1] - 1)]
            for offset in (-1, 0, 1, 2)
        ]
        rows.append(cubic(*values, u))
    return bilinear, triangle, cubic(*rows, v)


def representations(surface, metadata):
    rows, cols = np.indices(surface.shape)
    x, y = cols / INTERVALS * 2 - 1, rows / INTERVALS * 2 - 1
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)
    train = (rows % STRIDE == 0) & (cols % STRIDE == 0)
    coeff = np.linalg.lstsq(basis[train], surface[train], rcond=None)[0].round(5)
    yield "Quadratic", basis @ coeff, {**metadata, "coefficients_m": coeff.tolist()}
    nodes = (surface[::STRIDE, ::STRIDE] * 100).round().astype(int)
    payload = {**metadata, "stride": STRIDE, "height_cm": nodes.tolist()}
    names = ("Bilinear", "Fixed TIN", "Bicubic")
    for name, estimate in zip(
        names, interpolate(nodes / 100, rows / STRIDE, cols / STRIDE), strict=True
    ):
        yield name, estimate, payload
    fine = (surface[::4, ::4] * 100).round().astype(int)
    estimate = interpolate(fine / 100, rows / 4, cols / 4)[0]
    yield "Finer grid", estimate, {**metadata, "stride": 4, "height_cm": fine.tolist()}


def slopes(surface, dy, dx):
    north, east = np.gradient(surface, dy, dx)
    return np.degrees(np.arctan(np.hypot(north, east)))


def compare(args, task_ids):
    names, rows, images = [], [], []
    alarms = json.loads(Path("configs/rescue_alarms.json").read_text(encoding="utf-8"))
    yy, xx = np.indices((INTERVALS + 1, INTERVALS + 1))
    holdout = ~((yy % 4 == 0) & (xx % 4 == 0))
    interior = holdout & (yy > 0) & (xx > 0) & (yy < INTERVALS) & (xx < INTERVALS)
    for task_id in task_ids:
        path = args.output / f"{task_id}.npz"
        if args.refresh or not path.exists():
            capture(args, task_id, path)
        _, start, _ = load_task(args.esar, "task_data.json", task_id)
        with np.load(path) as sample:
            surface, dy, dx, north, east = patch(sample, alarms[str(task_id)], start)
        slope = slopes(surface, dy, dx)
        steep = (slope > 30) & interior
        scene = SCENES[task_id]
        names.append(scene)
        images.append([surface, slope])
        metadata = {
            "frame": "NED",
            "height": "-down",
            "north": round(north, 3),
            "east": round(east, 3),
            "pixel_m": [round(dy, 5), round(dx, 5)],
            "intervals": INTERVALS,
        }
        for method, estimate, data in representations(surface, metadata):
            payload = json.dumps(
                {"method": method, **data}, separators=(",", ":")
            ).encode()
            residual = abs(estimate - surface)
            predicted = slopes(estimate, dy, dx)
            results = {
                "task_id": task_id,
                "scene": scene,
                "method": method,
                "test_pixels": int(holdout.sum()),
                "payload_bytes": len(payload),
                "rmse_m": round(float(np.sqrt(np.mean(residual[holdout] ** 2))), 3),
                "p95_m": round(float(np.percentile(residual[holdout], 95)), 3),
                "slope_mae_deg": round(
                    float(np.mean(abs(predicted - slope)[interior])), 3
                ),
                "steep_pixels": int(steep.sum()),
                "missed_steep_pct": round(
                    float(np.mean(predicted[steep] <= 30) * 100), 2
                )
                if steep.any()
                else None,
            }
            rows.append(results)
            images[-1].append(residual)
            (
                args.output / f"{task_id}-{method.replace(' ', '-').lower()}.json"
            ).write_bytes(payload)
        print(scene, json.dumps(rows[-5:], ensure_ascii=True), flush=True)
    with (args.output / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    chart(names, images, args.output / "comparison.png")


def chart(names, images, path):
    columns = [
        "Surface",
        "Slope (0-45 deg)",
        "Quadratic",
        "Bilinear",
        "Fixed TIN",
        "Bicubic",
        "Finer grid",
    ]
    size, stride = 170, 180
    image = Image.new(
        "RGB", (len(columns) * stride + 20, 68 + len(images) * 200), "#111b27"
    )
    draw = ImageDraw.Draw(image)
    draw.text(
        (10, 9),
        "ESARBench | 450 m windows | absolute elevation error, 0 m (dark) to >= 5 m (light)",
        fill="white",
    )
    for col, title in enumerate(columns):
        draw.text((10 + col * stride, 32), title, fill="white")
    colors = np.array([[12, 21, 38], [34, 112, 163], [89, 205, 164], [255, 228, 142]])
    for row, (name, panels) in enumerate(zip(names, images, strict=True)):
        top = 50 + row * 200
        for col, panel in enumerate(panels):
            lo, hi = (
                (float(panel.min()), float(panel.max()))
                if col == 0
                else (0, 45 if col == 1 else 5)
            )
            scaled = ((panel - lo) / max(hi - lo, 1e-6)).clip(0, 1)
            rgb = np.stack(
                [
                    np.interp(scaled, np.linspace(0, 1, 4), colors[:, channel])
                    for channel in range(3)
                ],
                axis=-1,
            )
            image.paste(
                Image.fromarray(rgb.astype(np.uint8)).resize((size, size)),
                (10 + col * stride, top),
            )
        draw.text((10, top + size + 3), name, fill="white")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esar", type=Path, default=Path(".external/ESAR-pinned"))
    parser.add_argument("--maps", type=Path, default=Path(".external/ESARMaps_win"))
    parser.add_argument("--output", type=Path, default=Path("data/terrain-comparison"))
    parser.add_argument(
        "--tasks", type=int, nargs="+", choices=SCENES, default=list(SCENES)
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    compare(args, args.tasks)


if __name__ == "__main__":
    main()

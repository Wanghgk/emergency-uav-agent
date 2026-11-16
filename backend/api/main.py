"""Serve the ESARBench observation replay application."""

import asyncio
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.simulation.basemap import Basemap
from backend.simulation.observation_collector import (
    Observation,
    load_observation,
    map_point,
)
from backend.simulation.rescue_search import Alarm, person_results, track_detections

ROOT = Path(__file__).resolve().parents[2]
OBSERVATIONS = ROOT / "data" / "observations"
MAPS = ROOT / "data" / "maps"
MISSIONS = ROOT / "data" / "missions"
DIAGNOSTICS = ROOT / ".external" / "diagnostics"
BUILD = ROOT / "frontend" / "dist"
FRAME_ID = re.compile(r"[0-9a-f]{32}")

app = FastAPI(title="Emergency-UAV-Agent")


@app.get("/api/basemaps")
def list_basemaps():
    maps = []
    for manifest in MAPS.glob("*/basemap.json"):
        if not (manifest.parent / "terrain.webp").is_file():
            continue
        try:
            entry = Basemap.model_validate_json(manifest.read_bytes())
        except (ValueError, OSError):
            continue
        key = f"{entry.map_id}-{entry.task_id}"
        if key == manifest.parent.name and re.fullmatch(r"[a-z0-9]+-\d+", key):
            maps.append(
                {
                    **entry.bounds(),
                    "map_id": entry.map_id,
                    "task_id": entry.task_id,
                    "width_px": entry.width_px,
                    "height_px": entry.height_px,
                    "image_url": f"/api/basemaps/{key}/image",
                }
            )
    return maps


@app.get("/api/basemaps/{key}/image")
def terrain_image(key: str):
    if re.fullmatch(r"[a-z0-9]+-\d+", key):
        path = MAPS / key / "terrain.webp"
        if path.is_file():
            return FileResponse(path, media_type="image/webp")
    raise HTTPException(404, "地图底图不存在")


@app.get("/api/missions")
def list_missions():
    runs = []
    for manifest in MISSIONS.glob("*.json"):
        if not FRAME_ID.fullmatch(manifest.stem):
            continue
        try:
            run = json.loads(manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (
            run.get("alarm")
            and run.get("status") in {"finished", "interrupted"}
            and "people_results" not in run
        ):
            alarm = Alarm(
                **{key: value for key, value in run["alarm"].items() if key != "source"}
            )
            tracks = []
            origin = None
            for event in run["events"]:
                if event["type"] != "observation" or event["sector"] == "START":
                    continue
                frame_id = event["frame_id"]
                if not FRAME_ID.fullmatch(frame_id):
                    continue
                try:
                    frame = Observation.model_validate_json(
                        (OBSERVATIONS / frame_id / "observation.json").read_bytes()
                    )
                except (OSError, ValueError):
                    continue
                origin = frame.map_origin_ue_cm
                track_detections(tracks, frame_id, event["detections"], alarm)
            run["people_results"] = (
                person_results(tracks, origin) if origin is not None else []
            )
        recheck = DIAGNOSTICS / f"{manifest.stem}-rotation-recheck.json"
        if recheck.is_file():
            try:
                record = json.loads(recheck.read_text(encoding="utf-8"))
                if record["source_run"] == manifest.stem:
                    score = record["evaluation"]
                    run["offline_recheck"] = {
                        key: score[key]
                        for key in (
                            "official_people",
                            "verified_confirmed",
                            "verified_candidates",
                        )
                    }
            except (ValueError, OSError, KeyError):
                pass
        runs.append(run)
    return runs


@app.websocket("/api/updates")
async def updates(websocket: WebSocket):
    await websocket.accept()
    previous = None
    try:
        while True:
            reports = sorted(
                (path.name, path.stat().st_mtime_ns)
                for path in (
                    *MISSIONS.glob("*.json"),
                    *DIAGNOSTICS.glob("*-rotation-recheck.json"),
                )
            )
            if reports != previous:
                await websocket.send_json({"type": "mission_update"})
                previous = reports
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1)
            except TimeoutError:
                pass
    except WebSocketDisconnect:
        return


def manifest_for(frame_id: str) -> Path:
    if FRAME_ID.fullmatch(frame_id):
        path = OBSERVATIONS / frame_id / "observation.json"
        if path.is_file():
            return path
    raise HTTPException(404, "观测帧不存在")


@app.get("/api/observations", response_model=list[Observation])
def list_observations():
    frames = []
    for manifest in OBSERVATIONS.glob("*/observation.json"):
        if not FRAME_ID.fullmatch(manifest.parent.name):
            continue
        try:
            frame = Observation.model_validate_json(manifest.read_bytes())
        except (ValueError, OSError):
            continue
        if frame.frame_id == manifest.parent.name:
            frames.append(frame)
    return sorted(frames, key=lambda frame: (frame.timestamp, frame.frame_id))


@app.get("/api/observations/{frame_id}/rgb")
def rgb_image(frame_id: str):
    image = manifest_for(frame_id).parent / "rgb.png"
    if not image.is_file():
        raise HTTPException(404, "RGB 图像不存在")
    return FileResponse(image, media_type="image/png")


@app.get("/api/observations/{frame_id}/point")
def project_point(frame_id: str, u: int, v: int):
    manifest = manifest_for(frame_id)
    try:
        observation, _, depth = load_observation(manifest)
        result = map_point(observation, depth, u, v)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, "该像素没有可用深度，或观测记录无法配准") from exc
    return {**result, "depth_m": float(depth[v, u])}


if BUILD.is_dir():
    app.mount("/", StaticFiles(directory=BUILD, html=True), name="frontend")

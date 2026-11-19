"""Serve the ESARBench observation replay application."""

import asyncio
import json
import re
import time
from pathlib import Path
from threading import Lock

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.simulation.basemap import Basemap
from backend.simulation.observation_collector import (
    Observation,
    load_observation,
    map_point,
)
from backend.simulation.rescue_search import archived_people

from .launch import ACTIVE_STATUSES, active_mission_id, pause_mission
from .launch import router as launch_router

ROOT = Path(__file__).resolve().parents[2]
OBSERVATIONS = ROOT / "data" / "observations"
MAPS = ROOT / "data" / "maps"
MISSIONS = ROOT / "data" / "missions"
GROUND_ROUTES = ROOT / "data" / "ground-routes"
GROUND_TERRAIN = ROOT / "data" / "ground-terrain"
LAUNCH_LOGS = ROOT / "data" / "launch-logs"
DIAGNOSTICS = ROOT / ".external" / "diagnostics"
BUILD = ROOT / "frontend" / "dist"
FRAME_ID = re.compile(r"[0-9a-f]{32}")
delete_lock = Lock()

app = FastAPI(title="Emergency-UAV-Agent")
app.include_router(launch_router)


@app.get("/api/basemaps")
def list_basemaps():
    maps = []
    for manifest in MAPS.glob("*/basemap.json"):
        try:
            entry = Basemap.model_validate_json(manifest.read_bytes())
            version = (manifest.parent / "terrain.webp").stat().st_mtime_ns
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
                    "render_source": entry.render_source,
                    "image_url": f"/api/basemaps/{key}/image?v={version}",
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
            and (
                run.get("status") in {"finished", "interrupted"}
                or (run.get("profile") == "fleet" and run.get("status") == "failed")
            )
            and "people_results" not in run
        ):
            run["people_results"] = archived_people(run, OBSERVATIONS)
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
        try:
            ground = json.loads(
                (GROUND_ROUTES / manifest.name).read_text(encoding="utf-8")
            )
            if (
                isinstance(ground, dict)
                and ground.get("source_run_id") == manifest.stem
            ):
                run["ground_rescue"] = ground
        except (OSError, ValueError):
            pass
        run["can_pause"] = (
            run.get("status") in ACTIVE_STATUSES
            and manifest.stem == active_mission_id()
        )
        runs.append(run)
    return runs


def mission_observations(run_id):
    for observation in OBSERVATIONS.glob("*/observation.json"):
        if not FRAME_ID.fullmatch(observation.parent.name):
            continue
        try:
            frame = json.loads(observation.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if frame.get("run_id") == run_id:
            yield observation.parent


def delete_with_retry(operation):
    for attempt in range(12):
        try:
            operation()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.04)


def validate_mission_artifact(path, *, directory=False):
    resolved = path.resolve()
    if (
        path.is_symlink()
        or resolved != path.absolute()
        or not resolved.is_relative_to(ROOT.resolve())
        or (path.exists() and not (path.is_dir() if directory else path.is_file()))
    ):
        raise OSError(f"Unexpected mission artifact: {path}")


@app.delete("/api/missions/{run_id}", status_code=204)
def delete_mission(run_id: str, request: Request):
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "请从本机任务页面删除任务。")
    if not FRAME_ID.fullmatch(run_id):
        raise HTTPException(404, "任务不存在。")
    manifest = MISSIONS / f"{run_id}.json"
    with delete_lock:
        try:
            report = json.loads(manifest.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise HTTPException(404, "任务不存在。") from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(409, "任务记录暂时无法读取。") from exc
        if report.get("status") in ACTIVE_STATUSES:
            try:
                if run_id != active_mission_id():
                    raise HTTPException(
                        409, "这条任务的运行进程不归当前网页管理，无法安全停止并删除。"
                    )
                pause_mission(run_id, request)
            except HTTPException as exc:
                if exc.status_code != 409:
                    raise
                # The task may have finished while the delete request was in flight.
                try:
                    status = json.loads(manifest.read_text(encoding="utf-8"))["status"]
                except (OSError, ValueError, KeyError):
                    raise exc from None
                if status in ACTIVE_STATUSES:
                    raise
        files = [
            GROUND_ROUTES / manifest.name,
            GROUND_TERRAIN / f"{run_id}.npz",
            LAUNCH_LOGS / f"{run_id}.log",
            DIAGNOSTICS / f"{run_id}-rotation-recheck.json",
            manifest.with_suffix(".tmp"),
            manifest,
        ]
        try:
            for file in files:
                validate_mission_artifact(file)
            frames = {}
            for directory in mission_observations(run_id):
                validate_mission_artifact(directory, directory=True)
                frame_manifest = directory / "observation.json"
                payload = [
                    file for file in directory.iterdir() if file != frame_manifest
                ]
                frames[directory] = [*payload, frame_manifest]
                for file in frames[directory]:
                    validate_mission_artifact(file)
            # Remove payloads before their manifests so a locked file can be retried.
            for directory, frame_files in frames.items():
                for file in frame_files:
                    delete_with_retry(file.unlink)
                delete_with_retry(directory.rmdir)
            for file in files:
                delete_with_retry(file.unlink)
        except OSError as exc:
            raise HTTPException(
                409, "任务文件未全部删除，请关闭占用文件的程序后重试。"
            ) from exc
    return Response(status_code=204)


@app.websocket("/api/updates")
async def updates(websocket: WebSocket):
    await websocket.accept()
    previous = None
    try:
        while True:
            reports = []
            for path in (
                *MISSIONS.glob("*.json"),
                *GROUND_ROUTES.glob("*.json"),
                *MAPS.glob("*/basemap.json"),
                *DIAGNOSTICS.glob("*-rotation-recheck.json"),
            ):
                try:
                    reports.append((path.name, path.stat().st_mtime_ns))
                except FileNotFoundError:
                    continue
            reports.sort()
            if reports != previous:
                await websocket.send_json({"type": "mission_update"})
                previous = reports
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.25)
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

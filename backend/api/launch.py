"""Start a local ESARBench fleet from a validated web request."""

import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.brain.ground_rescue import configured_entry
from backend.simulation.esar_adapter import launch_args, load_task
from backend.simulation.fleet import fleet_roles
from backend.simulation.mission_manifest import write_manifest
from backend.simulation.session import HOST, PORT

ROOT = Path(__file__).resolve().parents[2]
ALARMS = ROOT / "configs" / "fleet_alarms.json"
ENTRIES = ROOT / "configs" / "rescue_entries.json"
ESAR = ROOT / ".external" / "ESAR-pinned"
MAPS = ROOT / ".external" / "ESARMaps_win"
MISSIONS = ROOT / "data" / "missions"
WEIGHTS = ROOT / ".external" / "models"
LABELS = {"lopnur": "罗布泊", "aotai2800": "鳌太线", "k2": "K2 雪山"}

router = APIRouter(prefix="/api")
lock = Lock()
active_run: tuple[str, subprocess.Popen] | None = None
ACTIVE_STATUSES = {"starting", "running"}


class LaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: int = Field(strict=True, ge=0)
    scouts: int = Field(default=2, strict=True, ge=1, le=6)
    relays: int = Field(default=1, strict=True, ge=0, le=4)
    deepseek: bool = False
    quick: bool = False


def tasks():
    alarms = json.loads(ALARMS.read_text(encoding="utf-8"))
    for key, alarm in alarms.items():
        task, _, _ = load_task(ESAR, "task_data.json", int(key))
        yield task, alarm


def validate_assets(task):
    launch_args(MAPS, ESAR / "settings" / "settings.json", task)
    return configured_entry(task.map_name, ENTRIES)


@router.get("/mission-options")
def mission_options():
    try:
        options = list(tasks())
    except (OSError, ValueError) as exc:
        raise HTTPException(
            503, "未找到 ESARBench 官方任务文件，请先完成 Windows 部署。"
        ) from exc
    result = []
    for task, alarm in options:
        try:
            validate_assets(task)
            available = True
        except (OSError, ValueError):
            available = False
        result.append(
            {
                "task_id": task.task_id,
                "map_id": task.map_name,
                "map_name": LABELS.get(task.map_name, task.map_name),
                "instruction": task.prompt,
                "available": available,
                "search_width_m": alarm["half_width"] * 2,
            }
        )
    return {
        "tasks": result,
        "deepseek_available": bool(os.getenv("DeepseekApikey", "").strip()),
    }


def record_failure(path, initial, message):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        report = initial
    if report["status"] not in ACTIVE_STATUSES:
        return
    report["status"] = "failed"
    report["events"].append(
        {"type": "error", "timestamp": datetime.now(UTC).isoformat(), "error": message}
    )
    write_manifest(path, report)


def monitor(process, path, report):
    global active_run
    code = process.wait()
    with lock:
        if active_run != (path.stem, process):
            return
        try:
            record_failure(
                path, report, f"任务进程提前退出（退出码 {code}）。请查看本地启动日志。"
            )
        finally:
            active_run = None


def active_mission_id():
    run = active_run
    return run[0] if run else None


@router.post("/missions/{run_id}/pause")
def pause_mission(run_id: str, request: Request):
    global active_run
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "请从本机任务页面暂停任务。")
    with lock:
        if active_run is None or active_run[0] != run_id:
            raise HTTPException(409, "该任务当前无法从网页暂停。")
        _, process = active_run
        path = MISSIONS / f"{run_id}.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["status"] not in ACTIVE_STATUSES or process.poll() is not None:
                raise HTTPException(409, "任务已结束。")
            result = subprocess.run(
                ["taskkill.exe", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0:
                raise HTTPException(409, "暂时无法停止仿真，请刷新任务状态。")
            process.wait(timeout=5)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["status"] in ACTIVE_STATUSES:
                report["status"] = "paused"
                report["events"].append(
                    {"type": "paused", "timestamp": datetime.now(UTC).isoformat()}
                )
                write_manifest(path, report)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(503, "暂停未完成，请刷新任务状态。") from exc
        active_run = None
        return {**report, "can_pause": False}


@router.post("/missions", status_code=202)
def launch_mission(payload: LaunchRequest, request: Request):
    global active_run
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "请从本机任务页面启动仿真。")
    with lock:
        if active_run is not None:
            raise HTTPException(409, "已有机队任务，请等待当前任务结束。")
        with socket.socket() as probe:
            probe.settimeout(0.3)
            if probe.connect_ex((HOST, PORT)) == 0:
                raise HTTPException(409, "已有 AirSim 场景正在运行，请先结束该任务。")
        try:
            selected = next(
                (
                    (task, alarm)
                    for task, alarm in tasks()
                    if task.task_id == payload.task_id
                ),
                None,
            )
            if selected is None:
                raise HTTPException(404, "所选任务未配置。")
            task, alarm = selected
            entry = validate_assets(task)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                503, "地图或任务文件未就绪，请先完成 Windows 部署。"
            ) from exc
        if any(
            not (WEIGHTS / name).is_file()
            for name in ("yolo11s-pose.onnx", "rtdetr_v2_r18vd.onnx")
        ):
            raise HTTPException(503, "缺少人员姿态复核模型权重，请先按部署指南下载。")
        if payload.deepseek and not os.getenv("DeepseekApikey", "").strip():
            raise HTTPException(422, "服务环境未加载 DeepseekApikey。")
        run_id = uuid4().hex
        path = MISSIONS / f"{run_id}.json"
        logs = MISSIONS.parent / "launch-logs"
        report = {
            "run_id": run_id,
            "task": task.model_dump(),
            "instruction": task.prompt,
            "profile": "fleet",
            "status": "starting",
            "start_ned_m": [entry["north"], entry["east"], entry["reference_down_m"]],
            "alarm": {**alarm, "source": "simulated_call_coarse_area"},
            "fleet": fleet_roles(payload.scouts, payload.relays),
            "events": [],
        }
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.run_fleet",
            "--run-id",
            run_id,
            "--task-id",
            str(payload.task_id),
            "--scouts",
            str(payload.scouts),
            "--relays",
            str(payload.relays),
        ]
        if payload.quick:
            command.extend(("--rounds", "1", "--points-per-scout", "6"))
        if payload.deepseek:
            command.append("--deepseek")
        try:
            MISSIONS.mkdir(parents=True, exist_ok=True)
            logs.mkdir(parents=True, exist_ok=True)
            write_manifest(path, report)
            with (logs / f"{run_id}.log").open("ab") as output:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError as exc:
            if path.is_file():
                record_failure(path, report, "无法启动本地任务进程。")
            raise HTTPException(503, "无法启动本地任务进程。") from exc
        active_run = (run_id, process)
        Thread(target=monitor, args=(process, path, report), daemon=True).start()
        return {**report, "can_pause": True}

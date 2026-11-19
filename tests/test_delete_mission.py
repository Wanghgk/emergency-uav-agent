import json
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request

from backend.api import launch, main


@pytest.mark.parametrize(
    "status",
    [
        "starting",
        "running",
        "completed",
        "finished",
        "paused",
        "failed",
        "interrupted",
        "partial",
    ],
)
def test_delete_mission_permanently_removes_only_its_artifacts(
    tmp_path, monkeypatch, status
):
    run_id = "a" * 32
    other_id = "b" * 32
    monkeypatch.setattr(main, "ROOT", tmp_path)
    for name in (
        "MISSIONS",
        "OBSERVATIONS",
        "GROUND_ROUTES",
        "GROUND_TERRAIN",
        "LAUNCH_LOGS",
        "DIAGNOSTICS",
    ):
        folder = tmp_path / name.lower()
        folder.mkdir()
        monkeypatch.setattr(main, name, folder)
    for name, state in ((run_id, status), (other_id, "running")):
        (main.MISSIONS / f"{name}.json").write_text(
            json.dumps({"status": state, "events": []})
        )
        frame = main.OBSERVATIONS / name
        frame.mkdir()
        (frame / "observation.json").write_text(json.dumps({"run_id": name}))
        (frame / "rgb.png").write_bytes(b"rgb fixture")
    artifacts = [
        main.GROUND_ROUTES / f"{run_id}.json",
        main.GROUND_TERRAIN / f"{run_id}.npz",
        main.LAUNCH_LOGS / f"{run_id}.log",
        main.DIAGNOSTICS / f"{run_id}-rotation-recheck.json",
        main.MISSIONS / f"{run_id}.tmp",
    ]
    for path in artifacts:
        path.touch()
    monkeypatch.setattr(launch, "MISSIONS", main.MISSIONS)
    process = Mock(pid=1234)
    process.poll.return_value = None
    stop = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(launch.subprocess, "run", stop)
    managed = status in launch.ACTIVE_STATUSES
    monkeypatch.setattr(launch, "active_run", (run_id, process) if managed else None)
    request = Request({"type": "http", "headers": []})
    with pytest.raises(HTTPException) as running:
        main.delete_mission(other_id, request)
    assert running.value.status_code == 409
    stop.assert_not_called()
    assert main.delete_mission(run_id, request).status_code == 204
    assert stop.call_count == int(managed)
    with pytest.raises(HTTPException) as missing:
        main.delete_mission(run_id, request)
    assert missing.value.status_code == 404
    assert all(not path.exists() for path in artifacts)
    assert not (tmp_path / "data" / "deleted-missions").exists()
    assert not (main.OBSERVATIONS / run_id).exists()
    assert (main.OBSERVATIONS / other_id / "rgb.png").is_file()
    assert (main.MISSIONS / f"{other_id}.json").is_file()

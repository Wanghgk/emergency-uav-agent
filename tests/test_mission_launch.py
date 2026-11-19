import json
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from backend.api import launch
from backend.api.launch import LaunchRequest, record_failure


@pytest.mark.parametrize(
    "options", [{"scouts": 7}, {"relays": -1}, {"task_id": "290"}, {"maps": "custom"}]
)
def test_launch_accepts_only_supported_options(options):
    with pytest.raises(ValidationError):
        LaunchRequest.model_validate({"task_id": 290, **options})


def test_interrupted_start_is_marked_failed_and_preserves_mission_data(tmp_path):
    path = tmp_path / "mission.json"
    report = {"status": "starting", "events": [], "run_id": "sample"}
    path.write_text(json.dumps(report), encoding="utf-8")
    record_failure(path, report, "simulator failed")
    failed = json.loads(path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["run_id"] == "sample"
    assert failed["events"][0]["error"] == "simulator failed"


@pytest.mark.parametrize("status", ["starting", "running"])
def test_pause_stops_the_owned_process_and_preserves_the_record(
    tmp_path, monkeypatch, status
):
    run_id = "a" * 32
    path = tmp_path / f"{run_id}.json"
    initial = {"status": status, "events": [{"type": "plan"}], "run_id": run_id}
    path.write_text(json.dumps(initial), encoding="utf-8")
    process = Mock(pid=1234)
    process.poll.return_value = None
    stop = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(launch, "MISSIONS", tmp_path)
    monkeypatch.setattr(launch, "active_run", (run_id, process))
    monkeypatch.setattr(launch.subprocess, "run", stop)
    request = Request({"type": "http", "headers": []})

    with pytest.raises(HTTPException) as unrelated:
        launch.pause_mission("b" * 32, request)
    assert unrelated.value.status_code == 409
    stop.assert_not_called()

    report = launch.pause_mission(run_id, request)
    assert stop.call_args.args[0] == ["taskkill.exe", "/F", "/T", "/PID", "1234"]
    process.wait.assert_called_once_with(timeout=5)
    assert report["status"] == "paused"
    assert report["can_pause"] is False
    assert [event["type"] for event in report["events"]] == ["plan", "paused"]
    assert launch.active_mission_id() is None

    path.unlink()
    launch.monitor(process, path, initial)
    assert not path.exists()

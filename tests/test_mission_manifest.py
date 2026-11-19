import json
from pathlib import Path

from backend.simulation.mission_manifest import write_manifest


def test_manifest_retries_a_transient_permission_error(tmp_path, monkeypatch):
    destination = tmp_path / "mission.json"
    replace = Path.replace
    attempts = []

    def temporarily_locked(path, target):
        attempts.append(target)
        if len(attempts) == 1:
            raise PermissionError("temporarily locked")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    write_manifest(destination, {"status": "completed"})
    assert attempts == [destination, destination]
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "completed"
    }

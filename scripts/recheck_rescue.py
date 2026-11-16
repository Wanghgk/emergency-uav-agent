"""Reprocess saved real search frames with rotated image tiles and score offline."""

import argparse
import json
from pathlib import Path
from uuid import UUID

from backend.simulation.esar_adapter import load_task
from backend.simulation.observation_collector import load_observation
from backend.simulation.rescue_search import Alarm, track_detections
from scripts.run_rescue import TiledPersonDetector, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=UUID)
    parser.add_argument("--esar", type=Path, default=Path(".external/ESAR-pinned"))
    parser.add_argument(
        "--weights", type=Path, default=Path(".external/models/yolox_tiny.onnx")
    )
    args = parser.parse_args()
    run_id = args.run_id.hex
    root = Path("data")
    run = json.loads((root / f"missions/{run_id}.json").read_text(encoding="utf-8"))
    alarm = Alarm(
        **{key: value for key, value in run["alarm"].items() if key != "source"}
    )
    tracks = []
    detector = TiledPersonDetector(args.weights, threshold=0.33)
    detector.orientations = (45, 135, 225, 315)
    reprocessed = []
    for item in run["events"]:
        if item["type"] != "observation" or item["sector"] == "START":
            continue
        frame_id = item["frame_id"]
        track_detections(tracks, frame_id, item["detections"], alarm)
        observation = load_observation(
            root / f"observations/{frame_id}/observation.json"
        )
        hits, _ = detector.detect(*observation)
        detections = [hit.model_dump() for hit in hits]
        track_detections(tracks, frame_id, detections, alarm)
        reprocessed.append({"frame_id": frame_id, "detections": detections})
    task, _, origin = load_task(args.esar, "task_data.json", run["task"]["task_id"])
    score = evaluate(args.esar, task, origin, tracks)
    report = {
        "source_run": run_id,
        "reprocessed_frames": len(reprocessed),
        "evaluation": score,
        "frames": reprocessed,
    }
    out = Path(f".external/diagnostics/{run_id}-rotation-recheck.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(out)
    print(json.dumps(score), out)


if __name__ == "__main__":
    main()

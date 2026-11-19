"""Publish complete mission snapshots even during brief Windows file locks."""

import json
import time


def write_manifest(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.04)

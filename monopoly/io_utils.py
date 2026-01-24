from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def tail_lines(path: Path, max_lines: int = 200) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    data = path.read_bytes()
    if not data:
        return []
    lines = data.splitlines()
    tail = lines[-max_lines:]
    return [line.decode("utf-8", errors="ignore") for line in tail]

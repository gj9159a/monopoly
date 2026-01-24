from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json_atomic

REQUIRED_STATUS_FIELDS = {
    "epoch",
    "best_fitness",
    "best_winrate_mean",
    "best_winrate_ci_low",
    "best_winrate_ci_high",
    "promoted_count",
    "last_promoted_epoch",
    "plateau_counter",
    "plateau_epochs",
    "total_games_simulated",
    "eval_seconds_last_epoch",
    "cache_hits_last_epoch",
    "current_phase",
    "started_at",
    "updated_at",
    "runs_dir",
    "best_params_path",
}


def validate_status(data: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_STATUS_FIELDS if field not in data]
    if missing:
        raise ValueError(f"status.json не содержит поля: {', '.join(sorted(missing))}")


def read_status(path: Path) -> dict[str, Any]:
    data = read_json(path, default={})
    if not isinstance(data, dict):
        raise ValueError("status.json должен содержать объект")
    validate_status(data)
    return data


def write_status(path: Path, data: dict[str, Any]) -> None:
    validate_status(data)
    write_json_atomic(path, data)


from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from monopoly.io_utils import tail_lines, write_json_atomic, write_text_atomic
from monopoly.run_utils import list_runs
from monopoly.status import REQUIRED_STATUS_FIELDS, read_status


def _sample_status() -> dict[str, object]:
    return {
        "epoch": 1,
        "best_fitness": 0.0,
        "best_winrate_mean": 0.5,
        "best_winrate_ci_low": 0.4,
        "best_winrate_ci_high": 0.6,
        "promoted_count": 1,
        "last_promoted_epoch": 1,
        "plateau_counter": 0,
        "plateau_epochs": 5,
        "total_games_simulated": 10,
        "eval_seconds_last_epoch": 1.0,
        "cache_hits_last_epoch": 0,
        "current_phase": "training",
        "started_at": "2026-01-24T00:00:00+00:00",
        "updated_at": "2026-01-24T00:00:01+00:00",
        "runs_dir": "runs/test",
        "best_params_path": "runs/test/best.json",
    }


def test_status_json_schema(tmp_path: Path) -> None:
    status = _sample_status()
    assert REQUIRED_STATUS_FIELDS.issubset(status.keys())
    path = tmp_path / "status.json"
    path.write_text(json.dumps(status), encoding="utf-8")
    loaded = read_status(path)
    assert loaded["epoch"] == 1


def test_tail_reader(tmp_path: Path) -> None:
    path = tmp_path / "progress.txt"
    lines = [f"line {idx}" for idx in range(1, 21)]
    path.write_text("\n".join(lines), encoding="utf-8")
    tail = tail_lines(path, max_lines=5)
    assert tail == lines[-5:]


def test_atomic_write_helpers(tmp_path: Path) -> None:
    json_path = tmp_path / "status.json"
    payload = {"ok": True, "value": 3}
    write_json_atomic(json_path, payload)
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload

    text_path = tmp_path / "summary.txt"
    write_text_atomic(text_path, "hello")
    assert text_path.read_text(encoding="utf-8") == "hello"


def test_runs_discovery(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20260101-000001").mkdir()
    (runs / "20260102-000001").mkdir()
    (runs / "20241231-235959").mkdir()
    found = list_runs(runs, limit=2)
    assert [path.name for path in found] == ["20260102-000001", "20260101-000001"]


def test_live_backend_smoke(tmp_path: Path) -> None:
    out_path = tmp_path / "live_state.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "monopoly.live",
            "--players",
            "2",
            "--seed",
            "1",
            "--mode",
            "fast",
            "--time-per-decision-sec",
            "0.01",
            "--horizon-turns",
            "5",
            "--out",
            str(out_path),
            "--max-steps",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["snapshot_index"] >= 2

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from monopoly.league import add_to_league
from monopoly.params import BotParams, save_params


def _cleanup_tmp(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _local_tmp() -> Path:
    base = Path(__file__).resolve().parent / "_tmp"
    base.mkdir(exist_ok=True)
    path = base / uuid4().hex
    path.mkdir()
    return path


def test_progress_smoke() -> None:
    tmp_path = _local_tmp()
    league_dir = tmp_path / "league"
    league_dir.mkdir()

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    save_params(BotParams(), baseline_path)
    save_params(BotParams(), candidate_path)
    add_to_league(candidate_path, 0.1, {"name": "cand", "note": "iter=1"}, league_dir)

    seeds_file = tmp_path / "seeds.txt"
    seeds_file.write_text("1\n2\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "monopoly.progress",
            "--league-dir",
            str(league_dir),
            "--baseline",
            str(baseline_path),
            "--games",
            "2",
            "--players",
            "2",
            "--max-steps",
            "20",
            "--keep",
            "1",
            "--seeds-file",
            str(seeds_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cand" in result.stdout
    _cleanup_tmp(tmp_path)

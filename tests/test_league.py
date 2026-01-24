from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from monopoly.league import add_to_league, load_index, prune_entries
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


def test_league_index_roundtrip() -> None:
    tmp_path = _local_tmp()
    league_dir = tmp_path / "league"
    league_dir.mkdir()

    params_a = tmp_path / "a.json"
    params_b = tmp_path / "b.json"
    save_params(BotParams(), params_a)
    save_params(BotParams(), params_b)

    add_to_league(params_a, "first", "iter=1", 0.1, league_dir)
    add_to_league(params_b, "second", "iter=2", 0.2, league_dir)

    entries = load_index(league_dir)
    assert len(entries) == 2
    assert (league_dir / "first.json").exists()
    assert (league_dir / "second.json").exists()

    prune_entries(league_dir, keep=1)
    entries_after = load_index(league_dir)
    assert len(entries_after) == 1
    assert not (league_dir / "first.json").exists()
    assert (league_dir / "second.json").exists()
    _cleanup_tmp(tmp_path)

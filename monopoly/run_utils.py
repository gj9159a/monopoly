from __future__ import annotations

from pathlib import Path


def list_runs(base_dir: Path, limit: int = 10) -> list[Path]:
    if not base_dir.exists():
        return []
    runs = [path for path in base_dir.iterdir() if path.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)
    if limit > 0:
        runs = runs[:limit]
    return runs


def latest_run(base_dir: Path) -> Path | None:
    runs = list_runs(base_dir, limit=1)
    return runs[0] if runs else None

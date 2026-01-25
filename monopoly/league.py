from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .params import BotParams, load_params, save_params


def _index_path(league_dir: Path) -> Path:
    return league_dir / "index.json"


def _normalize_index(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "entries" in data:
        data = data["entries"]
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError("Некорректный формат index.json")
    return data


def load_index(league_dir: Path) -> list[dict[str, Any]]:
    index_path = _index_path(league_dir)
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return _normalize_index(payload)


def save_index(league_dir: Path, entries: list[dict[str, Any]]) -> None:
    league_dir.mkdir(parents=True, exist_ok=True)
    index_path = _index_path(league_dir)
    index_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_entry_path(entry: dict[str, Any], league_dir: Path) -> Path:
    raw = Path(entry.get("path", ""))
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] != league_dir.name:
        return (Path.cwd() / raw).resolve()
    return (league_dir / raw).resolve()


def add_to_league(
    params_path: Path,
    name: str,
    meta: str,
    fitness: float | None,
    league_dir: Path,
) -> dict[str, Any]:
    if "/" in name or "\\" in name:
        raise ValueError("name не должен содержать разделители путей")
    league_dir.mkdir(parents=True, exist_ok=True)
    target_path = league_dir / f"{name}.json"
    params = load_params(params_path)
    save_params(params, target_path)

    created_at = datetime.now().isoformat(timespec="seconds")
    try:
        rel_path = target_path.relative_to(Path.cwd())
    except ValueError:
        rel_path = target_path
    entry = {
        "name": name,
        "path": rel_path.as_posix(),
        "created_at": created_at,
        "meta": meta,
        "fitness": fitness,
    }
    entries = load_index(league_dir)
    entries.append(entry)
    save_index(league_dir, entries)
    return entry


def list_entries(league_dir: Path) -> list[dict[str, Any]]:
    return load_index(league_dir)


def prune_entries(league_dir: Path, keep: int) -> list[dict[str, Any]]:
    entries = load_index(league_dir)
    if keep <= 0:
        keep = 0
    if len(entries) <= keep:
        return entries
    to_remove = entries[:-keep]
    for entry in to_remove:
        path = _resolve_entry_path(entry, league_dir)
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                try:
                    path.chmod(0o666)
                    path.unlink()
                except PermissionError:
                    raise
    entries = entries[-keep:]
    save_index(league_dir, entries)
    return entries


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Управление лигой параметров")
    parser.add_argument("--league-dir", type=Path, default=Path("monopoly/data/league"))
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    add_parser = subparsers.add_parser("add", help="Добавить параметры в лигу")
    add_parser.add_argument("--params", type=Path, required=True)
    add_parser.add_argument("--name", type=str, required=True)
    add_parser.add_argument("--meta", type=str, default="")
    add_parser.add_argument("--fitness", type=_parse_float, default=None)

    subparsers.add_parser("list", help="Показать лигу")

    prune_parser = subparsers.add_parser("prune", help="Удалить старые записи")
    prune_parser.add_argument("--keep", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    league_dir: Path = args.league_dir

    if args.cmd == "add":
        entry = add_to_league(
            params_path=args.params,
            name=args.name,
            meta=args.meta,
            fitness=args.fitness,
            league_dir=league_dir,
        )
        print(f"Добавлено: {entry['name']} -> {entry['path']}")
        return

    if args.cmd == "list":
        entries = list_entries(league_dir)
        if not entries:
            print("Лига пуста")
            return
        for entry in entries:
            print(
                f"{entry.get('name')} | {entry.get('created_at')} | "
                f"fitness={entry.get('fitness')} | {entry.get('meta')}"
            )
        return

    if args.cmd == "prune":
        entries = prune_entries(league_dir, args.keep)
        print(f"Осталось записей: {len(entries)}")
        return


if __name__ == "__main__":
    main()

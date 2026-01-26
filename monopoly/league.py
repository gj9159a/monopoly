from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import write_json_atomic
from .params import BotParams, load_params, save_params

INDEX_VERSION = 1
DEFAULT_TOP_K = 16


def _index_path(league_dir: Path) -> Path:
    return league_dir / "index.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_params(params: BotParams) -> str:
    payload = json.dumps(params.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _hash_text(payload)


def hash_params(params: BotParams) -> str:
    return _hash_params(params)


def _normalize_meta(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, str):
        meta = meta.strip()
        return {"note": meta} if meta else {}
    return {"note": str(meta)}


def _safe_name(name: str) -> str:
    cleaned = []
    for ch in name.strip():
        if ch.isascii() and (ch.isalnum() or ch in {"-", "_"}):
            cleaned.append(ch)
        elif ch.isspace() or ch in {"/", "\\"}:
            cleaned.append("_")
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def resolve_entry_path(entry: dict[str, Any], league_dir: Path) -> Path:
    raw = Path(str(entry.get("path", "")))
    if raw.is_absolute():
        return raw
    candidate = (Path.cwd() / raw).resolve()
    if candidate.exists():
        return candidate
    return (league_dir / raw).resolve()


def _coerce_fitness(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_created_at(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _sort_key(entry: dict[str, Any]) -> tuple[float, float, str, str]:
    fitness = _coerce_fitness(entry.get("fitness"))
    fitness_value = fitness if fitness is not None else float("-inf")
    created_ts = _parse_created_at(entry.get("created_at"))
    hash_value = str(entry.get("hash") or "")
    name_value = str(entry.get("name") or "")
    return (-fitness_value, -created_ts, hash_value, name_value)


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items_sorted = sorted(items, key=_sort_key)
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for entry in items_sorted:
        hash_value = str(entry.get("hash") or "")
        if hash_value:
            if hash_value in seen:
                continue
            seen.add(hash_value)
        result.append(entry)
    return result


def _ensure_entry_fields(entry: dict[str, Any], league_dir: Path) -> dict[str, Any]:
    normalized = dict(entry)
    if "name" not in normalized or not normalized.get("name"):
        path = Path(str(normalized.get("path", "")))
        normalized["name"] = path.stem if path.stem else "bot"
    if "path" not in normalized or not normalized.get("path"):
        normalized["path"] = f"{normalized['name']}.json"
    normalized["fitness"] = _coerce_fitness(normalized.get("fitness"))
    if "meta" not in normalized:
        normalized["meta"] = {}
    if not isinstance(normalized.get("meta"), dict):
        normalized["meta"] = _normalize_meta(normalized.get("meta"))
    if not normalized.get("hash"):
        path = resolve_entry_path(normalized, league_dir)
        if path.exists():
            try:
                params = load_params(path)
                normalized["hash"] = _hash_params(params)
            except Exception:
                normalized["hash"] = None
        else:
            normalized["hash"] = None
    if not normalized.get("params_hash") and normalized.get("hash"):
        normalized["params_hash"] = normalized.get("hash")
    if normalized.get("params_hash") and not normalized.get("hash"):
        normalized["hash"] = normalized.get("params_hash")
    if not normalized.get("eval_protocol_hash"):
        normalized["eval_protocol_hash"] = "unknown"
    if not normalized.get("bench_timestamp"):
        created = normalized.get("created_at")
        if created:
            normalized["bench_timestamp"] = created
    return normalized


def _normalize_index_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {"version": INDEX_VERSION, "top_k": DEFAULT_TOP_K, "items": []}
    if isinstance(payload, dict):
        if "items" in payload:
            items = payload.get("items")
        elif "entries" in payload:
            items = payload.get("entries")
        else:
            items = payload.get("items", [])
        top_k = payload.get("top_k", DEFAULT_TOP_K)
        return {"version": payload.get("version", INDEX_VERSION), "top_k": top_k, "items": items}
    if isinstance(payload, list):
        return {"version": INDEX_VERSION, "top_k": DEFAULT_TOP_K, "items": payload}
    raise ValueError("Некорректный формат index.json")


def _canonicalize_index(index: dict[str, Any], league_dir: Path) -> dict[str, Any]:
    top_k_raw = index.get("top_k", DEFAULT_TOP_K)
    try:
        top_k = int(top_k_raw)
    except (TypeError, ValueError):
        top_k = DEFAULT_TOP_K
    items_raw = index.get("items", [])
    if not isinstance(items_raw, list):
        raise ValueError("index.json: items должен быть списком")
    items = [_ensure_entry_fields(entry, league_dir) for entry in items_raw if isinstance(entry, dict)]
    items = _dedupe_items(items)
    items.sort(key=_sort_key)
    for idx, entry in enumerate(items, start=1):
        entry["rank"] = idx
    return {"version": INDEX_VERSION, "top_k": top_k, "items": items}


def load_index(league_dir: Path) -> dict[str, Any]:
    index_path = _index_path(league_dir)
    if not index_path.exists():
        return {"version": INDEX_VERSION, "top_k": DEFAULT_TOP_K, "items": []}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return _canonicalize_index(_normalize_index_payload(payload), league_dir)


def save_index(index: dict[str, Any], league_dir: Path) -> None:
    league_dir.mkdir(parents=True, exist_ok=True)
    canonical = _canonicalize_index(index, league_dir)
    write_json_atomic(_index_path(league_dir), canonical)


def _select_name(meta: dict[str, Any], params_hash: str) -> str:
    candidate = meta.get("name") if isinstance(meta, dict) else None
    if isinstance(candidate, str):
        safe = _safe_name(candidate)
        if safe:
            return safe
    return f"bot_{params_hash[:8]}"


def _ensure_unique_name(name: str, league_dir: Path, params_hash: str) -> str:
    base = _safe_name(name) or f"bot_{params_hash[:8]}"
    path = league_dir / f"{base}.json"
    if not path.exists():
        return base
    try:
        existing_hash = _hash_params(load_params(path))
        if existing_hash == params_hash:
            return base
    except Exception:
        pass
    suffix = params_hash[:8]
    alt = f"{base}_{suffix}"
    if not (league_dir / f"{alt}.json").exists():
        return alt
    for idx in range(2, 100):
        alt = f"{base}_{idx}"
        if not (league_dir / f"{alt}.json").exists():
            return alt
    return f"{base}_{suffix}_{datetime.now().strftime('%H%M%S')}"


def _load_params_any(params: BotParams | Path | str | dict[str, Any]) -> BotParams:
    if isinstance(params, BotParams):
        return params
    if isinstance(params, (str, Path)):
        return load_params(Path(params))
    if isinstance(params, dict):
        return BotParams.from_dict(params)
    raise TypeError("params должен быть BotParams, dict или путь")


def _top_hashes(items: list[dict[str, Any]], top_k: int) -> set[str]:
    if top_k <= 0:
        return set()
    return {str(entry.get("hash")) for entry in items[:top_k] if entry.get("hash")}


def _prune_index(index: dict[str, Any], league_dir: Path, top_k: int) -> dict[str, Any]:
    canonical = _canonicalize_index(index, league_dir)
    if top_k <= 0:
        top_k = 0
    items = canonical["items"]
    to_keep = items[:top_k] if top_k else []
    to_remove = items[top_k:] if top_k else items
    for entry in to_remove:
        path = resolve_entry_path(entry, league_dir)
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                try:
                    path.chmod(0o666)
                    path.unlink()
                except PermissionError:
                    raise
    for idx, entry in enumerate(to_keep, start=1):
        entry["rank"] = idx
    canonical["items"] = to_keep
    canonical["top_k"] = top_k
    return canonical


def add_to_league(
    params: BotParams | Path | str | dict[str, Any],
    fitness: float | None,
    meta: dict[str, Any] | str | None,
    league_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    entry_fields: dict[str, Any] | None = None,
) -> tuple[bool, bool, int | None]:
    league_dir = Path(league_dir)
    index = load_index(league_dir)
    items = index.get("items", [])
    prev_items = list(items)
    prev_sorted = sorted(prev_items, key=_sort_key)
    prev_top_hashes = _top_hashes(prev_sorted, top_k)

    params_obj = _load_params_any(params)
    params_hash = _hash_params(params_obj)

    for entry in prev_items:
        if str(entry.get("hash")) == params_hash:
            return False, False, entry.get("rank")

    meta_payload = _normalize_meta(meta)
    name = _ensure_unique_name(_select_name(meta_payload, params_hash), league_dir, params_hash)
    league_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.json"
    target_path = league_dir / filename
    save_params(params_obj, target_path)

    entry = {
        "rank": None,
        "name": name,
        "hash": params_hash,
        "params_hash": params_hash,
        "fitness": _coerce_fitness(fitness),
        "path": Path(filename).as_posix(),
        "created_at": _utc_now(),
        "meta": meta_payload,
    }
    if entry_fields:
        reserved = {"rank", "name", "hash", "params_hash", "fitness", "path", "created_at", "meta"}
        for key, value in entry_fields.items():
            if key in reserved:
                continue
            entry[key] = value

    items.append(entry)
    next_index = {"version": INDEX_VERSION, "top_k": top_k, "items": items}
    pruned = _prune_index(next_index, league_dir, top_k)
    save_index(pruned, league_dir)

    new_top_hashes = _top_hashes(pruned["items"], top_k)
    changed_topk = prev_top_hashes != new_top_hashes

    rank = None
    for item in pruned["items"]:
        if str(item.get("hash")) == params_hash:
            rank = item.get("rank")
            break

    return True, changed_topk, rank


def prune_to_top_k(index: dict[str, Any], league_dir: Path, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    pruned = _prune_index(index, league_dir, top_k)
    save_index(pruned, league_dir)
    return pruned


def list_entries(league_dir: Path) -> list[dict[str, Any]]:
    return load_index(league_dir).get("items", [])


def sample_from_league(
    n: int,
    league_dir: Path,
    exclude_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    exclude_hashes = exclude_hashes or set()
    index = load_index(league_dir)
    top_k = int(index.get("top_k", DEFAULT_TOP_K) or DEFAULT_TOP_K)
    items = index.get("items", [])
    pool = items[:top_k] if top_k > 0 else items
    filtered = [entry for entry in pool if str(entry.get("hash")) not in exclude_hashes]
    if len(filtered) <= n:
        random.shuffle(filtered)
        return filtered
    return random.sample(filtered, n)


def get_top1(league_dir: Path) -> dict[str, Any] | None:
    index = load_index(league_dir)
    items = index.get("items", [])
    if not items:
        return None
    return items[0]


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
    add_parser.add_argument("--name", type=str, default="")
    add_parser.add_argument("--meta", type=str, default="")
    add_parser.add_argument("--fitness", type=_parse_float, default=None)
    add_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    subparsers.add_parser("list", help="Показать лигу")

    prune_parser = subparsers.add_parser("prune", help="Удалить лишние записи")
    prune_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    league_dir: Path = args.league_dir

    if args.cmd == "add":
        meta: dict[str, Any] = {}
        if args.meta:
            meta["note"] = args.meta
        if args.name:
            meta["name"] = args.name
        added, changed_topk, rank = add_to_league(
            params=args.params,
            fitness=args.fitness,
            meta=meta,
            league_dir=league_dir,
            top_k=args.top_k,
        )
        status = "добавлено" if added else "уже есть"
        print(f"{status}: rank={rank}, changed_topk={changed_topk}")
        return

    if args.cmd == "list":
        entries = list_entries(league_dir)
        if not entries:
            print("Лига пуста")
            return
        for entry in entries:
            fitness = entry.get("fitness")
            rank = entry.get("rank")
            name = entry.get("name")
            created = entry.get("created_at")
            print(f"#{rank} {name} | {created} | fitness={fitness} | hash={entry.get('hash')}")
        return

    if args.cmd == "prune":
        index = load_index(league_dir)
        pruned = prune_to_top_k(index, league_dir, args.top_k)
        print(f"Осталось записей: {len(pruned.get('items', []))}")
        return


if __name__ == "__main__":
    main()

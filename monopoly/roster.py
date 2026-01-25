from __future__ import annotations

from pathlib import Path
from typing import Any

from .league import get_top1, resolve_entry_path, sample_from_league
from .params import BotParams, load_params

BOT_NAME_BASELINE = "Бот-baseline"


def format_bot_name(rank: int | None, is_baseline: bool) -> str:
    if is_baseline or rank is None:
        return BOT_NAME_BASELINE
    return f"Бот-#{rank}"


def _load_entry_params(entry: dict[str, Any], league_dir: Path) -> BotParams | None:
    path = resolve_entry_path(entry, league_dir)
    if not path.exists():
        return None
    try:
        return load_params(path)
    except Exception:
        return None


def build_roster_all_top1(
    league_dir: Path,
    baseline: BotParams,
    num_players: int,
) -> tuple[list[BotParams], list[str]]:
    top1 = get_top1(league_dir)
    if not top1:
        return [baseline] * num_players, [BOT_NAME_BASELINE] * num_players
    params = _load_entry_params(top1, league_dir)
    if params is None:
        return [baseline] * num_players, [BOT_NAME_BASELINE] * num_players
    rank = int(top1.get("rank", 1) or 1)
    name = format_bot_name(rank, is_baseline=False)
    return [params] * num_players, [name] * num_players


def build_roster_top1_plus_random(
    league_dir: Path,
    baseline: BotParams,
    num_players: int,
) -> tuple[list[BotParams], list[str]]:
    top1 = get_top1(league_dir)
    if not top1:
        return [baseline] * num_players, [BOT_NAME_BASELINE] * num_players

    params = _load_entry_params(top1, league_dir)
    if params is None:
        return [baseline] * num_players, [BOT_NAME_BASELINE] * num_players

    roster_params: list[BotParams] = [params]
    roster_names: list[str] = [format_bot_name(int(top1.get("rank", 1) or 1), is_baseline=False)]

    remaining = max(0, num_players - 1)
    exclude = {str(top1.get("hash") or "")}
    sampled = sample_from_league(remaining, league_dir, exclude_hashes=exclude) if remaining else []
    for entry in sampled:
        entry_params = _load_entry_params(entry, league_dir)
        if entry_params is None:
            continue
        roster_params.append(entry_params)
        rank = entry.get("rank")
        roster_names.append(format_bot_name(int(rank) if rank is not None else None, is_baseline=False))

    while len(roster_params) < num_players:
        roster_params.append(baseline)
        roster_names.append(BOT_NAME_BASELINE)

    return roster_params, roster_names

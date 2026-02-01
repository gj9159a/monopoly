from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autotrain import run_autotrain
from .bench import bench
from .io_utils import read_json, write_json_atomic
from .league import add_to_league, hash_params, load_index, resolve_entry_path, save_index
from .params import BotParams, ThinkingConfig, load_params
from .train import ADVANTAGE_SCALE, FITNESS_CONFIDENCE, FITNESS_COEFFS, PLACE_TO_SCORE, SCORING_VERSION

DEFAULT_TOP_K_POOL = 8
DEFAULT_LEAGUE_CAP = 16
DEFAULT_MAX_NEW_BESTS = 16
DEFAULT_META_PLATEAU_CYCLES = 3
DEFAULT_BOOTSTRAP_MIN = 8
DEFAULT_EPS_IMPROVEMENT = 1e-4
DEFAULT_GAMES_PER_CAND_MIN = 8
DEFAULT_GAMES_PER_CAND_MAX = 64
DEFAULT_GAMES_PER_CAND_TARGET_CI = 0.20
CYCLE_SEED_STEP = 10007
POOL_SNAPSHOT_STEP = 9973
REBENCH_LOG_FILE = "rebench_log.csv"

STATUS_FILE = "status.json"


@dataclass
class TopKStats:
    hashes: set[str]
    top1_fitness: float
    mean_fitness: float


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def eval_protocol_hash(protocol: dict[str, Any]) -> str:
    return _hash_payload(protocol)


def build_eval_protocol(
    *,
    protocol_kind: str,
    players: int,
    games_per_cand: int,
    max_steps: int,
    seat_rotation_policy: dict[str, Any],
    eval_seeds_policy: dict[str, Any],
    opponent_sampling_policy: dict[str, Any],
    scoring_version: str,
) -> dict[str, Any]:
    return {
        "protocol_kind": protocol_kind,
        "players": players,
        "games_per_cand": games_per_cand,
        "max_steps": max_steps,
        "seat_rotation_policy": seat_rotation_policy,
        "eval_seeds_policy": eval_seeds_policy,
        "opponent_sampling_policy": opponent_sampling_policy,
        "fitness_confidence": FITNESS_CONFIDENCE,
        "place_to_score": PLACE_TO_SCORE,
        "advantage_scale": ADVANTAGE_SCALE,
        "fitness_coeffs": FITNESS_COEFFS,
        "cutoff_outcome_policy": "placement_mapping",
        "scoring_version": scoring_version,
    }


def build_league_rebench_protocol(
    *,
    players: int,
    games_per_cand: int,
    max_steps: int,
    league_cap: int,
    seed: int,
    cand_seats: str,
) -> dict[str, Any]:
    return build_eval_protocol(
        protocol_kind="league_rebench",
        players=players,
        games_per_cand=games_per_cand,
        max_steps=max_steps,
        seat_rotation_policy={
            "mode": cand_seats,
            "start": "seed % players",
            "seed_source": "league_rebench_seed",
        },
        eval_seeds_policy={"mode": "seed + idx", "seed_source": "league_rebench_seed"},
        opponent_sampling_policy={
            "source": "league_topk_selfplay",
            "pool_size": league_cap,
            "pool_size_effective": "min(league_cap, league_size)",
            "exclude_self": True,
            "pool_fixed": True,
            "fallback_if_empty": "baseline",
        },
        scoring_version=SCORING_VERSION,
    ) | {"league_rebench_seed": seed}


def _build_training_eval_protocol(
    *,
    players: int,
    games_per_cand: int,
    max_steps: int,
    cand_seats: str,
    master_seed: int,
    opponents_source: str,
    top_k_pool: int,
    league_cap: int,
    pool_snapshot_step: int,
) -> dict[str, Any]:
    opponent_policy: dict[str, Any] = {
        "source": opponents_source,
        "pool_fixed_per_cycle": True,
    }
    if opponents_source == "league_snapshot":
        opponent_policy.update(
            {
                "top_k_pool": top_k_pool,
                "league_cap": league_cap,
                "pool_snapshot_seed_policy": f"master_seed + cycle_index*{pool_snapshot_step}",
            }
        )
    return build_eval_protocol(
        protocol_kind="training_eval",
        players=players,
        games_per_cand=games_per_cand,
        max_steps=max_steps,
        seat_rotation_policy={
            "mode": cand_seats,
            "start": "cycle_seed % players",
            "seed_source": "cycle_seed",
        },
        eval_seeds_policy={
            "mode": "cycle_seed + idx",
            "cycle_seed_policy": f"master_seed + cycle_index*{CYCLE_SEED_STEP}",
        },
        opponent_sampling_policy=opponent_policy,
        scoring_version=SCORING_VERSION,
    ) | {"master_seed": master_seed}


def _extract_eval_protocol_hash(entry: dict[str, Any]) -> str:
    value = entry.get("eval_protocol_hash")
    if isinstance(value, str) and value:
        return value
    return "unknown"


def _collect_eval_protocol_hashes(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in items:
        value = _extract_eval_protocol_hash(entry)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_fitness(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_k_entries(items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    return items[: min(top_k, len(items))]


def _top_k_stats(items: list[dict[str, Any]], top_k: int) -> TopKStats:
    entries = _top_k_entries(items, top_k)
    hashes = {str(entry.get("hash")) for entry in entries if entry.get("hash")}
    if not entries:
        return TopKStats(hashes=hashes, top1_fitness=float("-inf"), mean_fitness=float("-inf"))
    fitness_values = [float(entry.get("fitness", 0.0)) for entry in entries]
    top1 = fitness_values[0]
    mean_value = sum(fitness_values) / len(fitness_values)
    return TopKStats(hashes=hashes, top1_fitness=top1, mean_fitness=mean_value)


def _top_k_improved(prev: TopKStats, new: TopKStats, eps: float) -> bool:
    if prev.hashes != new.hashes:
        return True
    if new.top1_fitness > prev.top1_fitness + eps:
        return True
    if new.mean_fitness > prev.mean_fitness + eps:
        return True
    return False


def _fitness_is_valid(value: float | None) -> bool:
    if value is None:
        return False
    return math.isfinite(value)


def _find_entry_by_hash(items: list[dict[str, Any]], params_hash: str) -> dict[str, Any] | None:
    for entry in items:
        if str(entry.get("hash")) == params_hash:
            return entry
    return None


def _snapshot_pool(
    items: list[dict[str, Any]],
    rng: random.Random,
    pool_size: int,
    league_cap: int,
) -> list[dict[str, Any]]:
    if pool_size <= 0:
        return []
    candidates = items[: min(league_cap, len(items))]
    if not candidates:
        return []
    if pool_size >= len(candidates):
        return list(candidates)
    return rng.sample(list(candidates), pool_size)


def _load_pool_params(snapshot: list[dict[str, Any]], league_dir: Path) -> list[BotParams]:
    params_list: list[BotParams] = []
    for entry in snapshot:
        path = resolve_entry_path(entry, league_dir)
        if not path.exists():
            continue
        try:
            params = load_params(path).with_thinking(ThinkingConfig())
        except Exception:
            continue
        params_list.append(params)
    return params_list


def _load_league_pool_params(
    league_dir: Path,
    league_cap: int,
    baseline: BotParams,
) -> tuple[list[BotParams], bool]:
    index = load_index(league_dir)
    items = index.get("items", [])[: int(league_cap)]
    params_list: list[BotParams] = []
    for entry in items:
        path = resolve_entry_path(entry, league_dir)
        if not path.exists():
            continue
        try:
            params = load_params(path).with_thinking(ThinkingConfig())
        except Exception:
            continue
        params_list.append(params)
    if not params_list:
        return [baseline], True
    return params_list, False


def _append_progress_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        handle.write("\n")


def _load_best_params(best_path: Path) -> BotParams:
    return load_params(best_path).with_thinking(ThinkingConfig())


def _bump_reject_reason(status: dict[str, Any], reason: str) -> None:
    reasons = status.get("candidates_rejected_reason")
    if not isinstance(reasons, dict):
        reasons = {}
        status["candidates_rejected_reason"] = reasons
    reasons[reason] = int(reasons.get(reason, 0) or 0) + 1


def _write_dedup_example(
    cycle_dir: Path,
    candidate_hash: str,
    candidate_fitness: float | None,
    existing_entry: dict[str, Any],
) -> None:
    payload = {
        "candidate_hash": candidate_hash,
        "candidate_fitness": candidate_fitness,
        "existing": {
            "rank": existing_entry.get("rank"),
            "name": existing_entry.get("name"),
            "hash": existing_entry.get("hash"),
            "fitness": existing_entry.get("fitness"),
            "created_at": existing_entry.get("created_at"),
            "path": existing_entry.get("path"),
        },
    }
    write_json_atomic(cycle_dir / "dedup_example.json", payload)


def _rebench_worker(
    args: tuple[
        BotParams,
        list[BotParams],
        BotParams,
        Path,
        int,
        int,
        int,
        str,
        int,
    ]
) -> tuple[dict[str, float | str], bool]:
    candidate, pool, baseline, league_dir, games, seed, max_steps, cand_seats, players = args
    pool_fallback = False
    if not pool:
        pool = [baseline]
        pool_fallback = True
    result = bench(
        candidate=candidate,
        baseline=baseline,
        league_dir=league_dir,
        opponents="league",
        num_players=players,
        games=games,
        seed=seed,
        max_steps=max_steps,
        cand_seats=cand_seats,
        min_games=games,
        delta=0.0,
        seeds_file=None,
        opponents_pool=pool,
    )
    return result, pool_fallback


def _rebench_league(
    league_dir: Path,
    baseline_path: Path,
    runs_dir: Path,
    eval_protocol: dict[str, Any],
    eval_protocol_hash_value: str,
    league_cap: int,
    games: int,
    max_steps: int,
    seed: int,
    cand_seats: str,
    players: int,
    workers: int,
) -> dict[str, Any]:
    index = load_index(league_dir)
    items = index.get("items", [])[:league_cap]
    if not items:
        return index

    baseline = load_params(baseline_path).with_thinking(ThinkingConfig())

    entries: list[tuple[dict[str, Any], BotParams]] = []
    for entry in items:
        path = resolve_entry_path(entry, league_dir)
        if not path.exists():
            continue
        try:
            params = load_params(path).with_thinking(ThinkingConfig())
        except Exception:
            continue
        params_hash = hash_params(params)
        entry["hash"] = params_hash
        entry["params_hash"] = params_hash
        entries.append((entry, params))

    params_list = [params for _, params in entries]
    jobs: list[
        tuple[
            BotParams,
            list[BotParams],
            BotParams,
            Path,
            int,
            int,
            int,
            str,
            int,
        ]
    ] = []
    for idx, (_, params) in enumerate(entries):
        pool = params_list[:idx] + params_list[idx + 1 :]
        jobs.append(
            (
                params,
                pool,
                baseline,
                league_dir,
                int(games),
                int(seed),
                int(max_steps),
                str(cand_seats),
                int(players),
            )
        )

    if int(workers) > 1 and len(jobs) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(workers)) as pool:
            results = pool.map(_rebench_worker, jobs)
    else:
        results = [_rebench_worker(job) for job in jobs]

    log_path = runs_dir / REBENCH_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "hash",
                "games",
                "fitness",
                "win_rate",
                "ci_low",
                "ci_high",
                "eval_protocol_hash",
                "pool_fallback",
            ]
        )
        for (entry, _), (result, pool_fallback) in zip(entries, results, strict=False):
            fitness = float(result["fitness"])
            win_rate = float(result["win_rate"])
            ci_low = float(result["ci_low"])
            ci_high = float(result["ci_high"])
            win_lcb = float(result.get("win_lcb", ci_low))
            place_score = float(result.get("place_score", 0.0))
            advantage = float(result.get("advantage", 0.0))
            cutoff_rate = float(result.get("cutoff_rate", 0.0))
            entry.update(
                {
                    "fitness": fitness,
                    "win_rate": win_rate,
                    "win_rate_ci_low": ci_low,
                    "win_rate_ci_high": ci_high,
                    "win_lcb": win_lcb,
                    "place_score": place_score,
                    "advantage": advantage,
                    "cutoff_rate": cutoff_rate,
                    "bench_timestamp": _utc_now(),
                    "eval_protocol_hash": eval_protocol_hash_value,
                    "eval_protocol": eval_protocol,
                }
            )
            writer.writerow(
                [
                    entry.get("name"),
                    entry.get("hash"),
                    games,
                    f"{fitness:.6f}",
                    f"{win_rate:.6f}",
                    f"{ci_low:.6f}",
                    f"{ci_high:.6f}",
                    eval_protocol_hash_value,
                    int(pool_fallback),
                ]
            )

    updated = {"version": index.get("version", 1), "top_k": index.get("top_k", league_cap), "items": [e for e, _ in entries]}
    save_index(updated, league_dir)
    return load_index(league_dir)


def _ensure_status_defaults(status: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    status.setdefault("current_phase", "training")
    status.setdefault("started_at", _utc_now())
    status.setdefault("updated_at", status["started_at"])
    status.setdefault("runs_dir", str(runs_dir))
    status.setdefault("current_cycle", 0)
    status.setdefault("new_bests_count", 0)
    status.setdefault("meta_plateau", 0)
    status.setdefault("cycle_in_progress", False)
    status.setdefault("current_cycle_dir", "")
    status.setdefault("pool_snapshot", [])
    status.setdefault("bootstrap", False)
    status.setdefault("league_size", 0)
    status.setdefault("league_rebench_needed", False)
    status.setdefault("league_rebench_done", False)
    status.setdefault("league_eval_protocol_hash", "")
    status.setdefault("league_rebench_on_mismatch", True)
    status.setdefault("league_rebench_games", 0)
    status.setdefault("league_rebench_max_steps", 0)
    status.setdefault("league_rebench_seed", 0)
    status.setdefault("auto_games_per_cand", False)
    status.setdefault("games_per_cand_min", 0)
    status.setdefault("games_per_cand_max", 0)
    status.setdefault("games_per_cand_target_ci", 0.0)
    status.setdefault("games_per_cand_current", 0)
    status.setdefault("games_per_cand_prev", 0)
    status.setdefault("games_per_cand_prev_ci_width", None)
    status.setdefault("cycle_games_per_cand", 0)
    status.setdefault("bench_max_games", 0)
    status.setdefault("min_progress_games", 0)
    status.setdefault("candidates_produced", 0)
    status.setdefault("candidates_evaluated", 0)
    status.setdefault("candidates_eligible_for_league", 0)
    status.setdefault("candidates_added_to_league", 0)
    status.setdefault("candidates_deduped", 0)
    if not isinstance(status.get("candidates_rejected_reason"), dict):
        status["candidates_rejected_reason"] = {}
    return status


def _status_path(runs_dir: Path) -> Path:
    return runs_dir / STATUS_FILE


def _write_status(status_path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = _utc_now()
    write_json_atomic(status_path, status)


def _bench_seed_source() -> str:
    default_seeds = Path("monopoly/data/seeds.txt")
    if default_seeds.exists():
        return str(default_seeds)
    return "seed+idx"


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Некорректное значение bool: {value}")


def _derived_seed_policy(bench_seed_source: str) -> dict[str, str]:
    return {
        "cycle_seed": "master_seed + cycle_index*10007",
        "eval_seeds": "cycle_seed + idx (0..games_per_cand-1)",
        "seat_rotation": "start = cycle_seed % players (cand_seats=rotate)",
        "game_seed": "eval_seed passed to create_engine",
        "opponents_rng": "cycle_seed + game_seed*1013 + seat*917 + case_index*37",
        "pool_snapshot_rng": "master_seed + cycle_index*9973",
        "bench_seeds": bench_seed_source,
    }


def _append_seeds_used(
    path: Path,
    cycle_index: int,
    master_seed: int,
    cycle_seed: int,
    games_per_cand: int,
    pool_snapshot_seed: int,
    bench_seed_source: str,
) -> None:
    count = max(1, min(10, int(games_per_cand)))
    eval_seeds = [cycle_seed + idx for idx in range(count)]
    seeds_text = ",".join(str(value) for value in eval_seeds)
    line = (
        f"cycle={cycle_index:03d} master_seed={master_seed} cycle_seed={cycle_seed} "
        f"eval_seeds[:{count}]={seeds_text} pool_snapshot_seed={pool_snapshot_seed} "
        f"bench_seeds={bench_seed_source}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def _load_prev_bench_ci(runs_dir: Path, cycle_index: int) -> tuple[float | None, float | None]:
    if cycle_index <= 0:
        return None, None
    bench_path = runs_dir / f"cycle_{cycle_index:03d}" / "last_bench.json"
    payload = read_json(bench_path, default=None)
    if not isinstance(payload, dict):
        return None, None
    ci_low = payload.get("ci_low")
    ci_high = payload.get("ci_high")
    if not isinstance(ci_low, (int, float)) or not isinstance(ci_high, (int, float)):
        return None, None
    return float(ci_low), float(ci_high)


def _auto_games_per_cand(
    base_games: int,
    ci_low: float | None,
    ci_high: float | None,
    min_games: int,
    max_games: int,
    target_ci: float,
) -> tuple[int, float | None]:
    base_games = _clamp_int(base_games, min_games, max_games)
    if ci_low is None or ci_high is None:
        return base_games, None
    width = float(ci_high) - float(ci_low)
    if width <= 0 or target_ci <= 0:
        return base_games, width
    scale = width / target_ci
    next_games = _clamp_int(int(round(base_games * scale * scale)), min_games, max_games)
    return next_games, width


def _cycle_status_path(cycle_dir: Path) -> Path:
    return cycle_dir / "status.json"


def _read_cycle_status(cycle_dir: Path) -> dict[str, Any] | None:
    status = read_json(_cycle_status_path(cycle_dir), default=None)
    return status if isinstance(status, dict) else None


def _cycle_finished(cycle_dir: Path) -> bool:
    status = _read_cycle_status(cycle_dir)
    if not status:
        return False
    return status.get("current_phase") == "finished"


def run_autoevolve(
    seed: int,
    runs_dir: Path,
    league_dir: Path,
    baseline_path: Path,
    top_k_pool: int,
    league_cap: int,
    max_new_bests: int,
    meta_plateau_cycles: int,
    bootstrap_min_league_for_pool: int,
    league_rebench_on_mismatch: bool,
    league_rebench_games: int | None,
    league_rebench_max_steps: int | None,
    league_rebench_seed: int | None,
    population: int,
    elite: int,
    games_per_cand: int,
    epoch_iters: int,
    plateau_epochs: int,
    eps_winrate: float,
    eps_fitness: float,
    min_progress_games: int,
    bench_max_games: int,
    delta: float,
    max_steps: int,
    workers: int,
    resume: bool,
    auto_games_per_cand: bool = False,
    games_per_cand_min: int = DEFAULT_GAMES_PER_CAND_MIN,
    games_per_cand_max: int = DEFAULT_GAMES_PER_CAND_MAX,
    games_per_cand_target_ci: float = DEFAULT_GAMES_PER_CAND_TARGET_CI,
    eps_improvement: float = DEFAULT_EPS_IMPROVEMENT,
) -> None:
    if bench_max_games < min_progress_games:
        raise ValueError("bench_max_games должен быть >= min_progress_games")
    if int(games_per_cand_min) <= 0:
        raise ValueError("games_per_cand_min должен быть >= 1")
    if int(games_per_cand_max) < int(games_per_cand_min):
        raise ValueError("games_per_cand_max должен быть >= games_per_cand_min")
    if auto_games_per_cand and float(games_per_cand_target_ci) <= 0:
        raise ValueError("games_per_cand_target_ci должен быть > 0 при авто-режиме")
    runs_dir.mkdir(parents=True, exist_ok=True)
    status_path = _status_path(runs_dir)

    status: dict[str, Any] = {}
    if resume and status_path.exists():
        existing = read_json(status_path, default={})
        if isinstance(existing, dict):
            status = existing

    status = _ensure_status_defaults(status, runs_dir)
    bench_seed_source = _bench_seed_source()
    derived_policy = _derived_seed_policy(bench_seed_source)
    status.update(
        {
            "top_k_pool": int(top_k_pool),
            "league_cap": int(league_cap),
            "max_new_bests": int(max_new_bests),
            "meta_plateau_cycles": int(meta_plateau_cycles),
            "bootstrap_min_league_for_pool": int(bootstrap_min_league_for_pool),
            "league_rebench_on_mismatch": bool(league_rebench_on_mismatch),
            "league_rebench_games": int(league_rebench_games or 0),
            "league_rebench_max_steps": int(league_rebench_max_steps or 0),
            "league_rebench_seed": int(league_rebench_seed) if league_rebench_seed is not None else int(seed),
            "seed": int(seed),
            "population": int(population),
            "elite": int(elite),
            "games_per_cand": int(games_per_cand),
            "auto_games_per_cand": bool(auto_games_per_cand),
            "games_per_cand_min": int(games_per_cand_min),
            "games_per_cand_max": int(games_per_cand_max),
            "games_per_cand_target_ci": float(games_per_cand_target_ci),
            "games_per_cand_current": int(status.get("games_per_cand_current", games_per_cand) or games_per_cand),
            "epoch_iters": int(epoch_iters),
            "plateau_epochs": int(plateau_epochs),
            "eps_winrate": float(eps_winrate),
            "eps_fitness": float(eps_fitness),
            "min_progress_games": int(min_progress_games),
            "bench_max_games": int(bench_max_games),
            "delta": float(delta),
            "max_steps": int(max_steps),
            "workers": int(workers),
            "master_seed": int(seed),
            "derived_seed_policy": derived_policy,
        }
    )
    _write_status(status_path, status)
    print(
        "seed: "
        f"master={seed}; derived="
        "cycle_seed=master+cycle_index*10007; eval_seeds=cycle_seed+idx; "
        "seat_rotation=start=cycle_seed%players; game_seed=eval_seed; "
        "opponents_rng=cycle_seed+game_seed*1013+seat*917+case_index*37; "
        f"pool_snapshot_rng=master+cycle_index*9973; bench_seeds={bench_seed_source}"
    )

    rebench_games = int(league_rebench_games) if league_rebench_games else int(games_per_cand)
    rebench_max_steps = int(league_rebench_max_steps) if league_rebench_max_steps else int(max_steps)
    rebench_seed = int(league_rebench_seed) if league_rebench_seed is not None else int(seed)

    league_eval_protocol = build_league_rebench_protocol(
        players=4,
        games_per_cand=rebench_games,
        max_steps=rebench_max_steps,
        league_cap=int(league_cap),
        seed=rebench_seed,
        cand_seats="rotate",
    )
    league_eval_hash = eval_protocol_hash(league_eval_protocol)
    league_index = load_index(league_dir)
    league_items = league_index.get("items", [])[: int(league_cap)]
    hash_counts = _collect_eval_protocol_hashes(league_items)
    mismatched = sum(count for key, count in hash_counts.items() if key != league_eval_hash)
    rebench_needed = bool(league_items) and mismatched > 0

    status.update(
        {
            "league_eval_protocol_hash": league_eval_hash,
            "league_rebench_needed": bool(rebench_needed),
            "league_rebench_done": False,
            "league_rebench_on_mismatch": bool(league_rebench_on_mismatch),
            "league_rebench_games": rebench_games,
            "league_rebench_max_steps": rebench_max_steps,
            "league_rebench_seed": rebench_seed,
        }
    )
    _write_status(status_path, status)
    print(
        f"league eval protocol hash={league_eval_hash} hashes={hash_counts} "
        f"mismatched={mismatched} rebench_on_mismatch={bool(league_rebench_on_mismatch)}"
    )

    if league_rebench_on_mismatch and rebench_needed and not resume:
        league_index = _rebench_league(
            league_dir=league_dir,
            baseline_path=baseline_path,
            runs_dir=runs_dir,
            eval_protocol=league_eval_protocol,
            eval_protocol_hash_value=league_eval_hash,
            league_cap=int(league_cap),
            games=rebench_games,
            max_steps=rebench_max_steps,
            seed=rebench_seed,
            cand_seats="rotate",
            players=4,
            workers=workers,
        )
        status.update(
            {
                "league_rebench_done": True,
                "league_size": len(league_index.get("items", [])),
            }
        )
        _write_status(status_path, status)

    current_cycle = int(status.get("current_cycle", 0))
    new_bests_count = int(status.get("new_bests_count", 0))
    meta_plateau = int(status.get("meta_plateau", 0))

    while new_bests_count < max_new_bests and meta_plateau < meta_plateau_cycles:
        resume_cycle = False
        cycle_dir = None
        if resume and status.get("cycle_in_progress"):
            cycle_dir_raw = status.get("current_cycle_dir")
            if cycle_dir_raw:
                cycle_dir = Path(str(cycle_dir_raw))
                if cycle_dir.exists() and not _cycle_finished(cycle_dir):
                    resume_cycle = True

        if resume_cycle:
            cycle_index = current_cycle
            pool_snapshot = status.get("pool_snapshot", [])
            bootstrap = bool(status.get("bootstrap", False))
            cycle_seed = status.get("cycle_seed")
            if not isinstance(cycle_seed, int):
                cycle_seed = seed + cycle_index * CYCLE_SEED_STEP
            pool_snapshot_seed = status.get("pool_snapshot_seed")
            if not isinstance(pool_snapshot_seed, int):
                pool_snapshot_seed = seed + cycle_index * POOL_SNAPSHOT_STEP
            prev_topk = TopKStats(
                hashes=set(status.get("prev_topk_hashes", []) or []),
                top1_fitness=float(status.get("prev_top1_fitness", float("-inf"))),
                mean_fitness=float(status.get("prev_topk_mean", float("-inf"))),
            )
            cycle_games_per_cand = int(
                status.get("cycle_games_per_cand")
                or status.get("games_per_cand_current")
                or games_per_cand
            )
            if auto_games_per_cand:
                cycle_games_per_cand = _clamp_int(
                    cycle_games_per_cand,
                    int(games_per_cand_min),
                    int(games_per_cand_max),
                )
        else:
            cycle_index = current_cycle + 1
            cycle_dir = runs_dir / f"cycle_{cycle_index:03d}"
            index = load_index(league_dir)
            items = index.get("items", [])
            prev_topk_stats = _top_k_stats(items, top_k_pool)
            prev_topk = prev_topk_stats
            league_size = len(items)
            bootstrap = league_size < bootstrap_min_league_for_pool
            cycle_seed = seed + cycle_index * CYCLE_SEED_STEP
            pool_snapshot_seed = seed + cycle_index * POOL_SNAPSHOT_STEP
            if bootstrap:
                pool_snapshot = []
            else:
                rng = random.Random(pool_snapshot_seed)
                pool_snapshot = _snapshot_pool(items, rng, top_k_pool, league_cap)

            prev_games = int(status.get("games_per_cand_current") or games_per_cand)
            cycle_games_per_cand = prev_games
            prev_ci_width = None
            if auto_games_per_cand:
                ci_low, ci_high = _load_prev_bench_ci(runs_dir, current_cycle)
                cycle_games_per_cand, prev_ci_width = _auto_games_per_cand(
                    base_games=prev_games,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    min_games=int(games_per_cand_min),
                    max_games=int(games_per_cand_max),
                    target_ci=float(games_per_cand_target_ci),
                )

            status.update(
                {
                    "current_cycle": cycle_index,
                    "current_cycle_dir": str(cycle_dir),
                    "pool_snapshot": pool_snapshot,
                    "bootstrap": bootstrap,
                    "league_size": league_size,
                    "cycle_seed": cycle_seed,
                    "pool_snapshot_seed": pool_snapshot_seed,
                    "games_per_cand_prev": int(prev_games),
                    "games_per_cand_prev_ci_width": prev_ci_width,
                    "games_per_cand_current": int(cycle_games_per_cand),
                    "cycle_games_per_cand": int(cycle_games_per_cand),
                    "prev_topk_hashes": sorted(prev_topk.hashes),
                    "prev_top1_fitness": prev_topk.top1_fitness,
                    "prev_topk_mean": prev_topk.mean_fitness,
                }
            )
            seeds_used_path = runs_dir / "seeds_used.txt"
            _append_seeds_used(
                seeds_used_path,
                cycle_index=cycle_index,
                master_seed=int(seed),
                cycle_seed=int(cycle_seed),
                games_per_cand=int(cycle_games_per_cand),
                pool_snapshot_seed=int(pool_snapshot_seed),
                bench_seed_source=bench_seed_source,
            )

        if cycle_dir is None:
            cycle_dir = runs_dir / f"cycle_{cycle_index:03d}"

        baseline = load_params(baseline_path).with_thinking(ThinkingConfig())
        if bootstrap:
            opponents_pool = [baseline]
        else:
            opponents_pool = _load_pool_params(pool_snapshot, league_dir)
            if not opponents_pool:
                opponents_pool = [baseline]

        status.update(
            {
                "cycle_in_progress": True,
                "current_cycle": cycle_index,
                "current_cycle_dir": str(cycle_dir),
                "current_phase": "training",
                "cycle_seed": cycle_seed,
                "pool_snapshot_seed": pool_snapshot_seed,
                "games_per_cand_current": int(cycle_games_per_cand),
                "cycle_games_per_cand": int(cycle_games_per_cand),
            }
        )
        _write_status(status_path, status)

        run_autotrain(
            profile="train_deep",
            epoch_iters=epoch_iters,
            plateau_epochs=plateau_epochs,
            eps_winrate=eps_winrate,
            eps_fitness=eps_fitness,
            min_progress_games=min_progress_games,
            bench_max_games=bench_max_games,
            delta=delta,
            seed=cycle_seed,
            players=4,
            max_steps=max_steps,
            population=population,
            elite=elite,
            games_per_cand=cycle_games_per_cand,
            opponents="league",
            baseline_path=baseline_path,
            league_dir=league_dir,
            cand_seats="rotate",
            workers=workers,
            runs_dir=cycle_dir,
            max_hours=None,
            seeds_file=None,
            resume=resume_cycle,
            opponents_pool=opponents_pool,
        )

        resume = False
        status["cycle_in_progress"] = False
        status["current_phase"] = "cycle_done"
        _write_status(status_path, status)

        new_bests_count += 1
        cycle_status = _read_cycle_status(cycle_dir) or {}
        best_path = cycle_dir / "best.json"
        best_params = _load_best_params(best_path)
        train_best_fitness = _coerce_fitness(cycle_status.get("best_fitness"))

        promotion_games = int(rebench_games)
        promotion_max_steps = int(rebench_max_steps)
        promotion_seed = int(rebench_seed)
        promotion_pool, promotion_pool_fallback = _load_league_pool_params(
            league_dir=league_dir,
            league_cap=int(league_cap),
            baseline=baseline,
        )
        promotion_result = bench(
            candidate=best_params,
            baseline=baseline,
            league_dir=league_dir,
            opponents="league",
            num_players=4,
            games=promotion_games,
            seed=promotion_seed,
            max_steps=promotion_max_steps,
            cand_seats="rotate",
            min_games=promotion_games,
            delta=0.0,
            seeds_file=None,
            opponents_pool=promotion_pool,
            workers=workers,
        )

        best_fitness = float(promotion_result["fitness"])
        fitness_valid = _fitness_is_valid(best_fitness)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_note = f"cycle={cycle_index}; bench_fitness={best_fitness:.6f}"
        if train_best_fitness is not None:
            meta_note += f"; train_fitness={train_best_fitness:.6f}"
        meta = {
            "name": f"best_{timestamp}_c{cycle_index:03d}",
            "note": meta_note,
        }
        bench_win_rate = float(promotion_result["win_rate"])
        bench_ci_low = float(promotion_result["ci_low"])
        bench_ci_high = float(promotion_result["ci_high"])
        bench_win_lcb = float(promotion_result.get("win_lcb", bench_ci_low))
        bench_place_score = float(promotion_result.get("place_score", 0.0))
        bench_advantage = float(promotion_result.get("advantage", 0.0))
        bench_cutoff_rate = float(promotion_result.get("cutoff_rate", 0.0))
        bench_stop_reason = str(promotion_result.get("stop_reason", ""))

        entry_eval_protocol = build_league_rebench_protocol(
            players=4,
            games_per_cand=promotion_games,
            max_steps=promotion_max_steps,
            league_cap=int(league_cap),
            seed=promotion_seed,
            cand_seats="rotate",
        )
        entry_eval_hash = eval_protocol_hash(entry_eval_protocol)
        entry_fields: dict[str, Any] = {
            "eval_protocol_hash": entry_eval_hash,
            "eval_protocol": entry_eval_protocol,
            "bench_timestamp": _utc_now(),
            "source_run_id": runs_dir.name,
            "source_cycle": cycle_index,
        }
        entry_fields["win_rate"] = bench_win_rate
        entry_fields["win_rate_ci_low"] = bench_ci_low
        entry_fields["win_rate_ci_high"] = bench_ci_high
        entry_fields["win_lcb"] = bench_win_lcb
        entry_fields["place_score"] = bench_place_score
        entry_fields["advantage"] = bench_advantage
        entry_fields["cutoff_rate"] = bench_cutoff_rate
        status["candidates_produced"] = int(status.get("candidates_produced", 0) or 0) + 1
        status["candidates_evaluated"] = int(status.get("candidates_evaluated", 0) or 0) + 1

        added = False
        changed_topk = False
        rank = None
        params_hash = hash_params(best_params)
        index_before_add = load_index(league_dir)
        existing_entry = _find_entry_by_hash(index_before_add.get("items", []), params_hash)
        add_reason = "added"

        if not fitness_valid:
            _bump_reject_reason(status, "invalid_fitness")
            add_reason = "invalid_fitness"
        elif existing_entry is not None:
            status["candidates_deduped"] = int(status.get("candidates_deduped", 0) or 0) + 1
            _bump_reject_reason(status, "dedup")
            rank = existing_entry.get("rank")
            add_reason = "dedup"
            _write_dedup_example(cycle_dir, params_hash, best_fitness, existing_entry)
        else:
            status["candidates_eligible_for_league"] = int(
                status.get("candidates_eligible_for_league", 0) or 0
            ) + 1
            added, changed_topk, rank = add_to_league(
                params=best_params,
                fitness=best_fitness,
                meta=meta,
                league_dir=league_dir,
                top_k=league_cap,
                entry_fields=entry_fields,
            )
            if added and rank is not None:
                status["candidates_added_to_league"] = int(
                    status.get("candidates_added_to_league", 0) or 0
                ) + 1
                add_reason = "added"
            elif added and rank is None:
                _bump_reject_reason(status, "pruned_top_k")
                add_reason = "pruned_top_k"
            else:
                _bump_reject_reason(status, "not_added")
                add_reason = "not_added"

        short_hash = params_hash[:8]
        final_added = bool(added and rank is not None)
        progress_line = (
            "final bench | "
            f"fitness {best_fitness:.4f} | "
            f"win_rate {bench_win_rate:.3f} lcb {bench_win_lcb:.3f} | "
            f"cutoff {bench_cutoff_rate:.3f} | "
            f"games {promotion_games} | stop {bench_stop_reason}"
        )
        if promotion_pool_fallback:
            progress_line += " | pool_fallback=baseline"
        if train_best_fitness is not None:
            progress_line += f" | train_fitness {train_best_fitness:.4f}"
        progress_line += f" | added {'yes' if final_added else 'no'}"
        if rank is not None:
            progress_line += f" | rank {rank}"
        _append_progress_line(cycle_dir / "progress.txt", progress_line)
        print(
            "best: "
            f"fitness={best_fitness:.6f} hash={short_hash} add={'yes' if final_added else 'no'} "
            f"reason={add_reason}"
        )

        index_after = load_index(league_dir)
        items_after = index_after.get("items", [])
        new_topk = _top_k_stats(items_after, top_k_pool)
        improved = _top_k_improved(prev_topk, new_topk, eps_improvement)

        if improved:
            meta_plateau = 0
        else:
            meta_plateau += 1

        status.update(
            {
                "current_cycle": cycle_index,
                "new_bests_count": new_bests_count,
                "meta_plateau": meta_plateau,
                "league_size": len(items_after),
                "last_best_fitness": float(best_fitness),
                "last_best_added": final_added,
                "last_best_rank": rank,
                "last_best_changed_topk": bool(changed_topk),
                "topk_hashes": sorted(new_topk.hashes),
                "topk_top1_fitness": new_topk.top1_fitness,
                "topk_mean": new_topk.mean_fitness,
            }
        )
        _write_status(status_path, status)

        current_cycle = cycle_index

        if new_bests_count >= max_new_bests or meta_plateau >= meta_plateau_cycles:
            break

    status.update({"current_phase": "finished"})
    _write_status(status_path, status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-evolve: bootstrap -> league -> meta-cycle")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run auto-evolve loop")
    run_parser.add_argument("--seed", type=int, default=123)
    run_parser.add_argument("--runs-dir", type=Path, default=None)
    run_parser.add_argument("--league-dir", type=Path, default=Path("monopoly/data/league"))
    run_parser.add_argument("--baseline", type=Path, default=Path("monopoly/data/params_baseline.json"))
    run_parser.add_argument("--top-k-pool", type=int, default=DEFAULT_TOP_K_POOL)
    run_parser.add_argument("--league-cap", type=int, default=DEFAULT_LEAGUE_CAP)
    run_parser.add_argument("--max-new-bests", type=int, default=DEFAULT_MAX_NEW_BESTS)
    run_parser.add_argument("--meta-plateau-cycles", type=int, default=DEFAULT_META_PLATEAU_CYCLES)
    run_parser.add_argument("--bootstrap-min-league-for-pool", type=int, default=DEFAULT_BOOTSTRAP_MIN)
    run_parser.add_argument("--league-rebench-on-mismatch", type=_parse_bool, default=True)
    run_parser.add_argument("--league-rebench-games", type=int, default=None)
    run_parser.add_argument("--league-rebench-max-steps", type=int, default=None)
    run_parser.add_argument("--league-rebench-seed", type=int, default=None)
    run_parser.add_argument("--population", type=int, default=48)
    run_parser.add_argument("--elite", type=int, default=12)
    run_parser.add_argument("--games-per-cand", type=int, default=20)
    run_parser.add_argument("--auto-games-per-cand", type=_parse_bool, default=False)
    run_parser.add_argument("--games-per-cand-min", type=int, default=DEFAULT_GAMES_PER_CAND_MIN)
    run_parser.add_argument("--games-per-cand-max", type=int, default=DEFAULT_GAMES_PER_CAND_MAX)
    run_parser.add_argument("--games-per-cand-target-ci", type=float, default=DEFAULT_GAMES_PER_CAND_TARGET_CI)
    run_parser.add_argument("--epoch-iters", type=int, default=10)
    run_parser.add_argument("--plateau-epochs", type=int, default=10)
    run_parser.add_argument("--eps-winrate", type=float, default=0.01)
    run_parser.add_argument("--eps-fitness", type=float, default=0.02)
    run_parser.add_argument("--min-progress-games", type=int, default=128)
    run_parser.add_argument("--bench-max-games", type=int, default=512)
    run_parser.add_argument("--delta", type=float, default=0.005)
    run_parser.add_argument("--max-steps", type=int, default=2000)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return

    runs_dir = args.runs_dir
    if runs_dir is None:
        runs_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")

    run_autoevolve(
        seed=args.seed,
        runs_dir=runs_dir,
        league_dir=args.league_dir,
        baseline_path=args.baseline,
        top_k_pool=args.top_k_pool,
        league_cap=args.league_cap,
        max_new_bests=args.max_new_bests,
        meta_plateau_cycles=args.meta_plateau_cycles,
        bootstrap_min_league_for_pool=args.bootstrap_min_league_for_pool,
        league_rebench_on_mismatch=bool(args.league_rebench_on_mismatch),
        league_rebench_games=args.league_rebench_games,
        league_rebench_max_steps=args.league_rebench_max_steps,
        league_rebench_seed=args.league_rebench_seed,
        population=args.population,
        elite=args.elite,
        games_per_cand=args.games_per_cand,
        auto_games_per_cand=bool(args.auto_games_per_cand),
        games_per_cand_min=args.games_per_cand_min,
        games_per_cand_max=args.games_per_cand_max,
        games_per_cand_target_ci=args.games_per_cand_target_ci,
        epoch_iters=args.epoch_iters,
        plateau_epochs=args.plateau_epochs,
        eps_winrate=args.eps_winrate,
        eps_fitness=args.eps_fitness,
        min_progress_games=args.min_progress_games,
        bench_max_games=args.bench_max_games,
        delta=args.delta,
        max_steps=args.max_steps,
        workers=args.workers,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autotrain import run_autotrain
from .io_utils import read_json, write_json_atomic
from .league import add_to_league, load_index, resolve_entry_path
from .params import BotParams, ThinkingConfig, load_params

DEFAULT_TOP_K_POOL = 8
DEFAULT_LEAGUE_CAP = 16
DEFAULT_MAX_NEW_BESTS = 16
DEFAULT_META_PLATEAU_CYCLES = 5
DEFAULT_BOOTSTRAP_MIN = 4
DEFAULT_EPS_IMPROVEMENT = 1e-4

STATUS_FILE = "status.json"


@dataclass
class TopKStats:
    hashes: set[str]
    top1_fitness: float
    mean_fitness: float


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


def _load_best_params(best_path: Path) -> BotParams:
    return load_params(best_path).with_thinking(ThinkingConfig())


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
    return status


def _status_path(runs_dir: Path) -> Path:
    return runs_dir / STATUS_FILE


def _write_status(status_path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = _utc_now()
    write_json_atomic(status_path, status)


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
    population: int,
    elite: int,
    games_per_cand: int,
    epoch_iters: int,
    plateau_epochs: int,
    eps_winrate: float,
    eps_fitness: float,
    min_progress_games: int,
    delta: float,
    max_steps: int,
    workers: int,
    resume: bool,
    eps_improvement: float = DEFAULT_EPS_IMPROVEMENT,
) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    status_path = _status_path(runs_dir)

    status: dict[str, Any] = {}
    if resume and status_path.exists():
        existing = read_json(status_path, default={})
        if isinstance(existing, dict):
            status = existing

    status = _ensure_status_defaults(status, runs_dir)
    status.update(
        {
            "top_k_pool": int(top_k_pool),
            "league_cap": int(league_cap),
            "max_new_bests": int(max_new_bests),
            "meta_plateau_cycles": int(meta_plateau_cycles),
            "bootstrap_min_league_for_pool": int(bootstrap_min_league_for_pool),
            "seed": int(seed),
            "population": int(population),
            "elite": int(elite),
            "games_per_cand": int(games_per_cand),
            "epoch_iters": int(epoch_iters),
            "plateau_epochs": int(plateau_epochs),
            "eps_winrate": float(eps_winrate),
            "eps_fitness": float(eps_fitness),
            "min_progress_games": int(min_progress_games),
            "delta": float(delta),
            "max_steps": int(max_steps),
            "workers": int(workers),
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
            prev_topk = TopKStats(
                hashes=set(status.get("prev_topk_hashes", []) or []),
                top1_fitness=float(status.get("prev_top1_fitness", float("-inf"))),
                mean_fitness=float(status.get("prev_topk_mean", float("-inf"))),
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
            if bootstrap:
                pool_snapshot = []
            else:
                rng = random.Random(seed + cycle_index * 9973)
                pool_snapshot = _snapshot_pool(items, rng, top_k_pool, league_cap)

            status.update(
                {
                    "current_cycle": cycle_index,
                    "current_cycle_dir": str(cycle_dir),
                    "pool_snapshot": pool_snapshot,
                    "bootstrap": bootstrap,
                    "league_size": league_size,
                    "prev_topk_hashes": sorted(prev_topk.hashes),
                    "prev_top1_fitness": prev_topk.top1_fitness,
                    "prev_topk_mean": prev_topk.mean_fitness,
                }
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
            delta=delta,
            seed=seed,
            players=6,
            max_steps=max_steps,
            population=population,
            elite=elite,
            games_per_cand=games_per_cand,
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
        best_fitness = _coerce_fitness(cycle_status.get("best_fitness"))
        if best_fitness is None:
            best_fitness = 0.0
        best_path = cycle_dir / "best.json"
        best_params = _load_best_params(best_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta = {
            "name": f"best_{timestamp}_c{cycle_index:03d}",
            "note": f"cycle={cycle_index}; best_fitness={best_fitness:.6f}",
        }
        added, changed_topk, rank = add_to_league(
            params=best_params,
            fitness=best_fitness,
            meta=meta,
            league_dir=league_dir,
            top_k=league_cap,
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
                "last_best_added": bool(added),
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
    run_parser.add_argument("--population", type=int, default=48)
    run_parser.add_argument("--elite", type=int, default=12)
    run_parser.add_argument("--games-per-cand", type=int, default=20)
    run_parser.add_argument("--epoch-iters", type=int, default=10)
    run_parser.add_argument("--plateau-epochs", type=int, default=10)
    run_parser.add_argument("--eps-winrate", type=float, default=0.01)
    run_parser.add_argument("--eps-fitness", type=float, default=0.02)
    run_parser.add_argument("--min-progress-games", type=int, default=400)
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
        population=args.population,
        elite=args.elite,
        games_per_cand=args.games_per_cand,
        epoch_iters=args.epoch_iters,
        plateau_epochs=args.plateau_epochs,
        eps_winrate=args.eps_winrate,
        eps_fitness=args.eps_fitness,
        min_progress_games=args.min_progress_games,
        delta=args.delta,
        max_steps=args.max_steps,
        workers=args.workers,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .bench import bench
from .io_utils import write_json_atomic, write_text_atomic
from .params import (
    BotParams,
    PARAM_SPECS,
    ThinkingConfig,
    load_params,
    params_to_vector,
    save_params,
    vector_to_params,
)
from .status import REQUIRED_STATUS_FIELDS, write_status
from .train import build_eval_cases, build_opponent_pool, evaluate_candidates, load_league


@dataclass
class CEMState:
    rng: random.Random
    means: list[float]
    stds: list[float]
    min_stds: list[float]
    best_params: BotParams
    best_score: float
    iter_index: int = 0


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "train_deep": {
        "population": 48,
        "elite": 12,
        "games_per_cand": 20,
        "max_steps": 2000,
        "opponents": "mixed",
        "cand_seats": "rotate",
    }
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _init_cem_state(seed: int) -> CEMState:
    rng = random.Random(seed)
    means = params_to_vector(BotParams())
    stds = [spec.init_std for spec in PARAM_SPECS]
    min_stds = [(spec.max_value - spec.min_value) * 0.02 for spec in PARAM_SPECS]
    return CEMState(
        rng=rng,
        means=means,
        stds=stds,
        min_stds=min_stds,
        best_params=BotParams(),
        best_score=float("-inf"),
    )


def _sample_params(rng: random.Random, means: list[float], stds: list[float]) -> BotParams:
    values = [rng.gauss(mean_value, std_value) for mean_value, std_value in zip(means, stds, strict=False)]
    return vector_to_params(values)


def _update_cem(
    state: CEMState,
    candidates: list[BotParams],
    scores: list[float],
    elite: int,
) -> tuple[float, float]:
    ranked = list(zip(candidates, scores))
    ranked.sort(key=lambda item: item[1], reverse=True)
    elites = ranked[: max(1, elite)]
    elite_scores = [score for _, score in elites]
    elite_vectors = [params_to_vector(params) for params, _ in elites]

    state.means = [mean(values) for values in zip(*elite_vectors, strict=False)]
    state.stds = []
    for idx, spec in enumerate(PARAM_SPECS):
        values = [vec[idx] for vec in elite_vectors]
        std_value = pstdev(values) if len(values) > 1 else spec.init_std
        state.stds.append(max(state.min_stds[idx], std_value))

    if elites[0][1] > state.best_score:
        state.best_params = elites[0][0]
        state.best_score = elites[0][1]

    elite_mean = sum(elite_scores) / len(elite_scores)
    elite_std = pstdev(elite_scores) if len(elite_scores) > 1 else 0.0
    return elite_mean, elite_std


def _parse_workers(value: str) -> int:
    value = value.strip().lower()
    if value == "auto":
        return max(1, os.cpu_count() or 1)
    return max(1, int(value))


def _write_progress(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        handle.write("\n")


def _prepare_status(payload: dict[str, Any]) -> dict[str, Any]:
    for field in REQUIRED_STATUS_FIELDS:
        payload.setdefault(field, None)
    return payload


def _write_summary(path: Path, status: dict[str, Any], stop_reason: str) -> None:
    lines = [
        f"stop_reason: {stop_reason}",
        f"epoch: {status.get('epoch')}",
        f"best_fitness: {status.get('best_fitness')}",
        (
            "best_winrate: "
            f"{status.get('best_winrate_mean')} "
            f"[{status.get('best_winrate_ci_low')}, {status.get('best_winrate_ci_high')}]"
        ),
        f"total_games: {status.get('total_games_simulated')}",
        f"best_params_path: {status.get('best_params_path')}",
        f"runs_dir: {status.get('runs_dir')}",
    ]
    write_text_atomic(path, "\n".join(lines))


def run_autotrain(
    profile: str,
    epoch_iters: int,
    plateau_epochs: int,
    eps_winrate: float,
    eps_fitness: float,
    min_progress_games: int,
    delta: float,
    seed: int,
    players: int,
    max_steps: int,
    population: int,
    elite: int,
    games_per_cand: int,
    opponents: str,
    baseline_path: Path,
    league_dir: Path,
    cand_seats: str,
    workers: int,
    runs_dir: Path,
    max_hours: float | None,
    seeds_file: Path | None,
    resume: bool,
    opponents_pool: list[BotParams] | None = None,
) -> None:
    if elite > population:
        raise ValueError("elite должен быть <= population")
    if not 2 <= players <= 6:
        raise ValueError("players должен быть в диапазоне 2..6")

    runs_dir.mkdir(parents=True, exist_ok=True)
    status_path = runs_dir / "status.json"
    log_path = runs_dir / "train_log.csv"
    progress_path = runs_dir / "progress.txt"
    best_path = runs_dir / "best.json"
    last_bench_path = runs_dir / "last_bench.json"
    cache_path = runs_dir / "eval_cache.jsonl"
    mean_std_path = runs_dir / "mean_std.json"
    summary_path = runs_dir / "summary.txt"

    baseline = load_params(baseline_path).with_thinking(ThinkingConfig())
    if opponents_pool is None:
        league = [params.with_thinking(ThinkingConfig()) for params in load_league(league_dir)]
        opponents_pool = build_opponent_pool(opponents, baseline, league)
    else:
        opponents_pool = [params.with_thinking(ThinkingConfig()) for params in opponents_pool]

    seeds = [seed + idx for idx in range(games_per_cand)]
    cases = build_eval_cases(seeds, players, cand_seats, seed)
    games_per_candidate = len(cases)

    cem_state = _init_cem_state(seed)
    cache: dict[str, Any] = {}

    started_at = _utc_now()
    status: dict[str, Any]
    if resume and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status = _prepare_status(status)
        status["plateau_epochs"] = plateau_epochs
        if best_path.exists():
            cem_state.best_params = load_params(best_path).with_thinking(ThinkingConfig())
            cem_state.best_score = float(status.get("best_fitness", -1e9))
        if mean_std_path.exists():
            payload = json.loads(mean_std_path.read_text(encoding="utf-8"))
            means = payload.get("mean")
            stds = payload.get("std")
            if isinstance(means, list) and isinstance(stds, list):
                cem_state.means = [float(value) for value in means]
                cem_state.stds = [float(value) for value in stds]
        status["current_phase"] = "training"
        status["updated_at"] = _utc_now()
        write_status(status_path, status)
    else:
        status = _prepare_status(
            {
                "epoch": 0,
                "best_fitness": -1e9,
                "best_winrate_mean": 0.0,
                "best_winrate_ci_low": 0.0,
                "best_winrate_ci_high": 0.0,
                "promoted_count": 0,
                "last_promoted_epoch": 0,
                "plateau_counter": 0,
                "plateau_epochs": plateau_epochs,
                "total_games_simulated": 0,
                "eval_seconds_last_epoch": 0.0,
                "cache_hits_last_epoch": 0,
                "current_phase": "training",
                "started_at": started_at,
                "updated_at": started_at,
                "runs_dir": str(runs_dir),
                "best_params_path": str(best_path),
            }
        )
        write_status(status_path, status)

    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(
                [
                    "epoch",
                    "best_fitness",
                    "best_winrate_mean",
                    "best_winrate_ci_low",
                    "best_winrate_ci_high",
                    "best_win_lcb",
                    "best_place_score",
                    "best_advantage",
                    "best_cutoff_rate",
                    "plateau_counter",
                    "promoted_count",
                    "cache_hits",
                    "eval_seconds",
                    "games_simulated",
                ]
            )

        total_games = int(status.get("total_games_simulated", 0) or 0)
        promoted_count = int(status.get("promoted_count", 0) or 0)
        last_promoted_epoch = int(status.get("last_promoted_epoch", 0) or 0)
        plateau_counter = int(status.get("plateau_counter", 0) or 0)
        prev_best_fitness = float(status.get("best_fitness", -1e9))
        prev_best_winrate = float(status.get("best_winrate_mean", 0.0))
        stop_reason = "plateau"
        start_time = time.perf_counter()

        def _handle_stop(signum: int, frame: Any) -> None:
            phase = "stopped"
            if status_path.exists():
                existing = json.loads(status_path.read_text(encoding="utf-8"))
                phase = existing.get("current_phase", "stopped")
                if phase not in {"paused", "stopped"}:
                    phase = "stopped"
            status.update({"current_phase": phase, "updated_at": _utc_now()})
            write_status(status_path, status)
            _write_summary(summary_path, status, phase)
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        epoch = int(status.get("epoch", 0) or 0)
        while True:
            epoch += 1
            epoch_start = time.perf_counter()
            cache_hits_epoch = 0
            games_epoch = 0

            for _ in range(epoch_iters):
                cem_state.iter_index += 1
                iter_start = time.perf_counter()
                candidates = [
                    _sample_params(cem_state.rng, cem_state.means, cem_state.stds)
                    for _ in range(population)
                ]
                results, cache_hits = evaluate_candidates(
                    candidates=candidates,
                    seeds=seeds,
                    num_players=players,
                    max_steps=max_steps,
                    opponents_pool=opponents_pool,
                    cand_seats=cand_seats,
                    seed=seed,
                    workers=workers,
                    cache=cache,
                    cache_path=cache_path,
                )
                scores = [result.fitness for result in results]
                elite_mean, elite_std = _update_cem(cem_state, candidates, scores, elite)

                non_cached = max(0, population - cache_hits)
                games_epoch += non_cached * games_per_candidate
                cache_hits_epoch += cache_hits

                iter_elapsed = time.perf_counter() - iter_start
                _write_progress(
                    progress_path,
                    (
                        f"iter {cem_state.iter_index:03d} | best {cem_state.best_score:.4f} | "
                        f"mean(top) {elite_mean:.4f} | std(top) {elite_std:.4f} | "
                        f"cache {cache_hits} | {iter_elapsed:.2f}s"
                    ),
                )

            total_games += games_epoch

            status.update({"current_phase": "bench", "updated_at": _utc_now()})
            write_status(status_path, status)

            if seeds_file is None:
                default_seeds = Path("monopoly/data/seeds.txt")
                if default_seeds.exists():
                    seeds_file = default_seeds

            bench_result = bench(
                candidate=cem_state.best_params.with_thinking(ThinkingConfig()),
                baseline=baseline.with_thinking(ThinkingConfig()),
                league_dir=league_dir,
                opponents=opponents,
                num_players=players,
                games=min_progress_games,
                seed=seed,
                max_steps=max_steps,
                cand_seats=cand_seats,
                min_games=min_progress_games,
                delta=delta,
                seeds_file=seeds_file,
                opponents_pool=list(opponents_pool),
            )
            write_json_atomic(last_bench_path, bench_result)
            save_params(cem_state.best_params, best_path)

            best_fitness = float(cem_state.best_score)
            best_winrate = float(bench_result["win_rate"])
            best_ci_low = float(bench_result["ci_low"])
            best_ci_high = float(bench_result["ci_high"])
            best_win_lcb = float(bench_result.get("win_lcb", best_ci_low))
            best_place_score = float(bench_result.get("place_score", 0.0))
            best_advantage = float(bench_result.get("advantage", 0.0))
            best_cutoff_rate = float(bench_result.get("cutoff_rate", 0.0))

            if best_fitness > prev_best_fitness + eps_fitness:
                promoted_count += 1
                last_promoted_epoch = epoch

            plateau_increment = (
                (best_fitness - prev_best_fitness) < eps_fitness
                and (best_winrate - prev_best_winrate) < eps_winrate
            )
            if plateau_increment:
                plateau_counter += 1
            else:
                plateau_counter = 0

            epoch_elapsed = time.perf_counter() - epoch_start
            status.update(
                {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "best_winrate_mean": best_winrate,
                    "best_winrate_ci_low": best_ci_low,
                    "best_winrate_ci_high": best_ci_high,
                    "best_win_lcb": best_win_lcb,
                    "best_place_score": best_place_score,
                    "best_advantage": best_advantage,
                    "best_cutoff_rate": best_cutoff_rate,
                    "promoted_count": promoted_count,
                    "last_promoted_epoch": last_promoted_epoch,
                    "plateau_counter": plateau_counter,
                    "plateau_epochs": plateau_epochs,
                    "total_games_simulated": total_games,
                    "eval_seconds_last_epoch": epoch_elapsed,
                    "cache_hits_last_epoch": cache_hits_epoch,
                    "current_phase": "training",
                    "updated_at": _utc_now(),
                    "best_params_path": str(best_path),
                }
            )
            write_status(status_path, status)
            write_json_atomic(
                mean_std_path,
                {
                    "params": [spec.name for spec in PARAM_SPECS],
                    "mean": [float(value) for value in cem_state.means],
                    "std": [float(value) for value in cem_state.stds],
                },
            )

            writer.writerow(
                [
                    epoch,
                    f"{best_fitness:.6f}",
                    f"{best_winrate:.6f}",
                    f"{best_ci_low:.6f}",
                    f"{best_ci_high:.6f}",
                    f"{best_win_lcb:.6f}",
                    f"{best_place_score:.6f}",
                    f"{best_advantage:.6f}",
                    f"{best_cutoff_rate:.6f}",
                    plateau_counter,
                    promoted_count,
                    cache_hits_epoch,
                    f"{epoch_elapsed:.4f}",
                    total_games,
                ]
            )

            _write_progress(
                progress_path,
                (
                    f"epoch {epoch:03d} | best_fitness {best_fitness:.4f} | "
                    f"win_hat {best_winrate:.3f} lcb {best_win_lcb:.3f} | "
                    f"cutoff {best_cutoff_rate:.3f} | plateau {plateau_counter}/{plateau_epochs}"
                ),
            )

            prev_best_fitness = best_fitness
            prev_best_winrate = best_winrate

            if plateau_counter >= plateau_epochs:
                stop_reason = "plateau"
                break
            if max_hours is not None:
                elapsed_hours = (time.perf_counter() - start_time) / 3600
                if elapsed_hours >= max_hours:
                    stop_reason = "max_hours"
                    break

        status.update(
            {
                "current_phase": "finished" if stop_reason == "plateau" else "stopped",
                "updated_at": _utc_now(),
            }
        )
        write_status(status_path, status)
        _write_summary(summary_path, status, status["current_phase"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autotrain: CEM до плато")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Запустить автотренинг")
    run_parser.add_argument("--profile", type=str, default="train_deep")
    run_parser.add_argument("--epoch-iters", type=int, default=10)
    run_parser.add_argument("--plateau-epochs", type=int, default=10)
    run_parser.add_argument("--eps-winrate", type=float, default=0.01)
    run_parser.add_argument("--eps-fitness", type=float, default=0.02)
    run_parser.add_argument("--min-progress-games", type=int, default=400)
    run_parser.add_argument("--delta", type=float, default=0.005)
    run_parser.add_argument("--seed", type=int, default=123)
    run_parser.add_argument("--players", type=int, default=6)
    run_parser.add_argument("--max-steps", type=int, default=2000)
    run_parser.add_argument("--population", type=int, default=None)
    run_parser.add_argument("--elite", type=int, default=None)
    run_parser.add_argument("--games-per-cand", type=int, default=None)
    run_parser.add_argument("--opponents", type=str, choices=["baseline", "league", "mixed"], default=None)
    run_parser.add_argument("--league-dir", type=Path, default=Path("monopoly/data/league"))
    run_parser.add_argument("--baseline", type=Path, default=Path("monopoly/data/params_baseline.json"))
    run_parser.add_argument("--cand-seats", type=str, choices=["all", "rotate"], default=None)
    run_parser.add_argument("--workers", type=_parse_workers, default=_parse_workers("auto"))
    run_parser.add_argument("--runs-dir", type=Path, default=None)
    run_parser.add_argument("--max-hours", type=float, default=None)
    run_parser.add_argument("--seeds-file", type=Path, default=None)
    run_parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return

    profile_name = "train_deep" if args.profile == "deep" else args.profile
    profile_cfg = PROFILE_PRESETS.get(profile_name)
    if profile_cfg is None:
        raise ValueError(f"Неизвестный профиль: {args.profile}")

    population = args.population if args.population is not None else profile_cfg["population"]
    elite = args.elite if args.elite is not None else profile_cfg["elite"]
    games_per_cand = args.games_per_cand if args.games_per_cand is not None else profile_cfg["games_per_cand"]
    opponents = args.opponents if args.opponents is not None else profile_cfg["opponents"]
    cand_seats = args.cand_seats if args.cand_seats is not None else profile_cfg["cand_seats"]

    runs_dir = args.runs_dir
    if runs_dir is None:
        runs_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")

    run_autotrain(
        profile=profile_name,
        epoch_iters=args.epoch_iters,
        plateau_epochs=args.plateau_epochs,
        eps_winrate=args.eps_winrate,
        eps_fitness=args.eps_fitness,
        min_progress_games=args.min_progress_games,
        delta=args.delta,
        seed=args.seed,
        players=args.players,
        max_steps=args.max_steps,
        population=population,
        elite=elite,
        games_per_cand=games_per_cand,
        opponents=opponents,
        baseline_path=args.baseline,
        league_dir=args.league_dir,
        cand_seats=cand_seats,
        workers=args.workers,
        runs_dir=runs_dir,
        max_hours=args.max_hours,
        seeds_file=args.seeds_file,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

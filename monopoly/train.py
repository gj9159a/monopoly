from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence

from .engine import create_engine
from .models import GameState
from .params import BotParams, PARAM_SPECS, load_params, params_to_vector, save_params, vector_to_params


def _net_worth(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    total = player.money
    for cell in state.board:
        if cell.owner_id != player_id:
            continue
        if cell.mortgaged:
            total += cell.mortgage_value or 0
        else:
            total += cell.price or 0
        total += (cell.houses + cell.hotels) * (cell.house_cost or 0)
    return total


def score_player(state: GameState, player_id: int, first_bankrupt_id: int | None) -> float:
    score = 0.0
    if state.winner_id == player_id:
        score += 1.0
    score += 0.0001 * _net_worth(state, player_id)
    if first_bankrupt_id == player_id:
        score -= 0.2
    return score


def play_game(
    params: BotParams | list[BotParams],
    num_players: int,
    seed: int,
    max_steps: int,
) -> tuple[GameState, int | None, int]:
    engine = create_engine(num_players=num_players, seed=seed, bot_params=params)
    first_bankrupt_id = None
    steps = 0
    while not engine.state.game_over and steps < max_steps:
        events = engine.step()
        for event in events:
            if event.type == "BANKRUPTCY" and first_bankrupt_id is None:
                first_bankrupt_id = event.player_id
        steps += 1
    return engine.state, first_bankrupt_id, steps


def load_league(league_dir: Path) -> list[BotParams]:
    if not league_dir.exists():
        return []
    files = sorted([
        path
        for path in league_dir.iterdir()
        if path.suffix.lower() in {".json", ".yml", ".yaml"}
    ])
    params_list: list[BotParams] = []
    for path in files:
        params_list.append(load_params(path))
    return params_list


def build_opponent_pool(opponents: str, baseline: BotParams, league: list[BotParams]) -> list[BotParams]:
    if opponents == "baseline":
        return [baseline]
    if opponents == "league":
        if not league:
            raise ValueError("Лига пуста для режима opponents=league")
        return list(league)
    if opponents == "mixed":
        pool = [baseline]
        pool.extend(league)
        return pool
    raise ValueError(f"Неизвестный режим opponents: {opponents}")


def build_eval_cases(
    seeds: Sequence[int],
    num_players: int,
    cand_seats: str,
    seed: int,
) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    if cand_seats == "all":
        for game_seed in seeds:
            for seat in range(num_players):
                cases.append((game_seed, seat))
        return cases
    if cand_seats != "rotate":
        raise ValueError(f"Неизвестный режим cand_seats: {cand_seats}")
    start = seed % num_players
    for idx, game_seed in enumerate(seeds):
        seat = (start + idx) % num_players
        cases.append((game_seed, seat))
    return cases


def _select_opponents(rng: random.Random, pool: Sequence[BotParams], count: int) -> list[BotParams]:
    if not pool:
        raise ValueError("Пул оппонентов пуст")
    if len(pool) == 1:
        return [pool[0]] * count
    return [pool[rng.randrange(len(pool))] for _ in range(count)]


def evaluate_candidate(
    candidate: BotParams,
    seeds: Sequence[int],
    num_players: int,
    max_steps: int,
    opponents_pool: Sequence[BotParams],
    cand_seats: str,
    seed: int,
) -> float:
    cases = build_eval_cases(seeds, num_players, cand_seats, seed)
    scores: list[float] = []
    for idx, (game_seed, seat) in enumerate(cases):
        rng = random.Random(seed + game_seed * 1013 + seat * 917 + idx * 37)
        opponents = _select_opponents(rng, opponents_pool, num_players - 1)
        params_by_seat: list[BotParams] = []
        opp_iter = iter(opponents)
        for seat_idx in range(num_players):
            if seat_idx == seat:
                params_by_seat.append(candidate)
            else:
                params_by_seat.append(next(opp_iter))
        state, first_bankrupt_id, _ = play_game(params_by_seat, num_players, game_seed, max_steps)
        scores.append(score_player(state, seat, first_bankrupt_id))
    return sum(scores) / len(scores)


def _sample_params(rng: random.Random, means: list[float], stds: list[float]) -> BotParams:
    values: list[float] = []
    for mean_value, std_value in zip(means, stds, strict=False):
        values.append(rng.gauss(mean_value, std_value))
    return vector_to_params(values)


def _save_mean_std(path: Path, means: Sequence[float], stds: Sequence[float]) -> None:
    payload = {
        "params": [spec.name for spec in PARAM_SPECS],
        "mean": [float(value) for value in means],
        "std": [float(value) for value in stds],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _eval_worker(args: tuple[BotParams, list[int], int, int, list[BotParams], str, int]) -> float:
    candidate, seeds, num_players, max_steps, pool, cand_seats, seed = args
    return evaluate_candidate(candidate, seeds, num_players, max_steps, pool, cand_seats, seed)


def cem_train(
    iters: int,
    population: int,
    elite: int,
    games_per_candidate: int,
    num_players: int,
    seed: int,
    max_steps: int,
    opponents: str,
    baseline_path: Path,
    league_dir: Path,
    cand_seats: str,
    out_path: Path,
    checkpoint_dir: Path,
    checkpoint_every: int,
    workers: int,
) -> BotParams:
    rng = random.Random(seed)
    means = params_to_vector(BotParams())
    stds = [spec.init_std for spec in PARAM_SPECS]
    min_stds = [(spec.max_value - spec.min_value) * 0.02 for spec in PARAM_SPECS]

    eval_seeds = [seed + idx for idx in range(games_per_candidate)]

    baseline = load_params(baseline_path)
    league = load_league(league_dir)
    opponents_pool = build_opponent_pool(opponents, baseline, league)

    best_params = BotParams()
    best_score = float("-inf")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "train_log.csv"
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow([
                "iter",
                "best_score",
                "mean_elite",
                "params",
            ])

        for iteration in range(1, iters + 1):
            start = time.perf_counter()
            candidates = [_sample_params(rng, means, stds) for _ in range(population)]

            if workers > 1:
                ctx = get_context("spawn")
                with ctx.Pool(processes=workers) as pool:
                    scores = pool.map(
                        _eval_worker,
                        [
                            (
                                candidate,
                                eval_seeds,
                                num_players,
                                max_steps,
                                list(opponents_pool),
                                cand_seats,
                                seed,
                            )
                            for candidate in candidates
                        ],
                    )
            else:
                scores = [
                    evaluate_candidate(
                        candidate,
                        eval_seeds,
                        num_players,
                        max_steps,
                        opponents_pool,
                        cand_seats,
                        seed,
                    )
                    for candidate in candidates
                ]

            ranked = list(zip(candidates, scores))
            ranked.sort(key=lambda item: item[1], reverse=True)
            elites = ranked[: max(1, elite)]
            elite_scores = [score for _, score in elites]
            elite_vectors = [params_to_vector(params) for params, _ in elites]

            means = [mean(values) for values in zip(*elite_vectors, strict=False)]
            stds = []
            for idx, spec in enumerate(PARAM_SPECS):
                values = [vec[idx] for vec in elite_vectors]
                std_value = pstdev(values) if len(values) > 1 else spec.init_std
                stds.append(max(min_stds[idx], std_value))

            if elites[0][1] > best_score:
                best_params = elites[0][0]
                best_score = elites[0][1]

            elapsed = time.perf_counter() - start
            elite_mean = sum(elite_scores) / len(elite_scores)
            writer.writerow([iteration, best_score, elite_mean, json.dumps(best_params.to_dict())])
            print(
                f"iter {iteration:02d} | best {best_score:.4f} | mean(top) {elite_mean:.4f} | "
                f"{elapsed:.2f}s"
            )

            if checkpoint_every > 0 and iteration % checkpoint_every == 0:
                save_params(best_params, checkpoint_dir / "best_params.json")
                _save_mean_std(checkpoint_dir / "mean_std.json", means, stds)

    save_params(best_params, checkpoint_dir / "best_params.json")
    _save_mean_std(checkpoint_dir / "mean_std.json", means, stds)
    save_params(best_params, out_path)
    return best_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CEM-тренировка параметров бота")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite", type=int, default=12)
    parser.add_argument("--games-per-cand", type=int, default=20)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--opponents", type=str, choices=["baseline", "league", "mixed"], default="mixed")
    parser.add_argument("--league-dir", type=Path, default=Path("monopoly/data/league"))
    parser.add_argument("--baseline", type=Path, default=Path("monopoly/data/params_baseline.json"))
    parser.add_argument("--cand-seats", type=str, choices=["all", "rotate"], default="rotate")
    parser.add_argument("--out", type=Path, default=Path("trained_params.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.elite > args.population:
        raise ValueError("elite должен быть <= population")
    if not 2 <= args.players <= 6:
        raise ValueError("players должен быть в диапазоне 2..6")

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = Path("runs") / timestamp

    params = cem_train(
        iters=args.iters,
        population=args.population,
        elite=args.elite,
        games_per_candidate=args.games_per_cand,
        num_players=args.players,
        seed=args.seed,
        max_steps=args.max_steps,
        opponents=args.opponents,
        baseline_path=args.baseline,
        league_dir=args.league_dir,
        cand_seats=args.cand_seats,
        out_path=args.out,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        workers=args.workers,
    )
    print(f"Лучшие параметры сохранены в {args.out}")
    print(params.to_dict())


if __name__ == "__main__":
    main()

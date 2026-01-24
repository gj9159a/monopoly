from __future__ import annotations

import argparse
import json
import csv
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from .engine import create_engine
from .models import GameState
from .params import BotParams, PARAM_SPECS, params_to_vector, save_params, vector_to_params


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


def _score_game(state: GameState, first_bankrupt_id: int | None) -> float:
    scores: list[float] = []
    for player in state.players:
        score = 0.0
        if state.winner_id == player.player_id:
            score += 1.0
        score += 0.0001 * _net_worth(state, player.player_id)
        if first_bankrupt_id == player.player_id:
            score -= 0.2
        scores.append(score)
    return sum(scores) / len(scores)


def play_game(params: BotParams, num_players: int, seed: int, max_steps: int) -> tuple[GameState, int | None, int]:
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


def evaluate_params(
    params: BotParams,
    seeds: Iterable[int],
    num_players: int,
    max_steps: int,
) -> float:
    scores = []
    for seed in seeds:
        state, first_bankrupt_id, _ = play_game(params, num_players, seed, max_steps)
        scores.append(_score_game(state, first_bankrupt_id))
    return sum(scores) / len(scores)


def _sample_params(rng: random.Random, means: list[float], stds: list[float]) -> BotParams:
    values: list[float] = []
    for mean_value, std_value in zip(means, stds, strict=False):
        values.append(rng.gauss(mean_value, std_value))
    return vector_to_params(values)


def cem_train(
    iters: int,
    population: int,
    elite: int,
    games_per_candidate: int,
    num_players: int,
    seed: int,
    max_steps: int,
    out_path: Path,
    log_path: Path,
) -> BotParams:
    rng = random.Random(seed)
    means = params_to_vector(BotParams())
    stds = [spec.init_std for spec in PARAM_SPECS]
    min_stds = [(spec.max_value - spec.min_value) * 0.02 for spec in PARAM_SPECS]

    eval_seeds = [seed + idx for idx in range(games_per_candidate)]

    best_params = BotParams()
    best_score = float("-inf")

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
            candidates: list[tuple[BotParams, float]] = []
            for _ in range(population):
                params = _sample_params(rng, means, stds)
                score = evaluate_params(params, eval_seeds, num_players, max_steps)
                candidates.append((params, score))

            candidates.sort(key=lambda item: item[1], reverse=True)
            elites = candidates[: max(1, elite)]
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
            summary = {
                "iter": iteration,
                "best_score": round(best_score, 4),
                "mean_elite": round(elite_mean, 4),
                "params": best_params.to_dict(),
            }
            writer.writerow([iteration, best_score, elite_mean, json_dump(best_params.to_dict())])
            print(
                f"iter {iteration:02d} | best {best_score:.4f} | mean(top) {elite_mean:.4f} | "
                f"{elapsed:.2f}s"
            )
            print(f"params: {summary['params']}")

    save_params(best_params, out_path)
    return best_params


def json_dump(data: dict[str, float | int]) -> str:
    return json.dumps(data, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CEM-тренировка параметров бота")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite", type=int, default=12)
    parser.add_argument("--games-per-cand", type=int, default=20)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("trained_params.json"))
    parser.add_argument("--log", type=Path, default=Path("train_log.csv"))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.elite > args.population:
        raise ValueError("elite должен быть <= population")
    if not 2 <= args.players <= 6:
        raise ValueError("players должен быть в диапазоне 2..6")

    params = cem_train(
        iters=args.iters,
        population=args.population,
        elite=args.elite,
        games_per_candidate=args.games_per_cand,
        num_players=args.players,
        seed=args.seed,
        max_steps=args.max_steps,
        out_path=args.out,
        log_path=args.log,
    )
    print(f"Лучшие параметры сохранены в {args.out}")
    print(params.to_dict())


if __name__ == "__main__":
    main()

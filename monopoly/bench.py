from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mean

from .params import BotParams, ThinkingConfig, load_params
from .train import (
    build_eval_cases,
    build_opponent_pool,
    load_league,
    play_game,
    score_player,
    _net_worth,
)


def _win_rate_ci(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    p = wins / games
    variance = p * (1 - p) / games
    margin = z * (variance ** 0.5)
    return max(0.0, p - margin), min(1.0, p + margin)


def _mean_ci(sum_value: float, sum_sq: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 1:
        return sum_value, sum_value
    mean_value = sum_value / n
    variance = max(0.0, (sum_sq - n * mean_value * mean_value) / (n - 1))
    margin = z * (variance ** 0.5) / (n ** 0.5)
    return mean_value - margin, mean_value + margin


def load_seed_pack(seeds_file: Path | None, seed: int, games: int) -> list[int]:
    if seeds_file is not None:
        if not seeds_file.exists():
            raise FileNotFoundError(f"Файл с сид-списком не найден: {seeds_file}")
        seeds: list[int] = []
        for line in seeds_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(int(line))
        if not seeds:
            raise ValueError("Пустой список сидов")
        return seeds[:games] if games > 0 else seeds
    return [seed + idx for idx in range(games)]


def bench(
    candidate: BotParams,
    baseline: BotParams,
    league_dir: Path,
    opponents: str,
    num_players: int,
    games: int,
    seed: int,
    max_steps: int,
    cand_seats: str,
    min_games: int,
    delta: float,
    seeds_file: Path | None = None,
    opponents_pool: list[BotParams] | None = None,
) -> dict[str, float | str]:
    if opponents_pool is None:
        league = load_league(league_dir)
        pool = build_opponent_pool(opponents, baseline, league)
    else:
        pool = list(opponents_pool)
    seeds = load_seed_pack(seeds_file, seed, games)
    cases = build_eval_cases(seeds, num_players, cand_seats, seed)

    wins = 0
    scores: list[float] = []
    net_sum = 0.0
    net_sum_sq = 0.0
    steps_list: list[int] = []
    stop_reason = "full"

    for idx, (game_seed, seat) in enumerate(cases):
        rng_seed = seed + game_seed * 1013 + seat * 917 + idx * 37
        rng = __import__("random").Random(rng_seed)
        opponents_params = [pool[rng.randrange(len(pool))] for _ in range(num_players - 1)]
        params_by_seat = []
        opp_iter = iter(opponents_params)
        for seat_idx in range(num_players):
            if seat_idx == seat:
                params_by_seat.append(candidate)
            else:
                params_by_seat.append(next(opp_iter))
        state, first_bankrupt_id, steps = play_game(params_by_seat, num_players, game_seed, max_steps)
        if state.winner_id == seat:
            wins += 1
        scores.append(score_player(state, seat, first_bankrupt_id))
        net_worth = _net_worth(state, seat)
        net_sum += net_worth
        net_sum_sq += net_worth * net_worth
        steps_list.append(steps)
        games_played = len(scores)
        min_required = min(min_games, len(cases))
        if games_played >= min_required:
            ci_low, ci_high = _win_rate_ci(wins, games_played)
            if ci_low >= 0.5 + delta:
                stop_reason = "above"
                break
            if ci_high <= 0.5:
                stop_reason = "below"
                break

    games_played = len(scores)
    win_rate = wins / max(1, games_played)
    ci_low, ci_high = _win_rate_ci(wins, games_played)
    net_ci_low, net_ci_high = _mean_ci(net_sum, net_sum_sq, games_played)
    return {
        "games": float(games_played),
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "avg_score": mean(scores) if scores else 0.0,
        "avg_net_worth": net_sum / games_played if games_played else 0.0,
        "net_worth_ci_low": net_ci_low,
        "net_worth_ci_high": net_ci_high,
        "avg_steps": mean(steps_list) if steps_list else 0.0,
        "stop_reason": stop_reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Бенчмарк кандидата против baseline/league")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--min-games", type=int, default=50)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seeds-file", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("monopoly/data/params_baseline.json"))
    parser.add_argument("--league-dir", type=Path, default=Path("monopoly/data/league"))
    parser.add_argument("--opponents", type=str, choices=["baseline", "league", "mixed"], default="mixed")
    parser.add_argument("--cand-seats", type=str, choices=["all", "rotate"], default="rotate")
    parser.add_argument("--thinking", action="store_true", help="Включить thinking-mode для кандидата")
    parser.add_argument("--thinking-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--thinking-horizon", type=int, default=30)
    parser.add_argument("--thinking-rollouts", type=int, default=12)
    parser.add_argument("--thinking-time-ms", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 2 <= args.players <= 6:
        raise ValueError("players должен быть в диапазоне 2..6")

    candidate = load_params(args.candidate)
    baseline = load_params(args.baseline)
    if args.thinking:
        candidate = candidate.with_thinking(
            ThinkingConfig(
                enabled=True,
                horizon_turns=int(args.thinking_horizon),
                rollouts_per_action=int(args.thinking_rollouts),
                time_budget_ms=int(args.thinking_time_ms),
                workers=int(args.thinking_workers),
            )
        )
    else:
        candidate = candidate.with_thinking(ThinkingConfig())
    baseline = baseline.with_thinking(ThinkingConfig())

    result = bench(
        candidate=candidate,
        baseline=baseline,
        league_dir=args.league_dir,
        opponents=args.opponents,
        num_players=args.players,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        cand_seats=args.cand_seats,
        min_games=args.min_games,
        delta=args.delta,
        seeds_file=args.seeds_file,
    )

    games = int(result["games"])
    print(f"Games: {games}")
    print(f"Win rate: {result['win_rate']:.3f}")
    print(f"95% CI: [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
    print(f"Avg score: {result['avg_score']:.4f}")
    print(f"Avg net worth: {result['avg_net_worth']:.1f}")
    print(f"Net worth CI: [{result['net_worth_ci_low']:.1f}, {result['net_worth_ci_high']:.1f}]")
    print(f"Avg steps: {result['avg_steps']:.1f}")
    if result["stop_reason"] != "full":
        print(f"Early stop: {result['stop_reason']}")


if __name__ == "__main__":
    main()

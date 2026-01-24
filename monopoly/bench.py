from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from .params import BotParams, load_params
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
) -> dict[str, float | str]:
    league = load_league(league_dir)
    pool = build_opponent_pool(opponents, baseline, league)
    seeds = [seed + idx for idx in range(games)]
    cases = build_eval_cases(seeds, num_players, cand_seats, seed)

    wins = 0
    scores: list[float] = []
    net_worths: list[int] = []
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
        net_worths.append(_net_worth(state, seat))
        steps_list.append(steps)
        games_played = len(scores)
        if games_played >= min_games:
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
    return {
        "games": float(games_played),
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "avg_score": mean(scores) if scores else 0.0,
        "avg_net_worth": mean(net_worths) if net_worths else 0.0,
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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("monopoly/data/params_baseline.json"))
    parser.add_argument("--league-dir", type=Path, default=Path("monopoly/data/league"))
    parser.add_argument("--opponents", type=str, choices=["baseline", "league", "mixed"], default="mixed")
    parser.add_argument("--cand-seats", type=str, choices=["all", "rotate"], default="rotate")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 2 <= args.players <= 6:
        raise ValueError("players должен быть в диапазоне 2..6")

    candidate = load_params(args.candidate)
    baseline = load_params(args.baseline)

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
    )

    games = int(result["games"])
    print(f"Games: {games}")
    print(f"Win rate: {result['win_rate']:.3f}")
    print(f"95% CI: [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
    print(f"Avg score: {result['avg_score']:.4f}")
    print(f"Avg net worth: {result['avg_net_worth']:.1f}")
    print(f"Avg steps: {result['avg_steps']:.1f}")
    if result["stop_reason"] != "full":
        print(f"Early stop: {result['stop_reason']}")


if __name__ == "__main__":
    main()

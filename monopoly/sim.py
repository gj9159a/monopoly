from __future__ import annotations

import argparse
from pathlib import Path

from .params import BotParams, load_params
from .train import play_game


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Запуск симуляции с параметрами бота")
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 2 <= args.players <= 6:
        raise ValueError("players должен быть в диапазоне 2..6")
    params = BotParams()
    if args.params:
        params = load_params(args.params)

    winners: dict[int | None, int] = {}
    steps_total = 0
    for idx in range(args.games):
        state, _, steps = play_game(params, args.players, args.seed + idx, args.max_steps)
        winners[state.winner_id] = winners.get(state.winner_id, 0) + 1
        steps_total += steps

    avg_steps = steps_total / max(1, args.games)
    print(f"Игры: {args.games}, средние шаги: {avg_steps:.1f}")
    for winner_id, count in sorted(winners.items(), key=lambda item: item[0] or -1):
        label = f"P{winner_id + 1}" if winner_id is not None else "None"
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()

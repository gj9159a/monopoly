from __future__ import annotations

import argparse
from pathlib import Path

from .bench import bench, load_seed_pack
from .league import load_index
from .params import BotParams, load_params


def _resolve_entry_path(entry: dict[str, object], league_dir: Path) -> Path:
    raw = Path(str(entry.get("path", "")))
    if raw.is_absolute():
        return raw
    candidate = Path.cwd() / raw
    if candidate.exists():
        return candidate
    return (league_dir / raw).resolve()


def _load_league_candidates(league_dir: Path, keep: int) -> list[tuple[str, Path]]:
    entries = load_index(league_dir)
    if entries:
        selected = entries[-keep:] if keep > 0 else entries
        return [(entry["name"], _resolve_entry_path(entry, league_dir)) for entry in selected]
    # fallback: scan directory if index отсутствует
    files = sorted([path for path in league_dir.glob("*.json") if path.name != "index.json"])
    selected = files[-keep:] if keep > 0 else files
    return [(path.stem, path) for path in selected]


def _load_params_safe(path: Path) -> BotParams:
    return load_params(path)


def run_progress(
    league_dir: Path,
    baseline_path: Path,
    games: int,
    seed: int,
    max_steps: int,
    players: int,
    cand_seats: str,
    keep: int,
    seeds_file: Path | None,
    min_games: int,
    delta: float,
) -> None:
    baseline = load_params(baseline_path)
    candidates = _load_league_candidates(league_dir, keep)
    if not candidates:
        print("Лига пуста, нечего оценивать")
        return

    league_params = []
    for _, path in candidates:
        league_params.append(_load_params_safe(path))

    print("name | wr_baseline (CI) | wr_mix (CI) | net_worth (CI)")
    for idx, (name, path) in enumerate(candidates):
        candidate = load_params(path)
        others = [baseline]
        for j, params in enumerate(league_params):
            if j == idx:
                continue
            others.append(params)

        result_baseline = bench(
            candidate=candidate,
            baseline=baseline,
            league_dir=league_dir,
            opponents="baseline",
            num_players=players,
            games=games,
            seed=seed,
            max_steps=max_steps,
            cand_seats=cand_seats,
            min_games=min_games,
            delta=delta,
            seeds_file=seeds_file,
            opponents_pool=[baseline],
        )

        result_mix = bench(
            candidate=candidate,
            baseline=baseline,
            league_dir=league_dir,
            opponents="mixed",
            num_players=players,
            games=games,
            seed=seed,
            max_steps=max_steps,
            cand_seats=cand_seats,
            min_games=min_games,
            delta=delta,
            seeds_file=seeds_file,
            opponents_pool=others,
        )

        print(
            f"{name} | "
            f"{result_baseline['win_rate']:.3f} [{result_baseline['ci_low']:.3f},{result_baseline['ci_high']:.3f}] | "
            f"{result_mix['win_rate']:.3f} [{result_mix['ci_low']:.3f},{result_mix['ci_high']:.3f}] | "
            f"{result_mix['avg_net_worth']:.1f} [{result_mix['net_worth_ci_low']:.1f},{result_mix['net_worth_ci_high']:.1f}]"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Оценка прогресса лиги")
    parser.add_argument("--league-dir", type=Path, default=Path("monopoly/data/league"))
    parser.add_argument("--baseline", type=Path, default=Path("monopoly/data/params_baseline.json"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--cand-seats", type=str, choices=["all", "rotate"], default="rotate")
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--min-games", type=int, default=50)
    parser.add_argument("--delta", type=float, default=0.05)
    default_seeds = Path("monopoly/data/seeds.txt")
    parser.add_argument("--seeds-file", type=Path, default=default_seeds if default_seeds.exists() else None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_progress(
        league_dir=args.league_dir,
        baseline_path=args.baseline,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        players=args.players,
        cand_seats=args.cand_seats,
        keep=args.keep,
        seeds_file=args.seeds_file,
        min_games=args.min_games,
        delta=args.delta,
    )


if __name__ == "__main__":
    main()

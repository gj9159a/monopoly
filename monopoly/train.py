from __future__ import annotations

import argparse
import csv
import hashlib
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
from .league import add_to_league, load_index, resolve_entry_path
from .params import (
    BotParams,
    PARAM_SPECS,
    ThinkingConfig,
    load_params,
    params_to_vector,
    save_params,
    vector_to_params,
)


@dataclass(frozen=True)
class EvalResult:
    fitness: float
    win_rate: float
    avg_net_worth: float


LAST_TRAIN_THINKING_USED = False


def _disable_thinking(params: BotParams) -> BotParams:
    if not params.thinking.enabled and params.thinking == ThinkingConfig():
        return params
    return params.with_thinking(ThinkingConfig())


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


def _hash_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_params(params: BotParams) -> str:
    payload = json.dumps(params.to_dict(), sort_keys=True)
    return _hash_text(payload)


def _hash_params_list(params_list: Sequence[BotParams]) -> str:
    payload = "|".join(_hash_params(params) for params in params_list)
    return _hash_text(payload)


def _hash_seeds(seeds: Sequence[int]) -> str:
    payload = ",".join(str(seed) for seed in seeds)
    return _hash_text(payload)


def _make_cache_key(
    candidate: BotParams,
    eval_config: dict[str, int | str],
    seeds_hash: str,
    opponents_pool_hash: str,
) -> str:
    payload = {
        "candidate": _hash_params(candidate),
        "eval": eval_config,
        "seeds": seeds_hash,
        "opponents": opponents_pool_hash,
    }
    return _hash_text(json.dumps(payload, sort_keys=True))


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Некорректное значение bool: {value}")


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
    index = load_index(league_dir)
    items = index.get("items", [])
    top_k = int(index.get("top_k", 16) or 16)
    if top_k > 0:
        items = items[:top_k]
    params_list: list[BotParams] = []
    for entry in items:
        path = resolve_entry_path(entry, league_dir)
        if not path.exists():
            continue
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


def _build_params_by_seat(
    candidate: BotParams,
    opponents_pool: Sequence[BotParams],
    num_players: int,
    game_seed: int,
    seat: int,
    case_index: int,
    seed: int,
) -> list[BotParams]:
    rng = random.Random(seed + game_seed * 1013 + seat * 917 + case_index * 37)
    opponents = _select_opponents(rng, opponents_pool, num_players - 1)
    params_by_seat: list[BotParams] = []
    opp_iter = iter(opponents)
    for seat_idx in range(num_players):
        if seat_idx == seat:
            params_by_seat.append(candidate)
        else:
            params_by_seat.append(next(opp_iter))
    return params_by_seat


def _eval_case(
    task: tuple[
        int,
        BotParams,
        int,
        int,
        int,
        int,
        int,
        list[BotParams],
        int,
    ],
) -> tuple[int, float, int, int]:
    (
        cand_index,
        candidate,
        game_seed,
        seat,
        case_index,
        num_players,
        max_steps,
        opponents_pool,
        seed,
    ) = task
    params_by_seat = _build_params_by_seat(
        candidate,
        opponents_pool,
        num_players,
        game_seed,
        seat,
        case_index,
        seed,
    )
    state, first_bankrupt_id, _ = play_game(params_by_seat, num_players, game_seed, max_steps)
    score = score_player(state, seat, first_bankrupt_id)
    win = 1 if state.winner_id == seat else 0
    net_worth = _net_worth(state, seat)
    return cand_index, score, win, net_worth


def evaluate_candidates(
    candidates: Sequence[BotParams],
    seeds: Sequence[int],
    num_players: int,
    max_steps: int,
    opponents_pool: Sequence[BotParams],
    cand_seats: str,
    seed: int,
    workers: int,
    cache: dict[str, EvalResult],
    cache_path: Path | None = None,
) -> tuple[list[EvalResult], int]:
    sanitized_candidates = [_disable_thinking(params) for params in candidates]
    sanitized_opponents = [_disable_thinking(params) for params in opponents_pool]
    global LAST_TRAIN_THINKING_USED
    LAST_TRAIN_THINKING_USED = any(params.thinking.enabled for params in sanitized_candidates) or any(
        params.thinking.enabled for params in sanitized_opponents
    )

    cases = build_eval_cases(seeds, num_players, cand_seats, seed)
    eval_config = {
        "players": num_players,
        "max_steps": max_steps,
        "cand_seats": cand_seats,
        "seed": seed,
    }
    seeds_hash = _hash_seeds(seeds)
    opponents_hash = _hash_params_list(sanitized_opponents)

    results: list[EvalResult | None] = [None] * len(candidates)
    cache_hits = 0
    tasks: list[tuple[
        int,
        BotParams,
        int,
        int,
        int,
        int,
        int,
        list[BotParams],
        int,
    ]] = []
    keys: dict[int, str] = {}

    for cand_index, candidate in enumerate(sanitized_candidates):
        key = _make_cache_key(candidate, eval_config, seeds_hash, opponents_hash)
        cached = cache.get(key)
        if cached is not None:
            results[cand_index] = cached
            cache_hits += 1
            continue
        keys[cand_index] = key
        for case_index, (game_seed, seat) in enumerate(cases):
            tasks.append(
                (
                    cand_index,
                    candidate,
                    game_seed,
                    seat,
                    case_index,
                    num_players,
                    max_steps,
                    list(sanitized_opponents),
                    seed,
                )
            )

    if tasks:
        if workers > 1:
            ctx = get_context("spawn")
            try:
                with ctx.Pool(processes=workers) as pool:
                    raw_results = pool.map(_eval_case, tasks)
            except (OSError, PermissionError):
                raw_results = [_eval_case(task) for task in tasks]
        else:
            raw_results = [_eval_case(task) for task in tasks]

        sums: dict[int, dict[str, float]] = {
            cand_index: {"score": 0.0, "wins": 0.0, "net": 0.0, "count": 0.0}
            for cand_index in keys
        }
        for cand_index, score, win, net_worth in raw_results:
            sums[cand_index]["score"] += score
            sums[cand_index]["wins"] += win
            sums[cand_index]["net"] += net_worth
            sums[cand_index]["count"] += 1

        for cand_index, key in keys.items():
            count = int(sums[cand_index]["count"])
            if count == 0:
                raise RuntimeError("Пустая оценка кандидата")
            fitness = sums[cand_index]["score"] / count
            win_rate = sums[cand_index]["wins"] / count
            avg_net = sums[cand_index]["net"] / count
            result = EvalResult(fitness=fitness, win_rate=win_rate, avg_net_worth=avg_net)
            results[cand_index] = result
            cache[key] = result
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a", encoding="utf-8") as handle:
                    payload = {
                        "key": key,
                        "fitness": result.fitness,
                        "win_rate": result.win_rate,
                        "avg_net_worth": result.avg_net_worth,
                    }
                    handle.write(json.dumps(payload, sort_keys=True))
                    handle.write("\n")

    if any(result is None for result in results):
        raise RuntimeError("Не удалось получить оценку для всех кандидатов")
    return [result for result in results], cache_hits


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
    cache: dict[str, EvalResult] = {}
    cache_path = checkpoint_dir / "eval_cache.jsonl"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "train_log.csv"
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow([
                "iter",
                "best_fitness",
                "mean_elite",
                "std_elite",
                "cache_hits",
                "eval_seconds",
            ])

        for iteration in range(1, iters + 1):
            start = time.perf_counter()
            candidates = [_sample_params(rng, means, stds) for _ in range(population)]
            eval_results, cache_hits = evaluate_candidates(
                candidates=candidates,
                seeds=eval_seeds,
                num_players=num_players,
                max_steps=max_steps,
                opponents_pool=opponents_pool,
                cand_seats=cand_seats,
                seed=seed,
                workers=workers,
                cache=cache,
                cache_path=cache_path,
            )
            scores = [result.fitness for result in eval_results]

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
            elite_std = pstdev(elite_scores) if len(elite_scores) > 1 else 0.0
            writer.writerow([
                iteration,
                best_score,
                elite_mean,
                elite_std,
                cache_hits,
                f"{elapsed:.4f}",
            ])
            print(
                f"iter {iteration:02d} | best {best_score:.4f} | mean(top) {elite_mean:.4f} | "
                f"std(top) {elite_std:.4f} | cache {cache_hits} | {elapsed:.2f}s"
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
    parser.add_argument("--league-auto-add", type=_parse_bool, default=False)
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
    baseline = load_params(args.baseline)
    league = load_league(args.league_dir)
    opponents_pool = build_opponent_pool(args.opponents, baseline, league)
    eval_results, _ = evaluate_candidates(
        candidates=[params],
        seeds=[args.seed + idx for idx in range(args.games_per_cand)],
        num_players=args.players,
        max_steps=args.max_steps,
        opponents_pool=opponents_pool,
        cand_seats=args.cand_seats,
        seed=args.seed,
        workers=1,
        cache={},
        cache_path=None,
    )
    best_fitness = eval_results[0].fitness
    print(f"Best fitness: {best_fitness:.4f}")
    quick_cases = build_eval_cases(
        seeds=[args.seed + idx for idx in range(20)],
        num_players=args.players,
        cand_seats=args.cand_seats,
        seed=args.seed,
    )
    wins = 0
    scores: list[float] = []
    net_worths: list[int] = []
    for idx, (game_seed, seat) in enumerate(quick_cases):
        params_by_seat = _build_params_by_seat(
            params,
            [baseline],
            args.players,
            game_seed,
            seat,
            idx,
            args.seed,
        )
        state, first_bankrupt_id, _ = play_game(params_by_seat, args.players, game_seed, args.max_steps)
        if state.winner_id == seat:
            wins += 1
        scores.append(score_player(state, seat, first_bankrupt_id))
        net_worths.append(_net_worth(state, seat))
    win_rate = wins / max(1, len(quick_cases))
    avg_net = mean(net_worths) if net_worths else 0.0
    print(f"Quick bench vs baseline (20 игр): win_rate={win_rate:.3f}, avg_net_worth={avg_net:.1f}")

    if args.league_auto_add and args.opponents in {"mixed", "league"}:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        meta = (
            f"iter={args.iters}; best_fitness={best_fitness:.4f}; "
            f"baseline_bench=win_rate:{win_rate:.3f},net:{avg_net:.1f}"
        )
        meta_payload = {"name": f"best_{timestamp}", "note": meta}
        added, changed_topk, rank = add_to_league(
            params=params,
            fitness=best_fitness,
            meta=meta_payload,
            league_dir=args.league_dir,
            top_k=16,
        )
        print(f"League auto-add: added={added}, rank={rank}, changed_topk={changed_topk}")


if __name__ == "__main__":
    main()

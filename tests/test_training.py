from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from monopoly.engine import create_engine
from monopoly.features import jail_exit_heat_group, landing_prob_group
from monopoly.league import add_to_league
from monopoly.params import BotParams, ThinkingConfig, decide_build_actions, save_params
from monopoly.train import (
    FITNESS_COEFFS,
    FITNESS_CONFIDENCE,
    LAST_TRAIN_THINKING_USED,
    build_eval_cases,
    build_opponent_pool,
    evaluate_candidates,
    cem_train,
    fitness_from_components,
    load_league,
    place_to_score,
    placements_by_net_worth,
    win_like_outcome,
    wilson_interval,
)


def _cleanup_tmp(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _local_tmp() -> Path:
    base = Path(__file__).resolve().parent / "_tmp"
    base.mkdir(exist_ok=True)
    path = base / uuid4().hex
    path.mkdir()
    return path


def _write_params(path: Path, params: BotParams) -> None:
    save_params(params, path)


def test_params_roundtrip() -> None:
    params = BotParams.from_dict(
        {
            "cash_buffer_base": 123,
            "cash_buffer_per_house": 7,
            "max_bid_fraction": 0.75,
            "weights": {
                "auction": {"early": {"base_value": 1.4}},
                "jail": {"late": {"action_pay": 0.5}},
            },
        }
    )
    tmp_path = _local_tmp()
    path = tmp_path / "params.json"
    params.to_json(path)
    loaded = BotParams.from_json(path)
    assert loaded == params
    _cleanup_tmp(tmp_path)


def test_eval_determinism() -> None:
    tmp_path = _local_tmp()
    baseline = BotParams()
    league_dir = tmp_path / "league"
    league_dir.mkdir()

    _write_params(tmp_path / "baseline.json", baseline)
    add_to_league(BotParams(cash_buffer_base=151), 0.1, {"name": "l1"}, league_dir)
    add_to_league(BotParams(cash_buffer_base=152), 0.2, {"name": "l2"}, league_dir)

    opponents_pool = build_opponent_pool("mixed", baseline, load_league(league_dir))
    seeds = [1, 2, 3]

    results_a, _ = evaluate_candidates(
        candidates=[baseline],
        seeds=seeds,
        num_players=2,
        max_steps=30,
        opponents_pool=opponents_pool,
        cand_seats="rotate",
        seed=7,
        workers=1,
        cache={},
    )
    results_b, _ = evaluate_candidates(
        candidates=[baseline],
        seeds=seeds,
        num_players=2,
        max_steps=30,
        opponents_pool=opponents_pool,
        cand_seats="rotate",
        seed=7,
        workers=1,
        cache={},
    )
    assert results_a[0].fitness == results_b[0].fitness
    _cleanup_tmp(tmp_path)


def test_place_to_score_mapping() -> None:
    assert place_to_score(1) == 1.0
    assert place_to_score(2) == 0.5
    assert place_to_score(3) == 0.1
    assert place_to_score(4) == 0.0
    assert place_to_score(5) == 0.0
    assert place_to_score(6) == 0.0


def test_cutoff_outcome_uses_net_worth_ranking() -> None:
    engine = create_engine(num_players=4, seed=1)
    state = engine.state
    for idx, player in enumerate(state.players):
        player.money = 1000 - idx * 100
    placements = placements_by_net_worth(state)
    assert placements[0] == 1
    assert placements[1] == 2
    top_outcome = win_like_outcome(placements[0], winner=False, ended_by_cutoff=True)
    second_outcome = win_like_outcome(placements[1], winner=False, ended_by_cutoff=True)
    assert top_outcome == 1.0
    assert second_outcome == 0.5


def test_wilson_lcb_monotonicity() -> None:
    low, _ = wilson_interval(1.0, 10, FITNESS_CONFIDENCE)
    high, _ = wilson_interval(2.0, 10, FITNESS_CONFIDENCE)
    assert high >= low


def test_fitness_priority_win() -> None:
    fitness_a = fitness_from_components(
        win_lcb=0.6,
        place_score=0.0,
        advantage=-0.5,
        cutoff_rate=0.0,
        avg_steps_norm=0.5,
        coefficients=FITNESS_COEFFS,
    )
    fitness_b = fitness_from_components(
        win_lcb=0.5,
        place_score=0.0,
        advantage=2.0,
        cutoff_rate=0.0,
        avg_steps_norm=0.1,
        coefficients=FITNESS_COEFFS,
    )
    assert fitness_a > fitness_b


def test_group_heat_favors_orange_red() -> None:
    state = create_engine(num_players=2, seed=1).state
    orange_heat = jail_exit_heat_group(state, "orange")
    red_heat = jail_exit_heat_group(state, "red")
    light_blue_heat = jail_exit_heat_group(state, "light_blue")
    brown_heat = jail_exit_heat_group(state, "brown")
    assert orange_heat > light_blue_heat
    assert orange_heat > brown_heat
    assert red_heat > light_blue_heat


def test_landing_prob_group_biases_orange() -> None:
    state = create_engine(num_players=2, seed=1).state
    orange_prob = landing_prob_group(state, "orange")
    light_blue_prob = landing_prob_group(state, "light_blue")
    assert orange_prob > light_blue_prob


def test_league_rotation() -> None:
    seeds = [10, 11, 12, 13, 14, 15]
    cases = build_eval_cases(seeds, num_players=4, cand_seats="rotate", seed=123)
    seats = {seat for _, seat in cases}
    assert seats == set(range(4))


def test_bot_constraints() -> None:
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    player.money = 0

    actions = decide_build_actions(engine.state, player, BotParams())
    assert actions == []

    cell = engine.state.board[1]
    decision = engine.bots[0].decide(
        engine.state,
        {
            "type": "auction_bid",
            "player_id": 0,
            "cell": cell,
            "current_price": 0,
            "min_increment": 1,
        },
    )
    if decision.get("action") == "bid":
        assert decision["bid"] >= 0
        assert decision["bid"] <= player.money


def test_train_cli_smoke() -> None:
    tmp_path = _local_tmp()
    baseline_path = tmp_path / "baseline.json"
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    _write_params(baseline_path, BotParams())
    add_to_league(BotParams(cash_buffer_base=151), 0.1, {"name": "l1"}, league_dir)

    out_path = tmp_path / "trained.json"
    checkpoint_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "monopoly.train",
            "--iters",
            "1",
            "--population",
            "2",
            "--elite",
            "1",
            "--games-per-cand",
            "1",
            "--players",
            "2",
            "--seed",
            "1",
            "--max-steps",
            "20",
            "--baseline",
            str(baseline_path),
            "--league-dir",
            str(league_dir),
            "--opponents",
            "mixed",
            "--cand-seats",
            "rotate",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-every",
            "1",
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out_path.exists()
    assert (checkpoint_dir / "best_params.json").exists()
    assert (checkpoint_dir / "mean_std.json").exists()
    assert (checkpoint_dir / "train_log.csv").exists()
    assert (checkpoint_dir / "eval_cache.jsonl").exists()
    _cleanup_tmp(tmp_path)


def test_cem_determinism() -> None:
    tmp_path = _local_tmp()
    baseline = BotParams()
    baseline_path = tmp_path / "baseline.json"
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    _write_params(baseline_path, baseline)
    add_to_league(BotParams(cash_buffer_base=151), 0.1, {"name": "l1"}, league_dir)

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    checkpoint_a = tmp_path / "run_a"
    checkpoint_b = tmp_path / "run_b"

    params_a = cem_train(
        iters=2,
        population=4,
        elite=2,
        games_per_candidate=2,
        num_players=2,
        seed=7,
        max_steps=40,
        opponents="mixed",
        baseline_path=baseline_path,
        league_dir=league_dir,
        cand_seats="rotate",
        out_path=out_a,
        checkpoint_dir=checkpoint_a,
        checkpoint_every=1,
        workers=1,
    )
    params_b = cem_train(
        iters=2,
        population=4,
        elite=2,
        games_per_candidate=2,
        num_players=2,
        seed=7,
        max_steps=40,
        opponents="mixed",
        baseline_path=baseline_path,
        league_dir=league_dir,
        cand_seats="rotate",
        out_path=out_b,
        checkpoint_dir=checkpoint_b,
        checkpoint_every=1,
        workers=1,
    )

    assert params_a.to_dict() == params_b.to_dict()
    _cleanup_tmp(tmp_path)


def test_eval_workers_consistency() -> None:
    baseline = BotParams()
    seeds = [1, 2]
    opponents_pool = [baseline]
    results_seq, _ = evaluate_candidates(
        candidates=[baseline],
        seeds=seeds,
        num_players=2,
        max_steps=20,
        opponents_pool=opponents_pool,
        cand_seats="rotate",
        seed=7,
        workers=1,
        cache={},
    )
    results_mp, _ = evaluate_candidates(
        candidates=[baseline],
        seeds=seeds,
        num_players=2,
        max_steps=20,
        opponents_pool=opponents_pool,
        cand_seats="rotate",
        seed=7,
        workers=2,
        cache={},
    )
    assert results_seq[0].fitness == results_mp[0].fitness


def test_baseline_league_files_loadable(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    _write_params(baseline_path, BotParams())
    add_to_league(BotParams(cash_buffer_base=151), 0.1, {"name": "l1"}, league_dir)
    league = load_league(league_dir)
    assert league
    for params in league:
        assert isinstance(params, BotParams)


def test_bench_smoke() -> None:
    tmp_path = _local_tmp()
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    league_dir = tmp_path / "league"
    league_dir.mkdir()

    _write_params(baseline_path, BotParams())
    _write_params(candidate_path, BotParams())
    add_to_league(BotParams(cash_buffer_base=151), 0.1, {"name": "l1"}, league_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "monopoly.bench",
            "--games",
            "2",
            "--players",
            "2",
            "--seed",
            "1",
            "--candidate",
            str(candidate_path),
            "--baseline",
            str(baseline_path),
            "--league-dir",
            str(league_dir),
            "--opponents",
            "mixed",
            "--cand-seats",
            "rotate",
            "--max-steps",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    _cleanup_tmp(tmp_path)


def test_train_disables_thinking() -> None:
    candidate = BotParams().with_thinking(ThinkingConfig(enabled=True, horizon_turns=5))
    opponents_pool = [BotParams().with_thinking(ThinkingConfig(enabled=True, horizon_turns=5))]
    results, _ = evaluate_candidates(
        candidates=[candidate],
        seeds=[1],
        num_players=2,
        max_steps=10,
        opponents_pool=opponents_pool,
        cand_seats="rotate",
        seed=3,
        workers=1,
        cache={},
    )
    assert results
    assert LAST_TRAIN_THINKING_USED is False

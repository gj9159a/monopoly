from __future__ import annotations

from pathlib import Path

from monopoly.autoevolve import build_league_rebench_protocol, eval_protocol_hash, run_autoevolve
from monopoly.league import add_to_league, load_index, save_index
from monopoly.params import BotParams, ThinkingConfig, save_params
from monopoly.io_utils import read_json


def _write_params(path: Path, params: BotParams) -> None:
    save_params(params, path)


def _run_autoevolve(
    runs_dir: Path,
    league_dir: Path,
    baseline_path: Path,
    **overrides: object,
) -> None:
    cfg = {
        "seed": 1,
        "runs_dir": runs_dir,
        "league_dir": league_dir,
        "baseline_path": baseline_path,
        "top_k_pool": 2,
        "league_cap": 16,
        "max_new_bests": 1,
        "meta_plateau_cycles": 5,
        "bootstrap_min_league_for_pool": 4,
        "league_rebench_on_mismatch": False,
        "league_rebench_games": None,
        "league_rebench_max_steps": None,
        "league_rebench_seed": None,
        "population": 2,
        "elite": 1,
        "games_per_cand": 1,
        "epoch_iters": 1,
        "plateau_epochs": 1,
        "eps_winrate": 1e9,
        "eps_fitness": 1e9,
        "min_progress_games": 1,
        "bench_max_games": 1,
        "delta": 0.0,
        "max_steps": 10,
        "workers": 1,
        "resume": False,
    }
    cfg.update(overrides)
    run_autoevolve(**cfg)


def test_autoevolve_bootstrap_adds_best(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    runs_dir = tmp_path / "runs"
    _run_autoevolve(runs_dir, league_dir, baseline_path, max_new_bests=1, bootstrap_min_league_for_pool=4)

    index = load_index(league_dir)
    assert index["items"]
    status = read_json(runs_dir / "status.json", default={})
    assert status.get("new_bests_count") == 1
    assert status.get("master_seed") == 1
    policy = status.get("derived_seed_policy")
    assert isinstance(policy, dict)
    assert "opponents_rng" in policy
    seeds_used = (runs_dir / "seeds_used.txt").read_text(encoding="utf-8")
    assert "cycle=001" in seeds_used


def test_autoevolve_meta_cycle_updates_index(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    add_to_league(BotParams(cash_buffer_base=150), 0.5, {"name": "seed"}, league_dir)

    runs_dir = tmp_path / "runs"
    _run_autoevolve(runs_dir, league_dir, baseline_path, max_new_bests=1, bootstrap_min_league_for_pool=1)

    index = load_index(league_dir)
    fitness_values = [entry["fitness"] for entry in index["items"]]
    assert fitness_values == sorted(fitness_values, reverse=True)
    assert [entry["rank"] for entry in index["items"]] == list(range(1, len(index["items"]) + 1))


def test_autoevolve_stop_max_new_bests(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    runs_dir = tmp_path / "runs"
    _run_autoevolve(runs_dir, league_dir, baseline_path, max_new_bests=1, meta_plateau_cycles=10)

    status = read_json(runs_dir / "status.json", default={})
    assert status.get("new_bests_count") == 1
    assert status.get("current_phase") == "finished"


def test_autoevolve_stop_meta_plateau(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    add_to_league(BotParams(cash_buffer_base=999), 1e9, {"name": "top"}, league_dir)

    runs_dir = tmp_path / "runs"
    _run_autoevolve(
        runs_dir,
        league_dir,
        baseline_path,
        top_k_pool=1,
        max_new_bests=5,
        meta_plateau_cycles=1,
        bootstrap_min_league_for_pool=1,
    )

    status = read_json(runs_dir / "status.json", default={})
    assert status.get("meta_plateau") >= 1
    assert status.get("new_bests_count") == 1


def test_bootstrap_league_grows_to_min_pool(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    runs_dir = tmp_path / "runs"
    _run_autoevolve(
        runs_dir,
        league_dir,
        baseline_path,
        max_new_bests=6,
        meta_plateau_cycles=10,
        bootstrap_min_league_for_pool=3,
        population=1,
        elite=1,
        games_per_cand=1,
        epoch_iters=1,
        plateau_epochs=1,
        min_progress_games=1,
        max_steps=10,
    )

    index = load_index(league_dir)
    assert len(index["items"]) >= 3


def test_bootstrap_add_allows_non_top1(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    add_to_league(BotParams(cash_buffer_base=999), 1e9, {"name": "top"}, league_dir)
    index_before = load_index(league_dir)
    top1_hash = index_before["items"][0]["hash"]

    runs_dir = tmp_path / "runs"
    _run_autoevolve(
        runs_dir,
        league_dir,
        baseline_path,
        max_new_bests=1,
        bootstrap_min_league_for_pool=4,
        population=1,
        elite=1,
        games_per_cand=1,
        epoch_iters=1,
        plateau_epochs=1,
        min_progress_games=1,
        max_steps=10,
    )

    index_after = load_index(league_dir)
    assert len(index_after["items"]) == 2
    assert index_after["items"][0]["hash"] == top1_hash


def test_rebench_runs_on_protocol_mismatch(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    add_to_league(BotParams(cash_buffer_base=151), 0.1, {"name": "l1"}, league_dir)
    add_to_league(BotParams(cash_buffer_base=152), 0.2, {"name": "l2"}, league_dir)

    index = load_index(league_dir)
    for entry in index["items"]:
        entry["eval_protocol_hash"] = "old"
    save_index(index, league_dir)

    runs_dir = tmp_path / "runs"
    _run_autoevolve(
        runs_dir,
        league_dir,
        baseline_path,
        max_new_bests=0,
        bootstrap_min_league_for_pool=1,
        league_rebench_on_mismatch=True,
        league_rebench_games=1,
        league_rebench_max_steps=10,
        league_rebench_seed=7,
    )

    status = read_json(runs_dir / "status.json", default={})
    assert status.get("league_rebench_needed") is True
    assert status.get("league_rebench_done") is True
    eval_hash = status.get("league_eval_protocol_hash")
    index_after = load_index(league_dir)
    assert all(entry.get("eval_protocol_hash") == eval_hash for entry in index_after["items"])
    fitness_values = [entry["fitness"] for entry in index_after["items"]]
    assert fitness_values == sorted(fitness_values, reverse=True)
    assert [entry["rank"] for entry in index_after["items"]] == list(range(1, len(index_after["items"]) + 1))
    assert (runs_dir / "rebench_log.csv").exists()


def test_rebench_skipped_when_hashes_match(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    add_to_league(BotParams(cash_buffer_base=151), 0.1, {"name": "l1"}, league_dir)
    protocol = build_league_rebench_protocol(
        players=4,
        games_per_cand=1,
        max_steps=10,
        league_cap=16,
        seed=1,
        cand_seats="rotate",
    )
    index = load_index(league_dir)
    for entry in index["items"]:
        entry["eval_protocol_hash"] = eval_protocol_hash(protocol)
    save_index(index, league_dir)

    runs_dir = tmp_path / "runs"
    _run_autoevolve(
        runs_dir,
        league_dir,
        baseline_path,
        max_new_bests=0,
        bootstrap_min_league_for_pool=1,
        league_rebench_on_mismatch=True,
        league_rebench_games=1,
        league_rebench_max_steps=10,
        league_rebench_seed=1,
    )

    status = read_json(runs_dir / "status.json", default={})
    assert status.get("league_rebench_needed") is False
    assert status.get("league_rebench_done") is False
    assert not (runs_dir / "rebench_log.csv").exists()


def test_rebench_runs_on_unknown_hash(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_params(baseline_path, BotParams())

    add_to_league(BotParams(cash_buffer_base=151), 0.1, {"name": "l1"}, league_dir)

    runs_dir = tmp_path / "runs"
    _run_autoevolve(
        runs_dir,
        league_dir,
        baseline_path,
        max_new_bests=0,
        bootstrap_min_league_for_pool=1,
        league_rebench_on_mismatch=True,
        league_rebench_games=1,
        league_rebench_max_steps=10,
        league_rebench_seed=1,
    )

    status = read_json(runs_dir / "status.json", default={})
    assert status.get("league_rebench_needed") is True
    assert status.get("league_rebench_done") is True


def test_eval_protocol_hash_changes_on_params() -> None:
    protocol = build_league_rebench_protocol(
        players=4,
        games_per_cand=1,
        max_steps=10,
        league_cap=16,
        seed=1,
        cand_seats="rotate",
    )
    base_hash = eval_protocol_hash(protocol)
    modified = dict(protocol)
    modified["fitness_confidence"] = 0.9
    assert eval_protocol_hash(modified) != base_hash
    modified_steps = dict(protocol)
    modified_steps["max_steps"] = 20
    assert eval_protocol_hash(modified_steps) != base_hash
    modified_coeffs = dict(protocol)
    modified_coeffs["fitness_coeffs"] = {
        **protocol.get("fitness_coeffs", {}),
        "advantage": 2.0,
    }
    assert eval_protocol_hash(modified_coeffs) != base_hash


def test_autoevolve_thinking_disabled(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    baseline = BotParams().with_thinking(ThinkingConfig(enabled=True, horizon_turns=5))
    _write_params(baseline_path, baseline)

    add_to_league(BotParams().with_thinking(ThinkingConfig(enabled=True, horizon_turns=5)), 0.1, {"name": "l1"}, league_dir)

    from monopoly import train as train_module

    train_module.LAST_TRAIN_THINKING_USED = True

    runs_dir = tmp_path / "runs"
    _run_autoevolve(
        runs_dir,
        league_dir,
        baseline_path,
        max_new_bests=1,
        bootstrap_min_league_for_pool=1,
    )

    assert train_module.LAST_TRAIN_THINKING_USED is False

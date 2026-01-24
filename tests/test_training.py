from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4
import shutil

from monopoly.engine import create_engine
from monopoly.params import BotParams, decide_build_actions
from monopoly.train import cem_train





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

def test_params_roundtrip() -> None:
    params = BotParams(
        cash_buffer_base=123,
        cash_buffer_per_house=7,
        auction_value_mult_street=1.2,
        auction_value_mult_rail=1.1,
        auction_value_mult_utility=0.8,
        monopoly_completion_bonus=0.4,
        monopoly_block_bonus=0.2,
        build_aggressiveness=1.1,
        un_mortgage_priority_mult=1.3,
        jail_exit_aggressiveness=0.2,
        risk_aversion=0.6,
        max_bid_fraction=0.75,
    )
    tmp_path = _local_tmp()
    path = tmp_path / "params.json"
    params.to_json(path)
    loaded = BotParams.from_json(path)
    assert loaded == params
    _cleanup_tmp(tmp_path)


def test_cem_determinism() -> None:
    tmp_path = _local_tmp()
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    log_a = tmp_path / "a.csv"
    log_b = tmp_path / "b.csv"

    params_a = cem_train(
        iters=2,
        population=6,
        elite=2,
        games_per_candidate=2,
        num_players=2,
        seed=7,
        max_steps=40,
        out_path=out_a,
        log_path=log_a,
    )
    params_b = cem_train(
        iters=2,
        population=6,
        elite=2,
        games_per_candidate=2,
        num_players=2,
        seed=7,
        max_steps=40,
        out_path=out_b,
        log_path=log_b,
    )

    assert params_a.to_dict() == params_b.to_dict()
    _cleanup_tmp(tmp_path)


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
    out_path = tmp_path / "trained.json"
    log_path = tmp_path / "train_log.csv"
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
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out_path.exists()
    _cleanup_tmp(tmp_path)

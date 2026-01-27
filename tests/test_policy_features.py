from __future__ import annotations

from monopoly.engine import create_engine
from monopoly.params import BotParams, STAGE_HYSTERESIS_TICKS, decide_auction_bid, game_stage


def test_policy_determinism() -> None:
    engine = create_engine(num_players=2, seed=7)
    state = engine.state
    player = state.players[0]
    cell = state.board[1]

    bid_a = decide_auction_bid(state, player, cell, current_price=0, min_increment=1, params=BotParams())
    bid_b = decide_auction_bid(state, player, cell, current_price=0, min_increment=1, params=BotParams())
    assert bid_a == bid_b


def test_stage_transitions() -> None:
    engine = create_engine(num_players=2, seed=1)
    state = engine.state
    assert game_stage(state) == "early"

    for cell in state.board:
        if cell.cell_type in {"property", "railroad", "utility"}:
            cell.owner_id = 0

    for _ in range(STAGE_HYSTERESIS_TICKS - 1):
        state.turn_index += 1
        assert game_stage(state) == "early"
    state.turn_index += 1
    assert game_stage(state) == "mid"

    state.players[1].bankrupt = True
    for _ in range(STAGE_HYSTERESIS_TICKS - 1):
        state.turn_index += 1
        assert game_stage(state) == "mid"
    state.turn_index += 1
    assert game_stage(state) == "late"


def test_params_flat_weights() -> None:
    params = BotParams.from_dict({"auction_early_base_value": 1.8})
    assert params.weights["auction"]["early"]["base_value"] == 1.8

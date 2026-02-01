from __future__ import annotations

from monopoly.engine import create_engine
from monopoly.features import (
    denial_value,
    hotel_scarcity,
    house_scarcity,
    positional_threat_self,
    positional_threat_self_turns,
    railroad_synergy,
    utility_synergy,
)
from monopoly.params import BotParams, STAGES, STAGE_HYSTERESIS_TICKS, decide_jail_exit, game_stage


def _clear_payments_in_range(state, start_pos: int) -> None:
    board_size = len(state.board)
    for roll in range(2, 13):
        cell = state.board[(start_pos + roll) % board_size]
        cell.owner_id = None
        cell.mortgaged = False
        cell.houses = 0
        cell.hotels = 0
        if cell.cell_type == "tax":
            cell.tax_amount = 0


def test_positional_threat_self_nonzero_and_mortgage_zero() -> None:
    engine = create_engine(num_players=2, seed=1)
    state = engine.state
    player_id = 0
    state.players[player_id].position = 0

    _clear_payments_in_range(state, 0)
    target = state.board[3]
    target.owner_id = 1
    target.houses = 1
    target.hotels = 0
    state.players[1].in_jail = False

    threat = positional_threat_self(state, player_id)
    assert threat > 0

    target.mortgaged = True
    threat_mortgaged = positional_threat_self(state, player_id)
    assert threat_mortgaged == 0


def test_denial_value_grows_when_bank_houses_low() -> None:
    engine = create_engine(num_players=2, seed=1)
    state = engine.state
    for cell in state.board:
        cell.houses = 0
        cell.hotels = 0

    low_denial = denial_value(1, 0, house_scarcity(state), hotel_scarcity(state))

    remaining = max(0, state.rules.bank_houses - 1)
    for cell in state.board:
        if cell.cell_type != "property":
            continue
        if remaining <= 0:
            break
        cell.houses = 1
        remaining -= 1

    high_denial = denial_value(1, 0, house_scarcity(state), hotel_scarcity(state))
    assert high_denial > low_denial


def test_synergy_features() -> None:
    engine = create_engine(num_players=2, seed=1)
    state = engine.state
    player_id = 0
    railroads = [cell for cell in state.board if cell.cell_type == "railroad"]
    railroads[0].owner_id = player_id
    railroads[1].owner_id = player_id
    assert railroad_synergy(state, player_id, railroads[2]) == 0.5

    utilities = [cell for cell in state.board if cell.cell_type == "utility"]
    utilities[0].owner_id = player_id
    assert utility_synergy(state, player_id, utilities[1]) == 0.5

    non_special = next(cell for cell in state.board if cell.cell_type == "property")
    assert railroad_synergy(state, player_id, non_special) == 0.0
    assert utility_synergy(state, player_id, non_special) == 0.0


def test_jail_hr2_signals() -> None:
    params = BotParams()
    for stage in STAGES:
        for feature in params.weights["jail"][stage]:
            params.weights["jail"][stage][feature] = 0.0
    params.weights["jail"]["early"]["lost_income_if_stay"] = -5.0
    params.weights["jail"]["early"]["saved_risk_if_stay"] = 5.0

    engine = create_engine(num_players=2, seed=1)
    state = engine.state
    player = state.players[0]
    opponent = state.players[1]
    player.money = state.rules.jail_fine + 100
    player.jail_turns = 1
    player.position = 0
    opponent.position = 0

    _clear_payments_in_range(state, player.position)
    income_cell = state.board[3]
    income_cell.owner_id = player.player_id
    income_cell.houses = 1
    income_cell.hotels = 0

    decision = decide_jail_exit(state, player, params)
    assert decision == "pay"

    engine2 = create_engine(num_players=2, seed=1)
    state2 = engine2.state
    player2 = state2.players[0]
    opponent2 = state2.players[1]
    player2.money = state2.rules.jail_fine + 100
    player2.jail_turns = 1
    player2.position = 0
    opponent2.position = 0

    _clear_payments_in_range(state2, player2.position)
    risk_cell = state2.board[3]
    risk_cell.owner_id = opponent2.player_id
    risk_cell.houses = 1
    risk_cell.hotels = 0

    decision2 = decide_jail_exit(state2, player2, params)
    assert decision2 == "roll"


def test_params_backward_compat() -> None:
    params = BotParams.from_dict({"weights": {"auction": {"early": {"bias": 0.1}}}})
    defaults = BotParams()
    for feature in [
        "positional_threat_self",
        "positional_threat_others",
        "positional_threat_self_2",
        "positional_threat_others_2",
        "railroad_synergy",
        "utility_synergy",
        "opponent_cash_min_norm",
        "opponent_cash_pressure",
        "opp_owned_in_group_max",
        "opp_cash_after_bid_min",
        "short_term_rent_1",
        "short_term_rent_2",
    ]:
        assert (
            params.weights["auction"]["early"][feature]
            == defaults.weights["auction"]["early"][feature]
        )
    for feature in ["denial_value", "short_term_income_1", "short_term_income_2"]:
        assert (
            params.weights["build"]["early"][feature]
            == defaults.weights["build"]["early"][feature]
        )
    assert (
        params.weights["jail"]["early"]["lost_income_if_stay"]
        == defaults.weights["jail"]["early"]["lost_income_if_stay"]
    )
    assert (
        params.weights["mortgage"]["early"]["positional_threat_self"]
        == defaults.weights["mortgage"]["early"]["positional_threat_self"]
    )


def test_positional_threat_two_turns_far_cell() -> None:
    engine = create_engine(num_players=2, seed=1)
    state = engine.state
    player_id = 0
    opponent_id = 1

    for cell in state.board:
        cell.owner_id = None
        cell.mortgaged = False
        cell.houses = 0
        cell.hotels = 0
        if cell.cell_type == "tax":
            cell.tax_amount = 0

    target = next(cell for cell in state.board if cell.cell_type == "property")
    target.owner_id = opponent_id
    target.houses = 1
    target.hotels = 0
    if not target.rent_by_houses:
        target.rent_by_houses = [100, 100, 100, 100, 100, 100]

    board_size = len(state.board)
    state.players[player_id].position = (target.index - 20) % board_size

    threat_one = positional_threat_self_turns(state, player_id, turns=1)
    threat_two = positional_threat_self_turns(state, player_id, turns=2)

    assert threat_one == 0
    assert threat_two > 0


def test_game_stage_progression() -> None:
    engine = create_engine(num_players=4, seed=1)
    state = engine.state
    assert game_stage(state) == "early"

    buyables = [
        cell
        for cell in state.board
        if cell.cell_type in {"property", "railroad", "utility"}
    ]
    for cell in buyables:
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

    state.players[1].bankrupt = False
    state.turn_index += 1
    assert game_stage(state) == "late"

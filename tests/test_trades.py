from __future__ import annotations

from monopoly.engine import create_engine
from monopoly.models import TradeHistory


def test_trade_disallows_group_buildings() -> None:
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    cell_a = engine.state.board[1]
    cell_b = engine.state.board[3]
    cell_a.owner_id = player.player_id
    cell_b.owner_id = player.player_id
    player.properties = [cell_a.index, cell_b.index]
    cell_a.houses = 1

    offer = {
        "from_player": player.player_id,
        "to_player": 1,
        "give_props": [cell_b.index],
        "receive_props": [],
        "give_cash": 0,
        "receive_cash": 0,
        "give_cards": 0,
        "receive_cards": 0,
    }
    valid, reason = engine._validate_trade_offer(offer)

    assert not valid
    assert reason == "not_tradeable"


def test_trade_requires_interest_fee_cash() -> None:
    engine = create_engine(num_players=2, seed=2)
    from_player = engine.state.players[0]
    to_player = engine.state.players[1]
    cell = engine.state.board[1]
    cell.owner_id = from_player.player_id
    cell.mortgaged = True
    cell.mortgage_value = 100
    from_player.properties = [cell.index]
    to_player.money = 5
    engine.state.rules.interest_rate = 0.1

    offer = {
        "from_player": from_player.player_id,
        "to_player": to_player.player_id,
        "give_props": [cell.index],
        "receive_props": [],
        "give_cash": 0,
        "receive_cash": 0,
        "give_cards": 0,
        "receive_cards": 0,
    }
    valid, reason = engine._validate_trade_offer(offer)

    assert not valid
    assert reason == "interest_fee_to"


def test_trade_offer_requires_improvement() -> None:
    engine = create_engine(num_players=2, seed=3)
    from_id = 0
    to_id = 1
    engine.state.trade_history[(from_id, to_id)] = TradeHistory(
        last_reject_score=100.0,
        last_offer_hash="x",
        last_reject_turn=1,
        from_props=engine._player_property_snapshot(from_id),
        to_props=engine._player_property_snapshot(to_id),
        from_worth=engine._player_net_worth(from_id),
        to_worth=engine._player_net_worth(to_id),
    )

    assert engine._trade_offer_improved(from_id, to_id, 101.0) is False
    assert engine._trade_offer_improved(from_id, to_id, 103.0) is True


def test_trade_history_resets_on_property_change() -> None:
    engine = create_engine(num_players=2, seed=4)
    from_id = 0
    to_id = 1
    cell = engine.state.board[1]
    cell.owner_id = from_id
    engine.state.players[from_id].properties = [cell.index]
    engine.state.trade_history[(from_id, to_id)] = TradeHistory(
        last_reject_score=10.0,
        last_offer_hash="x",
        last_reject_turn=1,
        from_props=engine._player_property_snapshot(from_id),
        to_props=engine._player_property_snapshot(to_id),
        from_worth=engine._player_net_worth(from_id),
        to_worth=engine._player_net_worth(to_id),
    )

    cell.owner_id = to_id

    assert engine._trade_offer_improved(from_id, to_id, 0.0) is True

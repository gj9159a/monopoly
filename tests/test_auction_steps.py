from __future__ import annotations

from dataclasses import dataclass

from monopoly.engine import create_engine
from monopoly.params import BotParams
from monopoly.params import (
    auction_increment_for_remaining,
    choose_auction_bid,
    normalize_auction_price,
)


@dataclass
class FixedRNG:
    values: list[int]
    index: int = 0

    def randint(self, a: int, b: int) -> int:
        if self.index >= len(self.values):
            raise IndexError("Недостаточно значений для FixedRNG")
        value = self.values[self.index]
        self.index += 1
        return value


def test_auction_step_thresholds():
    increments = [5, 20, 50]
    assert auction_increment_for_remaining(120, increments) == 50
    assert auction_increment_for_remaining(45, increments) == 20
    assert auction_increment_for_remaining(12, increments) == 5
    assert auction_increment_for_remaining(9, increments) == 5
    assert auction_increment_for_remaining(4, increments) == 0


def test_auction_price_normalization():
    increments = [5, 20, 50]
    assert normalize_auction_price(0, increments) == 0
    assert normalize_auction_price(5, increments) == 5
    assert normalize_auction_price(7, increments) == 10


def test_auction_bid_respects_increments_and_max():
    increments = [5, 20, 50]
    bid = choose_auction_bid(target_max=300, current_price=200, increments=increments)
    assert bid == 250
    assert (bid - 200) in increments

    bid2 = choose_auction_bid(target_max=220, current_price=200, increments=increments)
    assert bid2 == 205
    assert (bid2 - 200) in increments

    bid3 = choose_auction_bid(target_max=204, current_price=200, increments=increments)
    assert bid3 == 0


def test_auction_finishes_with_winner():
    aggressive = BotParams.from_dict(
        {
            "auction_early_cash_after_bid": -1.0,
            "auction_early_liquidity_ratio": -0.5,
        }
    )
    engine = create_engine(num_players=3, seed=1, bot_params=aggressive)
    engine.state.rng = FixedRNG([1, 2])
    engine.state.players[0].position = 0
    cell = engine.state.board[3]
    cell.owner_id = None

    events = engine.step()

    assert any(event.type == "AUCTION_START" for event in events)
    assert any(event.type == "AUCTION_WIN" for event in events)
    assert cell.owner_id is not None


def test_auction_starts_at_zero_and_first_bid_is_min_increment():
    params = BotParams.from_dict({"max_bid_fraction": 1.0})
    engine = create_engine(num_players=2, seed=1, bot_params=params)
    engine.state.rng = FixedRNG([1, 2])
    min_increment = min(engine.state.rules.auction_increments)
    for player in engine.state.players:
        player.money = min_increment
    engine.state.players[0].position = 0
    cell = engine.state.board[3]
    cell.owner_id = None

    events = engine.step()

    bid_events = [event for event in events if event.type == "AUCTION_BID"]
    assert bid_events
    first_bid = bid_events[0]
    assert first_bid.payload["bid"] == min_increment
    assert first_bid.payload["increment"] == min_increment


def test_auction_pass_when_max_bid_below_min_increment():
    params = BotParams.from_dict({"max_bid_fraction": 1.0})
    engine = create_engine(num_players=2, seed=1, bot_params=params)
    engine.state.rng = FixedRNG([1, 2])
    min_increment = min(engine.state.rules.auction_increments)
    for player in engine.state.players:
        player.money = max(0, min_increment - 1)
    engine.state.players[0].position = 0
    cell = engine.state.board[3]
    cell.owner_id = None

    events = engine.step()

    assert not [event for event in events if event.type == "AUCTION_BID"]


def test_auction_bids_are_multiple_of_min_increment():
    engine = create_engine(num_players=3, seed=1)
    engine.state.rng = FixedRNG([1, 2])
    min_increment = min(engine.state.rules.auction_increments)
    engine.state.players[0].position = 0
    cell = engine.state.board[3]
    cell.owner_id = None

    events = engine.step()

    for event in events:
        if event.type == "AUCTION_BID":
            assert event.payload["bid"] % min_increment == 0
            assert event.payload["increment"] % min_increment == 0

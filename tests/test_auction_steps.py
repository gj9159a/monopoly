from __future__ import annotations

from dataclasses import dataclass

from monopoly.engine import create_engine
from monopoly.params import BotParams
from monopoly.params import auction_increment_for_remaining, choose_auction_bid


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
    assert auction_increment_for_remaining(9, increments) == 0


def test_auction_bid_respects_increments_and_max():
    increments = [5, 20, 50]
    bid = choose_auction_bid(target_max=300, current_price=200, increments=increments)
    assert bid == 250
    assert (bid - 200) in increments

    bid2 = choose_auction_bid(target_max=220, current_price=200, increments=increments)
    assert bid2 == 205
    assert (bid2 - 200) in increments

    bid3 = choose_auction_bid(target_max=206, current_price=200, increments=increments)
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

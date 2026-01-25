from __future__ import annotations

from dataclasses import dataclass

from monopoly.engine import create_engine
from monopoly.params import BotParams


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


def _engine_with_rng(values: list[int], num_players: int = 2, bot_params: BotParams | None = None):
    engine = create_engine(num_players=num_players, seed=1, bot_params=bot_params)
    engine.state.rng = FixedRNG(values)
    return engine


def _aggressive_auction_params() -> BotParams:
    return BotParams.from_dict(
        {
            "auction_early_cash_after_bid": -1.0,
            "auction_early_liquidity_ratio": -0.5,
        }
    )


def test_hr1_always_auction():
    engine = _engine_with_rng([1, 2], bot_params=_aggressive_auction_params())
    engine.state.players[0].position = 0
    cell = engine.state.board[3]
    cell.owner_id = None

    events = engine.step()

    assert any(event.type == "AUCTION_START" for event in events)
    assert cell.owner_id is not None


def test_hr2_no_rent_when_owner_in_jail():
    engine = _engine_with_rng([1, 2])
    engine.state.players[0].position = 0
    owner = engine.state.players[1]
    owner.in_jail = True

    cell = engine.state.board[3]
    cell.owner_id = owner.player_id

    money_before = engine.state.players[0].money
    events = engine.step()
    rent_events = [event for event in events if event.type == "PAY_RENT"]

    assert rent_events
    assert rent_events[-1].payload["amount"] == 0
    assert engine.state.players[0].money == money_before


def test_determinism_by_seed():
    def snapshot(seed: int):
        engine = create_engine(num_players=3, seed=seed)
        for _ in range(15):
            engine.step()
        players = [
            (p.position, p.money, p.in_jail, p.jail_turns) for p in engine.state.players
        ]
        tail_log = [(e.type, e.msg_ru) for e in engine.state.event_log[-10:]]
        return players, tail_log

    assert snapshot(42) == snapshot(42)


def test_three_doubles_send_to_jail():
    engine = _engine_with_rng([2, 2, 3, 3, 4, 4])
    engine.state.players[0].position = 0

    engine.step()
    engine.step()
    events = engine.step()

    player = engine.state.players[0]
    assert player.in_jail is True
    assert any(event.type == "GO_TO_JAIL" for event in events)


def test_game_end_on_bankruptcy():
    engine = _engine_with_rng([1, 2])
    player = engine.state.players[0]
    owner = engine.state.players[1]
    player.position = 0
    player.money = 10

    cell = engine.state.board[3]
    cell.owner_id = owner.player_id
    cell.rent_by_houses = [100, 100, 100, 100, 100, 100]

    events = engine.step()

    assert player.bankrupt is True
    assert engine.state.game_over is True
    assert engine.state.winner_id == owner.player_id
    assert any(event.type == "GAME_END" for event in events)

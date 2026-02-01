from __future__ import annotations

from dataclasses import dataclass

from monopoly.engine import Engine, create_game
from monopoly.params import BotParams


@dataclass
class PassiveBot:
    params: BotParams = BotParams()

    def decide(self, state, context):
        return {"action": "pass"}


class AuctionLiquidityBot:
    def __init__(self, mortgage_cell_index: int, bid: int, params: BotParams | None = None) -> None:
        self.params = params or BotParams()
        self.mortgage_cell_index = mortgage_cell_index
        self.bid = bid

    def decide(self, state, context):
        if context.get("type") == "liquidity":
            return {"actions": [{"action": "mortgage", "cell_index": self.mortgage_cell_index}]}
        if context.get("type") == "auction_bid":
            return {"action": "bid", "bid": self.bid}
        return {"action": "pass"}


class TradeLiquidityBot:
    def __init__(
        self,
        mortgage_cell_index: int,
        offer: dict,
        params: BotParams | None = None,
    ) -> None:
        self.params = params or BotParams()
        self.mortgage_cell_index = mortgage_cell_index
        self.offer = offer

    def decide(self, state, context):
        if context.get("type") == "liquidity":
            return {"actions": [{"action": "mortgage", "cell_index": self.mortgage_cell_index}]}
        if context.get("type") == "trade_offer":
            return {"action": "offer", "offer": self.offer}
        return {"action": "pass"}


class AcceptBot:
    def __init__(self, params: BotParams | None = None) -> None:
        self.params = params or BotParams()

    def decide(self, state, context):
        if context.get("type") == "trade_accept":
            return {"action": "accept", "score": 1.0, "valid": True, "mortgage_policy": "keep"}
        return {"action": "pass"}


def _pick_two_properties(state):
    props = [cell for cell in state.board if cell.cell_type == "property"]
    if len(props) < 2:
        raise AssertionError("Недостаточно property-клеток для теста")
    return props[0], props[1]


def test_liquidity_allows_auction_bid_with_mortgage():
    state = create_game(num_players=2, seed=1)
    player0 = state.players[0]
    player0.money = 10

    mortgage_cell, auction_cell = _pick_two_properties(state)
    mortgage_cell.owner_id = player0.player_id
    mortgage_cell.mortgaged = False
    mortgage_cell.mortgage_value = 200
    auction_cell.owner_id = None

    bid = 50
    bots = [AuctionLiquidityBot(mortgage_cell.index, bid), PassiveBot()]
    engine = Engine(state, bots)

    events = engine._run_auction(auction_cell, turn_index=0)

    assert auction_cell.owner_id == player0.player_id
    assert mortgage_cell.mortgaged is True
    assert player0.money == 10 + int(mortgage_cell.mortgage_value or 0) - bid
    assert any(event.type == "MORTGAGE" for event in events)


def test_liquidity_enables_trade_offer_cash():
    state = create_game(num_players=2, seed=2)
    state.rules.no_trades = False
    player0 = state.players[0]
    player1 = state.players[1]
    player0.money = 0

    mortgage_cell, trade_cell = _pick_two_properties(state)
    mortgage_cell.owner_id = player0.player_id
    mortgage_cell.mortgaged = False
    mortgage_cell.mortgage_value = 200
    trade_cell.owner_id = player1.player_id

    offer = {
        "from_player": player0.player_id,
        "to_player": player1.player_id,
        "give_props": [],
        "receive_props": [trade_cell.index],
        "give_cash": 100,
        "receive_cash": 0,
        "give_cards": 0,
        "receive_cards": 0,
    }

    bots = [
        TradeLiquidityBot(mortgage_cell.index, offer),
        AcceptBot(),
    ]
    engine = Engine(state, bots)

    events = engine._bot_trade_phase(player0, turn_index=0)

    assert any(event.type == "TRADE_ACCEPT" for event in events)
    assert trade_cell.owner_id == player0.player_id
    assert mortgage_cell.mortgaged is True
    assert player0.money == int(mortgage_cell.mortgage_value or 0) - offer["give_cash"]

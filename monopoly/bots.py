from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Cell, GameState, Player
from .params import (
    BotParams,
    compute_cash_buffer,
    decide_auction_bid,
    decide_build_actions,
    decide_jail_exit,
    decide_liquidation,
    estimate_asset_value,
)


@dataclass(frozen=True)
class Bot:
    params: BotParams

    def decide(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        decision_type = context.get("type")
        if decision_type == "auction_bid":
            return self._decide_auction_bid(state, context)
        if decision_type == "jail_decision":
            return self._decide_jail(state, context)
        if decision_type == "economy_phase":
            return self._decide_economy(state, context)
        if decision_type == "liquidation":
            return self._decide_liquidation(state, context)
        raise ValueError(f"Неизвестный тип контекста: {decision_type}")

    def _decide_auction_bid(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        cell: Cell = context["cell"]
        current_price = int(context["current_price"])
        min_increment = int(context["min_increment"])

        bid = decide_auction_bid(state, player, cell, current_price, min_increment, self.params)
        if bid <= 0:
            return {"action": "pass"}
        return {"action": "bid", "bid": bid}

    def _decide_jail(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        action = decide_jail_exit(state, player, self.params)
        if action == "use_card" and not player.get_out_of_jail_cards:
            action = "roll"
        return {"action": action}

    def _decide_economy(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        actions = decide_build_actions(state, player, self.params)
        return {"actions": actions}

    def _decide_liquidation(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        debt = int(context.get("debt", 0))
        actions = decide_liquidation(state, player, debt, self.params)
        return {"actions": actions}

    def prioritize_mortgage(self, cells: list[Cell], state: GameState, player: Player) -> list[Cell]:
        return sorted(
            cells,
            key=lambda cell: estimate_asset_value(state, player, cell, self.params),
        )

    def cash_buffer(self, state: GameState, player: Player) -> int:
        return compute_cash_buffer(state, player, self.params)


def create_bots(num_players: int, params: BotParams | list[BotParams] | None = None) -> list[Bot]:
    if params is None:
        params_list = [BotParams()] * num_players
    elif isinstance(params, BotParams):
        params_list = [params] * num_players
    else:
        params_list = list(params)
        if len(params_list) < num_players:
            params_list.extend([params_list[-1]] * (num_players - len(params_list)))
        params_list = params_list[:num_players]
    return [Bot(param) for param in params_list]

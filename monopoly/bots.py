from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Cell, GameState, Player
from .params import (
    BotParams,
    choose_auction_bid,
    compute_cash_buffer,
    decide_auction_bid,
    decide_trade_accept,
    decide_trade_offer,
    decide_build_actions,
    decide_jail_exit,
    decide_liquidation,
    estimate_asset_value,
)
from .thinking import choose_action


@dataclass
class Bot:
    params: BotParams
    last_thinking: dict[str, Any] = field(default_factory=dict)
    _thinking_cache: dict[str, float] = field(default_factory=dict, repr=False)

    def decide(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        decision_type = context.get("type")
        if self.params.thinking.enabled and decision_type in {
            "auction_bid",
            "jail_decision",
            "economy_phase",
            "liquidation",
        }:
            cache = None
            if self.params.thinking.cache_enabled:
                cache = self._thinking_cache
                cache_size = max(0, int(self.params.thinking.cache_size))
                if cache_size and len(cache) > cache_size:
                    cache.clear()
            action, stats = choose_action(
                state,
                context,
                self.params,
                self.params.thinking,
                cache=cache,
            )
            self.last_thinking = {
                "decision_type": stats.decision_type,
                "ms": stats.ms,
                "candidates": stats.candidates,
                "rollouts": stats.rollouts,
                "best_score": stats.best_score,
            }
            return action
        if decision_type == "auction_bid":
            return self._decide_auction_bid(state, context)
        if decision_type == "jail_decision":
            return self._decide_jail(state, context)
        if decision_type == "economy_phase":
            return self._decide_economy(state, context)
        if decision_type == "liquidation":
            return self._decide_liquidation(state, context)
        if decision_type == "trade_offer":
            return self._decide_trade_offer(state, context)
        if decision_type == "trade_accept":
            return self._decide_trade_accept(state, context)
        raise ValueError(f"Неизвестный тип контекста: {decision_type}")

    def _decide_auction_bid(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        cell: Cell = context["cell"]
        current_price = int(context["current_price"])
        min_increment = int(context["min_increment"])
        increments = getattr(state.rules, "auction_increments", None)
        if not increments:
            increments = [min_increment]
        target_max = decide_auction_bid(state, player, cell, current_price, min_increment, self.params)
        bid = choose_auction_bid(int(target_max), current_price, list(increments))
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

    def _decide_trade_offer(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        offer = decide_trade_offer(state, player, self.params)
        if offer is None:
            return {"action": "pass"}
        return {"action": "offer", "offer": offer}

    def _decide_trade_accept(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        offer = context.get("offer", {})
        if not isinstance(offer, dict):
            return {"action": "reject", "score": 0.0, "valid": False}
        return decide_trade_accept(state, player, offer, self.params)

    def prioritize_mortgage(self, cells: list[Cell], state: GameState, player: Player) -> list[Cell]:
        if self.params.thinking.enabled and cells:
            context = {"type": "mortgage", "player_id": player.player_id, "cells": cells}
            cache = None
            if self.params.thinking.cache_enabled:
                cache = self._thinking_cache
                cache_size = max(0, int(self.params.thinking.cache_size))
                if cache_size and len(cache) > cache_size:
                    cache.clear()
            action, stats = choose_action(
                state,
                context,
                self.params,
                self.params.thinking,
                cache=cache,
            )
            self.last_thinking = {
                "decision_type": stats.decision_type,
                "ms": stats.ms,
                "candidates": stats.candidates,
                "rollouts": stats.rollouts,
                "best_score": stats.best_score,
            }
            order = action.get("order")
            if isinstance(order, list) and order:
                ordered = []
                index_map = {cell.index: cell for cell in cells}
                for idx in order:
                    cell = index_map.get(int(idx))
                    if cell is not None and cell not in ordered:
                        ordered.append(cell)
                rest = [cell for cell in cells if cell not in ordered]
                rest.sort(key=lambda cell: estimate_asset_value(state, player, cell, self.params))
                return ordered + rest
            chosen_index = action.get("cell_index")
            if chosen_index is not None:
                preferred = [cell for cell in cells if cell.index == int(chosen_index)]
                rest = [cell for cell in cells if cell.index != int(chosen_index)]
                rest.sort(key=lambda cell: estimate_asset_value(state, player, cell, self.params))
                return preferred + rest
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

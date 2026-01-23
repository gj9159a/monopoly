from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Cell, GameState


@dataclass(frozen=True)
class BotProfile:
    name: str
    max_bid_multiplier: float
    reserve_cash: int


PROFILES: dict[str, BotProfile] = {
    "Aggressive": BotProfile(name="Aggressive", max_bid_multiplier=1.4, reserve_cash=50),
    "Conservative": BotProfile(name="Conservative", max_bid_multiplier=0.9, reserve_cash=150),
    "Builder": BotProfile(name="Builder", max_bid_multiplier=1.2, reserve_cash=100),
    "CashSaver": BotProfile(name="CashSaver", max_bid_multiplier=0.7, reserve_cash=200),
}


class BaseBot:
    def __init__(self, profile: BotProfile) -> None:
        self.profile = profile

    def decide(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        if context.get("type") == "auction_bid":
            return self._decide_auction_bid(state, context)
        if context.get("type") == "jail_decision":
            return self._decide_jail(state, context)
        raise ValueError(f"Неизвестный тип контекста: {context.get('type')}")

    def _decide_auction_bid(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        cell: Cell = context["cell"]
        current_price = int(context["current_price"])
        min_increment = int(context["min_increment"])

        if cell.price is None:
            return {"action": "pass"}

        max_bid = int(cell.price * self.profile.max_bid_multiplier)
        max_bid = min(max_bid, max(0, player.money - self.profile.reserve_cash))

        if max_bid < 1:
            return {"action": "pass"}

        next_bid = current_price + min_increment
        if next_bid <= max_bid and next_bid <= player.money:
            return {"action": "bid", "bid": next_bid}
        return {"action": "pass"}

    def _decide_jail(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        player_id = int(context["player_id"])
        player = state.players[player_id]
        fine = state.rules.jail_fine
        has_card = bool(context.get("has_card")) and player.get_out_of_jail_cards
        if has_card and player.money < fine + self.profile.reserve_cash:
            return {"action": "use_card"}
        if player.money - fine >= self.profile.reserve_cash:
            return {"action": "pay"}
        return {"action": "roll"}

    def prioritize_mortgage(self, cells: list[Cell]) -> list[Cell]:
        if self.profile.name in {"Aggressive", "Builder"}:
            return sorted(cells, key=lambda c: (c.price or 0, c.index))
        return sorted(cells, key=lambda c: (c.price or 0, c.index), reverse=True)


def create_bots(num_players: int, profile_names: list[str] | None = None) -> list[BaseBot]:
    if profile_names is None or not profile_names:
        profile_names = list(PROFILES.keys())
    bots: list[BaseBot] = []
    for idx in range(num_players):
        profile_name = profile_names[idx % len(profile_names)]
        profile = PROFILES.get(profile_name)
        if profile is None:
            raise ValueError(f"Неизвестный профиль бота: {profile_name}")
        bots.append(BaseBot(profile))
    return bots

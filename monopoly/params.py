from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .models import Cell, GameState, Player


@dataclass(frozen=True)
class BotParams:
    cash_buffer_base: int = 150
    cash_buffer_per_house: int = 20
    auction_value_mult_street: float = 1.0
    auction_value_mult_rail: float = 1.1
    auction_value_mult_utility: float = 0.9
    monopoly_completion_bonus: float = 0.6
    monopoly_block_bonus: float = 0.3
    build_aggressiveness: float = 1.0
    un_mortgage_priority_mult: float = 1.1
    jail_exit_aggressiveness: float = 0.6
    risk_aversion: float = 0.5
    max_bid_fraction: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotParams":
        normalized = dict(data)
        for spec in PARAM_SPECS:
            if spec.name in normalized:
                value = normalized[spec.name]
                if spec.value_type is int:
                    normalized[spec.name] = int(round(float(value)))
                else:
                    normalized[spec.name] = float(value)
        fields = {
            field: normalized.get(field, getattr(cls, field))
            for field in cls.__dataclass_fields__
        }
        return cls(**fields)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "BotParams":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_yaml(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: Path) -> "BotParams":
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ParamSpec:
    name: str
    min_value: float
    max_value: float
    init_std: float
    value_type: type


PARAM_SPECS: list[ParamSpec] = [
    ParamSpec("cash_buffer_base", 0, 600, 120, int),
    ParamSpec("cash_buffer_per_house", 0, 80, 20, int),
    ParamSpec("auction_value_mult_street", 0.4, 2.0, 0.4, float),
    ParamSpec("auction_value_mult_rail", 0.4, 2.0, 0.4, float),
    ParamSpec("auction_value_mult_utility", 0.4, 2.0, 0.4, float),
    ParamSpec("monopoly_completion_bonus", 0.0, 1.5, 0.3, float),
    ParamSpec("monopoly_block_bonus", 0.0, 1.2, 0.3, float),
    ParamSpec("build_aggressiveness", 0.0, 2.0, 0.4, float),
    ParamSpec("un_mortgage_priority_mult", 0.5, 2.0, 0.3, float),
    ParamSpec("jail_exit_aggressiveness", 0.0, 1.0, 0.3, float),
    ParamSpec("risk_aversion", 0.0, 1.0, 0.3, float),
    ParamSpec("max_bid_fraction", 0.1, 1.0, 0.2, float),
]


def params_to_vector(params: BotParams) -> list[float]:
    return [float(getattr(params, spec.name)) for spec in PARAM_SPECS]


def vector_to_params(values: Iterable[float]) -> BotParams:
    data: dict[str, Any] = {}
    for spec, value in zip(PARAM_SPECS, values, strict=False):
        clipped = min(spec.max_value, max(spec.min_value, float(value)))
        if spec.value_type is int:
            data[spec.name] = int(round(clipped))
        else:
            data[spec.name] = float(clipped)
    return BotParams.from_dict(data)


def load_params(path: str | Path) -> BotParams:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return BotParams.from_yaml(path)
    if path.suffix.lower() == ".json":
        return BotParams.from_json(path)
    raise ValueError(f"Неизвестный формат параметров: {path}")


def save_params(params: BotParams, path: str | Path) -> None:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        params.to_yaml(path)
        return
    if path.suffix.lower() == ".json":
        params.to_json(path)
        return
    raise ValueError(f"Неизвестный формат параметров: {path}")


def compute_cash_buffer(state: GameState, player: Player, params: BotParams) -> int:
    opponent_buildings = 0
    for cell in state.board:
        if cell.owner_id is None or cell.owner_id == player.player_id:
            continue
        opponent_buildings += cell.houses + cell.hotels * 5
    buffer_value = params.cash_buffer_base + params.cash_buffer_per_house * opponent_buildings
    buffer_value = int(buffer_value * (1 + params.risk_aversion))
    return max(0, buffer_value)


def estimate_asset_value(state: GameState, player: Player, cell: Cell, params: BotParams) -> float:
    base = float(cell.price or 0)
    if cell.cell_type == "property" and cell.group:
        group_cells = [c for c in state.board if c.group == cell.group]
        owned_by_player = [c for c in group_cells if c.owner_id == player.player_id]
        if len(owned_by_player) == len(group_cells) - 1:
            base += params.monopoly_completion_bonus * base
        for opponent in state.players:
            if opponent.player_id == player.player_id or opponent.bankrupt:
                continue
            if all(c.owner_id == opponent.player_id or c.index == cell.index for c in group_cells):
                base += params.monopoly_block_bonus * base
                break
    return base


def decide_auction_bid(
    state: GameState,
    player: Player,
    cell: Cell,
    current_price: int,
    min_increment: int,
    params: BotParams,
) -> int:
    if cell.price is None:
        return 0
    buffer_value = compute_cash_buffer(state, player, params)
    max_bid_cash = max(0, player.money - buffer_value)
    max_bid = int(max_bid_cash * params.max_bid_fraction)
    if cell.cell_type == "property":
        value = estimate_asset_value(state, player, cell, params) * params.auction_value_mult_street
    elif cell.cell_type == "railroad":
        value = estimate_asset_value(state, player, cell, params) * params.auction_value_mult_rail
    elif cell.cell_type == "utility":
        value = estimate_asset_value(state, player, cell, params) * params.auction_value_mult_utility
    else:
        value = 0
    target = int(min(value, max_bid))
    next_bid = current_price + min_increment
    if next_bid <= target and next_bid <= player.money:
        return next_bid
    return 0


def _property_level(houses: int, hotels: int) -> int:
    return houses + hotels * 5


def _group_cells(state: GameState, group: str | None) -> list[Cell]:
    if group is None:
        return []
    return [cell for cell in state.board if cell.group == group]


def _owns_group(state: GameState, player_id: int, group: str | None) -> bool:
    if group is None:
        return False
    group_cells = _group_cells(state, group)
    return bool(group_cells) and all(cell.owner_id == player_id for cell in group_cells)


def _can_build_on_cell(state: GameState, player_id: int, cell: Cell, houses: int, hotels: int) -> bool:
    if cell.owner_id != player_id or cell.cell_type != "property":
        return False
    if cell.group is None or not _owns_group(state, player_id, cell.group):
        return False
    group_cells = _group_cells(state, cell.group)
    if any(other.mortgaged for other in group_cells):
        return False
    if cell.house_cost is None:
        return False
    levels = [
        _property_level(
            houses if other.index == cell.index else other.houses,
            hotels if other.index == cell.index else other.hotels,
        )
        for other in group_cells
    ]
    min_level = min(levels)
    if _property_level(houses, hotels) != min_level:
        return False
    if _property_level(houses, hotels) >= 5:
        return False
    total_houses = sum(c.houses for c in state.board)
    total_hotels = sum(c.hotels for c in state.board)
    if _property_level(houses, hotels) == 4 and total_hotels >= state.rules.bank_hotels:
        return False
    if _property_level(houses, hotels) < 4 and total_houses >= state.rules.bank_houses:
        return False
    return True


def decide_build_actions(state: GameState, player: Player, params: BotParams) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if player.bankrupt:
        return actions
    buffer_value = compute_cash_buffer(state, player, params)
    effective_buffer = int(buffer_value * max(0.0, 1 - 0.5 * params.build_aggressiveness))
    available_cash = player.money

    mortgaged_cells = [
        cell
        for cell in state.board
        if cell.owner_id == player.player_id and cell.mortgaged
    ]
    mortgaged_cells.sort(
        key=lambda cell: estimate_asset_value(state, player, cell, params), reverse=True
    )
    for cell in mortgaged_cells:
        cost = int((cell.mortgage_value or 0) * (1 + state.rules.interest_rate))
        if available_cash - cost < buffer_value:
            continue
        if estimate_asset_value(state, player, cell, params) * params.un_mortgage_priority_mult < cost:
            continue
        actions.append({"action": "unmortgage", "cell_index": cell.index})
        available_cash -= cost

    local_houses = {cell.index: cell.houses for cell in state.board}
    local_hotels = {cell.index: cell.hotels for cell in state.board}

    planned = 0
    max_plans = 20
    while planned < max_plans:
        candidates: list[Cell] = []
        total_houses = sum(local_houses.values())
        total_hotels = sum(local_hotels.values())
        for cell in state.board:
            if cell.owner_id != player.player_id:
                continue
            if cell.cell_type != "property":
                continue
            if cell.group is None or not _owns_group(state, player.player_id, cell.group):
                continue
            group_cells = _group_cells(state, cell.group)
            if any(other.mortgaged for other in group_cells):
                continue
            if cell.house_cost is None:
                continue
            levels = [
                _property_level(local_houses[other.index], local_hotels[other.index])
                for other in group_cells
            ]
            if _property_level(local_houses[cell.index], local_hotels[cell.index]) != min(levels):
                continue
            if _property_level(local_houses[cell.index], local_hotels[cell.index]) >= 5:
                continue
            if _property_level(local_houses[cell.index], local_hotels[cell.index]) == 4 and total_hotels >= state.rules.bank_hotels:
                continue
            if _property_level(local_houses[cell.index], local_hotels[cell.index]) < 4 and total_houses >= state.rules.bank_houses:
                continue
            candidates.append(cell)
        if not candidates:
            break
        candidates.sort(
            key=lambda cell: estimate_asset_value(state, player, cell, params), reverse=True
        )
        cell = candidates[0]
        cost = cell.house_cost or 0
        if available_cash - cost < effective_buffer:
            break
        actions.append({"action": "build", "cell_index": cell.index})
        available_cash -= cost
        if _property_level(local_houses[cell.index], local_hotels[cell.index]) < 4:
            local_houses[cell.index] += 1
        else:
            local_hotels[cell.index] = 1
            local_houses[cell.index] = 0
        planned += 1
    return actions


def decide_liquidation(
    state: GameState, player: Player, debt: int, params: BotParams
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if debt <= 0:
        return actions
    sell_candidates = [
        cell
        for cell in state.board
        if cell.owner_id == player.player_id and (cell.houses > 0 or cell.hotels > 0)
    ]
    sell_candidates.sort(
        key=lambda cell: estimate_asset_value(state, player, cell, params)
    )
    for cell in sell_candidates:
        level = _property_level(cell.houses, cell.hotels)
        for _ in range(level):
            actions.append({"action": "sell_building", "cell_index": cell.index})

    mortgage_candidates = [
        cell
        for cell in state.board
        if cell.owner_id == player.player_id and not cell.mortgaged
    ]
    mortgage_candidates.sort(
        key=lambda cell: estimate_asset_value(state, player, cell, params)
    )
    for cell in mortgage_candidates:
        actions.append({"action": "mortgage", "cell_index": cell.index})

    return actions


def decide_jail_exit(state: GameState, player: Player, params: BotParams) -> str:
    fine = state.rules.jail_fine
    buffer_value = compute_cash_buffer(state, player, params)
    has_card = bool(player.get_out_of_jail_cards)

    if has_card and player.money < fine + buffer_value:
        return "use_card"
    if player.money - fine >= buffer_value and params.jail_exit_aggressiveness >= 0.4:
        return "pay"
    if params.jail_exit_aggressiveness >= 0.9 and player.money - fine >= 0:
        return "pay"
    return "roll"

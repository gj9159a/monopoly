from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .features import (
    cell_heat,
    denial_value,
    group_heat,
    group_heat_vs_base,
    jail_exit_heat_group,
    landing_prob_group,
    positional_threat_others,
    positional_threat_self,
    railroad_synergy,
    utility_synergy,
)
from .models import Cell, GameState, Player

STAGES = ("early", "mid", "late")
STAGE_ORDER = {"early": 0, "mid": 1, "late": 2}
STAGE_HYSTERESIS_TICKS = 6
EARLY_TO_MID_UNOWNED = 6
LATE_FREE_HOUSES = 10
LATE_MONOPOLY_LEVEL = 3

AUCTION_FEATURES = [
    "bias",
    "base_value",
    "group_strength",
    "group_heat",
    "group_heat_vs_base",
    "cell_heat",
    "landing_prob_group",
    "jail_exit_heat_group",
    "completes_monopoly",
    "blocks_opponent_monopoly",
    "cash_after_bid",
    "liquidity_ratio",
    "risk_of_ruin",
    "is_street",
    "is_railroad",
    "is_utility",
    "owned_in_group",
    "positional_threat_self",
    "positional_threat_others",
    "railroad_synergy",
    "utility_synergy",
    "opponent_cash_min_norm",
    "opponent_cash_pressure",
]

BUILD_FEATURES = [
    "bias",
    "roi",
    "rent_delta",
    "group_strength",
    "group_heat",
    "group_heat_vs_base",
    "cell_heat",
    "landing_prob_group",
    "jail_exit_heat_group",
    "cash_after_build",
    "enemy_threat",
    "level_norm",
    "to_hotel",
    "target_three",
    "has_monopoly",
    "bank_houses_ratio",
    "bank_hotels_ratio",
    "positional_threat_others",
    "house_scarcity",
    "hotel_scarcity",
    "denial_value",
    "opponent_cash_pressure",
]

MORTGAGE_FEATURES = [
    "bias",
    "asset_value",
    "low_value",
    "breaks_monopoly",
    "has_buildings",
    "group_heat",
    "group_heat_vs_base",
    "cell_heat",
    "landing_prob_group",
    "jail_exit_heat_group",
    "is_railroad",
    "is_utility",
    "cash_needed",
    "action_sell_building",
    "action_mortgage",
    "positional_threat_self",
]

JAIL_FEATURES = [
    "bias",
    "has_card",
    "cash_after_pay",
    "jail_turns",
    "danger",
    "action_pay",
    "action_use_card",
    "action_roll",
    "danger_if_pay",
    "danger_if_use_card",
    "lost_income_if_stay",
    "saved_risk_if_stay",
]

DECISION_FEATURES = {
    "auction": AUCTION_FEATURES,
    "build": BUILD_FEATURES,
    "mortgage": MORTGAGE_FEATURES,
    "jail": JAIL_FEATURES,
}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    min_value: float
    max_value: float
    init_std: float
    value_type: type


@dataclass(frozen=True)
class ThinkingConfig:
    enabled: bool = False
    horizon_turns: int = 30
    rollouts_per_action: int = 12
    time_budget_ms: int = 0
    workers: int = 1
    cache_enabled: bool = True
    cache_size: int = 4096


def _weight_key(decision: str, stage: str, feature: str) -> str:
    return f"{decision}_{stage}_{feature}"


def _stage_weights(
    base: dict[str, float],
    early: dict[str, float] | None = None,
    mid: dict[str, float] | None = None,
    late: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    return {
        "early": {**base, **(early or {})},
        "mid": {**base, **(mid or {})},
        "late": {**base, **(late or {})},
    }


def _default_weights() -> dict[str, dict[str, dict[str, float]]]:
    auction_base = {
        "bias": 0.0,
        "base_value": 1.0,
        "group_strength": 0.5,
        "group_heat": 0.0,
        "group_heat_vs_base": 0.0,
        "cell_heat": 0.0,
        "landing_prob_group": 0.0,
        "jail_exit_heat_group": 0.0,
        "completes_monopoly": 1.2,
        "blocks_opponent_monopoly": 0.6,
        "cash_after_bid": 0.7,
        "liquidity_ratio": 0.6,
        "risk_of_ruin": -1.1,
        "is_street": 0.3,
        "is_railroad": 0.35,
        "is_utility": 0.1,
        "owned_in_group": 0.6,
        "positional_threat_self": 0.0,
        "positional_threat_others": 0.0,
        "railroad_synergy": 0.0,
        "utility_synergy": 0.0,
        "opponent_cash_min_norm": 0.0,
        "opponent_cash_pressure": 0.0,
    }
    build_base = {
        "bias": 0.0,
        "roi": 1.0,
        "rent_delta": 0.4,
        "group_strength": 0.5,
        "group_heat": 0.0,
        "group_heat_vs_base": 0.0,
        "cell_heat": 0.0,
        "landing_prob_group": 0.0,
        "jail_exit_heat_group": 0.0,
        "cash_after_build": 0.7,
        "enemy_threat": -0.4,
        "level_norm": -0.1,
        "to_hotel": 0.2,
        "target_three": 0.4,
        "has_monopoly": 0.8,
        "bank_houses_ratio": 0.2,
        "bank_hotels_ratio": 0.2,
        "positional_threat_others": 0.0,
        "house_scarcity": 0.0,
        "hotel_scarcity": 0.0,
        "denial_value": 0.0,
        "opponent_cash_pressure": 0.0,
    }
    mortgage_base = {
        "bias": 0.0,
        "asset_value": -0.6,
        "low_value": 0.9,
        "breaks_monopoly": -1.2,
        "has_buildings": -0.8,
        "group_heat": 0.0,
        "group_heat_vs_base": 0.0,
        "cell_heat": 0.0,
        "landing_prob_group": 0.0,
        "jail_exit_heat_group": 0.0,
        "is_railroad": 0.1,
        "is_utility": 0.05,
        "cash_needed": 1.0,
        "action_sell_building": 0.7,
        "action_mortgage": 0.2,
        "positional_threat_self": 0.0,
    }
    jail_base = {
        "bias": 0.0,
        "has_card": 0.2,
        "cash_after_pay": 0.6,
        "jail_turns": 0.4,
        "danger": 0.0,
        "action_pay": 0.2,
        "action_use_card": 0.4,
        "action_roll": 0.1,
        "danger_if_pay": -0.7,
        "danger_if_use_card": -0.6,
        "lost_income_if_stay": 0.0,
        "saved_risk_if_stay": 0.0,
    }
    return {
        "auction": _stage_weights(
            auction_base,
            early={"cash_after_bid": 0.6, "completes_monopoly": 1.1},
            mid={"cash_after_bid": 0.8, "completes_monopoly": 1.3},
            late={"cash_after_bid": 0.95, "risk_of_ruin": -1.3},
        ),
        "build": _stage_weights(
            build_base,
            early={"cash_after_build": 0.6},
            mid={"cash_after_build": 0.7},
            late={"cash_after_build": 0.9, "enemy_threat": -0.6},
        ),
        "mortgage": _stage_weights(
            mortgage_base,
            early={"action_sell_building": 0.8},
            mid={"action_mortgage": 0.25},
            late={"action_mortgage": 0.35, "breaks_monopoly": -1.4},
        ),
        "jail": _stage_weights(
            jail_base,
            early={"action_pay": 0.3, "action_roll": 0.05},
            mid={"action_pay": 0.2},
            late={"action_roll": 0.2, "danger_if_pay": -0.9},
        ),
    }


_WEIGHT_KEYS: dict[str, tuple[str, str, str]] = {}
for decision, feature_list in DECISION_FEATURES.items():
    for stage in STAGES:
        for feature in feature_list:
            key = _weight_key(decision, stage, feature)
            _WEIGHT_KEYS[key] = (decision, stage, feature)


PARAM_SPECS: list[ParamSpec] = [
    ParamSpec("cash_buffer_base", 0, 600, 120, int),
    ParamSpec("cash_buffer_per_house", 0, 80, 20, int),
    ParamSpec("max_bid_fraction", 0.2, 1.0, 0.2, float),
]

for key in _WEIGHT_KEYS:
    PARAM_SPECS.append(ParamSpec(key, -2.5, 2.5, 0.4, float))


@dataclass(frozen=True)
class BotParams:
    weights: dict[str, dict[str, dict[str, float]]] = field(default_factory=_default_weights)
    cash_buffer_base: int = 150
    cash_buffer_per_house: int = 20
    max_bid_fraction: float = 0.95
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "cash_buffer_base": self.cash_buffer_base,
            "cash_buffer_per_house": self.cash_buffer_per_house,
            "max_bid_fraction": self.max_bid_fraction,
            "weights": self.weights,
        }
        if self.thinking.enabled or self.thinking != ThinkingConfig():
            payload["thinking"] = {
                "enabled": self.thinking.enabled,
                "horizon_turns": self.thinking.horizon_turns,
                "rollouts_per_action": self.thinking.rollouts_per_action,
                "time_budget_ms": self.thinking.time_budget_ms,
                "workers": self.thinking.workers,
                "cache_enabled": self.thinking.cache_enabled,
                "cache_size": self.thinking.cache_size,
            }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotParams":
        normalized = dict(data)
        weights = _default_weights()

        raw_weights = normalized.get("weights")
        if isinstance(raw_weights, dict):
            for decision, stages in raw_weights.items():
                if decision not in weights or not isinstance(stages, dict):
                    continue
                for stage, features in stages.items():
                    if stage not in weights[decision] or not isinstance(features, dict):
                        continue
                    for feature, value in features.items():
                        if feature not in weights[decision][stage]:
                            continue
                        weights[decision][stage][feature] = float(value)

        for key, value in normalized.items():
            if key in _WEIGHT_KEYS:
                decision, stage, feature = _WEIGHT_KEYS[key]
                weights[decision][stage][feature] = float(value)

        thinking_data = normalized.get("thinking")
        if isinstance(thinking_data, dict):
            thinking = ThinkingConfig(
                enabled=bool(thinking_data.get("enabled", False)),
                horizon_turns=int(thinking_data.get("horizon_turns", ThinkingConfig().horizon_turns)),
                rollouts_per_action=int(
                    thinking_data.get("rollouts_per_action", ThinkingConfig().rollouts_per_action)
                ),
                time_budget_ms=int(thinking_data.get("time_budget_ms", ThinkingConfig().time_budget_ms)),
                workers=int(thinking_data.get("workers", ThinkingConfig().workers)),
                cache_enabled=bool(thinking_data.get("cache_enabled", ThinkingConfig().cache_enabled)),
                cache_size=int(thinking_data.get("cache_size", ThinkingConfig().cache_size)),
            )
        else:
            thinking = ThinkingConfig()

        cash_buffer_base = int(normalized.get("cash_buffer_base", cls.cash_buffer_base))
        cash_buffer_per_house = int(
            normalized.get("cash_buffer_per_house", cls.cash_buffer_per_house)
        )
        max_bid_fraction = float(normalized.get("max_bid_fraction", cls.max_bid_fraction))

        return cls(
            weights=weights,
            cash_buffer_base=cash_buffer_base,
            cash_buffer_per_house=cash_buffer_per_house,
            max_bid_fraction=max_bid_fraction,
            thinking=thinking,
        )

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

    def with_thinking(self, config: ThinkingConfig) -> "BotParams":
        return BotParams(
            weights=self.weights,
            cash_buffer_base=self.cash_buffer_base,
            cash_buffer_per_house=self.cash_buffer_per_house,
            max_bid_fraction=self.max_bid_fraction,
            thinking=config,
        )


BASE_FIELDS = {"cash_buffer_base", "cash_buffer_per_house", "max_bid_fraction"}


def params_to_vector(params: BotParams) -> list[float]:
    values: list[float] = []
    for spec in PARAM_SPECS:
        if spec.name in BASE_FIELDS:
            values.append(float(getattr(params, spec.name)))
            continue
        if spec.name in _WEIGHT_KEYS:
            decision, stage, feature = _WEIGHT_KEYS[spec.name]
            values.append(float(params.weights[decision][stage][feature]))
            continue
        values.append(0.0)
    return values


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


def _buyable_cells(state: GameState) -> list[Cell]:
    return [
        cell
        for cell in state.board
        if cell.cell_type in {"property", "railroad", "utility"}
    ]


def _monopoly_groups(state: GameState) -> list[list[Cell]]:
    groups: dict[str, list[Cell]] = {}
    for cell in state.board:
        if cell.cell_type != "property" or not cell.group:
            continue
        groups.setdefault(cell.group, []).append(cell)
    monopolies: list[list[Cell]] = []
    for cells in groups.values():
        owner = cells[0].owner_id
        if owner is None:
            continue
        if all(cell.owner_id == owner for cell in cells):
            monopolies.append(cells)
    return monopolies


def _desired_stage(state: GameState) -> str:
    buyables = _buyable_cells(state)
    unowned = sum(1 for cell in buyables if cell.owner_id is None)
    monopolies = _monopoly_groups(state)
    m = len(monopolies)
    houses = sum(cell.houses for cell in state.board)
    free_houses = max(0, int(state.rules.bank_houses) - houses)
    bankruptcies = sum(1 for player in state.players if player.bankrupt)

    max_monopoly_level = 0.0
    for cells in monopolies:
        levels = [_property_level(cell.houses, cell.hotels) for cell in cells]
        if levels:
            max_monopoly_level = max(max_monopoly_level, sum(levels) / len(levels))

    desired = "early"
    if m >= 1 or houses > 0 or unowned <= EARLY_TO_MID_UNOWNED:
        desired = "mid"
    if (
        bankruptcies > 0
        or free_houses <= LATE_FREE_HOUSES
        or max_monopoly_level >= LATE_MONOPOLY_LEVEL
    ):
        desired = "late"
    return desired


def game_stage(state: GameState) -> str:
    if state.stage_last_turn == state.turn_index:
        return state.stage
    state.stage_last_turn = state.turn_index

    desired = _desired_stage(state)
    current = state.stage
    desired_index = STAGE_ORDER.get(desired, 0)
    current_index = STAGE_ORDER.get(current, 0)

    if desired_index <= current_index:
        state.stage = current
        state.stage_candidate = current
        state.stage_candidate_ticks = 0
        return state.stage

    if state.stage_candidate != desired:
        state.stage_candidate = desired
        state.stage_candidate_ticks = 1
    else:
        state.stage_candidate_ticks += 1

    if state.stage_candidate_ticks >= STAGE_HYSTERESIS_TICKS:
        state.stage = desired
        state.stage_candidate = desired
        state.stage_candidate_ticks = 0
    return state.stage


def compute_cash_buffer(state: GameState, player: Player, params: BotParams) -> int:
    opponent_buildings = 0
    for cell in state.board:
        if cell.owner_id is None or cell.owner_id == player.player_id:
            continue
        opponent_buildings += cell.houses + cell.hotels * 5
    buffer_value = params.cash_buffer_base + params.cash_buffer_per_house * opponent_buildings
    return max(0, int(buffer_value))


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


def _estimate_rent(state: GameState, cell: Cell) -> float:
    if cell.owner_id is None:
        return 0.0
    owner = state.players[cell.owner_id]
    if owner.in_jail:
        return 0.0
    if cell.mortgaged:
        return 0.0
    if cell.cell_type == "property" and cell.rent_by_houses:
        level = _property_level(cell.houses, cell.hotels)
        level = max(0, min(level, len(cell.rent_by_houses) - 1))
        rent = float(cell.rent_by_houses[level])
        if level == 0 and cell.group and _owns_group(state, cell.owner_id, cell.group):
            rent *= 2
        return rent
    if cell.cell_type == "railroad" and cell.rent:
        owned = sum(
            1
            for c in state.board
            if c.cell_type == "railroad" and c.owner_id == cell.owner_id
        )
        idx = max(0, min(owned - 1, len(cell.rent) - 1))
        return float(cell.rent[idx])
    if cell.cell_type == "utility" and cell.rent_multiplier:
        owned = sum(
            1
            for c in state.board
            if c.cell_type == "utility" and c.owner_id == cell.owner_id
        )
        idx = max(0, min(owned - 1, len(cell.rent_multiplier) - 1))
        multiplier = float(cell.rent_multiplier[idx])
        return multiplier * 7.0
    return 0.0


def _max_opponent_rent(state: GameState, player_id: int) -> float:
    max_rent = 0.0
    for cell in state.board:
        if cell.owner_id is None or cell.owner_id == player_id:
            continue
        rent = _estimate_rent(state, cell)
        if rent > max_rent:
            max_rent = rent
    return max_rent


def _opponent_cash_metrics(state: GameState, player_id: int) -> tuple[float, float]:
    opponent_cash = [
        opponent.money
        for opponent in state.players
        if opponent.player_id != player_id and not opponent.bankrupt
    ]
    if not opponent_cash:
        return 0.0, 0.0
    min_cash = min(opponent_cash)
    min_norm = min_cash / (state.players[player_id].money + 1)
    min_norm = min(2.0, max(0.0, min_norm))
    pressure = 1.0 - min(1.0, max(0.0, min_cash / 500.0))
    return min_norm, pressure


def _group_strength(state: GameState, group: str | None) -> float:
    if not group:
        return 0.0
    group_cells = _group_cells(state, group)
    if not group_cells:
        return 0.0
    base = 0.0
    for cell in group_cells:
        if cell.rent_by_houses:
            base += float(cell.rent_by_houses[0])
    return base / max(1.0, len(group_cells)) / 10.0


def estimate_asset_value(state: GameState, player: Player, cell: Cell, params: BotParams) -> float:
    base = float(cell.price or 0)
    if cell.cell_type == "property" and cell.rent_by_houses:
        base += float(cell.rent_by_houses[0]) * 5
    elif cell.cell_type == "railroad" and cell.rent:
        base += float(cell.rent[0]) * 4
    elif cell.cell_type == "utility" and cell.rent_multiplier:
        base += float(cell.price or 0) * 0.5
    return base


def _score_action(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def _normalize_auction_increments(increments: list[int] | None) -> list[int]:
    if not increments:
        return [1]
    cleaned = sorted({int(value) for value in increments if int(value) > 0})
    return cleaned or [1]


def normalize_auction_price(current_price: int, increments: list[int]) -> int:
    incs = _normalize_auction_increments(increments)
    if not incs:
        return current_price
    min_increment = incs[0]
    if min_increment <= 0:
        return current_price
    remainder = current_price % min_increment
    if remainder == 0:
        return current_price
    return current_price + (min_increment - remainder)


def auction_increment_for_remaining(remaining: int, increments: list[int]) -> int:
    incs = _normalize_auction_increments(increments)
    if not incs:
        return 0
    if len(incs) == 1:
        step = incs[0]
        return step if remaining >= step else 0
    if len(incs) == 2:
        small, large = incs[0], incs[1]
        if remaining >= large * 2:
            return large
        if remaining >= small:
            return small
        return 0
    small = incs[0]
    mid = incs[len(incs) // 2]
    large = incs[-1]
    if remaining >= large * 2:
        return large
    if remaining >= mid * 2:
        return mid
    if remaining >= small:
        return small
    return 0


def choose_auction_bid(
    target_max: int,
    current_price: int,
    increments: list[int],
) -> int:
    incs = _normalize_auction_increments(increments)
    current_price = normalize_auction_price(current_price, incs)
    min_increment = incs[0] if incs else 1
    if target_max < min_increment:
        return 0
    if target_max <= 0 or current_price >= target_max:
        return 0
    remaining = target_max - current_price
    step = auction_increment_for_remaining(remaining, incs)
    if step <= 0:
        return 0
    bid = current_price + step
    if bid > target_max:
        return 0
    return bid


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
    stage = game_stage(state)
    weights = params.weights["auction"][stage]
    max_bid = min(player.money, int(player.money * params.max_bid_fraction))
    min_bid = current_price + min_increment
    if max_bid < min_bid:
        return 0

    candidates: list[int] = []
    spread = max(1, (max_bid - min_bid) // 3)
    for bid in (min_bid, min_bid + spread, min_bid + spread * 2, max_bid):
        bid = min(max_bid, max(min_bid, bid))
        if bid not in candidates:
            candidates.append(bid)

    best_bid = 0
    best_score = 0.0
    start_cash = max(1, state.rules.starting_cash)
    risk = _max_opponent_rent(state, player.player_id) / max(1.0, player.money)
    threat_self = positional_threat_self(state, player.player_id) / start_cash
    threat_others = positional_threat_others(state, player.player_id) / start_cash
    opponent_cash_min_norm, opponent_cash_pressure = _opponent_cash_metrics(
        state, player.player_id
    )
    rr_synergy = railroad_synergy(state, player.player_id, cell)
    util_synergy = utility_synergy(state, player.player_id, cell)

    for bid in candidates:
        cash_after = player.money - bid
        if cash_after < 0:
            continue
        group_strength = _group_strength(state, cell.group)
        group_heat_score = group_heat(state, cell.group)
        group_heat_delta = group_heat_vs_base(state, cell.group)
        cell_heat_score = cell_heat(state, cell.index)
        landing_prob = landing_prob_group(state, cell.group)
        jail_heat = jail_exit_heat_group(state, cell.group)
        completes = 0.0
        blocks = 0.0
        owned_in_group = 0.0
        if cell.cell_type == "property" and cell.group:
            group_cells = _group_cells(state, cell.group)
            owned_by_player = [c for c in group_cells if c.owner_id == player.player_id]
            owned_in_group = len(owned_by_player) / max(1, len(group_cells))
            if len(owned_by_player) == len(group_cells) - 1:
                completes = 1.0
            for opponent in state.players:
                if opponent.player_id == player.player_id or opponent.bankrupt:
                    continue
                if all(
                    c.owner_id == opponent.player_id or c.index == cell.index
                    for c in group_cells
                ):
                    blocks = 1.0
                    break
        features = {
            "bias": 1.0,
            "base_value": float(cell.price) / 100.0,
            "group_strength": group_strength,
            "group_heat": group_heat_score,
            "group_heat_vs_base": group_heat_delta,
            "cell_heat": cell_heat_score,
            "landing_prob_group": landing_prob,
            "jail_exit_heat_group": jail_heat,
            "completes_monopoly": completes,
            "blocks_opponent_monopoly": blocks,
            "cash_after_bid": cash_after / start_cash,
            "liquidity_ratio": cash_after / max(1.0, player.money),
            "risk_of_ruin": risk,
            "is_street": 1.0 if cell.cell_type == "property" else 0.0,
            "is_railroad": 1.0 if cell.cell_type == "railroad" else 0.0,
            "is_utility": 1.0 if cell.cell_type == "utility" else 0.0,
            "owned_in_group": owned_in_group,
            "positional_threat_self": threat_self,
            "positional_threat_others": threat_others,
            "railroad_synergy": rr_synergy,
            "utility_synergy": util_synergy,
            "opponent_cash_min_norm": opponent_cash_min_norm,
            "opponent_cash_pressure": opponent_cash_pressure,
        }
        score = _score_action(weights, features)
        if score > best_score:
            best_score = score
            best_bid = bid

    if best_bid <= 0 or best_score <= 0.0:
        return 0
    return best_bid


def decide_build_actions(state: GameState, player: Player, params: BotParams) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if player.bankrupt:
        return actions
    stage = game_stage(state)
    weights = params.weights["build"][stage]
    start_cash = max(1, state.rules.starting_cash)
    enemy_threat = _max_opponent_rent(state, player.player_id) / max(1.0, player.money)
    threat_others = positional_threat_others(state, player.player_id) / start_cash
    _, opponent_cash_pressure = _opponent_cash_metrics(state, player.player_id)

    available_cash = player.money
    local_houses = {cell.index: cell.houses for cell in state.board}
    local_hotels = {cell.index: cell.hotels for cell in state.board}

    planned = 0
    max_plans = 20
    while planned < max_plans:
        candidates: list[tuple[float, Cell]] = []
        total_houses = sum(local_houses.values())
        total_hotels = sum(local_hotels.values())
        bank_houses_ratio = (state.rules.bank_houses - total_houses) / max(1, state.rules.bank_houses)
        bank_hotels_ratio = (state.rules.bank_hotels - total_hotels) / max(1, state.rules.bank_hotels)
        house_scarcity_value = 1.0 - bank_houses_ratio
        hotel_scarcity_value = 1.0 - bank_hotels_ratio

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
            if not _can_build_on_cell(state, player.player_id, cell, local_houses[cell.index], local_hotels[cell.index]):
                continue

            current_level = _property_level(local_houses[cell.index], local_hotels[cell.index])
            next_level = min(5, current_level + 1)
            if not cell.rent_by_houses:
                continue
            rent_current = float(cell.rent_by_houses[min(current_level, len(cell.rent_by_houses) - 1)])
            rent_next = float(cell.rent_by_houses[min(next_level, len(cell.rent_by_houses) - 1)])
            rent_delta = max(0.0, rent_next - rent_current)
            cost = int(cell.house_cost or 0)
            if available_cash - cost < 0:
                continue
            houses_to_take = 1 if current_level < 4 else 0
            hotels_to_take = 1 if current_level == 4 else 0

            features = {
                "bias": 1.0,
                "roi": rent_delta / max(1.0, cost),
                "rent_delta": rent_delta / 100.0,
                "group_strength": _group_strength(state, cell.group),
                "group_heat": group_heat(state, cell.group),
                "group_heat_vs_base": group_heat_vs_base(state, cell.group),
                "cell_heat": cell_heat(state, cell.index),
                "landing_prob_group": landing_prob_group(state, cell.group),
                "jail_exit_heat_group": jail_exit_heat_group(state, cell.group),
                "cash_after_build": (available_cash - cost) / start_cash,
                "enemy_threat": enemy_threat,
                "level_norm": current_level / 4.0,
                "to_hotel": 1.0 if current_level == 4 else 0.0,
                "target_three": 1.0 if next_level == 3 else 0.0,
                "has_monopoly": 1.0,
                "bank_houses_ratio": bank_houses_ratio,
                "bank_hotels_ratio": bank_hotels_ratio,
                "positional_threat_others": threat_others,
                "house_scarcity": house_scarcity_value,
                "hotel_scarcity": hotel_scarcity_value,
                "denial_value": denial_value(
                    houses_to_take, hotels_to_take, house_scarcity_value, hotel_scarcity_value
                ),
                "opponent_cash_pressure": opponent_cash_pressure,
            }
            score = _score_action(weights, features)
            candidates.append((score, cell))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_cell = candidates[0]
        if best_score <= 0:
            break
        cost = best_cell.house_cost or 0
        if available_cash - cost < 0:
            break
        actions.append({"action": "build", "cell_index": best_cell.index})
        available_cash -= cost
        if _property_level(local_houses[best_cell.index], local_hotels[best_cell.index]) < 4:
            local_houses[best_cell.index] += 1
        else:
            local_hotels[best_cell.index] = 1
            local_houses[best_cell.index] = 0
        planned += 1

    return actions


def decide_liquidation(
    state: GameState, player: Player, debt: int, params: BotParams
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if debt <= 0:
        return actions
    stage = game_stage(state)
    weights = params.weights["mortgage"][stage]
    start_cash = max(1, state.rules.starting_cash)
    threat_self = positional_threat_self(state, player.player_id) / start_cash

    available_cash = player.money
    local_houses = {cell.index: cell.houses for cell in state.board}
    local_hotels = {cell.index: cell.hotels for cell in state.board}
    local_mortgaged = {cell.index: cell.mortgaged for cell in state.board}

    guard = 0
    while available_cash < debt and guard < 200:
        guard += 1
        candidates: list[tuple[float, dict[str, Any], Cell]] = []
        cash_needed = max(0, debt - available_cash) / 100.0

        for cell in state.board:
            if cell.owner_id != player.player_id:
                continue
            if local_hotels[cell.index] > 0 or local_houses[cell.index] > 0:
                features = {
                    "bias": 1.0,
                    "asset_value": estimate_asset_value(state, player, cell, params) / 100.0,
                    "low_value": 1.0 / (1.0 + estimate_asset_value(state, player, cell, params) / 100.0),
                    "breaks_monopoly": 1.0 if cell.group and _owns_group(state, player.player_id, cell.group) else 0.0,
                    "has_buildings": 1.0,
                    "group_heat": group_heat(state, cell.group),
                    "group_heat_vs_base": group_heat_vs_base(state, cell.group),
                    "cell_heat": cell_heat(state, cell.index),
                    "landing_prob_group": landing_prob_group(state, cell.group),
                    "jail_exit_heat_group": jail_exit_heat_group(state, cell.group),
                    "is_railroad": 1.0 if cell.cell_type == "railroad" else 0.0,
                    "is_utility": 1.0 if cell.cell_type == "utility" else 0.0,
                    "cash_needed": cash_needed,
                    "action_sell_building": 1.0,
                    "action_mortgage": 0.0,
                    "positional_threat_self": threat_self,
                }
                score = _score_action(weights, features)
                candidates.append((score, {"action": "sell_building", "cell_index": cell.index}, cell))
            if local_mortgaged[cell.index]:
                continue
            if local_hotels[cell.index] > 0 or local_houses[cell.index] > 0:
                continue
            mortgage_value = int(cell.mortgage_value or 0)
            if mortgage_value <= 0:
                continue
            features = {
                "bias": 1.0,
                "asset_value": estimate_asset_value(state, player, cell, params) / 100.0,
                "low_value": 1.0 / (1.0 + estimate_asset_value(state, player, cell, params) / 100.0),
                "breaks_monopoly": 1.0 if cell.group and _owns_group(state, player.player_id, cell.group) else 0.0,
                "has_buildings": 0.0,
                "group_heat": group_heat(state, cell.group),
                "group_heat_vs_base": group_heat_vs_base(state, cell.group),
                "cell_heat": cell_heat(state, cell.index),
                "landing_prob_group": landing_prob_group(state, cell.group),
                "jail_exit_heat_group": jail_exit_heat_group(state, cell.group),
                "is_railroad": 1.0 if cell.cell_type == "railroad" else 0.0,
                "is_utility": 1.0 if cell.cell_type == "utility" else 0.0,
                "cash_needed": cash_needed,
                "action_sell_building": 0.0,
                "action_mortgage": 1.0,
                "positional_threat_self": threat_self,
            }
            score = _score_action(weights, features)
            candidates.append((score, {"action": "mortgage", "cell_index": cell.index}, cell))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, action, cell = candidates[0]
        if best_score <= 0:
            break
        if action["action"] == "sell_building":
            refund = int((cell.house_cost or 0) / 2)
            if local_hotels[cell.index] > 0:
                local_hotels[cell.index] = 0
                local_houses[cell.index] = 4
            else:
                local_houses[cell.index] = max(0, local_houses[cell.index] - 1)
            available_cash += refund
        elif action["action"] == "mortgage":
            local_mortgaged[cell.index] = True
            available_cash += int(cell.mortgage_value or 0)
        actions.append(action)

    return actions


def decide_jail_exit(state: GameState, player: Player, params: BotParams) -> str:
    stage = game_stage(state)
    weights = params.weights["jail"][stage]
    fine = state.rules.jail_fine
    has_card = bool(player.get_out_of_jail_cards)
    danger = _max_opponent_rent(state, player.player_id) / max(1.0, player.money)
    start_cash = max(1, state.rules.starting_cash)
    lost_income_if_stay = positional_threat_others(state, player.player_id) / start_cash
    saved_risk_if_stay = positional_threat_self(state, player.player_id) / start_cash

    candidates: list[tuple[float, str]] = []

    if player.money >= fine:
        features_pay = {
            "bias": 1.0,
            "has_card": 1.0 if has_card else 0.0,
            "cash_after_pay": (player.money - fine) / start_cash,
            "jail_turns": player.jail_turns / 3.0,
            "danger": danger,
            "action_pay": 1.0,
            "action_use_card": 0.0,
            "action_roll": 0.0,
            "danger_if_pay": danger,
            "danger_if_use_card": 0.0,
            "lost_income_if_stay": 0.0,
            "saved_risk_if_stay": 0.0,
        }
        candidates.append((_score_action(weights, features_pay), "pay"))

    if has_card:
        features_card = {
            "bias": 1.0,
            "has_card": 1.0,
            "cash_after_pay": player.money / start_cash,
            "jail_turns": player.jail_turns / 3.0,
            "danger": danger,
            "action_pay": 0.0,
            "action_use_card": 1.0,
            "action_roll": 0.0,
            "danger_if_pay": 0.0,
            "danger_if_use_card": danger,
            "lost_income_if_stay": 0.0,
            "saved_risk_if_stay": 0.0,
        }
        candidates.append((_score_action(weights, features_card), "use_card"))

    features_roll = {
        "bias": 1.0,
        "has_card": 1.0 if has_card else 0.0,
        "cash_after_pay": player.money / start_cash,
        "jail_turns": player.jail_turns / 3.0,
        "danger": danger,
        "action_pay": 0.0,
        "action_use_card": 0.0,
        "action_roll": 1.0,
        "danger_if_pay": 0.0,
        "danger_if_use_card": 0.0,
        "lost_income_if_stay": lost_income_if_stay,
        "saved_risk_if_stay": saved_risk_if_stay,
    }
    candidates.append((_score_action(weights, features_roll), "roll"))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]

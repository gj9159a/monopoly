from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

from .models import Cell, GameState

DICE_SUM_PROBS: tuple[tuple[int, float], ...] = (
    (2, 1 / 36),
    (3, 2 / 36),
    (4, 3 / 36),
    (5, 4 / 36),
    (6, 5 / 36),
    (7, 6 / 36),
    (8, 5 / 36),
    (9, 4 / 36),
    (10, 3 / 36),
    (11, 2 / 36),
    (12, 1 / 36),
)

DICE_OUTCOMES: tuple[tuple[int, int, float], ...] = tuple(
    (die1, die2, 1 / 36) for die1 in range(1, 7) for die2 in range(1, 7)
)

JAIL_BIAS_WEIGHT = 0.25
HEATMAP_MAX_ITERS = 5000
HEATMAP_TOL = 1e-12


@dataclass(frozen=True)
class Heatmap:
    rules_hash: str
    cell: tuple[float, ...]
    group: dict[str, float]
    group_base: dict[str, float]


def dice_sum_probs_2d6() -> Iterable[tuple[int, float]]:
    return DICE_SUM_PROBS


def _group_cells(state: GameState, group: str | None) -> list[int]:
    if not group:
        return []
    return [idx for idx, cell in enumerate(state.board) if cell.group == group]


def _jail_index(state: GameState) -> int | None:
    for idx, cell in enumerate(state.board):
        if cell.cell_type == "jail":
            return idx
    return None


def jail_exit_heat_group(state: GameState, group: str | None) -> float:
    cells = _group_cells(state, group)
    if not cells:
        return 0.0
    jail_idx = _jail_index(state)
    if jail_idx is None:
        return 0.0
    board_size = len(state.board)
    total = 0.0
    for roll, prob in DICE_SUM_PROBS:
        idx = (jail_idx + roll) % board_size
        if idx in cells:
            total += prob
    return total


def landing_prob_group(state: GameState, group: str | None) -> float:
    cells = _group_cells(state, group)
    if not cells:
        return 0.0
    board_size = len(state.board)
    base_prob = len(cells) / max(1.0, board_size)
    jail_bias = jail_exit_heat_group(state, group)
    return (1.0 - JAIL_BIAS_WEIGHT) * base_prob + JAIL_BIAS_WEIGHT * jail_bias


_HEATMAP_CACHE: dict[str, Heatmap] = {}
_HEATMAP_LOGGED: set[str] = set()


def _heatmap_rules_hash(state: GameState) -> str:
    payload = {
        "rules": asdict(state.rules),
        "board": [
            {"cell_type": cell.cell_type, "group": cell.group} for cell in state.board
        ],
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_heatmap(state: GameState, rules_hash: str) -> Heatmap:
    board_size = len(state.board)
    jail_idx = _jail_index(state)
    if jail_idx is None:
        raise ValueError("На поле нет клетки jail для heatmap")
    go_to_jail = {idx for idx, cell in enumerate(state.board) if cell.cell_type == "go_to_jail"}
    group_cells: dict[str, list[int]] = {}
    for idx, cell in enumerate(state.board):
        if not cell.group:
            continue
        group_cells.setdefault(cell.group, []).append(idx)

    states: list[tuple[int, int, int]] = []
    for pos in range(board_size):
        for doubles in range(3):
            states.append((pos, 0, doubles))
    for jail_state in range(1, 4):
        states.append((jail_idx, jail_state, 0))

    state_index = {state: idx for idx, state in enumerate(states)}
    transitions: list[list[tuple[int, float]]] = [[] for _ in states]

    def after_move(pos: int, doubles_count: int) -> tuple[int, int, int]:
        if pos in go_to_jail:
            return (jail_idx, 1, 0)
        return (pos, 0, doubles_count)

    for idx, (pos, jail_state, doubles_count) in enumerate(states):
        next_probs: dict[int, float] = {}
        if jail_state == 0:
            for die1, die2, prob in DICE_OUTCOMES:
                total = die1 + die2
                is_double = die1 == die2
                if is_double and doubles_count >= 2:
                    next_state = (jail_idx, 1, 0)
                else:
                    next_pos = (pos + total) % board_size
                    next_doubles = doubles_count + 1 if is_double else 0
                    next_state = after_move(next_pos, next_doubles)
                next_idx = state_index[next_state]
                next_probs[next_idx] = next_probs.get(next_idx, 0.0) + prob
        else:
            for die1, die2, prob in DICE_OUTCOMES:
                total = die1 + die2
                is_double = die1 == die2
                if is_double or jail_state >= 3:
                    next_pos = (jail_idx + total) % board_size
                    next_state = after_move(next_pos, 0)
                else:
                    next_state = (jail_idx, jail_state + 1, 0)
                next_idx = state_index[next_state]
                next_probs[next_idx] = next_probs.get(next_idx, 0.0) + prob
        transitions[idx] = list(next_probs.items())

    state_count = len(states)
    dist = [1.0 / state_count] * state_count
    for _ in range(HEATMAP_MAX_ITERS):
        updated = [0.0] * state_count
        for src_idx, edges in enumerate(transitions):
            mass = dist[src_idx]
            if mass == 0:
                continue
            for dst_idx, prob in edges:
                updated[dst_idx] += mass * prob
        total = sum(updated)
        if total > 0:
            updated = [value / total for value in updated]
        delta = max(abs(updated[i] - dist[i]) for i in range(state_count))
        dist = updated
        if delta < HEATMAP_TOL:
            break

    heat_cell = [0.0] * board_size
    for (pos, _jail, _doubles), prob in zip(states, dist, strict=False):
        heat_cell[pos] += prob

    group_heat = {group: sum(heat_cell[idx] for idx in cells) for group, cells in group_cells.items()}
    group_base = {group: len(cells) / max(1.0, board_size) for group, cells in group_cells.items()}

    return Heatmap(
        rules_hash=rules_hash,
        cell=tuple(heat_cell),
        group=group_heat,
        group_base=group_base,
    )


def heatmap_for_state(state: GameState) -> Heatmap:
    rules_hash = _heatmap_rules_hash(state)
    cached = _HEATMAP_CACHE.get(rules_hash)
    if cached is None:
        cached = _build_heatmap(state, rules_hash)
        _HEATMAP_CACHE[rules_hash] = cached
        if rules_hash not in _HEATMAP_LOGGED:
            print(f"heatmap: computed (rules_hash={rules_hash})")
            _HEATMAP_LOGGED.add(rules_hash)
        return cached
    if rules_hash not in _HEATMAP_LOGGED:
        print(f"heatmap: cached (rules_hash={rules_hash})")
        _HEATMAP_LOGGED.add(rules_hash)
    return cached


def cell_heat(state: GameState, cell_index: int) -> float:
    heatmap = heatmap_for_state(state)
    if cell_index < 0 or cell_index >= len(heatmap.cell):
        return 0.0
    return heatmap.cell[cell_index]


def group_heat(state: GameState, group: str | None) -> float:
    if not group:
        return 0.0
    heatmap = heatmap_for_state(state)
    return heatmap.group.get(group, 0.0)


def group_heat_vs_base(state: GameState, group: str | None) -> float:
    if not group:
        return 0.0
    heatmap = heatmap_for_state(state)
    return heatmap.group.get(group, 0.0) - heatmap.group_base.get(group, 0.0)


def _owns_group(state: GameState, owner_id: int, group: str | None) -> bool:
    if not group:
        return False
    for cell in state.board:
        if cell.group == group and cell.owner_id != owner_id:
            return False
    return True


def estimate_cell_payment(state: GameState, mover_id: int, cell_index: int) -> int:
    cell = state.board[int(cell_index)]
    if cell.cell_type == "tax":
        return int(cell.tax_amount or 0)
    if cell.cell_type not in {"property", "railroad", "utility"}:
        return 0
    if cell.owner_id is None or cell.owner_id == mover_id:
        return 0
    if cell.mortgaged:
        return 0
    owner = state.players[cell.owner_id]
    if state.rules.hr2_no_rent_in_jail and owner.in_jail:
        return 0
    if cell.cell_type == "property":
        if not cell.rent_by_houses:
            return 0
        level = cell.houses + cell.hotels * 5
        level = max(0, min(level, len(cell.rent_by_houses) - 1))
        rent = float(cell.rent_by_houses[level])
        if level == 0 and cell.group and _owns_group(state, cell.owner_id, cell.group):
            rent *= 2
        return int(rent)
    if cell.cell_type == "railroad":
        if not cell.rent:
            return 0
        owned = sum(
            1
            for other in state.board
            if other.cell_type == "railroad" and other.owner_id == cell.owner_id
        )
        idx = max(0, min(owned - 1, len(cell.rent) - 1))
        return int(cell.rent[idx])
    if cell.cell_type == "utility":
        if not cell.rent_multiplier:
            return 0
        owned = sum(
            1
            for other in state.board
            if other.cell_type == "utility" and other.owner_id == cell.owner_id
        )
        multiplier = cell.rent_multiplier[1] if owned >= 2 else cell.rent_multiplier[0]
        return int(multiplier * 7)
    return 0


def positional_threat_self(state: GameState, player_id: int) -> float:
    player = state.players[player_id]
    board_size = len(state.board)
    total = 0.0
    for roll, prob in DICE_SUM_PROBS:
        idx = (player.position + roll) % board_size
        total += prob * estimate_cell_payment(state, player_id, idx)
    return total


def positional_threat_others(state: GameState, player_id: int) -> float:
    total = 0.0
    board_size = len(state.board)
    for opponent in state.players:
        if opponent.player_id == player_id or opponent.bankrupt:
            continue
        for roll, prob in DICE_SUM_PROBS:
            idx = (opponent.position + roll) % board_size
            if state.board[idx].owner_id != player_id:
                continue
            total += prob * estimate_cell_payment(state, opponent.player_id, idx)
    return total


def house_scarcity(state: GameState) -> float:
    max_houses = max(1, int(state.rules.bank_houses))
    total_houses = sum(cell.houses for cell in state.board)
    used = min(max_houses, max(0, total_houses))
    return min(1.0, max(0.0, used / max_houses))


def hotel_scarcity(state: GameState) -> float:
    max_hotels = max(1, int(state.rules.bank_hotels))
    total_hotels = sum(cell.hotels for cell in state.board)
    used = min(max_hotels, max(0, total_hotels))
    return min(1.0, max(0.0, used / max_hotels))


def denial_value(
    houses_to_take: int,
    hotels_to_take: int,
    house_scarcity_value: float,
    hotel_scarcity_value: float,
) -> float:
    return houses_to_take * house_scarcity_value + hotels_to_take * hotel_scarcity_value


def railroad_synergy(state: GameState, player_id: int, cell: Cell) -> float:
    if cell.cell_type != "railroad":
        return 0.0
    owned = sum(
        1
        for other in state.board
        if other.cell_type == "railroad" and other.owner_id == player_id
    )
    return owned / 4.0


def utility_synergy(state: GameState, player_id: int, cell: Cell) -> float:
    if cell.cell_type != "utility":
        return 0.0
    owned = sum(
        1
        for other in state.board
        if other.cell_type == "utility" and other.owner_id == player_id
    )
    return owned / 2.0

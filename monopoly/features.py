from __future__ import annotations

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


def dice_sum_probs_2d6() -> Iterable[tuple[int, float]]:
    return DICE_SUM_PROBS


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

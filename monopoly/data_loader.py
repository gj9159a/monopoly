from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import Cell, Rules

VALID_CELL_TYPES = {
    "go",
    "property",
    "community",
    "chance",
    "tax",
    "railroad",
    "utility",
    "jail",
    "free_parking",
    "go_to_jail",
}


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл данных: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_rules(path: Path) -> Rules:
    data = _load_yaml(path)
    required = [
        "hr1_always_auction",
        "hr2_no_rent_in_jail",
        "free_parking_empty",
        "no_trades",
        "go_salary",
        "jail_fine",
        "starting_cash",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"В rules.yaml нет ключа '{key}'")
    return Rules(
        hr1_always_auction=bool(data["hr1_always_auction"]),
        hr2_no_rent_in_jail=bool(data["hr2_no_rent_in_jail"]),
        free_parking_empty=bool(data["free_parking_empty"]),
        no_trades=bool(data["no_trades"]),
        go_salary=int(data["go_salary"]),
        jail_fine=int(data["jail_fine"]),
        starting_cash=int(data["starting_cash"]),
    )


def load_board(path: Path) -> list[Cell]:
    data = _load_yaml(path)
    if not isinstance(data, list):
        raise ValueError("board.yaml должен содержать список клеток")
    if len(data) != 40:
        raise ValueError("board.yaml должен содержать 40 клеток")

    cells: list[Cell] = []
    for idx, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"Клетка {idx} должна быть объектом")
        if raw.get("index") != idx:
            raise ValueError(f"Клетка {idx} должна иметь index={idx}")
        cell_type = raw.get("type")
        if cell_type not in VALID_CELL_TYPES:
            raise ValueError(f"Неизвестный тип клетки: {cell_type}")
        name = raw.get("name")
        if not name:
            raise ValueError(f"Клетка {idx} должна иметь name")

        cell = Cell(
            index=idx,
            name=str(name),
            cell_type=str(cell_type),
            group=raw.get("group"),
            price=raw.get("price"),
            rent=raw.get("rent"),
            house_cost=raw.get("house_cost"),
            mortgage=raw.get("mortgage"),
            rent_multiplier=raw.get("rent_multiplier"),
            tax_amount=raw.get("amount"),
        )

        _validate_cell(cell)
        cells.append(cell)
    return cells


def _validate_cell(cell: Cell) -> None:
    if cell.cell_type == "property":
        _require_fields(cell, ["group", "price", "rent", "house_cost", "mortgage"])
        if not isinstance(cell.rent, list) or len(cell.rent) != 6:
            raise ValueError(f"Клетка {cell.index}: rent должен иметь 6 значений")
    elif cell.cell_type == "railroad":
        _require_fields(cell, ["group", "price", "rent", "mortgage"])
        if not isinstance(cell.rent, list) or len(cell.rent) != 4:
            raise ValueError(f"Клетка {cell.index}: rent должен иметь 4 значения")
    elif cell.cell_type == "utility":
        _require_fields(cell, ["group", "price", "rent_multiplier", "mortgage"])
        if not isinstance(cell.rent_multiplier, list) or len(cell.rent_multiplier) != 2:
            raise ValueError(f"Клетка {cell.index}: rent_multiplier должен иметь 2 значения")
    elif cell.cell_type == "tax":
        if cell.tax_amount is None:
            raise ValueError(f"Клетка {cell.index}: налог должен иметь amount")


def _require_fields(cell: Cell, fields: list[str]) -> None:
    for field_name in fields:
        if getattr(cell, field_name) is None:
            raise ValueError(f"Клетка {cell.index}: обязательное поле {field_name} не задано")

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import Card, Cell, Rules

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
        "interest_rate",
        "bank_houses",
        "bank_hotels",
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
        interest_rate=float(data["interest_rate"]),
        bank_houses=int(data["bank_houses"]),
        bank_hotels=int(data["bank_hotels"]),
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
            rent_by_houses=raw.get("rent_by_houses"),
            house_cost=raw.get("house_cost"),
            mortgage=raw.get("mortgage"),
            mortgage_value=raw.get("mortgage_value"),
            rent_multiplier=raw.get("rent_multiplier"),
            tax_amount=raw.get("amount"),
        )

        if cell.mortgage_value is None and cell.mortgage is not None:
            cell.mortgage_value = cell.mortgage
        if cell.mortgage_value is None and cell.price is not None:
            cell.mortgage_value = int(cell.price / 2)
        if cell.rent_by_houses is None and cell.rent is not None:
            cell.rent_by_houses = cell.rent

        _validate_cell(cell)
        cells.append(cell)
    return cells


def load_cards(path: Path, deck: str) -> list[Card]:
    data = _load_yaml(path)
    if not isinstance(data, list):
        raise ValueError(f"{path.name} должен содержать список карточек")
    overrides = _load_card_text_overrides(path.parent / "cards_texts_ru_official.yaml")
    cards: list[Card] = []
    for idx, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"Карточка {idx} должна быть объектом")
        card_id = raw.get("id")
        text_ru = raw.get("text_ru")
        effect = raw.get("effect")
        if not card_id or not text_ru:
            raise ValueError(f"Карточка {idx} должна иметь id и text_ru")
        if not isinstance(effect, dict) or "type" not in effect:
            raise ValueError(f"Карточка {idx} должна иметь effect с type")
        if str(card_id) in overrides:
            text_ru = overrides[str(card_id)]
        cards.append(
            Card(
                card_id=str(card_id),
                text_ru=str(text_ru),
                effect={k: v for k, v in effect.items()},
                deck=deck,
            )
        )
    if not cards:
        raise ValueError(f"{path.name} должен содержать хотя бы одну карточку")
    return cards


def _load_card_text_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("cards_texts_ru_official.yaml должен содержать словарь id -> text_ru")
    result: dict[str, str] = {}
    for key, value in data.items():
        if key is None or value is None:
            continue
        result[str(key)] = str(value)
    return result


def _validate_cell(cell: Cell) -> None:
    if cell.cell_type == "property":
        _require_fields(cell, ["group", "price", "rent_by_houses", "house_cost", "mortgage_value"])
        if not isinstance(cell.rent_by_houses, list) or len(cell.rent_by_houses) != 6:
            raise ValueError(f"Клетка {cell.index}: rent_by_houses должен иметь 6 значений")
    elif cell.cell_type == "railroad":
        _require_fields(cell, ["group", "price", "rent", "mortgage_value"])
        if not isinstance(cell.rent, list) or len(cell.rent) != 4:
            raise ValueError(f"Клетка {cell.index}: rent должен иметь 4 значения")
    elif cell.cell_type == "utility":
        _require_fields(cell, ["group", "price", "rent_multiplier", "mortgage_value"])
        if not isinstance(cell.rent_multiplier, list) or len(cell.rent_multiplier) != 2:
            raise ValueError(f"Клетка {cell.index}: rent_multiplier должен иметь 2 значения")
    elif cell.cell_type == "tax":
        if cell.tax_amount is None:
            raise ValueError(f"Клетка {cell.index}: налог должен иметь amount")


def _require_fields(cell: Cell, fields: list[str]) -> None:
    for field_name in fields:
        if getattr(cell, field_name) is None:
            raise ValueError(f"Клетка {cell.index}: обязательное поле {field_name} не задано")

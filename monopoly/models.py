from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Card:
    card_id: str
    text_ru: str
    effect: dict[str, Any]
    deck: str


@dataclass(slots=True)
class DeckState:
    draw_pile: list[Card] = field(default_factory=list)
    discard: list[Card] = field(default_factory=list)


@dataclass(slots=True)
class Event:
    type: str
    turn_index: int
    player_id: int | None
    msg_ru: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Rules:
    hr1_always_auction: bool
    hr2_no_rent_in_jail: bool
    free_parking_empty: bool
    no_trades: bool
    go_salary: int
    jail_fine: int
    starting_cash: int
    interest_rate: float
    bank_houses: int
    bank_hotels: int


@dataclass(slots=True)
class Cell:
    index: int
    name: str
    cell_type: str
    group: str | None = None
    price: int | None = None
    rent: list[int] | None = None
    rent_by_houses: list[int] | None = None
    house_cost: int | None = None
    mortgage: int | None = None
    mortgage_value: int | None = None
    rent_multiplier: list[int] | None = None
    tax_amount: int | None = None
    owner_id: int | None = None
    houses: int = 0
    hotels: int = 0
    mortgaged: bool = False


@dataclass(slots=True)
class Player:
    player_id: int
    name: str
    position: int = 0
    money: int = 0
    in_jail: bool = False
    jail_turns: int = 0
    doubles_count: int = 0
    bankrupt: bool = False
    properties: list[int] = field(default_factory=list)
    get_out_of_jail_cards: list[Card] = field(default_factory=list)


@dataclass(slots=True)
class GameState:
    seed: int
    rng: random.Random
    rules: Rules
    board: list[Cell]
    players: list[Player]
    turn_index: int = 0
    current_player: int = 0
    event_log: list[Event] = field(default_factory=list)
    game_over: bool = False
    winner_id: int | None = None
    decks: dict[str, DeckState] = field(default_factory=dict)

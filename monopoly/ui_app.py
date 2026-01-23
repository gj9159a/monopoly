from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass
class MockEvent:
    type: str
    msg_ru: str
    payload: dict[str, Any]


@dataclass
class MockPlayer:
    player_id: int
    name: str
    money: int
    position: int
    in_jail: bool
    properties: list[str]


@dataclass
class MockCell:
    name: str
    cell_type: str
    owner_id: int | None = None
    mortgaged: bool = False
    houses: int = 0
    hotels: int = 0


@dataclass
class MockState:
    turn_index: int
    board: list[MockCell]
    players: list[MockPlayer]
    events: list[MockEvent]


def _cell_catalog() -> list[tuple[str, str]]:
    return [
        ("Старт", "go"),
        ("Улица 1", "property"),
        ("Казна", "community"),
        ("Улица 2", "property"),
        ("Налог", "tax"),
        ("Вокзал 1", "railroad"),
        ("Улица 3", "property"),
        ("Шанс", "chance"),
        ("Улица 4", "property"),
        ("Улица 5", "property"),
        ("Тюрьма / В гостях", "jail"),
        ("Улица 6", "property"),
        ("Электростанция", "utility"),
        ("Улица 7", "property"),
        ("Улица 8", "property"),
        ("Вокзал 2", "railroad"),
        ("Улица 9", "property"),
        ("Казна", "community"),
        ("Улица 10", "property"),
        ("Улица 11", "property"),
        ("Бесплатная парковка", "free_parking"),
        ("Улица 12", "property"),
        ("Шанс", "chance"),
        ("Улица 13", "property"),
        ("Улица 14", "property"),
        ("Вокзал 3", "railroad"),
        ("Улица 15", "property"),
        ("Улица 16", "property"),
        ("Водоканал", "utility"),
        ("Улица 17", "property"),
        ("В тюрьму", "go_to_jail"),
        ("Улица 18", "property"),
        ("Улица 19", "property"),
        ("Казна", "community"),
        ("Улица 20", "property"),
        ("Вокзал 4", "railroad"),
        ("Шанс", "chance"),
        ("Улица 21", "property"),
        ("Налог", "tax"),
        ("Улица 22", "property"),
    ]


def _make_mock_state(num_players: int, seed: int) -> MockState:
    rng = random.Random(seed)
    cells = [MockCell(name, cell_type) for name, cell_type in _cell_catalog()]

    players: list[MockPlayer] = []
    for pid in range(num_players):
        players.append(
            MockPlayer(
                player_id=pid,
                name=f"Бот {pid + 1}",
                money=rng.randint(1200, 1700),
                position=rng.randint(0, 39),
                in_jail=rng.choice([False, False, False, True]),
                properties=[],
            )
        )

    property_positions = [i for i, cell in enumerate(cells) if cell.cell_type == "property"]
    rng.shuffle(property_positions)
    for idx, pos in enumerate(property_positions[: max(1, num_players * 2)]):
        owner = idx % num_players
        cells[pos].owner_id = owner
        cells[pos].houses = rng.choice([0, 0, 1, 2])
        players[owner].properties.append(cells[pos].name)

    events = [
        MockEvent(
            type="TURN_START",
            msg_ru="Старт хода: Бот 1",
            payload={"player_id": 0, "turn_index": 0},
        ),
        MockEvent(
            type="DICE_ROLL",
            msg_ru="Бот 1 бросил 5 и 3",
            payload={"player_id": 0, "dice": [5, 3]},
        ),
        MockEvent(
            type="MOVE",
            msg_ru="Бот 1 переместился на 8",
            payload={"player_id": 0, "position": 8},
        ),
        MockEvent(
            type="AUCTION_START",
            msg_ru="Запущен аукцион за Улица 4",
            payload={"cell": "Улица 4"},
        ),
        MockEvent(
            type="AUCTION_WIN",
            msg_ru="Бот 2 выиграл аукцион за 140",
            payload={"player_id": 1, "price": 140},
        ),
    ]

    return MockState(turn_index=0, board=cells, players=players, events=events)


def _perimeter_coords() -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for col in range(10, -1, -1):
        coords.append((10, col))
    for row in range(9, 0, -1):
        coords.append((row, 0))
    for col in range(0, 11):
        coords.append((0, col))
    for row in range(1, 10):
        coords.append((row, 10))
    return coords


def _render_board(board: list[MockCell], players: list[MockPlayer]) -> None:
    coords = _perimeter_coords()
    grid = [["" for _ in range(11)] for _ in range(11)]

    players_at = {pos: [] for pos in range(40)}
    for player in players:
        players_at[player.position].append(player)

    for idx, cell in enumerate(board):
        row, col = coords[idx]
        owner_text = ""
        if cell.owner_id is not None:
            owner_text = f"Вл: Бот {cell.owner_id + 1}"
        mort_text = "Ипотека" if cell.mortgaged else ""
        build_text = ""
        if cell.hotels:
            build_text = f"Отели: {cell.hotels}"
        elif cell.houses:
            build_text = f"Дома: {cell.houses}"
        tokens = " ".join([f"P{p.player_id + 1}" for p in players_at[idx]])
        players_text = f"Игроки: {tokens}" if tokens else ""
        parts = [f"<div class='name'>{cell.name}</div>"]
        for line in [owner_text, mort_text, build_text, players_text]:
            if line:
                parts.append(f"<div class='meta'>{line}</div>")
        grid[row][col] = "".join(parts)

    html_rows = []
    for row in grid:
        html_cells = [f"<div class='cell'>{cell}</div>" for cell in row]
        html_rows.append("".join(html_cells))
    html = f\"\"\"\n    <style>\n      .board {{\n        display: grid;\n        grid-template-columns: repeat(11, minmax(60px, 1fr));\n        grid-auto-rows: 70px;\n        gap: 2px;\n        background: #e6e0d6;\n        padding: 6px;\n        border-radius: 10px;\n      }}\n      .cell {{\n        background: #f7f2ea;\n        border: 1px solid #c9bfae;\n        padding: 4px;\n        font-size: 10px;\n        line-height: 1.1;\n        overflow: hidden;\n      }}\n      .name {{\n        font-weight: 700;\n        font-size: 11px;\n      }}\n      .meta {{\n        color: #4a3e2d;\n      }}\n    </style>\n    <div class='board'>\n      {\"\".join(html_rows)}\n    </div>\n    \"\"\"\n    st.markdown(html, unsafe_allow_html=True)


def _render_players(players: list[MockPlayer]) -> None:
    table = [
        {
            "Игрок": player.name,
            "Деньги": player.money,
            "Позиция": player.position,
            "В тюрьме": "Да" if player.in_jail else "Нет",
            "Собственность": len(player.properties),
        }
        for player in players
    ]
    st.dataframe(table, use_container_width=True)

    for player in players:
        with st.expander(f"{player.name} — собственность"):
            if player.properties:
                st.write(", ".join(player.properties))
            else:
                st.write("Нет собственности")


def _render_log(events: list[MockEvent]) -> None:
    st.subheader("Лента событий")
    for event in reversed(events[-30:]):
        st.write(f"- {event.msg_ru}")


def main() -> None:
    st.set_page_config(page_title="Монополия — наблюдатель", layout="wide")
    st.title("Монополия — наблюдатель")
    st.caption("UI-черновик: данные пока моковые, движок появится на следующем шаге.")

    with st.sidebar:
        st.header("Управление")
        num_players = st.slider("Число ботов", min_value=2, max_value=6, value=4, step=1)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        st.multiselect(
            "Профили ботов",
            options=["Aggressive", "Conservative", "Builder", "CashSaver"],
            default=["Aggressive", "Builder"],
        )
        new_game = st.button("Новая игра", type="primary")
        step_once = st.button("Шаг")
        step_ten = st.button("+10 шагов")
        step_hundred = st.button("+100 шагов")
        st.button("Авто (скоро)")
        st.button("Сброс")

    if "state" not in st.session_state or new_game:
        st.session_state.state = _make_mock_state(num_players, seed)

    if step_once or step_ten or step_hundred:
        steps = 1 if step_once else 10 if step_ten else 100
        for _ in range(steps):
            st.session_state.state.turn_index += 1
            st.session_state.state.events.append(
                MockEvent(
                    type="TURN_START",
                    msg_ru=f"Сделан шаг {st.session_state.state.turn_index}",
                    payload={"turn_index": st.session_state.state.turn_index},
                )
            )

    state: MockState = st.session_state.state

    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.subheader("Поле")
        _render_board(state.board, state.players)
        _render_log(state.events)

    with col_right:
        st.subheader("Игроки")
        _render_players(state.players)


if __name__ == "__main__":
    main()

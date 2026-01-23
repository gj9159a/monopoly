from __future__ import annotations

import streamlit as st

from .bots import PROFILES
from .engine import create_engine
from .models import Cell, Event, GameState, Player


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


def _render_board(board: list[Cell], players: list[Player]) -> None:
    coords = _perimeter_coords()
    grid = [["" for _ in range(11)] for _ in range(11)]

    players_at = {pos: [] for pos in range(40)}
    for player in players:
        players_at[player.position].append(player)

    for idx, cell in enumerate(board):
        row, col = coords[idx]
        owner_text = ""
        if cell.owner_id is not None:
            owner_name = players[cell.owner_id].name if cell.owner_id < len(players) else "?"
            owner_text = f"Вл: {owner_name}"
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
    html = f"""
    <style>
      .board {{
        display: grid;
        grid-template-columns: repeat(11, minmax(60px, 1fr));
        grid-auto-rows: 70px;
        gap: 2px;
        background: #e6e0d6;
        padding: 6px;
        border-radius: 10px;
      }}
      .cell {{
        background: #f7f2ea;
        border: 1px solid #c9bfae;
        padding: 4px;
        font-size: 10px;
        line-height: 1.1;
        overflow: hidden;
      }}
      .name {{
        font-weight: 700;
        font-size: 11px;
      }}
      .meta {{
        color: #4a3e2d;
      }}
    </style>
    <div class='board'>
      {"".join(html_rows)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_players(players: list[Player], board: list[Cell]) -> None:
    owned_map: dict[int, list[str]] = {player.player_id: [] for player in players}
    for cell in board:
        if cell.owner_id is not None:
            owned_map[cell.owner_id].append(cell.name)
    table = [
        {
            "Игрок": player.name,
            "Деньги": player.money,
            "Позиция": player.position,
            "В тюрьме": "Да" if player.in_jail else "Нет",
            "Собственность": len(owned_map[player.player_id]),
        }
        for player in players
    ]
    st.dataframe(table, use_container_width=True)

    for player in players:
        with st.expander(f"{player.name} — собственность"):
            names = owned_map[player.player_id]
            if names:
                st.write(", ".join(names))
            else:
                st.write("Нет собственности")


def _render_log(events: list[Event]) -> None:
    st.subheader("Лента событий")
    for event in reversed(events[-30:]):
        st.write(f"- {event.msg_ru}")


def main() -> None:
    st.set_page_config(page_title="Монополия — наблюдатель", layout="wide")
    st.title("Монополия — наблюдатель")
    st.caption("UI подключен к движку: кнопки управляют симуляцией.")

    with st.sidebar:
        st.header("Управление")
        num_players = st.slider("Число ботов", min_value=2, max_value=6, value=4, step=1)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        profiles = st.multiselect(
            "Профили ботов",
            options=list(PROFILES.keys()),
            default=["Aggressive", "Builder"],
        )
        new_game = st.button("Новая игра", type="primary")
        step_once = st.button("Шаг")
        step_ten = st.button("+10 шагов")
        step_hundred = st.button("+100 шагов")
        st.button("Авто (скоро)")
        st.button("Сброс")

    if "engine" not in st.session_state or new_game:
        st.session_state.engine = create_engine(num_players, seed, bot_profiles=profiles)

    if step_once or step_ten or step_hundred:
        steps = 1 if step_once else 10 if step_ten else 100
        for _ in range(steps):
            if st.session_state.engine.state.game_over:
                break
            st.session_state.engine.step()

    engine = st.session_state.engine
    state: GameState = engine.state

    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.subheader("Поле")
        if state.game_over:
            winner_name = (
                state.players[state.winner_id].name if state.winner_id is not None else "не определен"
            )
            st.success(f"Игра окончена. Победитель: {winner_name}")
        _render_board(state.board, state.players)
        _render_log(state.event_log)

    with col_right:
        st.subheader("Игроки")
        _render_players(state.players, state.board)


if __name__ == "__main__":
    main()

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


def _render_board(board: list[Cell], players: list[Player], active_player_id: int | None) -> None:
    coords = _perimeter_coords()
    grid = [["" for _ in range(11)] for _ in range(11)]

    players_at = {pos: [] for pos in range(40)}
    for player in players:
        players_at[player.position].append(player)

    active_position = None
    if active_player_id is not None and 0 <= active_player_id < len(players):
        active_position = players[active_player_id].position

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
        tokens = " ".join(
            [
                f"P{p.player_id + 1}" + ("★" if p.player_id == active_player_id else "")
                for p in players_at[idx]
            ]
        )
        players_text = f"Игроки: {tokens}" if tokens else ""
        parts = [f"<div class='name'>{cell.name}</div>"]
        for line in [owner_text, mort_text, build_text, players_text]:
            if line:
                parts.append(f"<div class='meta'>{line}</div>")
        cell_class = "cell active" if active_position == idx else "cell"
        grid[row][col] = f"<div class='{cell_class}'>" + "".join(parts) + "</div>"

    html_rows = []
    for row in grid:
        html_cells = [cell if cell else "<div class='cell'></div>" for cell in row]
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
      .cell.active {{
        border: 2px solid #c44d29;
        box-shadow: inset 0 0 0 2px #f2d3c7;
        background: #fff3ee;
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
            "Активы": len(owned_map[player.player_id]),
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


def _render_turn_panel(state: GameState) -> None:
    st.subheader("Текущий ход")
    if state.game_over:
        winner_name = (
            state.players[state.winner_id].name if state.winner_id is not None else "не определен"
        )
        st.success(f"Игра окончена. Победитель: {winner_name}")
        return

    active_player = state.players[state.current_player]
    current_cell = state.board[active_player.position]

    last_roll = next(
        (
            event
            for event in reversed(state.event_log)
            if event.type in {"DICE_ROLL", "JAIL_ROLL"}
        ),
        None,
    )
    last_roll_text = last_roll.msg_ru if last_roll else "—"

    st.write(f"**Активный игрок:** {active_player.name}")
    st.write(f"**Клетка:** {current_cell.name}")
    st.write(f"**Последний бросок:** {last_roll_text}")

    tail_events = state.event_log[-3:]
    if tail_events:
        st.write("**Последние события:**")
        for event in tail_events:
            st.write(f"- {event.msg_ru}")
    else:
        st.write("**Последние события:** —")


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
        run_to_end = st.button("До конца игры")
        st.button("Сброс")

    if "engine" not in st.session_state or new_game:
        st.session_state.engine = create_engine(num_players, seed, bot_profiles=profiles)
        st.session_state.run_info = ""

    if step_once or step_ten or step_hundred:
        steps = 1 if step_once else 10 if step_ten else 100
        for _ in range(steps):
            if st.session_state.engine.state.game_over:
                break
            st.session_state.engine.step()

    if run_to_end:
        max_steps = 5000
        steps_done = 0
        while steps_done < max_steps and not st.session_state.engine.state.game_over:
            st.session_state.engine.step()
            steps_done += 1
        if st.session_state.engine.state.game_over:
            st.session_state.run_info = f"Игра завершена за {steps_done} шагов."
        else:
            st.session_state.run_info = f"Достигнут лимит {max_steps} шагов."

    engine = st.session_state.engine
    state: GameState = engine.state

    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.subheader("Поле")
        _render_board(state.board, state.players, state.current_player)
        _render_log(state.event_log)

    with col_right:
        _render_turn_panel(state)
        if st.session_state.get("run_info"):
            st.info(st.session_state.run_info)
        st.subheader("Игроки")
        _render_players(state.players, state.board)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from .engine import create_engine
from .io_utils import read_json, tail_lines, write_json_atomic
from .params import BotParams, load_params
from .status import read_status

ROOT_DIR = Path(__file__).resolve().parents[1]


try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - fallback when dependency отсутствует

    def st_autorefresh(interval: int = 1000, key: str | None = None) -> None:  # type: ignore[override]
        components.html(
            f"""
            <script>
              const timeout = {interval};
              setTimeout(() => window.location.reload(), timeout);
            </script>
            """,
            height=0,
        )
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(obj: Any, key: str, default: Any | None = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("type", ""))
    return str(getattr(event, "type", ""))


def _event_msg(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("msg_ru", ""))
    return str(getattr(event, "msg_ru", ""))


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


def _render_board(board: list[Any], players: list[Any], active_player_id: int | None) -> None:
    coords = _perimeter_coords()
    grid = [["" for _ in range(11)] for _ in range(11)]

    players_at = {pos: [] for pos in range(40)}
    for player in players:
        pos = int(_get(player, "position", 0))
        players_at[pos].append(player)

    active_position = None
    if active_player_id is not None and 0 <= active_player_id < len(players):
        active_position = int(_get(players[active_player_id], "position", 0))

    for idx, cell in enumerate(board):
        row, col = coords[idx]
        owner_text = ""
        owner_id = _get(cell, "owner_id")
        if owner_id is not None and 0 <= int(owner_id) < len(players):
            owner_name = _get(players[int(owner_id)], "name", "?")
            owner_text = f"Вл: {owner_name}"
        mort_text = "ИП" if _get(cell, "mortgaged", False) else ""
        build_text = ""
        if _get(cell, "hotels", 0):
            build_text = "🏨"
        elif _get(cell, "houses", 0):
            build_text = f"🏠x{_get(cell, 'houses', 0)}"
        tokens = " ".join(
            [
                f"P{_get(p, 'player_id', 0) + 1}"
                + ("★" if _get(p, "player_id") == active_player_id else "")
                for p in players_at[idx]
            ]
        )
        players_text = f"Игроки: {tokens}" if tokens else ""
        parts = [f"<div class='name'>{_get(cell, 'name', '')}</div>"]
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


def _render_players(players: list[Any], board: list[Any]) -> None:
    owned_map: dict[int, list[str]] = {}
    for cell in board:
        owner_id = _get(cell, "owner_id")
        if owner_id is None:
            continue
        owner_id = int(owner_id)
        owned_map.setdefault(owner_id, []).append(str(_get(cell, "name", "")))

    table = []
    for player in players:
        player_id = int(_get(player, "player_id", 0))
        table.append(
            {
                "Игрок": _get(player, "name", ""),
                "Деньги": _get(player, "money", 0),
                "Позиция": _get(player, "position", 0),
                "В тюрьме": "Да" if _get(player, "in_jail", False) else "Нет",
                "Активы": len(owned_map.get(player_id, [])),
            }
        )
    st.dataframe(table, use_container_width=True)

    for player in players:
        player_id = int(_get(player, "player_id", 0))
        with st.expander(f"{_get(player, 'name', '')} — собственность"):
            names = owned_map.get(player_id, [])
            if names:
                st.write(", ".join(names))
            else:
                st.write("Нет собственности")


def _render_log(events: list[Any]) -> None:
    st.subheader("Лента событий")
    for event in reversed(events[-30:]):
        st.write(f"- {_event_msg(event)}")


def _render_turn_panel(state: Any) -> None:
    st.subheader("Текущий ход")
    if _get(state, "game_over", False):
        winner_id = _get(state, "winner_id")
        players = _get(state, "players", [])
        winner_name = "не определен"
        if winner_id is not None and 0 <= int(winner_id) < len(players):
            winner_name = _get(players[int(winner_id)], "name", "не определен")
        st.success(f"Игра окончена. Победитель: {winner_name}")
        return

    players = _get(state, "players", [])
    board = _get(state, "board", [])
    active_id = int(_get(state, "current_player", 0))
    if not players or not board:
        st.write("Нет данных о состоянии.")
        return
    active_player = players[active_id]
    current_cell = board[int(_get(active_player, "position", 0))]

    event_log = _get(state, "event_log", [])
    last_roll = next(
        (event for event in reversed(event_log) if _event_type(event) in {"DICE_ROLL", "JAIL_ROLL"}),
        None,
    )
    last_card = next(
        (event for event in reversed(event_log) if _event_type(event) == "DRAW_CARD"),
        None,
    )
    last_effect = next(
        (event for event in reversed(event_log) if _event_type(event) == "CARD_EFFECT"),
        None,
    )

    st.write(f"**Активный игрок:** {_get(active_player, 'name', '')}")
    st.write(f"**Клетка:** {_get(current_cell, 'name', '')}")
    st.write(f"**Последний бросок:** {_event_msg(last_roll) if last_roll else '—'}")
    st.write(f"**Последняя карта:** {_event_msg(last_card) if last_card else '—'}")
    st.write(f"**Результат карты:** {_event_msg(last_effect) if last_effect else '—'}")

    tail_events = event_log[-3:]
    if tail_events:
        st.write("**Последние события:**")
        for event in tail_events:
            st.write(f"- {_event_msg(event)}")
    else:
        st.write("**Последние события:** —")


def _start_autotrain(profile: str = "deep") -> Path:
    runs_dir = ROOT_DIR / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    cmd = [
        sys.executable,
        "-m",
        "monopoly.autotrain",
        "run",
        "--profile",
        profile,
        "--workers",
        "auto",
        "--runs-dir",
        str(runs_dir),
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
    st.session_state.train_proc = proc
    st.session_state.train_runs_dir = str(runs_dir)
    return runs_dir


def _stop_autotrain() -> None:
    proc = st.session_state.get("train_proc")
    if proc and proc.poll() is None:
        proc.terminate()
        st.session_state.train_proc = None
    runs_dir_raw = st.session_state.get("train_runs_dir")
    if not runs_dir_raw:
        return
    status_path = Path(runs_dir_raw) / "status.json"
    data = read_json(status_path, default=None)
    if isinstance(data, dict):
        data["current_phase"] = "stopped"
        data["updated_at"] = _utc_now()
        write_json_atomic(status_path, data)


def _start_live_match(runs_dir: Path, params_path: Path) -> Path:
    out_path = runs_dir / "live_state.json"
    cmd = [
        sys.executable,
        "-m",
        "monopoly.live",
        "--players",
        "6",
        "--params",
        str(params_path),
        "--mode",
        "deep",
        "--workers",
        "auto",
        "--time-per-decision-sec",
        "3.0",
        "--horizon-turns",
        "60",
        "--seed",
        "42",
        "--out",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
    st.session_state.live_proc = proc
    st.session_state.live_state_path = str(out_path)
    return out_path


def render_game_mode() -> None:
    with st.sidebar:
        st.header("Управление")
        num_players = st.slider("Число ботов", min_value=2, max_value=6, value=4, step=1)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        params_path = st.text_input(
            "Путь к параметрам бота (json/yaml)",
            value="",
            placeholder="trained_params.json",
        )
        new_game = st.button("Новая игра", type="primary")
        step_once = st.button("Шаг")
        step_ten = st.button("+10 шагов")
        step_hundred = st.button("+100 шагов")
        run_to_end = st.button("До конца игры")

    if "engine" not in st.session_state or new_game:
        bot_params = BotParams()
        if params_path:
            try:
                bot_params = load_params(params_path)
            except Exception as exc:
                st.warning(f"Не удалось загрузить параметры: {exc}. Использую дефолтные.")
        st.session_state.engine = create_engine(num_players, seed, bot_params=bot_params)
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
    state = engine.state

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


def render_training_mode() -> None:
    with st.sidebar:
        st.header("Тренировка")
        start_btn = st.button("Старт тренировки", type="primary")
        stop_btn = st.button("Стоп тренировки")

    if start_btn:
        proc = st.session_state.get("train_proc")
        if proc and proc.poll() is None:
            st.warning("Тренировка уже запущена.")
        else:
            _start_autotrain("deep")

    if stop_btn:
        _stop_autotrain()

    runs_dir_raw = st.session_state.get("train_runs_dir")
    if runs_dir_raw:
        st.caption(f"runs_dir: {runs_dir_raw}")

    runs_dir = Path(runs_dir_raw) if runs_dir_raw else None
    status = None
    if runs_dir and (runs_dir / "status.json").exists():
        try:
            status = read_status(runs_dir / "status.json")
        except Exception as exc:
            st.warning(f"status.json прочитать не удалось: {exc}")

    if status:
        st_autorefresh(interval=1000, key="train_refresh")
        col1, col2, col3 = st.columns(3)
        col1.metric("Epoch", status["epoch"])
        col2.metric(
            "Best win-rate",
            f"{status['best_winrate_mean']:.3f}",
            help=f"CI [{status['best_winrate_ci_low']:.3f}, {status['best_winrate_ci_high']:.3f}]",
        )
        col3.metric("Best fitness", f"{status['best_fitness']:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Total games", int(status["total_games_simulated"]))
        col5.metric("Cache hits", int(status["cache_hits_last_epoch"]))
        col6.metric(
            "Plateau",
            f"{status['plateau_counter']}/{status['plateau_epochs']}",
        )

        st.write(f"Фаза: **{status['current_phase']}**")

        log_path = runs_dir / "train_log.csv"
        if log_path.exists():
            epochs: list[int] = []
            winrates: list[float] = []
            fitness: list[float] = []
            with log_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    epochs.append(int(row["epoch"]))
                    winrates.append(float(row["best_winrate_mean"]))
                    fitness.append(float(row["best_fitness"]))
            if epochs:
                fig1, ax1 = plt.subplots()
                ax1.plot(epochs, winrates)
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Best win-rate")
                st.pyplot(fig1, clear_figure=True)

                fig2, ax2 = plt.subplots()
                ax2.plot(epochs, fitness)
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Best fitness")
                st.pyplot(fig2, clear_figure=True)

        progress_path = runs_dir / "progress.txt"
        tail = tail_lines(progress_path, max_lines=200)
        if tail:
            st.subheader("Лог тренировки")
            st.code("\n".join(tail))

        if status["current_phase"] == "finished":
            best_path = Path(status["best_params_path"])
            if best_path.exists():
                if st.button("Запустить live матч 6 ботов"):
                    _start_live_match(runs_dir, best_path)


def render_live_mode() -> None:
    with st.sidebar:
        st.header("Live матч")
        default_path = st.session_state.get("live_state_path", "")
        live_path = st.text_input("Путь к live_state.json", value=default_path)
        st.session_state.live_state_path = live_path
        stop_btn = st.button("Остановить live матч")

    if stop_btn:
        proc = st.session_state.get("live_proc")
        if proc and proc.poll() is None:
            proc.terminate()
            st.session_state.live_proc = None

    if not live_path:
        st.info("Укажите путь к live_state.json")
        return

    payload = read_json(Path(live_path), default=None)
    if not isinstance(payload, dict):
        st.warning("live_state.json еще не создан.")
        return

    st_autorefresh(interval=1000, key="live_refresh")

    if payload.get("thinking"):
        st.info(
            "Думает... "
            f"Контекст: {payload.get('decision_context', '')} | "
            f"Rollouts: {payload.get('rollouts_done', 0)} | "
            f"Time left: {payload.get('time_left_sec', 0.0):.2f}s"
        )

    board = payload.get("board", [])
    players = payload.get("players", [])
    active_id = payload.get("current_player")

    col_left, col_right = st.columns([2, 1], gap="large")
    with col_left:
        st.subheader("Поле")
        _render_board(board, players, active_id)
        _render_log(payload.get("event_log", []))

    with col_right:
        _render_turn_panel(payload)
        st.subheader("Игроки")
        _render_players(players, board)


def main() -> None:
    st.set_page_config(page_title="Монополия — наблюдатель", layout="wide")
    st.title("Монополия — наблюдатель")
    st.caption("Режимы: игра, тренировка, live матч.")

    with st.sidebar:
        st.header("Режим")
        mode = st.radio("", ["Игра", "Тренировка", "Live матч"], index=0)

    if mode == "Игра":
        render_game_mode()
    elif mode == "Тренировка":
        render_training_mode()
    else:
        render_live_mode()


if __name__ == "__main__":
    main()

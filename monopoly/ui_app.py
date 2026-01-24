from __future__ import annotations

import csv
import html as html_lib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import yaml

from .engine import create_engine
from .io_utils import read_json, tail_lines, write_json_atomic
from .params import BotParams, load_params
from .run_utils import latest_run, list_runs
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


CELL_TYPE_LABELS = {
    "go": "Старт",
    "property": "Улица",
    "railroad": "Вокзал",
    "utility": "Коммун.",
    "tax": "Налог",
    "chance": "Шанс",
    "community": "Казна",
    "jail": "Тюрьма",
    "free_parking": "Парковка",
    "go_to_jail": "В тюрьму",
}

CELL_TYPE_ICONS = {
    "go": "▶",
    "property": "■",
    "railroad": "🚆",
    "utility": "⚡",
    "tax": "¤",
    "chance": "?",
    "community": "✚",
    "jail": "⛓",
    "free_parking": "P",
    "go_to_jail": "⇢",
}

GROUP_COLORS = {
    "brown": "#8b5a2b",
    "light_blue": "#89c5f5",
    "pink": "#f2a4c8",
    "orange": "#f39c4a",
    "red": "#e2574c",
    "yellow": "#f0d45b",
    "green": "#3c9a5f",
    "blue": "#3665d2",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 1) - 2)


def _html_escape(text: str) -> str:
    return html_lib.escape(str(text), quote=True)


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


def _cell_type_label(cell_type: str) -> str:
    return CELL_TYPE_LABELS.get(cell_type, "Клетка")


def _cell_icon(cell_type: str) -> str:
    return CELL_TYPE_ICONS.get(cell_type, "")


def _load_card_ids(data_dir: Path) -> set[str]:
    ids: set[str] = set()
    for filename in ("cards_chance.yaml", "cards_community.yaml"):
        path = data_dir / filename
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("id"):
                    ids.add(str(item["id"]))
    return ids


def _official_texts_status(data_dir: Path) -> dict[str, Any]:
    override_path = data_dir / "cards_texts_ru_official.yaml"
    all_ids = _load_card_ids(data_dir)
    if not override_path.exists():
        return {
            "available": False,
            "missing": sorted(all_ids),
            "error": None,
            "path": str(override_path),
        }
    try:
        raw = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "available": True,
            "missing": sorted(all_ids),
            "error": str(exc),
            "path": str(override_path),
        }
    if not isinstance(raw, dict):
        return {
            "available": True,
            "missing": sorted(all_ids),
            "error": "Файл должен содержать словарь id -> text_ru",
            "path": str(override_path),
        }
    override_ids = {str(k) for k in raw.keys() if k is not None}
    missing = sorted(all_ids - override_ids)
    return {
        "available": True,
        "missing": missing,
        "error": None,
        "path": str(override_path),
    }


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


def _player_badge(player: Any, active_player_id: int | None) -> str:
    pid = int(_get(player, "player_id", 0))
    token = f"P{pid + 1}"
    if active_player_id is not None and pid == active_player_id:
        token += "★"
    if _get(player, "in_jail", False):
        token += "🔒"
    return token


def _compute_net_worth(state: Any, player_id: int) -> int:
    players = _get(state, "players", [])
    board = _get(state, "board", [])
    if player_id >= len(players):
        return 0
    player = players[player_id]
    total = int(_get(player, "money", 0))
    for cell in board:
        if _get(cell, "owner_id") != player_id:
            continue
        if _get(cell, "mortgaged", False):
            total += int(_get(cell, "mortgage_value", 0) or 0)
        else:
            total += int(_get(cell, "price", 0) or 0)
        total += (int(_get(cell, "houses", 0)) + int(_get(cell, "hotels", 0)) * 5) * int(
            _get(cell, "house_cost", 0) or 0
        )
    return total


def _build_center_panel(
    state: Any,
    mode: str,
    thinking: dict[str, Any] | None = None,
    cards_status: dict[str, Any] | None = None,
) -> str:
    players = _get(state, "players", [])
    board = _get(state, "board", [])
    event_log = _get(state, "event_log", [])

    current_player_id = int(_get(state, "current_player", 0)) if players else 0
    active_player = players[current_player_id] if players else None
    active_cell = None
    if active_player and board:
        active_cell = board[int(_get(active_player, "position", 0))]

    last_roll = next(
        (event for event in reversed(event_log) if _event_type(event) in {"DICE_ROLL", "JAIL_ROLL"}),
        None,
    )
    last_card = next(
        (event for event in reversed(event_log) if _event_type(event) == "DRAW_CARD"),
        None,
    )

    thinking_html = ""
    if thinking:
        if thinking.get("thinking"):
            thinking_html = (
                f"<div class='center-meta'>Думает… {thinking.get('decision_context','')}</div>"
                f"<div class='center-meta'>Rollouts: {thinking.get('rollouts_done',0)} | "
                f"Time left: {thinking.get('time_left_sec',0):.2f}s</div>"
            )

    events_tail = event_log[-12:]
    events_html = "".join(
        f"<div class='event-line'>{_event_msg(ev)}</div>" for ev in events_tail
    )
    if last_card:
        events_html = f"<div class='event-highlight'>{_event_msg(last_card)}</div>" + events_html

    players_rows = []
    for player in players:
        pid = int(_get(player, "player_id", 0))
        houses = sum(
            int(_get(cell, "houses", 0))
            for cell in board
            if _get(cell, "owner_id") == pid
        )
        hotels = sum(
            int(_get(cell, "hotels", 0))
            for cell in board
            if _get(cell, "owner_id") == pid
        )
        mortgages = sum(
            1
            for cell in board
            if _get(cell, "owner_id") == pid and _get(cell, "mortgaged", False)
        )
        props = sum(1 for cell in board if _get(cell, "owner_id") == pid)
        net_worth = _compute_net_worth(state, pid)
        jail = "Да" if _get(player, "in_jail", False) else "Нет"
        players_rows.append(
            "<tr>"
            f"<td>{_get(player, 'name', '')}</td>"
            f"<td>{int(_get(player, 'money', 0))}</td>"
            f"<td>{net_worth}</td>"
            f"<td>{props}</td>"
            f"<td>{houses}/{hotels}</td>"
            f"<td>{mortgages}</td>"
            f"<td>{jail}</td>"
            "</tr>"
        )

    active_cell_name = _get(active_cell, "name", "—") if active_cell else "—"
    active_player_name = _get(active_player, "name", "—") if active_player else "—"
    last_roll_text = _event_msg(last_roll) if last_roll else "—"
    jail_text = "Да" if active_player and _get(active_player, "in_jail", False) else "Нет"

    legend_icons = (
        "■ улица, 🚆 вокзал, ⚡ коммуналка, ¤ налог, ? шанс, ✚ казна, ⛓ тюрьма, ▶ старт, P парковка, ⇢ в тюрьму"
    )
    legend_badges = (
        "<span class='badge badge-mort'>ИП</span> ипотека, "
        "<span class='badge badge-build'>Д1–Д4</span> дома, "
        "<span class='badge badge-build'>Н</span> отель"
    )
    official_texts = "не подключены"
    official_class = "status-bad"
    if cards_status and cards_status.get("available"):
        if cards_status.get("missing"):
            official_texts = "неполные"
            official_class = "status-warn"
        elif cards_status.get("error"):
            official_texts = "ошибка"
            official_class = "status-warn"
        else:
            official_texts = "подключены"
            official_class = "status-ok"

    return f"""
    <div class='center-grid'>
      <div class='center-block'>
        <div class='center-title'>Текущий ход</div>
        <div class='center-value'>{active_player_name}</div>
        <div class='center-meta'>Клетка: {active_cell_name}</div>
        <div class='center-meta'>Последний бросок: {last_roll_text}</div>
        <div class='center-meta'>Тюрьма: {jail_text}</div>
        {thinking_html}
      </div>
      <div class='center-block'>
        <div class='center-title'>События</div>
        <div class='event-list'>{events_html}</div>
      </div>
      <div class='center-block'>
        <div class='center-title'>Игроки</div>
        <table class='players-table'>
          <thead>
            <tr><th>Игрок</th><th>Cash</th><th>Net</th><th>Участки</th><th>Д/О</th><th>ИП</th><th>Тюрьма</th></tr>
          </thead>
          <tbody>
            {''.join(players_rows)}
          </tbody>
        </table>
      </div>
      <div class='center-block'>
        <div class='center-title'>Управление</div>
        <div class='center-meta'>Шаг / +10 / +100 / До конца — кнопки под доской</div>
        <div class='center-meta'>Периметр = 40 клеток, центр 9×9 под панель.</div>
      </div>
      <div class='center-block'>
        <div class='center-title'>Легенда</div>
        <div class='center-meta'>{legend_badges}</div>
        <div class='center-meta'>{legend_icons}</div>
        <div class='center-meta'>Офиц. тексты: <span class='{official_class}'>{official_texts}</span></div>
      </div>
    </div>
    """


# ---------------------------------------------------------
# Board rendering
# ---------------------------------------------------------

def _render_board(
    board: list[Any],
    players: list[Any],
    active_player_id: int | None,
    center_html: str,
) -> None:
    coords = _perimeter_coords()

    players_at = {pos: [] for pos in range(40)}
    for player in players:
        pos = int(_get(player, "position", 0))
        players_at[pos].append(player)

    active_position = None
    if active_player_id is not None and 0 <= int(active_player_id) < len(players):
        active_position = int(_get(players[int(active_player_id)], "position", 0))

    html_cells = []
    for idx, cell in enumerate(board):
        row, col = coords[idx]
        owner_text = ""
        owner_id = _get(cell, "owner_id")
        cell_type = str(_get(cell, "cell_type", ""))
        ownable = cell_type in {"property", "railroad", "utility"}
        if owner_id is not None and 0 <= int(owner_id) < len(players):
            owner_text = f"Вл: P{int(owner_id)+1}"
        elif ownable:
            owner_text = "Вл: Банк"
        mort_text = "<span class='badge badge-mort'>ИП</span>" if _get(cell, "mortgaged", False) else ""
        build_text = ""
        if _get(cell, "hotels", 0):
            build_text = "<span class='badge badge-build'>Н</span>"
        elif _get(cell, "houses", 0):
            build_text = f"<span class='badge badge-build'>Д{_get(cell, 'houses', 0)}</span>"
        tokens = " ".join(
            [_player_badge(p, active_player_id) for p in players_at[idx]]
        )
        players_text = f"{tokens}" if tokens else ""
        type_label = _cell_type_label(cell_type)
        type_icon = _cell_icon(cell_type)
        color = GROUP_COLORS.get(str(_get(cell, "group", "")), "")
        color_strip = (
            f"<div class='color-strip' style='background:{color}'></div>" if color else ""
        )
        corner_class = "corner" if idx in {0, 10, 20, 30} else ""
        active_class = "active" if active_position == idx else ""
        cell_name_raw = _get(cell, "name", "")
        cell_name = _html_escape(cell_name_raw)
        html_cells.append(
            f"""
            <div class='cell {corner_class} {active_class}' style='grid-row:{row + 1}; grid-column:{col + 1};'>
              {color_strip}
              <div class='cell-title' title='{cell_name}'>{cell_name}</div>
              <div class='cell-type'>{type_icon} {type_label}</div>
              <div class='cell-meta'>{owner_text}</div>
              <div class='cell-meta'>{mort_text} {build_text}</div>
              <div class='cell-players'>{players_text}</div>
            </div>
            """
        )

    html = f"""
    <style>
      .board-grid {{
        display: grid;
        grid-template-columns: repeat(11, minmax(70px, 1fr));
        grid-template-rows: repeat(11, minmax(70px, 1fr));
        gap: 2px;
        background: #e6e0d6;
        padding: 6px;
        border-radius: 12px;
        position: relative;
      }}
      .cell {{
        background: #f7f2ea;
        border: 1px solid #c9bfae;
        padding: 6px;
        font-size: 11px;
        line-height: 1.2;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
      }}
      .cell.active {{
        border: 2px solid #c44d29;
        box-shadow: inset 0 0 0 2px #f2d3c7;
        background: #fff3ee;
      }}
      .cell.corner {{
        background: #f2e8d8;
        border: 2px solid #bfae98;
        font-weight: 700;
        text-transform: uppercase;
      }}
      .cell.corner .cell-title {{
        font-size: 13px;
        text-align: center;
      }}
      .cell-title {{
        font-weight: 700;
        font-size: 12px;
        min-height: 28px;
        word-break: break-word;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .cell-type {{
        font-size: 10px;
        color: #5a4c3c;
      }}
      .cell-meta {{
        font-size: 10px;
        color: #4a3e2d;
        min-height: 14px;
      }}
      .cell-players {{
        font-size: 10px;
        color: #2f2a24;
        font-weight: 600;
      }}
      .badge {{
        display: inline-block;
        padding: 1px 4px;
        border-radius: 6px;
        font-size: 9px;
        line-height: 1.2;
        font-weight: 700;
      }}
      .badge-mort {{
        background: #e8c6c6;
        color: #8c1f1f;
      }}
      .badge-build {{
        background: #d6e8c6;
        color: #1f6b1f;
      }}
      .color-strip {{
        height: 6px;
        border-radius: 2px;
        margin-bottom: 4px;
      }}
      .board-center {{
        grid-row: 2 / span 9;
        grid-column: 2 / span 9;
        background: #fdf7ef;
        border: 1px solid #d7cbb7;
        border-radius: 10px;
        padding: 10px;
        overflow: hidden;
      }}
      .center-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        grid-auto-rows: minmax(80px, auto);
        gap: 10px;
        font-size: 11px;
      }}
      .center-block {{
        background: #fffdf6;
        border: 1px solid #e4d6c2;
        border-radius: 8px;
        padding: 9px;
      }}
      .center-title {{
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 6px;
      }}
      .center-value {{
        font-size: 13px;
        font-weight: 700;
      }}
      .center-meta {{
        font-size: 10.5px;
        color: #5d5141;
      }}
      .event-list {{
        max-height: 140px;
        overflow: hidden;
      }}
      .event-line {{
        font-size: 10.5px;
        color: #4b4035;
      }}
      .event-highlight {{
        font-size: 10.5px;
        font-weight: 700;
        margin-bottom: 4px;
      }}
      .players-table {{
        width: 100%;
        font-size: 10px;
        border-collapse: collapse;
      }}
      .players-table th, .players-table td {{
        text-align: left;
        padding: 2px 4px;
        border-bottom: 1px solid #eee2d1;
      }}
      @media (max-width: 900px) {{
        .board-grid {{
          grid-template-columns: repeat(11, minmax(50px, 1fr));
          grid-template-rows: repeat(11, minmax(50px, 1fr));
        }}
        .cell-title {{
          font-size: 10px;
          min-height: 20px;
        }}
        .center-grid {{
          grid-template-columns: 1fr;
        }}
        .event-list {{
          max-height: 100px;
        }}
      }}
      .status-ok {{
        color: #1f6b1f;
        font-weight: 700;
      }}
      .status-warn {{
        color: #8c4b1f;
        font-weight: 700;
      }}
      .status-bad {{
        color: #8c1f1f;
        font-weight: 700;
      }}
    </style>
    <div class='board-grid'>
      <div class='board-center'>
        {center_html}
      </div>
      {"".join(html_cells)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------
# Actions and modes
# ---------------------------------------------------------

def _start_autotrain(
    profile: str,
    workers: int,
    plateau_epochs: int,
    epoch_iters: int,
    min_games: int,
    delta: float,
    seeds_file: str,
    league_dir: str,
    runs_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    if runs_dir is None:
        runs_dir = ROOT_DIR / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    error_log = runs_dir / "error.log"
    cmd = [
        sys.executable,
        "-m",
        "monopoly.autotrain",
        "run",
        "--profile",
        profile,
        "--workers",
        str(workers),
        "--plateau-epochs",
        str(plateau_epochs),
        "--epoch-iters",
        str(epoch_iters),
        "--min-progress-games",
        str(min_games),
        "--delta",
        str(delta),
        "--league-dir",
        league_dir,
        "--runs-dir",
        str(runs_dir),
    ]
    if seeds_file:
        cmd.extend(["--seeds-file", seeds_file])
    if resume:
        cmd.append("--resume")
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("ab") as err_handle:
        proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=err_handle, stderr=err_handle)
    st.session_state.train_proc = proc
    st.session_state.train_runs_dir = str(runs_dir)
    st.session_state.train_error_log = str(error_log)
    return runs_dir


def _stop_autotrain(phase: str) -> None:
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
        data["current_phase"] = phase
        data["updated_at"] = _utc_now()
        write_json_atomic(status_path, data)


def _start_live_match(
    runs_dir: Path,
    params_path: Path,
    workers: int,
    time_per_decision: float,
    horizon: int,
    seed: int,
) -> Path:
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
        str(workers),
        "--time-per-decision-sec",
        str(time_per_decision),
        "--horizon-turns",
        str(horizon),
        "--seed",
        str(seed),
        "--out",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
    st.session_state.live_proc = proc
    st.session_state.live_state_path = str(out_path)
    return out_path


# ---------------------------------------------------------
# Modes
# ---------------------------------------------------------

def render_game_mode(cards_status: dict[str, Any] | None = None) -> None:
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

    center_html = _build_center_panel(state, mode="game", cards_status=cards_status)
    _render_board(state.board, state.players, state.current_player, center_html)

    st.subheader("Управление")
    cols = st.columns(4)
    if cols[0].button("Шаг", key="step_main"):
        st.session_state.engine.step()
    if cols[1].button("+10", key="step_10_main"):
        for _ in range(10):
            if st.session_state.engine.state.game_over:
                break
            st.session_state.engine.step()
    if cols[2].button("+100", key="step_100_main"):
        for _ in range(100):
            if st.session_state.engine.state.game_over:
                break
            st.session_state.engine.step()
    if cols[3].button("До конца", key="step_end_main"):
        max_steps = 5000
        steps_done = 0
        while steps_done < max_steps and not st.session_state.engine.state.game_over:
            st.session_state.engine.step()
            steps_done += 1

    if st.session_state.get("run_info"):
        st.info(st.session_state.run_info)


def render_training_mode() -> None:
    runs_base = ROOT_DIR / "runs"
    runs = list_runs(runs_base, limit=10)
    run_names = [run.name for run in runs]
    latest = latest_run(runs_base)

    with st.sidebar:
        st.header("Тренировка")
        preset = "Deep (максимально умно)"
        st.caption(f"Preset: {preset}")
        workers = st.number_input("Workers", min_value=1, value=_default_workers(), step=1)
        time_per_decision = st.number_input("Time per decision (sec)", min_value=0.1, value=3.0, step=0.5)
        horizon = st.number_input("Horizon turns", min_value=1, value=60, step=1)
        plateau_epochs = st.number_input("Plateau epochs", min_value=1, value=5, step=1)
        epoch_iters = st.number_input("Epoch iters", min_value=1, value=10, step=1)
        min_games = st.number_input("Min games", min_value=10, value=200, step=10)
        delta = st.number_input("Delta", min_value=0.0, value=0.05, step=0.01, format="%.2f")
        seeds_file = st.text_input(
            "Seeds file",
            value=str(ROOT_DIR / "monopoly" / "data" / "seeds.txt"),
        )
        league_dir = st.text_input(
            "League dir",
            value=str(ROOT_DIR / "monopoly" / "data" / "league"),
        )

        st.subheader("Runs")
        selected_name = None
        if run_names:
            default_index = 0
            current_runs = st.session_state.get("train_runs_dir")
            if current_runs:
                try:
                    default_index = run_names.index(Path(current_runs).name)
                except ValueError:
                    default_index = 0
            selected_name = st.selectbox("Последние runs", run_names, index=default_index)
        open_last = st.button("Открыть последний")

        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        start_btn = col_btn1.button("Старт", type="primary")
        pause_btn = col_btn2.button("Пауза")
        stop_btn = col_btn3.button("Стоп")
        resume_btn = col_btn4.button("Продолжить")

    if open_last and latest is not None:
        st.session_state.train_runs_dir = str(latest)
    elif selected_name:
        st.session_state.train_runs_dir = str(runs_base / selected_name)

    if start_btn:
        proc = st.session_state.get("train_proc")
        if proc and proc.poll() is None:
            st.warning("Тренировка уже запущена.")
        else:
            _start_autotrain(
                profile="deep",
                workers=int(workers),
                plateau_epochs=int(plateau_epochs),
                epoch_iters=int(epoch_iters),
                min_games=int(min_games),
                delta=float(delta),
                seeds_file=seeds_file,
                league_dir=league_dir,
            )

    if pause_btn:
        _stop_autotrain("paused")

    if stop_btn:
        _stop_autotrain("stopped")

    if resume_btn:
        runs_dir_raw = st.session_state.get("train_runs_dir")
        if runs_dir_raw:
            _start_autotrain(
                profile="deep",
                workers=int(workers),
                plateau_epochs=int(plateau_epochs),
                epoch_iters=int(epoch_iters),
                min_games=int(min_games),
                delta=float(delta),
                seeds_file=seeds_file,
                league_dir=league_dir,
                runs_dir=Path(runs_dir_raw),
                resume=True,
            )

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

    proc = st.session_state.get("train_proc")
    if proc and proc.poll() is not None and proc.returncode not in (0, None):
        error_log = st.session_state.get("train_error_log")
        if error_log:
            tail = tail_lines(Path(error_log), max_lines=100)
            if tail:
                st.error("Тренировка завершилась с ошибкой. stderr:")
                st.code("\n".join(tail))

    if status:
        st_autorefresh(interval=1000, key="train_refresh")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epoch", status["epoch"])
        col2.metric(
            "Best win-rate",
            f"{status['best_winrate_mean']:.3f}",
            help=f"CI [{status['best_winrate_ci_low']:.3f}, {status['best_winrate_ci_high']:.3f}]",
        )
        col3.metric("Best fitness", f"{status['best_fitness']:.4f}")
        col4.metric("Games simulated", int(status["total_games_simulated"]))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Phase", status["current_phase"])
        col6.metric(
            "Plateau counter",
            f"{status['plateau_counter']}/{status['plateau_epochs']}",
        )
        col7.metric("Promoted", int(status["promoted_count"]))
        col8.metric("Last promoted", int(status["last_promoted_epoch"]))

        st.write(f"Cache hits: **{int(status['cache_hits_last_epoch'])}**")

        log_path = runs_dir / "train_log.csv"
        if log_path.exists():
            epochs: list[int] = []
            winrates: list[float] = []
            fitness: list[float] = []
            plateau_vals: list[int] = []
            with log_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    epochs.append(int(row.get("epoch", 0)))
                    winrates.append(float(row.get("best_winrate_mean", 0.0)))
                    fitness.append(float(row.get("best_fitness", 0.0)))
                    if "plateau_counter" in row:
                        plateau_vals.append(int(float(row.get("plateau_counter", 0))))
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

                if plateau_vals:
                    fig3, ax3 = plt.subplots()
                    ax3.plot(epochs, plateau_vals)
                    ax3.set_xlabel("Epoch")
                    ax3.set_ylabel("Plateau counter")
                    st.pyplot(fig3, clear_figure=True)

        progress_path = runs_dir / "progress.txt"
        tail = tail_lines(progress_path, max_lines=200)
        if tail:
            st.subheader("Лог тренировки")
            st.code("\n".join(tail))
            try:
                full_log = progress_path.read_text(encoding="utf-8")
            except Exception:
                full_log = "\n".join(tail)
            st.download_button(
                "Скачать лог",
                data=full_log,
                file_name="progress.txt",
            )

        if status["current_phase"] == "finished":
            best_path = Path(status["best_params_path"])
            if best_path.exists():
                if st.button("Запустить live матч сейчас"):
                    _start_live_match(
                        runs_dir,
                        best_path,
                        workers=int(workers),
                        time_per_decision=float(time_per_decision),
                        horizon=int(horizon),
                        seed=42,
                    )


def render_live_mode(cards_status: dict[str, Any] | None = None) -> None:
    runs_base = ROOT_DIR / "runs"
    latest = latest_run(runs_base)
    selected_runs = st.session_state.get("train_runs_dir")
    default_best = ""
    if selected_runs:
        candidate = Path(selected_runs) / "best.json"
        if candidate.exists():
            default_best = str(candidate)

    with st.sidebar:
        st.header("Live матч")
        params_path = st.text_input("Путь к params", value=default_best)
        workers = st.number_input("Workers", min_value=1, value=_default_workers(), step=1)
        time_per_decision = st.number_input("Time per decision (sec)", min_value=0.1, value=3.0, step=0.5)
        horizon = st.number_input("Horizon turns", min_value=1, value=60, step=1)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        refresh_ms = st.slider("Обновление UI (мс)", min_value=500, max_value=1500, value=1000, step=100)
        start_btn = st.button("Запустить матч 6 deep-ботов", type="primary")
        stop_btn = st.button("Остановить live матч")
        live_path = st.text_input(
            "Путь к live_state.json",
            value=st.session_state.get("live_state_path", ""),
        )
        st.session_state.live_state_path = live_path

    if start_btn:
        if params_path:
            runs_dir = Path(params_path).parent
        else:
            runs_dir = latest or runs_base
        params = Path(params_path) if params_path else (runs_dir / "best.json")
        _start_live_match(
            runs_dir,
            params,
            workers=int(workers),
            time_per_decision=float(time_per_decision),
            horizon=int(horizon),
            seed=int(seed),
        )

    if stop_btn:
        proc = st.session_state.get("live_proc")
        if proc and proc.poll() is None:
            proc.terminate()
            st.session_state.live_proc = None

    if not live_path:
        st.info("Ожидание live_state.json...")
        return

    try:
        payload = read_json(Path(live_path), default=None)
    except Exception:
        st.warning("live_state.json еще не готов (поврежден или в процессе записи).")
        return
    if not isinstance(payload, dict):
        st.warning("live_state.json еще не создан.")
        return

    st_autorefresh(interval=refresh_ms, key="live_refresh")

    thinking = {
        "thinking": payload.get("thinking"),
        "decision_context": payload.get("decision_context", ""),
        "rollouts_done": payload.get("rollouts_done", 0),
        "time_left_sec": payload.get("time_left_sec", 0.0),
    }

    center_html = _build_center_panel(payload, mode="live", thinking=thinking, cards_status=cards_status)
    _render_board(payload.get("board", []), payload.get("players", []), payload.get("current_player"), center_html)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Монополия — наблюдатель", layout="wide")
    st.title("Монополия — наблюдатель")
    st.caption("Режимы: игра, тренировка, live матч.")

    data_dir = ROOT_DIR / "monopoly" / "data"
    cards_status = _official_texts_status(data_dir)
    if cards_status.get("available"):
        if cards_status.get("error"):
            st.warning(f"cards_texts_ru_official.yaml: {cards_status['error']}")
        elif cards_status.get("missing"):
            missing = cards_status.get("missing", [])
            preview = ", ".join(missing[:10])
            tail = f" (+{len(missing) - 10})" if len(missing) > 10 else ""
            st.warning(f"cards_texts_ru_official.yaml: не хватает id: {preview}{tail}")

    with st.sidebar:
        st.header("Режим")
        mode = st.radio("", ["Игра", "Тренировка", "Live матч"], index=0)

    if mode == "Игра":
        render_game_mode(cards_status=cards_status)
    elif mode == "Тренировка":
        render_training_mode()
    else:
        render_live_mode(cards_status=cards_status)


if __name__ == "__main__":
    main()

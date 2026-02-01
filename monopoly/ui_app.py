from __future__ import annotations

import csv
import html as html_lib
import os
import random
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import yaml

from .engine import create_engine
from .io_utils import read_json, tail_lines, write_json_atomic
from .league import load_index
from .params import BotParams, ThinkingConfig, load_params
from .roster import BOT_NAME_BASELINE, build_roster_all_top1, build_roster_top1_plus_random
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
    "brown": "#8b4a2f",
    "light_blue": "#7fc8f0",
    "pink": "#d764a7",
    "orange": "#f08a24",
    "red": "#e04b3f",
    "yellow": "#f2d24b",
    "green": "#2f8f4e",
    "blue": "#1f4fb2",
}

PLAYER_COLORS = [
    "#d45d4c",
    "#2d7dd2",
    "#2a9d8f",
    "#f4a261",
    "#6d597a",
    "#1b9aaa",
]

BOARD_STYLE = {
    "cell_w_min": 100,
    "cell_w_max": 135,
    "cell_h_min": 88,
    "cell_h_max": 118,
    "cell_vh": 8.2,
    "gap": 2,
    "pad": 6,
    "font_base": 14,
    "font_small": 12,
    "color_h": 18,
    "color_side": 7,
    "event_h": 260,
    "tint_alpha": 0.08,
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 1) - 8)


def _html_escape(text: str) -> str:
    return html_lib.escape(str(text), quote=True)


def _short_hash(value: str, length: int = 8) -> str:
    value = str(value or "")
    return value[:length]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _rgba_from_hex(value: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(value)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _unknown_group_ids(board: list[Any]) -> list[str]:
    unknown: set[str] = set()
    for cell in board:
        if str(_get(cell, "cell_type", "")) != "property":
            continue
        group = _get(cell, "group")
        if not group:
            continue
        group_str = str(group)
        if group_str not in GROUP_COLORS:
            unknown.add(group_str)
    return sorted(unknown)


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


def _icon_html(cell_type: str) -> str:
    icon_names = {
        "railroad": "train",
        "utility": "bolt",
        "tax": "payments",
        "chance": "casino",
        "community": "volunteer_activism",
        "jail": "local_police",
        "go": "flag",
        "free_parking": "local_parking",
        "go_to_jail": "gavel",
        "property": "home",
    }
    fallback = {
        "railroad": "🚆",
        "utility": "⚡",
        "tax": "💰",
        "chance": "❓",
        "community": "🎁",
        "jail": "⛓",
        "go": "▶",
        "free_parking": "🅿️",
        "go_to_jail": "↘",
        "property": "■",
    }
    name = icon_names.get(cell_type)
    if not name:
        return _cell_icon(cell_type)
    return (
        "<span class='icon-wrap'>"
        f"<span class='ms-icon' aria-hidden='true'>{name}</span>"
        f"<span class='icon-fallback' aria-hidden='true'>{fallback.get(cell_type, '')}</span>"
        "</span>"
    )


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
    unknown_groups: list[str] | None = None,
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
    if thinking and thinking.get("thinking"):
        decision_ctx = thinking.get("decision_context", "")
        if "ms" in thinking:
            ms = float(thinking.get("ms", 0.0))
            candidates = int(thinking.get("candidates", 0))
            rollouts = int(thinking.get("rollouts", 0))
            best_score = float(thinking.get("best_score", 0.0))
            thinking_html = (
                f"<div class='center-meta'>Думал: {ms:.0f} ms · "
                f"Канд.: {candidates} · Rollouts: {rollouts}</div>"
                f"<div class='center-meta'>Контекст: {decision_ctx} · "
                f"Best: {best_score:.3f}</div>"
            )
        else:
            thinking_html = (
                f"<div class='center-meta'>Думает… {decision_ctx}</div>"
                f"<div class='center-meta'>Rollouts: {thinking.get('rollouts_done',0)} | "
                f"Time left: {thinking.get('time_left_sec',0):.2f}s</div>"
            )

    events_tail = event_log[-30:]
    events_html = "".join(f"<div class='event-line'>{_event_msg(ev)}</div>" for ev in events_tail)
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
    active_money = int(_get(active_player, "money", 0)) if active_player else 0
    last_roll_text = _event_msg(last_roll) if last_roll else "—"
    jail_text = "Да" if active_player and _get(active_player, "in_jail", False) else "Нет"

    legend_icons = (
        f"{_icon_html('property')} улица, {_icon_html('railroad')} ЖД, {_icon_html('utility')} коммуналка, "
        f"{_icon_html('tax')} налог, {_icon_html('chance')} шанс, {_icon_html('community')} казна, "
        f"{_icon_html('jail')} тюрьма, {_icon_html('go')} старт, {_icon_html('free_parking')} парковка, "
        f"{_icon_html('go_to_jail')} в тюрьму"
    )
    legend_badges = (
        "<span class='badge badge-mort'>ИП</span> ипотека, "
        "<span class='house'></span><span class='house'></span><span class='house'></span><span class='house'></span> дома, "
        "<span class='hotel'>★</span> отель"
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

    unknown_html = ""
    if unknown_groups:
        safe_groups = ", ".join(_html_escape(group) for group in unknown_groups)
        unknown_html = f"<div class='center-alert'>Неизвестные group_id: {safe_groups}</div>"

    return f"""
    <div class='center-grid'>
      {unknown_html}
      <div class='center-block center-current'>
        <div class='center-title'>Текущий ход</div>
        <div class='center-value'>{active_player_name}</div>
        <div class='center-meta'>Деньги: {active_money}</div>
        <div class='center-meta'>Клетка: {active_cell_name}</div>
        <div class='center-meta'>Кубики: {last_roll_text}</div>
        <div class='center-meta'>Тюрьма: {jail_text}</div>
        {thinking_html}
      </div>
      <div class='center-block center-events'>
        <div class='center-title'>События</div>
        <div class='event-list'>{events_html}</div>
      </div>
      <div class='center-block center-players'>
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
      <div class='center-block center-legend'>
        <details class='legend'>
          <summary>Легенда</summary>
          <div class='center-meta'>{legend_badges}</div>
          <div class='center-meta'>{legend_icons}</div>
          <div class='center-meta'>Периметр = 40 клеток, центр 9×9 под панель.</div>
          <div class='center-meta'>Шаг аукциона: 5/20/50 (адаптивно).</div>
          <div class='center-meta'>Офиц. тексты: <span class='{official_class}'>{official_texts}</span></div>
          <div class='center-meta'>Управление — кнопки под доской.</div>
        </details>
      </div>
    </div>
    """


# ---------------------------------------------------------
# Board rendering
# ---------------------------------------------------------

def _build_board_html(
    board: list[Any],
    players: list[Any],
    active_player_id: int | None,
    center_html: str,
    show_group_id: bool,
) -> tuple[str, int]:
    rows = 11
    cell_w_min_px = int(BOARD_STYLE["cell_w_min"])
    cell_w_max_px = int(BOARD_STYLE["cell_w_max"])
    cell_h_min_px = int(BOARD_STYLE["cell_h_min"])
    cell_h_max_px = int(BOARD_STYLE["cell_h_max"])
    cell_vh = float(BOARD_STYLE["cell_vh"])
    cell_w_vh = cell_vh + 0.9
    gap_px = int(BOARD_STYLE["gap"])
    pad_px = int(BOARD_STYLE["pad"])
    font_base = float(BOARD_STYLE["font_base"])
    font_small = float(BOARD_STYLE["font_small"])
    color_h = int(BOARD_STYLE["color_h"])
    color_side = int(BOARD_STYLE["color_side"])
    event_h = int(BOARD_STYLE["event_h"])
    tint_alpha = float(BOARD_STYLE["tint_alpha"])
    extra_px = 32
    min_iframe_height = 700

    iframe_height = rows * cell_h_max_px + (rows - 1) * gap_px + 2 * pad_px + extra_px
    if iframe_height < min_iframe_height:
        iframe_height = min_iframe_height

    coords = _perimeter_coords()

    players_at = {pos: [] for pos in range(40)}
    for player in players:
        pos = int(_get(player, "position", 0))
        players_at[pos].append(player)

    active_position = None
    if active_player_id is not None and 0 <= int(active_player_id) < len(players):
        active_position = int(_get(players[int(active_player_id)], "position", 0))

    html_cells: list[str] = []
    for idx, cell in enumerate(board):
        row, col = coords[idx]
        owner_id = _get(cell, "owner_id")
        cell_type = str(_get(cell, "cell_type", ""))

        mort_text = "<span class='badge badge-mort'>ИП</span>" if _get(cell, "mortgaged", False) else ""
        owner_ribbon = ""
        if owner_id is not None and 0 <= int(owner_id) < len(players):
            owner_idx = int(owner_id)
            owner_color = PLAYER_COLORS[owner_idx]
            owner_label = f"Вл P{owner_idx + 1}"
            owner_ribbon = (
                f"<div class='owner-ribbon' style='--ownerColor: {owner_color};'>"
                f"<span class='owner-label'>{owner_label}</span>"
                f"{mort_text}"
                "</div>"
            )

        build_text = ""
        if _get(cell, "hotels", 0):
            build_text = "<span class='hotel'>★</span>"
        elif _get(cell, "houses", 0):
            build_text = "<span class='house'></span>" * int(_get(cell, "houses", 0))

        tokens = " ".join(
            [
                f"<span class='presence p{int(_get(p, 'player_id', 0)) + 1} {'active' if int(_get(p, 'player_id', 0)) == active_player_id else ''}'>{int(_get(p, 'player_id', 0)) + 1}</span>"
                for p in players_at[idx]
            ]
        )
        presence_text = f"{tokens}" if tokens else ""

        type_label = _cell_type_label(cell_type)
        type_icon = _icon_html(cell_type)
        group_id = str(_get(cell, "group", "")) if _get(cell, "group") else ""
        group_color = GROUP_COLORS.get(group_id) if cell_type == "property" else ""
        group_tint = _rgba_from_hex(group_color, tint_alpha) if group_color else ""
        color_cap = "<div class='color-cap'></div>" if group_color else ""
        color_side_html = "<div class='color-side'></div>" if group_color else ""
        corner_class = "corner" if idx in {0, 10, 20, 30} else ""
        active_class = "active" if active_position == idx else ""
        type_class = f"type-{cell_type}"
        street_class = "street" if cell_type == "property" else ""
        cell_name_raw = _get(cell, "name", "")
        cell_name = _html_escape(cell_name_raw)
        meta_parts = []
        if _get(cell, "price") is not None:
            meta_parts.append(f"Цена {int(_get(cell, 'price'))}")
        if _get(cell, "tax_amount") is not None:
            meta_parts.append(f"Налог {int(_get(cell, 'tax_amount'))}")
        meta_text = " · ".join(meta_parts)
        group_debug = (
            f"<div class='cell-group'>{_html_escape(group_id)}</div>"
            if show_group_id and group_id and cell_type == "property"
            else ""
        )
        style_parts = [f"grid-row:{row + 1}; grid-column:{col + 1};"]
        if group_color:
            style_parts.append(f"--groupColor: {group_color}; --groupTint: {group_tint};")
        style_attr = " ".join(style_parts)
        html_cells.append(
            f"""
            <div class='cell {type_class} {street_class} {corner_class} {active_class}' style='{style_attr}'>
              {color_side_html}
              {color_cap}
              {owner_ribbon}
              <div class='cell-body'>
                <div class='cell-title' title='{cell_name}'>{cell_name}</div>
                {group_debug}
                <div class='cell-type'>{type_icon}<span class='cell-type-label'>{type_label}</span></div>
                <div class='cell-meta'>{meta_text}</div>
              </div>
              <div class='cell-buildings'>{build_text}</div>
              <div class='cell-presence'>{presence_text}</div>
            </div>
            """
        )

    html = f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Material+Symbols+Outlined&display=swap">
    <style>
      :root {{
        color-scheme: light;
      }}
      *, *::before, *::after {{
        box-sizing: border-box;
      }}
      html, body {{
        margin: 0;
        padding: 0;
      }}
      .board-grid {{
        --cellW: clamp({cell_w_min_px}px, {cell_w_vh}vh, {cell_w_max_px}px);
        --cellH: clamp({cell_h_min_px}px, {cell_vh}vh, {cell_h_max_px}px);
        --gap: {gap_px}px;
        --pad: {pad_px}px;
        --fontBase: {font_base}px;
        --fontSmall: {font_small}px;
        --colorH: {color_h}px;
        --colorSide: {color_side}px;
        display: grid;
        grid-template-columns: repeat({rows}, minmax(var(--cellW), 1fr));
        grid-template-rows: repeat({rows}, var(--cellH));
        gap: var(--gap);
        background: #e9e2d5;
        padding: var(--pad);
        border-radius: 16px;
        position: relative;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        color: #2f2a24;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
      }}
      .icon-wrap {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
      }}
      .ms-icon {{
        font-family: 'Material Symbols Outlined', 'Segoe UI Symbol', sans-serif;
        font-size: 14px;
        line-height: 1;
      }}
      .icon-fallback {{
        display: none;
        font-size: 13px;
        line-height: 1;
      }}
      .no-icon-font .ms-icon {{
        display: none;
      }}
      .no-icon-font .icon-fallback {{
        display: inline;
      }}
      .cell {{
        background: #f8f4ee;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        padding: 7px;
        font-size: var(--fontSmall);
        line-height: 1.15;
        display: flex;
        flex-direction: column;
        gap: 4px;
        position: relative;
      }}
      .cell.street {{
        background: var(--groupTint, #f8f4ee);
        padding-left: calc(7px + var(--colorSide));
      }}
      .cell.active {{
        border: 2px solid #c24b2a;
        box-shadow: 0 0 0 2px rgba(194, 75, 42, 0.15);
        background: #fff6ef;
      }}
      .cell.corner {{
        background: #f3eadc;
        border: 2px solid rgba(0, 0, 0, 0.18);
        font-weight: 700;
      }}
      .cell.corner .cell-title {{
        font-size: 13px;
        text-align: center;
      }}
      .color-cap {{
        height: var(--colorH);
        border-radius: 4px;
        background: var(--groupColor, transparent);
        margin-bottom: 2px;
      }}
      .color-side {{
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: var(--colorSide);
        background: var(--groupColor, transparent);
        border-radius: 8px 0 0 8px;
      }}
      .owner-ribbon {{
        position: absolute;
        top: 6px;
        right: 6px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 6px;
        border-radius: 10px;
        background: var(--ownerColor, #8b8b8b);
        color: #fff;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.1px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.18);
        z-index: 2;
      }}
      .cell.street .owner-ribbon {{
        top: calc(var(--colorH) + 6px);
      }}
      .owner-label {{
        white-space: nowrap;
      }}
      .cell-body {{
        display: flex;
        flex-direction: column;
        gap: 3px;
        padding-bottom: 20px;
      }}
      .cell-title {{
        font-weight: 700;
        font-size: calc(var(--fontBase) + 0.5px);
        line-height: 1.15;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .cell-group {{
        font-size: 9px;
        color: #7b6a59;
      }}
      .cell-type {{
        font-size: var(--fontSmall);
        color: #5a4c3c;
        display: inline-flex;
        align-items: center;
        gap: 4px;
      }}
      .cell-type-label {{
        text-transform: none;
        letter-spacing: 0;
      }}
      .cell-meta {{
        font-size: var(--fontSmall);
        color: #6b5b4b;
      }}
      .cell-buildings {{
        position: absolute;
        right: 6px;
        bottom: 6px;
        display: flex;
        gap: 2px;
        min-height: 12px;
      }}
      .cell-presence {{
        position: absolute;
        left: 6px;
        bottom: 6px;
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
      }}
      .presence {{
        width: 15px;
        height: 15px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 9px;
        color: #fff;
        border: 1px solid rgba(0, 0, 0, 0.22);
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.7) inset;
      }}
      .presence.p1 {{ background: {PLAYER_COLORS[0]}; }}
      .presence.p2 {{ background: {PLAYER_COLORS[1]}; }}
      .presence.p3 {{ background: {PLAYER_COLORS[2]}; }}
      .presence.p4 {{ background: {PLAYER_COLORS[3]}; }}
      .presence.p5 {{ background: {PLAYER_COLORS[4]}; }}
      .presence.p6 {{ background: {PLAYER_COLORS[5]}; }}
      .presence.active {{
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.9), 0 0 6px rgba(0, 0, 0, 0.25);
      }}
      .badge {{
        display: inline-flex;
        align-items: center;
        padding: 1px 5px;
        border-radius: 8px;
        font-size: 9px;
        line-height: 1.2;
        font-weight: 700;
      }}
      .badge-mort {{
        background: #f3c7c7;
        color: #8c1f1f;
        padding: 0 4px;
        font-size: 8px;
      }}
      .house {{
        width: 8px;
        height: 8px;
        background: #2f7d2f;
        border-radius: 2px;
        display: inline-block;
      }}
      .hotel {{
        color: #b22222;
        font-size: 12px;
        line-height: 1;
      }}
      .cell.type-railroad {{ background: #f2f4f7; }}
      .cell.type-utility {{ background: #eef3f9; }}
      .cell.type-tax {{ background: #f9eeee; }}
      .cell.type-chance {{ background: #f2eef9; }}
      .cell.type-community {{ background: #eef7ef; }}
      .cell.type-jail {{ background: #f3f1ec; }}
      .cell.type-free_parking {{ background: #f4f0ea; }}
      .cell.type-go_to_jail {{ background: #f6efe9; }}
      .board-center {{
        grid-row: 2 / span 9;
        grid-column: 2 / span 9;
        background: #fdf8f1;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 12px;
        padding: 14px;
        overflow: hidden;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.6);
      }}
      .center-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        grid-template-areas:
          "alert alert"
          "current events"
          "players players"
          "legend legend";
        gap: 12px;
        font-size: var(--fontSmall);
      }}
      .center-alert {{
        grid-area: alert;
        background: #fff1e0;
        border: 1px solid #e7b894;
        color: #8a4a2f;
        border-radius: 10px;
        padding: 6px 10px;
        font-weight: 600;
      }}
      .center-block {{
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 10px;
        padding: 10px;
      }}
      .center-current {{ grid-area: current; }}
      .center-events {{ grid-area: events; }}
      .center-players {{ grid-area: players; }}
      .center-legend {{ grid-area: legend; }}
      .center-title {{
        font-weight: 700;
        font-size: var(--fontSmall);
        margin-bottom: 6px;
      }}
      .center-value {{
        font-size: calc(var(--fontBase) + 1px);
        font-weight: 700;
      }}
      .center-meta {{
        font-size: var(--fontSmall);
        color: #5d5141;
      }}
      .event-list {{
        max-height: {event_h}px;
        overflow-y: auto;
        padding-right: 4px;
      }}
      .event-line {{
        font-size: var(--fontSmall);
        color: #4b4035;
        margin-bottom: 2px;
      }}
      .event-highlight {{
        font-size: var(--fontSmall);
        font-weight: 700;
        margin-bottom: 6px;
      }}
      .players-table {{
        width: 100%;
        font-size: var(--fontSmall);
        border-collapse: collapse;
      }}
      .players-table th, .players-table td {{
        text-align: left;
        padding: 3px 4px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
      }}
      details.legend summary {{
        cursor: pointer;
        font-weight: 700;
        margin-bottom: 6px;
      }}
      @media (max-width: 900px) {{
        .center-grid {{
          grid-template-columns: 1fr;
          grid-template-areas:
            "alert"
            "current"
            "events"
            "players"
            "legend";
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
    <script>
      try {{
        const hasIconFont = document.fonts && document.fonts.check("12px 'Material Symbols Outlined'");
        if (!hasIconFont) {{
          document.documentElement.classList.add('no-icon-font');
        }}
      }} catch (e) {{
        document.documentElement.classList.add('no-icon-font');
      }}
    </script>
     <div class='board-grid'>
       <div class='board-center'>
         {center_html}
       </div>
       {"".join(html_cells)}
     </div>
     """
    return textwrap.dedent(html).strip(), iframe_height


def _render_board(
    board: list[Any],
    players: list[Any],
    active_player_id: int | None,
    center_html: str,
    show_group_id: bool,
) -> None:
    html, iframe_height = _build_board_html(
        board,
        players,
        active_player_id,
        center_html,
        show_group_id,
    )
    if html.count("class='cell") < 40:
        st.error("board html empty")
        return
    components.html(html, height=iframe_height, scrolling=False)


# ---------------------------------------------------------
# Actions and modes
# ---------------------------------------------------------

def _start_autoevolve(
    workers: int,
    population: int,
    elite: int,
    epoch_iters: int,
    plateau_epochs: int,
    bench_min_games: int,
    bench_max_games: int,
    plateau_delta: float,
    games_per_cand: int,
    auto_games_per_cand: bool,
    games_per_cand_min: int,
    games_per_cand_max: int,
    games_per_cand_target_ci: float,
    max_steps: int,
    seed: int,
    top_k_pool: int,
    league_cap: int,
    max_new_bests: int,
    meta_plateau_cycles: int,
    bootstrap_min_league_for_pool: int,
    league_rebench_on_mismatch: bool,
    league_rebench_games: int,
    league_dir: str,
    baseline_path: str,
    runs_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    if runs_dir is None:
        runs_dir = ROOT_DIR / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    error_log = runs_dir / "error.log"
    cmd = [
        sys.executable,
        "-m",
        "monopoly.autoevolve",
        "run",
        "--workers",
        str(workers),
        "--plateau-epochs",
        str(plateau_epochs),
        "--epoch-iters",
        str(epoch_iters),
        "--population",
        str(population),
        "--elite",
        str(elite),
        "--min-progress-games",
        str(bench_min_games),
        "--bench-max-games",
        str(bench_max_games),
        "--eps-winrate",
        str(plateau_delta),
        "--eps-fitness",
        str(plateau_delta),
        "--delta",
        str(plateau_delta),
        "--games-per-cand",
        str(games_per_cand),
        "--auto-games-per-cand",
        str(auto_games_per_cand).lower(),
        "--games-per-cand-min",
        str(games_per_cand_min),
        "--games-per-cand-max",
        str(games_per_cand_max),
        "--games-per-cand-target-ci",
        str(games_per_cand_target_ci),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--top-k-pool",
        str(top_k_pool),
        "--league-cap",
        str(league_cap),
        "--max-new-bests",
        str(max_new_bests),
        "--meta-plateau-cycles",
        str(meta_plateau_cycles),
        "--bootstrap-min-league-for-pool",
        str(bootstrap_min_league_for_pool),
        "--league-rebench-on-mismatch",
        str(league_rebench_on_mismatch).lower(),
        "--league-rebench-games",
        str(league_rebench_games),
        "--league-dir",
        league_dir,
        "--baseline",
        baseline_path,
        "--runs-dir",
        str(runs_dir),
    ]
    if resume:
        cmd.append("--resume")
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("ab") as err_handle:
        proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=err_handle, stderr=err_handle)
    st.session_state.train_proc = proc
    st.session_state.train_runs_dir = str(runs_dir)
    st.session_state.train_error_log = str(error_log)
    return runs_dir


def _stop_autoevolve(phase: str) -> None:
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
    thinking_enabled: bool,
    workers: int,
    time_per_decision: float,
    horizon: int,
    rollouts_per_action: int,
    cache_enabled: bool,
    cache_size: int,
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
        "--workers",
        str(workers),
        "--time-per-decision-sec",
        str(time_per_decision),
        "--horizon-turns",
        str(horizon),
        "--rollouts-per-action",
        str(rollouts_per_action),
        "--cache-size",
        str(cache_size),
        "--seed",
        str(seed),
        "--out",
        str(out_path),
    ]
    if thinking_enabled:
        cmd.append("--thinking")
    else:
        cmd.extend(["--mode", "fast"])
    if cache_enabled:
        cmd.append("--cache")
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
        st.subheader("Состав ботов")
        roster_mode = st.radio(
            "Режим состава",
            ["Все TOP-1", "TOP-1 + 5 случайных из лиги", "Baseline/кастом"],
            index=0,
        )
        league_dir = st.text_input(
            "League dir",
            value=str(ROOT_DIR / "monopoly" / "data" / "league"),
        )
        baseline_path = st.text_input(
            "Путь к baseline/кастом параметрам (json/yaml)",
            value=str(ROOT_DIR / "monopoly" / "data" / "params_baseline.json"),
            placeholder="trained_params.json",
        )
        st.subheader("Thinking-mode")
        thinking_enabled = st.checkbox("Включить thinking-mode", value=True)
        thinking_workers = st.number_input(
            "Thinking workers",
            min_value=1,
            value=_default_workers(),
            step=1,
            disabled=not thinking_enabled,
        )
        thinking_horizon = st.number_input(
            "Horizon turns",
            min_value=1,
            value=60,
            step=1,
            disabled=not thinking_enabled,
        )
        thinking_rollouts = st.number_input(
            "Rollouts per action (0 = auto)",
            min_value=0,
            value=0,
            step=1,
            disabled=not thinking_enabled,
        )
        thinking_time_ms = st.number_input(
            "Time budget (ms, 0 = off)",
            min_value=0,
            value=2000,
            step=100,
            disabled=not thinking_enabled,
        )
        thinking_cache = st.checkbox("Cache", value=True, disabled=not thinking_enabled)
        thinking_cache_size = st.number_input(
            "Cache size",
            min_value=0,
            value=4096,
            step=256,
            disabled=not thinking_enabled or not thinking_cache,
        )
        new_game = st.button("Новая игра", type="primary")
        step_once = st.button("Шаг")
        step_ten = st.button("+10 шагов")
        step_hundred = st.button("+100 шагов")
        run_to_end = st.button("До конца игры")

    if "engine" not in st.session_state or new_game:
        baseline_params = BotParams()
        if baseline_path:
            try:
                baseline_params = load_params(baseline_path)
            except Exception as exc:
                st.warning(f"Не удалось загрузить параметры: {exc}. Использую дефолтные.")
                baseline_params = BotParams()

        league_dir_path = Path(league_dir) if league_dir else (ROOT_DIR / "monopoly" / "data" / "league")
        if roster_mode == "Все TOP-1":
            roster_params, roster_names = build_roster_all_top1(
                league_dir_path,
                baseline_params,
                num_players,
            )
        elif roster_mode == "TOP-1 + 5 случайных из лиги":
            roster_params, roster_names = build_roster_top1_plus_random(
                league_dir_path,
                baseline_params,
                num_players,
            )
        else:
            roster_params = [baseline_params] * num_players
            roster_names = [BOT_NAME_BASELINE] * num_players

        if thinking_enabled:
            rollouts_value = int(thinking_rollouts)
            if rollouts_value <= 0:
                rollouts_value = 12
            config = ThinkingConfig(
                enabled=True,
                horizon_turns=int(thinking_horizon),
                rollouts_per_action=rollouts_value,
                time_budget_ms=int(thinking_time_ms),
                workers=int(thinking_workers),
                cache_enabled=bool(thinking_cache),
                cache_size=int(thinking_cache_size),
            )
            roster_params = [params.with_thinking(config) for params in roster_params]
        else:
            roster_params = [params.with_thinking(ThinkingConfig()) for params in roster_params]
        st.session_state.engine = create_engine(num_players, seed, bot_params=roster_params)
        for player, name in zip(st.session_state.engine.state.players, roster_names, strict=False):
            player.name = name
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

    unknown_groups = _unknown_group_ids(state.board)
    show_group_id = bool(st.session_state.get("show_group_id", False))
    thinking_info = None
    if hasattr(engine, "bots") and engine.bots:
        try:
            bot = engine.bots[state.current_player]
            last = getattr(bot, "last_thinking", None)
            if isinstance(last, dict) and last:
                thinking_info = {
                    "thinking": True,
                    "decision_context": last.get("decision_type", ""),
                    "ms": last.get("ms", 0.0),
                    "candidates": last.get("candidates", 0),
                    "rollouts": last.get("rollouts", 0),
                    "best_score": last.get("best_score", 0.0),
                }
        except Exception:
            thinking_info = None

    center_html = _build_center_panel(
        state,
        mode="game",
        thinking=thinking_info,
        cards_status=cards_status,
        unknown_groups=unknown_groups,
    )
    _render_board(state.board, state.players, state.current_player, center_html, show_group_id)

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
        st.caption("Auto-evolve: bootstrap -> league -> meta-cycle")
        top_k_pool = st.number_input("Top-K pool", min_value=1, value=16, step=1)
        league_cap = st.number_input("League cap", min_value=1, value=16, step=1)
        max_new_bests = st.number_input("Max new bests", min_value=1, value=16, step=1)
        meta_plateau_cycles = st.number_input("Meta-plateau cycles", min_value=1, value=3, step=1)
        bootstrap_min_league_for_pool = st.number_input(
            "Bootstrap min league",
            min_value=0,
            value=8,
            step=1,
        )
        league_rebench_on_mismatch = st.checkbox(
            "Re-benchmark при несовпадении протокола fitness",
            value=True,
        )
        league_dir = st.text_input(
            "League dir",
            value=str(ROOT_DIR / "monopoly" / "data" / "league"),
        )
        baseline_path = st.text_input(
            "Baseline params",
            value=str(ROOT_DIR / "monopoly" / "data" / "params_baseline.json"),
        )

        with st.expander("Advanced"):
            if "train_seed" not in st.session_state:
                st.session_state.train_seed = random.randint(0, 999999)
            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=999999,
                value=int(st.session_state.train_seed),
                step=1,
                key="train_seed",
            )
            workers = st.number_input("Workers", min_value=1, value=_default_workers(), step=1)
            population = st.number_input("Population", min_value=16, value=64, step=1)
            elite = st.number_input("Elite", min_value=4, value=16, step=1)
            auto_games_per_cand = st.checkbox("Авто games per candidate", value=True)
            games_label = "Games per candidate (стартовое)" if auto_games_per_cand else "Games per candidate"
            games_per_cand = st.number_input(games_label, min_value=1, value=32, step=1)
            games_per_cand_min = 8
            games_per_cand_max = 128
            games_per_cand_target_ci = 0.10
            if auto_games_per_cand:
                games_per_cand_min = st.number_input("Games per candidate min", min_value=1, value=8, step=1)
                games_per_cand_max = st.number_input(
                    "Games per candidate max",
                    min_value=int(games_per_cand_min),
                    value=128,
                    step=1,
                )
                games_per_cand_target_ci = st.number_input(
                    "Target win-rate CI width",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.10,
                    step=0.01,
                    format="%.2f",
                )
                if not (games_per_cand_min <= games_per_cand <= games_per_cand_max):
                    st.warning("Стартовое Games per candidate вне диапазона min/max; будет приведено к границам.")
                st.caption("Авто использует ширину CI win-rate из предыдущего цикла (last_bench.json).")
            league_rebench_games = st.number_input(
                "Games for re-benchmark",
                min_value=10,
                value=256,
                step=10,
                disabled=not league_rebench_on_mismatch,
            )
            bench_max_games = st.number_input("Bench games (max)", min_value=32, value=512, step=32)
            bench_min_games = st.number_input("Early-stop min games", min_value=10, value=128, step=10)
            max_steps = st.number_input("Max steps", min_value=128, value=2048, step=1)
            epoch_iters = st.number_input("Epoch iters", min_value=1, value=10, step=1)
            plateau_epochs = st.number_input("Plateau epochs", min_value=1, value=1, step=1)
            plateau_delta = st.number_input(
                "Plateau delta",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )
            if bench_min_games > bench_max_games:
                st.warning("Early-stop min games должен быть <= Bench games (max).")

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
        elif bench_min_games > bench_max_games:
            st.error("Early-stop min games должен быть <= Bench games (max).")
        else:
            _start_autoevolve(
                workers=int(workers),
                population=int(population),
                elite=int(elite),
                epoch_iters=int(epoch_iters),
                plateau_epochs=int(plateau_epochs),
                bench_min_games=int(bench_min_games),
                bench_max_games=int(bench_max_games),
                plateau_delta=float(plateau_delta),
                games_per_cand=int(games_per_cand),
                auto_games_per_cand=bool(auto_games_per_cand),
                games_per_cand_min=int(games_per_cand_min),
                games_per_cand_max=int(games_per_cand_max),
                games_per_cand_target_ci=float(games_per_cand_target_ci),
                max_steps=int(max_steps),
                seed=int(seed),
                top_k_pool=int(top_k_pool),
                league_cap=int(league_cap),
                max_new_bests=int(max_new_bests),
                meta_plateau_cycles=int(meta_plateau_cycles),
                bootstrap_min_league_for_pool=int(bootstrap_min_league_for_pool),
                league_rebench_on_mismatch=bool(league_rebench_on_mismatch),
                league_rebench_games=int(league_rebench_games),
                league_dir=league_dir,
                baseline_path=baseline_path,
            )

    if pause_btn:
        _stop_autoevolve("paused")

    if stop_btn:
        _stop_autoevolve("stopped")

    if resume_btn:
        runs_dir_raw = st.session_state.get("train_runs_dir")
        if runs_dir_raw:
            if bench_min_games > bench_max_games:
                st.error("Early-stop min games должен быть <= Bench games (max).")
            else:
                _start_autoevolve(
                    workers=int(workers),
                    population=int(population),
                    elite=int(elite),
                    epoch_iters=int(epoch_iters),
                    plateau_epochs=int(plateau_epochs),
                    bench_min_games=int(bench_min_games),
                    bench_max_games=int(bench_max_games),
                    plateau_delta=float(plateau_delta),
                    games_per_cand=int(games_per_cand),
                    auto_games_per_cand=bool(auto_games_per_cand),
                    games_per_cand_min=int(games_per_cand_min),
                    games_per_cand_max=int(games_per_cand_max),
                    games_per_cand_target_ci=float(games_per_cand_target_ci),
                    max_steps=int(max_steps),
                    seed=int(seed),
                    top_k_pool=int(top_k_pool),
                    league_cap=int(league_cap),
                    max_new_bests=int(max_new_bests),
                    meta_plateau_cycles=int(meta_plateau_cycles),
                    bootstrap_min_league_for_pool=int(bootstrap_min_league_for_pool),
                    league_rebench_on_mismatch=bool(league_rebench_on_mismatch),
                    league_rebench_games=int(league_rebench_games),
                    league_dir=league_dir,
                    baseline_path=baseline_path,
                    runs_dir=Path(runs_dir_raw),
                    resume=True,
                )

    runs_dir_raw = st.session_state.get("train_runs_dir")
    if runs_dir_raw:
        st.caption(f"runs_dir: {runs_dir_raw}")

    runs_dir = Path(runs_dir_raw) if runs_dir_raw else None
    meta_status = None
    cycle_status = None
    cycle_dir = None
    if runs_dir and (runs_dir / "status.json").exists():
        meta_status = read_json(runs_dir / "status.json", default=None)
        if isinstance(meta_status, dict):
            cycle_dir_raw = meta_status.get("current_cycle_dir")
            if cycle_dir_raw:
                cycle_dir = Path(str(cycle_dir_raw))
                if not cycle_dir.is_absolute():
                    cycle_dir = (ROOT_DIR / cycle_dir).resolve()
                if (cycle_dir / "status.json").exists():
                    try:
                        cycle_status = read_status(cycle_dir / "status.json")
                    except Exception as exc:
                        st.warning(f"cycle status прочитать не удалось: {exc}")

    proc = st.session_state.get("train_proc")
    if proc and proc.poll() is not None and proc.returncode not in (0, None):
        error_log = st.session_state.get("train_error_log")
        if error_log:
            tail = tail_lines(Path(error_log), max_lines=100)
            if tail:
                st.error("Тренировка завершилась с ошибкой. stderr:")
                st.code("\n".join(tail))

    if isinstance(meta_status, dict):
        st_autorefresh(interval=1000, key="train_refresh")
        new_bests = int(meta_status.get("new_bests_count", 0) or 0)
        max_bests = int(meta_status.get("max_new_bests", 0) or 0)
        meta_plateau = int(meta_status.get("meta_plateau", 0) or 0)
        meta_plateau_cap = int(meta_status.get("meta_plateau_cycles", 0) or 0)
        league_size = int(meta_status.get("league_size", 0) or 0)
        pool_snapshot = meta_status.get("pool_snapshot", [])
        pool_size = len(pool_snapshot) if isinstance(pool_snapshot, list) else 0
        bootstrap_mode = bool(meta_status.get("bootstrap", False))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Meta cycle", int(meta_status.get("current_cycle", 0) or 0))
        col2.metric("New bests", f"{new_bests}/{max_bests}")
        col3.metric("Meta plateau", f"{meta_plateau}/{meta_plateau_cap}")
        col4.metric("League size", league_size)

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Bootstrap", "Да" if bootstrap_mode else "Нет")
        col6.metric("Pool size", pool_size)
        col7.metric("Top-K pool", int(meta_status.get("top_k_pool", 0) or 0))
        col8.metric("League cap", int(meta_status.get("league_cap", 0) or 0))

        rebench_needed = bool(meta_status.get("league_rebench_needed", False))
        rebench_done = bool(meta_status.get("league_rebench_done", False))
        col_rb1, col_rb2 = st.columns(2)
        col_rb1.metric("Re-benchmark needed", "Да" if rebench_needed else "Нет")
        col_rb2.metric("Re-benchmark done", "Да" if rebench_done else "Нет")

        auto_games = bool(meta_status.get("auto_games_per_cand", False))
        games_current = int(
            meta_status.get("games_per_cand_current")
            or meta_status.get("games_per_cand", 0)
            or 0
        )
        if auto_games:
            prev_ci_width = meta_status.get("games_per_cand_prev_ci_width")
            target_ci = meta_status.get("games_per_cand_target_ci")
            prev_ci_text = f"{prev_ci_width:.3f}" if isinstance(prev_ci_width, (int, float)) else "—"
            target_ci_text = f"{float(target_ci):.2f}" if isinstance(target_ci, (int, float)) else "—"
            st.caption(
                "Games per candidate (auto): "
                f"{games_current} | prev CI width: {prev_ci_text} | target CI: {target_ci_text}"
            )
        else:
            st.caption(f"Games per candidate: {games_current}")

        if cycle_status:
            col9, col10, col11, col12 = st.columns(4)
            col9.metric("Epoch", cycle_status["epoch"])
            col10.metric(
                "Best win-rate",
                f"{cycle_status['best_winrate_mean']:.3f}",
                help=(
                    f"CI [{cycle_status['best_winrate_ci_low']:.3f}, "
                    f"{cycle_status['best_winrate_ci_high']:.3f}]"
                ),
            )
            col11.metric("Best fitness", f"{cycle_status['best_fitness']:.4f}")
            col12.metric("Games simulated", int(cycle_status["total_games_simulated"]))

            col13, col14, col15, col16 = st.columns(4)
            col13.metric("Phase", cycle_status["current_phase"])
            col14.metric(
                "Plateau counter",
                f"{cycle_status['plateau_counter']}/{cycle_status['plateau_epochs']}",
            )
            col15.metric("Promoted", int(cycle_status["promoted_count"]))
            col16.metric("Last promoted", int(cycle_status["last_promoted_epoch"]))

            st.write(f"Cache hits: **{int(cycle_status['cache_hits_last_epoch'])}**")

        league_items = []
        try:
            league_index = load_index(Path(league_dir))
            league_items = league_index.get("items", [])
        except Exception as exc:
            st.warning(f"league/index.json прочитать не удалось: {exc}")

        if league_items:
            st.subheader("Top-16 лиги")
            rows = [
                {
                    "rank": entry.get("rank"),
                    "fitness": entry.get("fitness"),
                    "hash": _short_hash(str(entry.get("hash", ""))),
                    "created_at": entry.get("created_at", ""),
                }
                for entry in league_items
            ]
            st.table(rows)

        st.subheader("Top-K snapshot (pool)")
        if isinstance(pool_snapshot, list) and pool_snapshot:
            pool_rows = [
                {
                    "rank": entry.get("rank"),
                    "fitness": entry.get("fitness"),
                    "hash": _short_hash(str(entry.get("hash", ""))),
                }
                for entry in pool_snapshot
            ]
            st.table(pool_rows)
        else:
            st.caption("Пул не сформирован (bootstrap или пустая лига).")

        if cycle_dir:
            log_path = cycle_dir / "train_log.csv"
        else:
            log_path = None
        if log_path and log_path.exists():
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

        if cycle_dir:
            progress_path = cycle_dir / "progress.txt"
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

        if cycle_status and cycle_status["current_phase"] == "finished":
            best_path = Path(cycle_status["best_params_path"])
            if best_path.exists():
                if st.button("Запустить live матч сейчас"):
                    _start_live_match(
                        runs_dir,
                        best_path,
                        thinking_enabled=True,
                        workers=int(_default_workers()),
                        time_per_decision=2.0,
                        horizon=60,
                        rollouts_per_action=0,
                        cache_enabled=True,
                        cache_size=4096,
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
        thinking_enabled = st.checkbox("Thinking-mode", value=True)
        workers = st.number_input("Workers", min_value=1, value=_default_workers(), step=1)
        time_per_decision = st.number_input("Time per decision (sec)", min_value=0.1, value=2.0, step=0.5)
        horizon = st.number_input("Horizon turns", min_value=1, value=60, step=1)
        rollouts_per_action = st.number_input(
            "Rollouts per action (0 = auto)",
            min_value=0,
            value=0,
            step=1,
            disabled=not thinking_enabled,
        )
        cache_enabled = st.checkbox("Cache", value=True, disabled=not thinking_enabled)
        cache_size = st.number_input(
            "Cache size",
            min_value=0,
            value=4096,
            step=256,
            disabled=not thinking_enabled or not cache_enabled,
        )
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        refresh_ms = st.slider("Обновление UI (мс)", min_value=500, max_value=1500, value=500, step=100)
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
            thinking_enabled=bool(thinking_enabled),
            workers=int(workers),
            time_per_decision=float(time_per_decision),
            horizon=int(horizon),
            rollouts_per_action=int(rollouts_per_action),
            cache_enabled=bool(cache_enabled),
            cache_size=int(cache_size),
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

    unknown_groups = _unknown_group_ids(payload.get("board", []))
    show_group_id = bool(st.session_state.get("show_group_id", False))
    center_html = _build_center_panel(
        payload,
        mode="live",
        thinking=thinking,
        cards_status=cards_status,
        unknown_groups=unknown_groups,
    )
    _render_board(
        payload.get("board", []),
        payload.get("players", []),
        payload.get("current_player"),
        center_html,
        show_group_id,
    )


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
        mode = st.radio("Режим", ["Игра", "Тренировка", "Live матч"], index=0, label_visibility="collapsed")
        st.checkbox("Показать group_id", value=False, key="show_group_id")

    if mode == "Игра":
        render_game_mode(cards_status=cards_status)
    elif mode == "Тренировка":
        render_training_mode()
    else:
        render_live_mode(cards_status=cards_status)


if __name__ == "__main__":
    main()

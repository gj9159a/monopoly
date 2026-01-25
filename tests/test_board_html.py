from __future__ import annotations

from pathlib import Path

from monopoly.data_loader import load_board
from monopoly.ui_app import GROUP_COLORS, _build_board_html


def test_board_html_has_cells() -> None:
    board_path = Path(__file__).resolve().parents[1] / "monopoly" / "data" / "board.yaml"
    board = load_board(board_path)
    html, height = _build_board_html(board, [], None, "<div>center</div>", False)

    assert html.count("<div class='cell") >= 40
    assert "<div class='board-center'>" in html
    assert height >= 600


def test_street_group_ids_mapped() -> None:
    board_path = Path(__file__).resolve().parents[1] / "monopoly" / "data" / "board.yaml"
    board = load_board(board_path)
    for cell in board:
        if cell.cell_type != "property":
            continue
        assert cell.group in GROUP_COLORS

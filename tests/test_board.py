from __future__ import annotations

from pathlib import Path

from monopoly.data_loader import load_board


def test_board_layout_and_types():
    board_path = Path(__file__).resolve().parents[1] / "monopoly" / "data" / "board.yaml"
    board = load_board(board_path)

    assert len(board) == 40
    assert [cell.index for cell in board] == list(range(40))

    expected_types = {
        0: "go",
        2: "community",
        4: "tax",
        5: "railroad",
        7: "chance",
        10: "jail",
        12: "utility",
        15: "railroad",
        17: "community",
        20: "free_parking",
        22: "chance",
        25: "railroad",
        28: "utility",
        30: "go_to_jail",
        33: "community",
        35: "railroad",
        36: "chance",
        38: "tax",
    }
    for idx, cell_type in expected_types.items():
        assert board[idx].cell_type == cell_type

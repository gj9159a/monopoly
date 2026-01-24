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


def test_board_canonical_values():
    board_path = Path(__file__).resolve().parents[1] / "monopoly" / "data" / "board.yaml"
    board = load_board(board_path)

    cell_1 = board[1]
    assert cell_1.price == 60
    assert cell_1.mortgage_value == 30
    assert cell_1.house_cost == 50
    assert cell_1.rent_by_houses is not None
    assert cell_1.rent_by_houses[0] == 2

    cell_39 = board[39]
    assert cell_39.price == 400
    assert cell_39.mortgage_value == 200
    assert cell_39.house_cost == 200
    assert cell_39.rent_by_houses is not None
    assert cell_39.rent_by_houses[0] == 50

    for idx in [4, 38]:
        assert board[idx].cell_type == "tax"
    for idx in [5, 15, 25, 35]:
        assert board[idx].cell_type == "railroad"
    for idx in [12, 28]:
        assert board[idx].cell_type == "utility"

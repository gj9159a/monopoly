from __future__ import annotations

from monopoly.roster import BOT_NAME_BASELINE, format_bot_name


def test_ui_bot_names_mapping() -> None:
    assert format_bot_name(14, is_baseline=False) == "Бот-#14"
    assert format_bot_name(None, is_baseline=True) == BOT_NAME_BASELINE

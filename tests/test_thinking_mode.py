from __future__ import annotations

from monopoly.engine import create_engine
from monopoly.params import BotParams, ThinkingConfig
from monopoly.thinking import choose_action


def _make_auction_context():
    engine = create_engine(num_players=2, seed=7)
    state = engine.state
    cell = state.board[1]
    context = {
        "type": "auction_bid",
        "player_id": 0,
        "cell": cell,
        "current_price": 10,
        "min_increment": 5,
    }
    return state, context


def test_thinking_determinism_workers1():
    state, context = _make_auction_context()
    params = BotParams()
    config = ThinkingConfig(enabled=True, horizon_turns=3, rollouts_per_action=2, time_budget_ms=0, workers=1)

    action1, _ = choose_action(state, context, params, config, workers=1)
    action2, _ = choose_action(state, context, params, config, workers=1)

    assert action1 == action2


def test_thinking_workers_consistency():
    state, context = _make_auction_context()
    params = BotParams()
    config = ThinkingConfig(enabled=True, horizon_turns=2, rollouts_per_action=1, time_budget_ms=0, workers=2)

    action1, _ = choose_action(state, context, params, config, workers=1)
    action2, _ = choose_action(state, context, params, config, workers=2)

    assert action1 == action2


def test_thinking_smoke_economy_phase():
    engine = create_engine(num_players=2, seed=11)
    state = engine.state
    params = BotParams()
    config = ThinkingConfig(enabled=True, horizon_turns=2, rollouts_per_action=1, time_budget_ms=0, workers=1)

    action, stats = choose_action(state, {"type": "economy_phase", "player_id": 0}, params, config, workers=1)

    assert isinstance(action, dict)
    assert stats.candidates >= 1

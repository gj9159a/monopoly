from __future__ import annotations

from monopoly.engine import create_engine


def test_smoke_500_steps():
    engine = create_engine(num_players=4, seed=123)
    for _ in range(500):
        engine.step()

    assert len(engine.state.event_log) > 0
    for player in engine.state.players:
        assert 0 <= player.position < 40

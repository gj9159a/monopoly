from __future__ import annotations

import math

from monopoly.engine import create_engine
from monopoly.features import heatmap_for_state
from monopoly.params import BotParams


def test_heatmap_deterministic() -> None:
    state = create_engine(num_players=2, seed=1).state
    heat_a = heatmap_for_state(state)
    heat_b = heatmap_for_state(state)
    assert heat_a.cell == heat_b.cell
    assert heat_a.group == heat_b.group


def test_heatmap_sums_to_one() -> None:
    state = create_engine(num_players=2, seed=1).state
    heat = heatmap_for_state(state)
    assert math.isclose(sum(heat.cell), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_heatmap_not_uniform_and_orange_red_warm() -> None:
    state = create_engine(num_players=2, seed=1).state
    heat = heatmap_for_state(state)
    assert max(heat.cell) - min(heat.cell) > 1e-4

    orange = heat.group.get("orange", 0.0)
    red = heat.group.get("red", 0.0)
    brown = heat.group.get("brown", 0.0)
    assert orange > brown
    assert red > brown


def test_backward_compat_league_params_load() -> None:
    params = BotParams.from_dict({"weights": {"auction": {"early": {"bias": 0.1}}}})
    assert params.weights["auction"]["early"]["group_heat"] == 0.0
    assert params.weights["auction"]["early"]["group_heat_vs_base"] == 0.0
    assert params.weights["auction"]["early"]["cell_heat"] == 0.0

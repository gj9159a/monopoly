from __future__ import annotations

from pathlib import Path

from monopoly.league import add_to_league, get_top1, hash_params, load_index, sample_from_league
from monopoly.params import BotParams


def _make_params(seed: int) -> BotParams:
    return BotParams(cash_buffer_base=150 + seed, cash_buffer_per_house=20 + seed)


def test_league_sorted_and_ranked(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()

    fitness_values = [0.2, 1.5, 0.9, -0.1, 0.7]
    for idx, fitness in enumerate(fitness_values):
        params = _make_params(idx)
        add_to_league(params, fitness, {"name": f"bot_{idx}"}, league_dir)

    index = load_index(league_dir)
    items = index["items"]
    assert [entry["fitness"] for entry in items] == sorted(fitness_values, reverse=True)
    assert [entry["rank"] for entry in items] == list(range(1, len(items) + 1))


def test_league_prune_top16(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()

    fitness_values = list(range(20))
    for idx, fitness in enumerate(fitness_values):
        params = _make_params(idx)
        add_to_league(params, float(fitness), {"name": f"bot_{idx}"}, league_dir, top_k=16)

    index = load_index(league_dir)
    items = index["items"]
    assert len(items) == 16
    expected = sorted(fitness_values, reverse=True)[:16]
    assert [entry["fitness"] for entry in items] == expected


def test_sample_unique_excluding_top1(tmp_path: Path) -> None:
    league_dir = tmp_path / "league"
    league_dir.mkdir()

    for idx in range(8):
        params = _make_params(idx)
        add_to_league(params, float(idx), {"name": f"bot_{idx}"}, league_dir)

    top1 = get_top1(league_dir)
    assert top1 is not None
    exclude = {str(top1.get("hash"))}
    sample = sample_from_league(5, league_dir, exclude_hashes=exclude)
    hashes = [str(entry.get("hash")) for entry in sample]
    assert len(sample) == 5
    assert len(hashes) == len(set(hashes))
    assert not exclude.intersection(hashes)


def test_hash_not_collapsing_on_small_param_change() -> None:
    base = BotParams(max_bid_fraction=0.7)
    changed = BotParams(max_bid_fraction=0.701)
    assert hash_params(base) != hash_params(changed)

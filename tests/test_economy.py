from __future__ import annotations

from dataclasses import dataclass

from monopoly.engine import create_engine


@dataclass
class FixedRNG:
    values: list[int]
    index: int = 0

    def randint(self, a: int, b: int) -> int:
        if self.index >= len(self.values):
            raise IndexError("Недостаточно значений для FixedRNG")
        value = self.values[self.index]
        self.index += 1
        return value


def _setup_brown_monopoly(engine):
    player = engine.state.players[0]
    cell_a = engine.state.board[1]
    cell_b = engine.state.board[3]
    cell_a.owner_id = player.player_id
    cell_b.owner_id = player.player_id
    return player, cell_a, cell_b


def _last_rent_amount(events):
    rent_events = [event for event in events if event.type == "PAY_RENT"]
    assert rent_events
    return rent_events[-1].payload["amount"]


def test_mortgage_disables_rent():
    engine = create_engine(num_players=2, seed=1)
    tenant = engine.state.players[0]
    owner = engine.state.players[1]
    cell = engine.state.board[3]
    cell.owner_id = owner.player_id
    cell.mortgaged = True

    tenant.position = 0
    events = engine.step()
    assert _last_rent_amount(events) == 0


def test_unmortgage_cost_includes_interest():
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    cell = engine.state.board[1]
    cell.owner_id = player.player_id
    cell.mortgaged = True
    cell.mortgage_value = 100
    engine.state.rules.interest_rate = 0.1
    player.money = 110

    events = engine._unmortgage_property(player, cell, 0)

    assert events
    assert cell.mortgaged is False
    assert player.money == 0


def test_cannot_mortgage_with_houses():
    engine = create_engine(num_players=2, seed=1)
    player, cell, _ = _setup_brown_monopoly(engine)
    cell.houses = 1

    events = engine._mortgage_property(player, cell, 0)

    assert events == []
    assert cell.mortgaged is False


def test_cannot_build_when_group_has_mortgage():
    engine = create_engine(num_players=2, seed=1)
    player, cell_a, cell_b = _setup_brown_monopoly(engine)
    cell_a.mortgaged = True
    player.money = 500

    events = engine._build_on_property(player, cell_b, 0)

    assert events == []
    assert cell_b.houses == 0


def test_even_building_rule():
    engine = create_engine(num_players=2, seed=1)
    player, cell_a, cell_b = _setup_brown_monopoly(engine)
    player.money = 500

    events_first = engine._build_on_property(player, cell_a, 0)
    events_second = engine._build_on_property(player, cell_a, 0)
    events_third = engine._build_on_property(player, cell_b, 0)

    assert events_first
    assert events_second == []
    assert events_third
    assert cell_a.houses == 1
    assert cell_b.houses == 1


def test_bank_house_limits():
    engine = create_engine(num_players=2, seed=1)
    player, cell_a, cell_b = _setup_brown_monopoly(engine)
    engine.state.rules.bank_houses = 1
    engine.state.rules.bank_hotels = 0
    player.money = 500

    events_first = engine._build_on_property(player, cell_a, 0)
    events_second = engine._build_on_property(player, cell_b, 0)

    assert events_first
    assert events_second == []

    cell_a.houses = 4
    cell_b.houses = 4
    events_hotel = engine._build_on_property(player, cell_a, 0)
    assert events_hotel == []


def test_mortgage_before_bankruptcy():
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    cell = engine.state.board[1]
    cell.owner_id = player.player_id
    cell.mortgage_value = 100
    player.money = 10

    events = engine._process_payment(
        player=player,
        amount=50,
        creditor_id=None,
        turn_index=0,
        reason="тест",
        event_type="PAY_TAX",
        message="Тест",
        cell_index=None,
    )

    assert any(event.type == "MORTGAGE" for event in events)
    assert player.bankrupt is False


def test_monopoly_double_rent_no_buildings():
    engine = create_engine(num_players=2, seed=1)
    tenant = engine.state.players[0]
    owner = engine.state.players[1]
    tenant.position = 0

    cell_a = engine.state.board[1]
    cell_b = engine.state.board[3]
    cell_a.owner_id = owner.player_id
    cell_b.owner_id = owner.player_id

    engine.state.rng = FixedRNG([1, 2])

    events = engine.step()
    assert _last_rent_amount(events) == 8


def test_monopoly_double_rent_mortgaged_priority():
    engine = create_engine(num_players=2, seed=1)
    tenant = engine.state.players[0]
    owner = engine.state.players[1]
    tenant.position = 0

    cell_a = engine.state.board[1]
    cell_b = engine.state.board[3]
    cell_a.owner_id = owner.player_id
    cell_b.owner_id = owner.player_id
    cell_b.mortgaged = True

    engine.state.rng = FixedRNG([1, 2])

    events = engine.step()
    assert _last_rent_amount(events) == 0


def test_monopoly_double_rent_owner_in_jail_priority():
    engine = create_engine(num_players=2, seed=1)
    tenant = engine.state.players[0]
    owner = engine.state.players[1]
    tenant.position = 0

    cell_a = engine.state.board[1]
    cell_b = engine.state.board[3]
    cell_a.owner_id = owner.player_id
    cell_b.owner_id = owner.player_id
    owner.in_jail = True

    engine.state.rng = FixedRNG([1, 2])

    events = engine.step()
    assert _last_rent_amount(events) == 0


def test_no_double_rent_for_railroads():
    engine = create_engine(num_players=2, seed=1)
    tenant = engine.state.players[0]
    owner = engine.state.players[1]
    tenant.position = 3

    for idx in [5, 15, 25, 35]:
        engine.state.board[idx].owner_id = owner.player_id

    engine.state.rng = FixedRNG([1, 1])

    events = engine.step()
    assert _last_rent_amount(events) == 200


def test_no_double_rent_for_utilities():
    engine = create_engine(num_players=2, seed=1)
    tenant = engine.state.players[0]
    owner = engine.state.players[1]
    tenant.position = 10

    engine.state.board[12].owner_id = owner.player_id
    engine.state.board[28].owner_id = owner.player_id

    engine.state.rng = FixedRNG([1, 1])

    events = engine.step()
    assert _last_rent_amount(events) == 20

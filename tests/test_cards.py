from __future__ import annotations

from pathlib import Path

from monopoly.data_loader import load_cards
from monopoly.engine import create_engine
from monopoly.models import Card, DeckState


def test_card_draw_determinism_by_seed():
    engine_a = create_engine(num_players=2, seed=123)
    engine_b = create_engine(num_players=2, seed=123)

    def draw_ids(engine, count: int):
        deck = engine.state.decks["chance"]
        ids = []
        for _ in range(count):
            card = engine._draw_from_deck("chance")
            ids.append(card.card_id)
            deck.discard.append(card)
        return ids

    assert draw_ids(engine_a, 3) == draw_ids(engine_b, 3)


def test_card_go_to_jail_effect():
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    card = Card(
        card_id="test_jail",
        text_ru="Тест: тюрьма",
        effect={"type": "go_to_jail"},
        deck="chance",
    )
    engine.state.decks["chance"] = DeckState(draw_pile=[card], discard=[])

    events = engine._draw_card("chance", player, 0)

    assert player.in_jail is True
    assert player.position == engine.jail_index
    assert any(event.type == "CARD_EFFECT" for event in events)


def test_get_out_of_jail_cycle():
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    card = Card(
        card_id="test_get_out",
        text_ru="Тест: выход из тюрьмы",
        effect={"type": "get_out_of_jail"},
        deck="chance",
    )
    engine.state.decks["chance"] = DeckState(draw_pile=[card], discard=[])

    events = engine._draw_card("chance", player, 0)
    assert any(event.type == "DRAW_CARD" for event in events)
    assert len(player.get_out_of_jail_cards) == 1
    assert not engine.state.decks["chance"].discard

    use_events = engine._use_get_out_of_jail_card(player, 1)
    assert any(event.type == "JAIL_CARD_USED" for event in use_events)
    assert not player.get_out_of_jail_cards
    assert len(engine.state.decks["chance"].discard) == 1

    events2 = engine._draw_card("chance", player, 2)
    assert any(event.payload.get("card_id") == "test_get_out" for event in events2 if event.type == "DRAW_CARD")


def test_pay_each_and_receive_from_each():
    engine = create_engine(num_players=3, seed=1)
    player = engine.state.players[0]
    p1 = engine.state.players[1]
    p2 = engine.state.players[2]
    player.money = 200
    p1.money = 100
    p2.money = 100

    engine.state.decks["chance"] = DeckState(draw_pile=[], discard=[])
    pay_card = Card(
        card_id="test_pay_each",
        text_ru="Тест: плати каждому",
        effect={"type": "pay_each", "amount": 10},
        deck="chance",
    )
    engine._apply_card_effect(pay_card, player, 0)
    assert player.money == 180
    assert p1.money == 110
    assert p2.money == 110

    receive_card = Card(
        card_id="test_receive_each",
        text_ru="Тест: получи от каждого",
        effect={"type": "receive_from_each", "amount": 5},
        deck="chance",
    )
    engine._apply_card_effect(receive_card, player, 1)
    assert player.money == 190
    assert p1.money == 105
    assert p2.money == 105


def test_move_to_next_railroad_wraps():
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    player.position = 36
    player.money = 0

    engine.state.decks["chance"] = DeckState(draw_pile=[], discard=[])
    card = Card(
        card_id="test_next_rail",
        text_ru="Тест: следующее депо",
        effect={"type": "move_to_next", "kind": "railroad", "auction_if_unowned": False},
        deck="chance",
    )
    engine._apply_card_effect(card, player, 0)

    assert player.position == 5
    assert player.money == engine.state.rules.go_salary


def test_deck_sizes_and_unique_ids():
    root = Path(__file__).resolve().parents[1]
    chance_cards = load_cards(root / "monopoly" / "data" / "cards_chance.yaml", "chance")
    community_cards = load_cards(root / "monopoly" / "data" / "cards_community.yaml", "community")
    assert len(chance_cards) == 16
    assert len(community_cards) == 16
    assert len({card.card_id for card in chance_cards}) == 16
    assert len({card.card_id for card in community_cards}) == 16


def test_rent_mode_railroad_double_and_hr2_mortgage():
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    owner = engine.state.players[1]
    rail = engine.state.board[5]
    rail.owner_id = owner.player_id
    owner.properties.append(rail.index)
    player.position = 4
    player.money = 500
    owner.money = 100

    card = Card(
        card_id="test_next_rail_double",
        text_ru="Тест: двойная рента",
        effect={"type": "move_to_next", "kind": "railroad", "rent_mode": "double"},
        deck="chance",
    )
    engine._apply_card_effect(card, player, 0, dice_total=7)
    assert player.position == 5
    assert player.money == 450
    assert owner.money == 150

    rail.mortgaged = True
    player.money = 500
    owner.money = 100
    player.position = 4
    engine._apply_card_effect(card, player, 1, dice_total=7)
    assert player.money == 500
    assert owner.money == 100

    rail.mortgaged = False
    owner.in_jail = True
    player.money = 500
    owner.money = 100
    player.position = 4
    engine._apply_card_effect(card, player, 2, dice_total=7)
    assert player.money == 500
    assert owner.money == 100


def test_repairs_effect():
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    cell_a = engine.state.board[1]
    cell_b = engine.state.board[3]
    cell_a.owner_id = player.player_id
    cell_b.owner_id = player.player_id
    cell_a.houses = 2
    cell_b.hotels = 1
    player.money = 500

    card = Card(
        card_id="test_repairs",
        text_ru="Тест: ремонт",
        effect={"type": "repairs", "per_house": 25, "per_hotel": 100},
        deck="community",
    )
    engine._apply_card_effect(card, player, 0, dice_total=None)
    assert player.money == 350


def test_go_back_three_spaces_lands_and_pays_tax():
    engine = create_engine(num_players=2, seed=1)
    player = engine.state.players[0]
    player.position = 7
    player.money = 500
    card = Card(
        card_id="test_go_back",
        text_ru="Тест: назад на 3",
        effect={"type": "move_relative", "steps": -3},
        deck="chance",
    )
    engine._apply_card_effect(card, player, 0, dice_total=None)
    assert player.position == 4
    assert player.money == 300

from __future__ import annotations

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
        effect={"type": "move_to_next", "kind": "railroad"},
        deck="chance",
    )
    engine._apply_card_effect(card, player, 0)

    assert player.position == 5
    assert player.money == engine.state.rules.go_salary

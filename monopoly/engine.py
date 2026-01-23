from __future__ import annotations

import random
from pathlib import Path

from .bots import BaseBot, create_bots
from .data_loader import load_board, load_cards, load_rules
from .models import Card, Cell, DeckState, Event, GameState, Player


class Engine:
    def __init__(self, state: GameState, bots: list[BaseBot]) -> None:
        if len(bots) != len(state.players):
            raise ValueError("Число ботов должно совпадать с числом игроков")
        self.state = state
        self.bots = bots
        self.jail_index = _find_cell_index(state.board, "jail")

    def step(self) -> list[Event]:
        state = self.state
        player = state.players[state.current_player]
        events: list[Event] = []
        turn_index = state.turn_index
        if state.game_over:
            return events

        if player.bankrupt:
            events.append(
                Event(
                    type="TURN_SKIP",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} пропускает ход (банкрот)",
                    payload={"player_id": player.player_id},
                )
            )
            events.extend(self._check_game_end(turn_index))
            self._advance_player()
            state.event_log.extend(events)
            state.turn_index += 1
            return events

        events.append(
            Event(
                type="TURN_START",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"Старт хода: {player.name}",
                payload={"player_id": player.player_id},
            )
        )

        if player.in_jail:
            jail_events, advance_player = self._handle_jail_turn(player, turn_index)
            events.extend(jail_events)
            events.extend(self._check_game_end(turn_index))
            state.event_log.extend(events)
            state.turn_index += 1
            if advance_player:
                self._advance_player()
            return events

        roll_events, extra_turn, went_to_jail = self._roll_and_move(
            player, turn_index, allow_extra_turn=True, count_doubles=True
        )
        events.extend(roll_events)
        events.extend(self._check_game_end(turn_index))

        state.event_log.extend(events)
        state.turn_index += 1
        if went_to_jail or not extra_turn:
            self._advance_player()
        return events

    def _advance_player(self) -> None:
        self.state.current_player = (self.state.current_player + 1) % len(self.state.players)

    def _roll_and_move(
        self, player: Player, turn_index: int, allow_extra_turn: bool, count_doubles: bool
    ) -> tuple[list[Event], bool, bool]:
        state = self.state
        events: list[Event] = []
        die1, die2 = _roll_dice(state.rng)
        total = die1 + die2
        is_double = die1 == die2

        events.append(
            Event(
                type="DICE_ROLL",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} бросил {die1} и {die2}",
                payload={"dice": [die1, die2], "total": total},
            )
        )

        if is_double:
            events.append(
                Event(
                    type="DOUBLE_ROLL",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} выбросил дубль",
                    payload={"dice": [die1, die2]},
                )
            )

        if count_doubles:
            if is_double:
                player.doubles_count += 1
            else:
                player.doubles_count = 0
            if player.doubles_count >= 3:
                player.doubles_count = 0
                events.extend(self._send_to_jail(player, turn_index, reason="три дубля подряд"))
                return events, False, True
        else:
            player.doubles_count = 0

        cell, move_events = self._move_player(player, total, turn_index)
        events.extend(move_events)
        events.extend(self._handle_landing(player, cell, turn_index, total))

        went_to_jail = player.in_jail
        extra_turn = allow_extra_turn and is_double and not went_to_jail
        if extra_turn:
            events.append(
                Event(
                    type="EXTRA_TURN",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} получает дополнительный ход",
                    payload={},
                )
            )
        return events, extra_turn, went_to_jail

    def _move_player(self, player: Player, steps: int, turn_index: int) -> tuple[Cell, list[Event]]:
        state = self.state
        events: list[Event] = []
        old_pos = player.position
        new_pos = (old_pos + steps) % len(state.board)
        if new_pos < old_pos:
            player.money += state.rules.go_salary
            events.append(
                Event(
                    type="PASS_GO",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} получил {state.rules.go_salary} за проход 'Старт'",
                    payload={"amount": state.rules.go_salary},
                )
            )

        player.position = new_pos
        events.append(
            Event(
                type="MOVE",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} переместился на {new_pos}",
                payload={"from": old_pos, "to": new_pos, "steps": steps},
            )
        )

        cell = state.board[new_pos]
        events.append(
            Event(
                type="LAND",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} попал на '{cell.name}'",
                payload={"cell_index": cell.index, "cell_type": cell.cell_type},
            )
        )
        return cell, events

    def _send_to_jail(self, player: Player, turn_index: int, reason: str) -> list[Event]:
        player.in_jail = True
        player.jail_turns = 0
        player.position = self.jail_index
        player.doubles_count = 0
        return [
            Event(
                type="GO_TO_JAIL",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} отправлен в тюрьму ({reason})",
                payload={"jail_index": self.jail_index, "reason": reason},
            )
        ]

    def _handle_jail_turn(self, player: Player, turn_index: int) -> tuple[list[Event], bool]:
        state = self.state
        events: list[Event] = []
        fine = state.rules.jail_fine
        decision = self.bots[player.player_id].decide(
            state,
            {
                "type": "jail_decision",
                "player_id": player.player_id,
                "has_card": bool(player.get_out_of_jail_cards),
            },
        )
        action = decision.get("action")

        if action == "use_card" and player.get_out_of_jail_cards:
            events.extend(self._use_get_out_of_jail_card(player, turn_index))
            player.in_jail = False
            player.jail_turns = 0
            player.doubles_count = 0
            roll_events, extra_turn, went_to_jail = self._roll_and_move(
                player, turn_index, allow_extra_turn=True, count_doubles=True
            )
            events.extend(roll_events)
            return events, went_to_jail or not extra_turn

        if action == "pay":
            events.extend(
                self._process_payment(
                    player=player,
                    amount=fine,
                    creditor_id=None,
                    turn_index=turn_index,
                    reason="штраф тюрьмы",
                    event_type="JAIL_PAY",
                    message=f"{player.name} заплатил {fine} и вышел из тюрьмы",
                    cell_index=None,
                )
            )
            if player.bankrupt:
                return events, True
            player.in_jail = False
            player.jail_turns = 0
            player.doubles_count = 0
            roll_events, extra_turn, went_to_jail = self._roll_and_move(
                player, turn_index, allow_extra_turn=True, count_doubles=True
            )
            events.extend(roll_events)
            return events, went_to_jail or not extra_turn

        die1, die2 = _roll_dice(state.rng)
        total = die1 + die2
        is_double = die1 == die2
        events.append(
            Event(
                type="JAIL_ROLL",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} бросил {die1} и {die2} в тюрьме",
                payload={"dice": [die1, die2], "total": total},
            )
        )

        if is_double:
            player.in_jail = False
            player.jail_turns = 0
            player.doubles_count = 0
            events.append(
                Event(
                    type="JAIL_EXIT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} выбросил дубль и вышел из тюрьмы",
                    payload={},
                )
            )
            cell, move_events = self._move_player(player, total, turn_index)
            events.extend(move_events)
            events.extend(self._handle_landing(player, cell, turn_index, total))
            return events, True

        player.jail_turns += 1
        events.append(
            Event(
                type="JAIL_TURN",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} остается в тюрьме (попытка {player.jail_turns}/3)",
                payload={"jail_turns": player.jail_turns},
            )
        )

        if player.jail_turns >= 3:
            events.extend(
                self._process_payment(
                    player=player,
                    amount=fine,
                    creditor_id=None,
                    turn_index=turn_index,
                    reason="штраф тюрьмы",
                    event_type="JAIL_PAY_FORCED",
                    message=f"{player.name} заплатил {fine} после 3 попыток и вышел из тюрьмы",
                    cell_index=None,
                )
            )
            if player.bankrupt:
                return events, True
            player.in_jail = False
            player.jail_turns = 0
            cell, move_events = self._move_player(player, total, turn_index)
            events.extend(move_events)
            events.extend(self._handle_landing(player, cell, turn_index, total))
        return events, True

    def _handle_landing(self, player: Player, cell: Cell, turn_index: int, dice_total: int | None) -> list[Event]:
        events: list[Event] = []
        if cell.cell_type == "go_to_jail":
            events.extend(self._send_to_jail(player, turn_index, reason="клетка 'В тюрьму'"))
            return events
        if cell.cell_type == "chance":
            events.extend(self._draw_card("chance", player, turn_index))
            return events
        if cell.cell_type == "community":
            events.extend(self._draw_card("community", player, turn_index))
            return events
        if cell.cell_type == "tax":
            amount = cell.tax_amount or 0
            events.extend(
                self._process_payment(
                    player=player,
                    amount=amount,
                    creditor_id=None,
                    turn_index=turn_index,
                    reason="налог",
                    event_type="PAY_TAX",
                    message=f"{player.name} заплатил налог {amount}",
                    cell_index=cell.index,
                )
            )
            return events
        if cell.cell_type in {"property", "railroad", "utility"}:
            if cell.owner_id is None and self.state.rules.hr1_always_auction:
                events.extend(self._run_auction(cell, turn_index))
            elif cell.owner_id is not None and cell.owner_id != player.player_id:
                owner = self.state.players[cell.owner_id]
                rent = self._calculate_rent(cell, owner.player_id, dice_total)
                reason = ""
                if cell.mortgaged:
                    rent = 0
                    reason = " (ипотека)"
                elif self.state.rules.hr2_no_rent_in_jail and owner.in_jail:
                    rent = 0
                    reason = " (владелец в тюрьме)"
                events.extend(
                    self._process_payment(
                        player=player,
                        amount=rent,
                        creditor_id=owner.player_id,
                        turn_index=turn_index,
                        reason="рента",
                        event_type="PAY_RENT",
                        message=f"{player.name} заплатил ренту {rent} владельцу {owner.name}{reason}",
                        cell_index=cell.index,
                    )
                )
        return events

    def _draw_card(self, deck_name: str, player: Player, turn_index: int) -> list[Event]:
        card = self._draw_from_deck(deck_name)
        events: list[Event] = [
            Event(
                type="DRAW_CARD",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} тянет карту: {card.text_ru}",
                payload={"deck": deck_name, "card_id": card.card_id, "text_ru": card.text_ru},
            )
        ]
        events.extend(self._apply_card_effect(card, player, turn_index))
        return events

    def _draw_from_deck(self, deck_name: str) -> Card:
        deck = self.state.decks.get(deck_name)
        if deck is None:
            raise ValueError(f"Нет колоды {deck_name}")
        if not deck.draw_pile:
            if not deck.discard:
                raise ValueError(f"Колода {deck_name} пуста")
            deck.draw_pile = deck.discard
            deck.discard = []
            self.state.rng.shuffle(deck.draw_pile)
        return deck.draw_pile.pop(0)

    def _apply_card_effect(self, card: Card, player: Player, turn_index: int) -> list[Event]:
        effect = card.effect
        effect_type = str(effect.get("type"))
        events: list[Event] = []
        payload: dict[str, int | str | None] = {
            "card_id": card.card_id,
            "deck": card.deck,
            "effect_type": effect_type,
        }

        if effect_type == "money":
            amount = int(effect.get("amount", 0))
            payload["amount"] = amount
            if amount >= 0:
                player.money += amount
                events.append(
                    Event(
                        type="CARD_EFFECT",
                        turn_index=turn_index,
                        player_id=player.player_id,
                        msg_ru=f"{player.name} получает {amount} по карте",
                        payload=payload,
                    )
                )
            else:
                events.extend(
                    self._process_payment(
                        player=player,
                        amount=-amount,
                        creditor_id=None,
                        turn_index=turn_index,
                        reason="карта",
                        event_type="CARD_EFFECT",
                        message=f"{player.name} платит {-amount} по карте",
                        cell_index=None,
                        extra_payload=payload,
                    )
                )

        elif effect_type == "pay_each":
            amount = int(effect.get("amount", 0))
            others = [p for p in self.state.players if p.player_id != player.player_id and not p.bankrupt]
            total = amount * len(others)
            payload["amount"] = amount
            payload["total"] = total
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} платит каждому по {amount}",
                    payload=payload,
                )
            )
            if total > 0:
                if player.money < total:
                    events.extend(self._liquidate_buildings(player, turn_index))
                if player.money >= total:
                    player.money -= total
                    for other in others:
                        other.money += amount
                else:
                    events.extend(self._bankrupt_player(player, None, turn_index, "карта pay_each"))

        elif effect_type == "receive_from_each":
            amount = int(effect.get("amount", 0))
            payload["amount"] = amount
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} получает от каждого по {amount}",
                    payload=payload,
                )
            )
            for other in self.state.players:
                if other.player_id == player.player_id or other.bankrupt:
                    continue
                events.extend(
                    self._process_payment(
                        player=other,
                        amount=amount,
                        creditor_id=player.player_id,
                        turn_index=turn_index,
                        reason="карта receive_from_each",
                        event_type="CARD_PAYMENT",
                        message=f"{other.name} платит {amount} игроку {player.name}",
                        cell_index=None,
                    )
                )

        elif effect_type == "move_to":
            target = int(effect.get("cell_index", 0))
            payload["cell_index"] = target
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} перемещается на клетку {target}",
                    payload=payload,
                )
            )
            cell, move_events = self._move_player(player, (target - player.position) % len(self.state.board), turn_index)
            events.extend(move_events)
            events.extend(self._handle_landing(player, cell, turn_index, None))

        elif effect_type == "move_relative":
            steps = int(effect.get("steps", 0))
            payload["steps"] = steps
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} перемещается на {steps} клеток",
                    payload=payload,
                )
            )
            cell, move_events = self._move_relative(player, steps, turn_index)
            events.extend(move_events)
            events.extend(self._handle_landing(player, cell, turn_index, None))

        elif effect_type == "go_to_jail":
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} отправляется в тюрьму по карте",
                    payload=payload,
                )
            )
            events.extend(self._send_to_jail(player, turn_index, reason="карта"))

        elif effect_type == "get_out_of_jail":
            player.get_out_of_jail_cards.append(card)
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} получает карту выхода из тюрьмы",
                    payload=payload,
                )
            )

        elif effect_type == "move_to_next":
            kind = str(effect.get("kind", ""))
            target = self._find_next_cell_index(player.position, kind)
            payload["kind"] = kind
            payload["cell_index"] = target
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} перемещается на следующую клетку {kind}",
                    payload=payload,
                )
            )
            cell, move_events = self._move_player(
                player, (target - player.position) % len(self.state.board), turn_index
            )
            events.extend(move_events)
            events.extend(self._handle_landing(player, cell, turn_index, None))

        else:
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} получает неизвестный эффект карты",
                    payload=payload,
                )
            )

        events.extend(self._check_game_end(turn_index))
        if effect_type != "get_out_of_jail":
            self.state.decks[card.deck].discard.append(card)
        return events

    def _use_get_out_of_jail_card(self, player: Player, turn_index: int) -> list[Event]:
        if not player.get_out_of_jail_cards:
            return []
        card = player.get_out_of_jail_cards.pop(0)
        deck = self.state.decks.get(card.deck)
        if deck is not None:
            deck.discard.append(card)
        return [
            Event(
                type="JAIL_CARD_USED",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} использует карту выхода из тюрьмы",
                payload={"deck": card.deck, "card_id": card.card_id},
            )
        ]

    def _move_relative(self, player: Player, steps: int, turn_index: int) -> tuple[Cell, list[Event]]:
        state = self.state
        events: list[Event] = []
        old_pos = player.position
        new_pos = (old_pos + steps) % len(state.board)
        if steps >= 0 and new_pos < old_pos:
            player.money += state.rules.go_salary
            events.append(
                Event(
                    type="PASS_GO",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} получил {state.rules.go_salary} за проход 'Старт'",
                    payload={"amount": state.rules.go_salary},
                )
            )
        player.position = new_pos
        events.append(
            Event(
                type="MOVE",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} переместился на {new_pos}",
                payload={"from": old_pos, "to": new_pos, "steps": steps},
            )
        )
        cell = state.board[new_pos]
        events.append(
            Event(
                type="LAND",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} попал на '{cell.name}'",
                payload={"cell_index": cell.index, "cell_type": cell.cell_type},
            )
        )
        return cell, events

    def _find_next_cell_index(self, current_pos: int, kind: str) -> int:
        for offset in range(1, len(self.state.board) + 1):
            idx = (current_pos + offset) % len(self.state.board)
            cell = self.state.board[idx]
            if cell.cell_type == kind:
                return idx
        raise ValueError(f"На поле нет клетки типа {kind}")

    def _process_payment(
        self,
        player: Player,
        amount: int,
        creditor_id: int | None,
        turn_index: int,
        reason: str,
        event_type: str,
        message: str,
        cell_index: int | None,
        extra_payload: dict[str, int | str | None] | None = None,
    ) -> list[Event]:
        events: list[Event] = []
        amount_due = max(0, int(amount))
        payload_base = {
            "amount": 0,
            "due": amount_due,
            "owner_id": creditor_id,
            "cell_index": cell_index,
        }
        if extra_payload:
            payload_base.update(extra_payload)

        if amount_due == 0:
            events.append(
                Event(
                    type=event_type,
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=message,
                    payload=payload_base,
                )
            )
            return events

        if player.money < amount_due:
            events.extend(self._liquidate_buildings(player, turn_index))

        paid = min(amount_due, player.money)
        player.money -= paid
        if creditor_id is not None:
            self.state.players[creditor_id].money += paid

        events.append(
            Event(
                type=event_type,
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=message,
                payload={**payload_base, "amount": paid},
            )
        )

        if paid < amount_due:
            events.extend(self._bankrupt_player(player, creditor_id, turn_index, reason))
        return events

    def _liquidate_buildings(self, player: Player, turn_index: int) -> list[Event]:
        events: list[Event] = []
        for cell in self.state.board:
            if cell.owner_id != player.player_id:
                continue
            if cell.houses <= 0 and cell.hotels <= 0:
                continue
            if cell.house_cost is None:
                continue
            units = cell.houses + cell.hotels * 5
            refund = int(units * cell.house_cost / 2)
            if refund <= 0:
                continue
            player.money += refund
            events.append(
                Event(
                    type="SELL_BUILDING",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} продает строения на '{cell.name}' за {refund}",
                    payload={"cell_index": cell.index, "refund": refund},
                )
            )
            cell.houses = 0
            cell.hotels = 0
        return events

    def _bankrupt_player(
        self, player: Player, creditor_id: int | None, turn_index: int, reason: str
    ) -> list[Event]:
        if player.bankrupt:
            return []
        player.bankrupt = True
        player.in_jail = False
        player.jail_turns = 0
        player.doubles_count = 0
        player.money = 0

        transferred: list[int] = []
        for cell in self.state.board:
            if cell.owner_id != player.player_id:
                continue
            cell.houses = 0
            cell.hotels = 0
            cell.mortgaged = False
            if creditor_id is None:
                cell.owner_id = None
            else:
                cell.owner_id = creditor_id
                self.state.players[creditor_id].properties.append(cell.index)
                transferred.append(cell.index)

        player.properties.clear()
        while player.get_out_of_jail_cards:
            card = player.get_out_of_jail_cards.pop(0)
            deck = self.state.decks.get(card.deck)
            if deck is not None:
                deck.discard.append(card)
        events = [
            Event(
                type="BANKRUPTCY",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} объявлен банкротом ({reason})",
                payload={"creditor_id": creditor_id, "properties": transferred},
            )
        ]
        events.extend(self._check_game_end(turn_index))
        return events

    def _check_game_end(self, turn_index: int) -> list[Event]:
        if self.state.game_over:
            return []
        active = [p for p in self.state.players if not p.bankrupt]
        if len(active) == 1:
            winner = active[0]
            self.state.game_over = True
            self.state.winner_id = winner.player_id
            return [
                Event(
                    type="GAME_END",
                    turn_index=turn_index,
                    player_id=winner.player_id,
                    msg_ru=f"Игра окончена. Победитель: {winner.name}",
                    payload={"winner_id": winner.player_id},
                )
            ]
        if len(active) == 0:
            self.state.game_over = True
            return [
                Event(
                    type="GAME_END",
                    turn_index=turn_index,
                    player_id=None,
                    msg_ru="Игра окончена. Победитель не определен",
                    payload={"winner_id": None},
                )
            ]
        return []

    def _calculate_rent(self, cell: Cell, owner_id: int, dice_total: int | None) -> int:
        if cell.cell_type == "property":
            if cell.rent is None:
                return 0
            houses = cell.houses
            if cell.hotels > 0:
                base_rent = cell.rent[5]
            elif houses > 0:
                base_rent = cell.rent[houses]
            else:
                base_rent = cell.rent[0]
                if cell.group and self._owns_group(owner_id, cell.group):
                    base_rent *= 2
            return int(base_rent)
        if cell.cell_type == "railroad":
            if cell.rent is None:
                return 0
            count = sum(
                1
                for owned in self.state.board
                if owned.cell_type == "railroad" and owned.owner_id == owner_id
            )
            count = max(1, min(count, len(cell.rent)))
            return int(cell.rent[count - 1])
        if cell.cell_type == "utility":
            if dice_total is None or cell.rent_multiplier is None:
                return 0
            count = sum(
                1
                for owned in self.state.board
                if owned.cell_type == "utility" and owned.owner_id == owner_id
            )
            multiplier = cell.rent_multiplier[1] if count >= 2 else cell.rent_multiplier[0]
            return int(dice_total * multiplier)
        return 0

    def _owns_group(self, owner_id: int, group: str) -> bool:
        group_cells = [cell for cell in self.state.board if cell.group == group]
        if not group_cells:
            return False
        return all(cell.owner_id == owner_id for cell in group_cells)

    def _run_auction(self, cell: Cell, turn_index: int) -> list[Event]:
        state = self.state
        events: list[Event] = []
        participants = [p.player_id for p in state.players if not p.bankrupt]
        active = participants[:]
        current_price = 0
        min_increment = 1
        last_bidder: int | None = None

        events.append(
            Event(
                type="AUCTION_START",
                turn_index=turn_index,
                player_id=None,
                msg_ru=f"Запущен аукцион за '{cell.name}'",
                payload={"cell_index": cell.index, "participants": participants},
            )
        )

        rounds = 0
        while len(active) > 1:
            for player_id in active[:]:
                player = state.players[player_id]
                context = {
                    "type": "auction_bid",
                    "player_id": player_id,
                    "cell": cell,
                    "current_price": current_price,
                    "min_increment": min_increment,
                }
                decision = self.bots[player_id].decide(state, context)
                if decision.get("action") == "bid":
                    bid = int(decision.get("bid", 0))
                    if bid <= current_price or bid > player.money:
                        events.append(
                            Event(
                                type="AUCTION_PASS",
                                turn_index=turn_index,
                                player_id=player_id,
                                msg_ru=f"{player.name} пас (некорректная ставка)",
                                payload={"bid": bid, "current_price": current_price},
                            )
                        )
                        active.remove(player_id)
                    else:
                        current_price = bid
                        last_bidder = player_id
                        events.append(
                            Event(
                                type="AUCTION_BID",
                                turn_index=turn_index,
                                player_id=player_id,
                                msg_ru=f"{player.name} сделал ставку {bid}",
                                payload={"bid": bid},
                            )
                        )
                else:
                    events.append(
                        Event(
                            type="AUCTION_PASS",
                            turn_index=turn_index,
                            player_id=player_id,
                            msg_ru=f"{player.name} пас",
                            payload={"current_price": current_price},
                        )
                    )
                    active.remove(player_id)
                if len(active) <= 1:
                    break
            rounds += 1
            if rounds > 200:
                break

        winner_id = None
        if len(active) == 1:
            winner_id = active[0]
        elif last_bidder is not None:
            winner_id = last_bidder

        if winner_id is not None and current_price > 0:
            winner = state.players[winner_id]
            winner.money -= current_price
            cell.owner_id = winner_id
            winner.properties.append(cell.index)
            events.append(
                Event(
                    type="AUCTION_WIN",
                    turn_index=turn_index,
                    player_id=winner_id,
                    msg_ru=f"{winner.name} выиграл аукцион за {current_price}",
                    payload={"price": current_price, "cell_index": cell.index},
                )
            )
        else:
            events.append(
                Event(
                    type="AUCTION_END",
                    turn_index=turn_index,
                    player_id=None,
                    msg_ru=f"Аукцион за '{cell.name}' завершен без победителя",
                    payload={"cell_index": cell.index},
                )
            )

        return events


def create_game(num_players: int, seed: int, data_dir: Path | None = None) -> GameState:
    if num_players < 2 or num_players > 6:
        raise ValueError("Число игроков должно быть от 2 до 6")
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"

    rules = load_rules(data_dir / "rules.yaml")
    board = load_board(data_dir / "board.yaml")
    chance_cards = load_cards(data_dir / "cards_chance.yaml", "chance")
    community_cards = load_cards(data_dir / "cards_community.yaml", "community")
    rng = random.Random(seed)

    players = [
        Player(
            player_id=pid,
            name=f"Бот {pid + 1}",
            position=0,
            money=rules.starting_cash,
        )
        for pid in range(num_players)
    ]

    decks = {
        "chance": _init_deck(chance_cards, rng),
        "community": _init_deck(community_cards, rng),
    }

    return GameState(
        seed=seed,
        rng=rng,
        rules=rules,
        board=board,
        players=players,
        turn_index=0,
        current_player=0,
        event_log=[],
        decks=decks,
    )


def create_engine(
    num_players: int, seed: int, data_dir: Path | None = None, bot_profiles: list[str] | None = None
) -> Engine:
    state = create_game(num_players=num_players, seed=seed, data_dir=data_dir)
    bots = create_bots(num_players, bot_profiles)
    return Engine(state, bots)


def _roll_dice(rng: random.Random) -> tuple[int, int]:
    return rng.randint(1, 6), rng.randint(1, 6)


def _find_cell_index(board: list[Cell], cell_type: str) -> int:
    for cell in board:
        if cell.cell_type == cell_type:
            return cell.index
    raise ValueError(f"На поле нет клетки типа {cell_type}")


def _init_deck(cards: list[Card], rng: random.Random) -> DeckState:
    draw_pile = list(cards)
    rng.shuffle(draw_pile)
    return DeckState(draw_pile=draw_pile, discard=[])

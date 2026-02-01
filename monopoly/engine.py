from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .bots import Bot, create_bots
from .data_loader import load_board, load_cards, load_rules
from .params import BotParams, evaluate_trade_for_player, normalize_auction_price
from .models import Card, Cell, DeckState, Event, GameState, Player, TradeHistory


class Engine:
    def __init__(self, state: GameState, bots: list[Bot]) -> None:
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
            if not player.bankrupt:
                if not player.in_jail:
                    events.extend(self._bot_economy_phase(player, turn_index))
                events.extend(self._bot_trade_phase(player, turn_index))
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
        if not player.bankrupt and not player.in_jail:
            events.extend(self._bot_economy_phase(player, turn_index))
        if not player.bankrupt:
            events.extend(self._bot_trade_phase(player, turn_index))
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
            events.extend(self._draw_card("chance", player, turn_index, dice_total))
            return events
        if cell.cell_type == "community":
            events.extend(self._draw_card("community", player, turn_index, dice_total))
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
                reason = ""
                if cell.mortgaged:
                    rent = 0
                    reason = " (ипотека)"
                elif self.state.rules.hr2_no_rent_in_jail and owner.in_jail:
                    rent = 0
                    reason = " (владелец в тюрьме)"
                else:
                    rent = self._calculate_rent(cell, owner.player_id, dice_total)
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

    def _draw_card(
        self, deck_name: str, player: Player, turn_index: int, dice_total: int | None = None
    ) -> list[Event]:
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
        events.extend(self._apply_card_effect(card, player, turn_index, dice_total))
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

    def _apply_card_effect(
        self, card: Card, player: Player, turn_index: int, dice_total: int | None = None
    ) -> list[Event]:
        effect = card.effect
        effect_type = str(effect.get("type"))
        events: list[Event] = []
        payload: dict[str, int | str | None] = {
            "card_id": card.card_id,
            "deck": card.deck,
            "effect_type": effect_type,
        }

        if effect_type in {"money", "pay_bank", "receive_bank"}:
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
                    events.extend(self._liquidate_buildings(player, turn_index, total))
                if player.money < total:
                    events.extend(self._mortgage_properties(player, turn_index, total))
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
            pass_go_raw = effect.get("pass_go", "auto")
            pass_go = None
            if isinstance(pass_go_raw, bool):
                pass_go = pass_go_raw
            elif isinstance(pass_go_raw, str) and pass_go_raw.lower() not in {"auto", ""}:
                pass_go = pass_go_raw.lower() in {"1", "true", "yes", "y"}
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
            cell, move_events = self._move_player_to(player, target, turn_index, pass_go)
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
            rent_mode = effect.get("rent_mode")
            auction_if_unowned = effect.get("auction_if_unowned", True)
            target = self._find_next_cell_index(player.position, kind)
            payload["kind"] = kind
            payload["cell_index"] = target
            payload["rent_mode"] = str(rent_mode) if rent_mode is not None else None
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} перемещается на следующую клетку {kind}",
                    payload=payload,
                )
            )
            cell, move_events = self._move_player_to(player, target, turn_index, True)
            events.extend(move_events)
            if cell.owner_id is None:
                if auction_if_unowned and self.state.rules.hr1_always_auction:
                    events.extend(self._run_auction(cell, turn_index))
            elif cell.owner_id != player.player_id:
                owner = self.state.players[cell.owner_id]
                rent, rent_events, reason = self._special_rent_amount(
                    player, cell, owner, rent_mode, dice_total, turn_index
                )
                events.extend(rent_events)
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

        elif effect_type == "repairs":
            per_house = int(effect.get("per_house", 0))
            per_hotel = int(effect.get("per_hotel", 0))
            houses = sum(
                cell.houses
                for cell in self.state.board
                if cell.owner_id == player.player_id
            )
            hotels = sum(
                cell.hotels
                for cell in self.state.board
                if cell.owner_id == player.player_id
            )
            amount = houses * per_house + hotels * per_hotel
            payload["per_house"] = per_house
            payload["per_hotel"] = per_hotel
            payload["amount"] = amount
            events.append(
                Event(
                    type="CARD_EFFECT",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} платит за ремонт {amount}",
                    payload=payload,
                )
            )
            if amount > 0:
                events.extend(
                    self._process_payment(
                        player=player,
                        amount=amount,
                        creditor_id=None,
                        turn_index=turn_index,
                        reason="ремонт",
                        event_type="CARD_EFFECT",
                        message=f"{player.name} платит {amount} за ремонт",
                        cell_index=None,
                        extra_payload=payload,
                    )
                )

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

    def _move_player_to(
        self, player: Player, target: int, turn_index: int, pass_go: bool | None
    ) -> tuple[Cell, list[Event]]:
        state = self.state
        events: list[Event] = []
        old_pos = player.position
        new_pos = int(target) % len(state.board)
        wrapped = new_pos < old_pos
        should_collect = pass_go is True or (pass_go is None and wrapped)
        if wrapped and should_collect:
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
        steps = (new_pos - old_pos) % len(state.board)
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

    def _special_rent_amount(
        self,
        player: Player,
        cell: Cell,
        owner: Player,
        rent_mode: str | None,
        dice_total: int | None,
        turn_index: int,
    ) -> tuple[int, list[Event], str]:
        events: list[Event] = []
        reason = ""
        if cell.mortgaged:
            return 0, events, " (ипотека)"
        if self.state.rules.hr2_no_rent_in_jail and owner.in_jail:
            return 0, events, " (владелец в тюрьме)"
        rent = self._calculate_rent(cell, owner.player_id, dice_total)
        if rent_mode == "double" and cell.cell_type == "railroad":
            rent *= 2
        elif rent_mode == "utility_10x" and cell.cell_type == "utility":
            total = dice_total
            if total is None:
                die1, die2 = _roll_dice(self.state.rng)
                total = die1 + die2
                events.append(
                    Event(
                        type="CARD_DICE_ROLL",
                        turn_index=turn_index,
                        player_id=player.player_id,
                        msg_ru=f"Бросок по карте: {die1} и {die2}",
                        payload={"dice": [die1, die2], "total": total},
                    )
                )
            rent = int(total * 10)
        return int(rent), events, reason

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
            events.extend(self._liquidate_buildings(player, turn_index, amount_due))
        if player.money < amount_due:
            events.extend(self._mortgage_properties(player, turn_index, amount_due))

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

    def _liquidate_buildings(self, player: Player, turn_index: int, target_cash: int) -> list[Event]:
        events: list[Event] = []
        if target_cash <= 0:
            return events
        while player.money < target_cash:
            allow_hotel_sale = self._bank_houses_available() >= 4
            cell = self._find_sell_candidate(player.player_id, allow_hotel=allow_hotel_sale)
            if cell is None or cell.house_cost is None:
                break
            refund = int(cell.house_cost / 2)
            if cell.hotels > 0:
                if not allow_hotel_sale:
                    break
                cell.hotels = 0
                cell.houses = 4
                building = "hotel"
            else:
                cell.houses = max(0, cell.houses - 1)
                building = "house"
            player.money += refund
            events.append(
                Event(
                    type="SELL_BUILDING",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} продает {building} на '{cell.name}' за {refund}",
                    payload={"cell_index": cell.index, "refund": refund, "building": building},
                )
            )
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
        auction_events: list[Event] = []
        for cell in self.state.board:
            if cell.owner_id != player.player_id:
                continue
            cell.houses = 0
            cell.hotels = 0
            if creditor_id is None:
                cell.owner_id = None
                cell.mortgaged = False
                if cell.cell_type in {"property", "railroad", "utility"}:
                    auction_events.extend(self._run_auction(cell, turn_index))
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
        events.extend(auction_events)
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
            if cell.rent_by_houses is None:
                return 0
            houses = cell.houses
            if cell.hotels > 0:
                base_rent = cell.rent_by_houses[5]
            elif houses > 0:
                base_rent = cell.rent_by_houses[houses]
            else:
                base_rent = cell.rent_by_houses[0]
                if cell.group and self._owns_group(owner_id, cell.group):
                    group_cells = [c for c in self.state.board if c.group == cell.group]
                    if all(c.houses == 0 and c.hotels == 0 for c in group_cells):
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

    def _group_cells(self, owner_id: int, group: str) -> list[Cell]:
        return [cell for cell in self.state.board if cell.group == group and cell.owner_id == owner_id]

    def _property_level(self, cell: Cell) -> int:
        return cell.houses + cell.hotels * 5

    def _find_sell_candidate(self, owner_id: int, allow_hotel: bool = True) -> Cell | None:
        candidates: list[Cell] = []
        for cell in self.state.board:
            if cell.owner_id != owner_id or cell.cell_type != "property":
                continue
            if self._property_level(cell) <= 0:
                continue
            if cell.hotels > 0 and not allow_hotel:
                continue
            group_cells = self._group_cells(owner_id, cell.group or "")
            if not group_cells:
                continue
            max_level = max(self._property_level(c) for c in group_cells)
            if self._property_level(cell) == max_level:
                candidates.append(cell)
        if not candidates:
            return None
        candidates.sort(key=lambda c: (-self._property_level(c), c.index))
        return candidates[0]

    def _mortgage_properties(self, player: Player, turn_index: int, target_cash: int) -> list[Event]:
        events: list[Event] = []
        if target_cash <= 0:
            return events
        candidates = self._mortgage_candidates(player.player_id)
        for cell in candidates:
            if player.money >= target_cash:
                break
            events.extend(self._mortgage_property(player, cell, turn_index))
        return events

    def _mortgage_candidates(self, owner_id: int) -> list[Cell]:
        candidates: list[Cell] = []
        for cell in self.state.board:
            if cell.owner_id != owner_id:
                continue
            if cell.cell_type not in {"property", "railroad", "utility"}:
                continue
            if self._can_mortgage_property(cell, owner_id):
                candidates.append(cell)
        if owner_id < len(self.bots):
            bot = self.bots[owner_id]
            prioritizer = getattr(bot, "prioritize_mortgage", None)
            if callable(prioritizer):
                return prioritizer(candidates, self.state, self.state.players[owner_id])
        candidates.sort(key=lambda c: (c.price or 0, c.index))
        return candidates

    def _can_mortgage_property(self, cell: Cell, owner_id: int) -> bool:
        if cell.owner_id != owner_id:
            return False
        if cell.mortgaged:
            return False
        if cell.houses > 0 or cell.hotels > 0:
            return False
        if cell.group and cell.cell_type == "property":
            group_cells = self._group_cells(owner_id, cell.group)
            if any(self._property_level(c) > 0 for c in group_cells):
                return False
        return True

    def _mortgage_property(self, player: Player, cell: Cell, turn_index: int) -> list[Event]:
        if not self._can_mortgage_property(cell, player.player_id):
            return []
        mortgage_value = cell.mortgage_value or 0
        cell.mortgaged = True
        player.money += mortgage_value
        return [
            Event(
                type="MORTGAGE",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} заложил '{cell.name}' за {mortgage_value}",
                payload={"cell_index": cell.index, "amount": mortgage_value},
            )
        ]

    def _unmortgage_cost(self, cell: Cell) -> int:
        mortgage_value = cell.mortgage_value or 0
        return int(mortgage_value * (1 + self.state.rules.interest_rate))

    def _unmortgage_property(self, player: Player, cell: Cell, turn_index: int) -> list[Event]:
        if cell.owner_id != player.player_id or not cell.mortgaged:
            return []
        cost = self._unmortgage_cost(cell)
        if player.money < cost:
            return []
        player.money -= cost
        cell.mortgaged = False
        return [
            Event(
                type="UNMORTGAGE",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} выкупил ипотеку '{cell.name}' за {cost}",
                payload={"cell_index": cell.index, "amount": cost},
            )
        ]

    def _bank_houses_available(self) -> int:
        total = sum(cell.houses for cell in self.state.board)
        return self.state.rules.bank_houses - total

    def _bank_hotels_available(self) -> int:
        total = sum(cell.hotels for cell in self.state.board)
        return self.state.rules.bank_hotels - total

    def _can_build_on_cell(self, owner_id: int, cell: Cell) -> bool:
        if cell.owner_id != owner_id or cell.cell_type != "property":
            return False
        if cell.group is None or not self._owns_group(owner_id, cell.group):
            return False
        group_cells = self._group_cells(owner_id, cell.group)
        if any(c.mortgaged for c in group_cells):
            return False
        if cell.house_cost is None:
            return False
        levels = [self._property_level(c) for c in group_cells]
        min_level = min(levels)
        if self._property_level(cell) != min_level:
            return False
        if self._property_level(cell) >= 5:
            return False
        if self._property_level(cell) == 4 and self._bank_hotels_available() <= 0:
            return False
        if self._property_level(cell) < 4 and self._bank_houses_available() <= 0:
            return False
        return True

    def _build_on_property(self, player: Player, cell: Cell, turn_index: int) -> list[Event]:
        if not self._can_build_on_cell(player.player_id, cell):
            return []
        cost = cell.house_cost or 0
        if player.money < cost:
            return []
        level = self._property_level(cell)
        if level < 4:
            cell.houses += 1
            building = "house"
        else:
            cell.hotels = 1
            cell.houses = 0
            building = "hotel"
        player.money -= cost
        return [
            Event(
                type="BUILD",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} построил {building} на '{cell.name}' за {cost}",
                payload={"cell_index": cell.index, "building": building, "cost": cost},
            )
        ]

    def _build_candidates(self, owner_id: int) -> list[Cell]:
        candidates = [
            cell for cell in self.state.board if self._can_build_on_cell(owner_id, cell)
        ]
        candidates.sort(key=lambda c: (c.group or "", self._property_level(c), c.index))
        return candidates

    def _bot_economy_phase(self, player: Player, turn_index: int) -> list[Event]:
        if player.bankrupt or self.state.game_over:
            return []
        bot = self.bots[player.player_id]
        decision = bot.decide(
            self.state,
            {"type": "economy_phase", "player_id": player.player_id},
        )
        events: list[Event] = []
        for action in decision.get("actions", []):
            action_type = action.get("action")
            cell_index = action.get("cell_index")
            if cell_index is None:
                continue
            cell = self.state.board[int(cell_index)]
            if action_type == "unmortgage":
                events.extend(self._unmortgage_property(player, cell, turn_index))
            elif action_type == "build":
                events.extend(self._build_on_property(player, cell, turn_index))
        return events

    def _bot_trade_phase(self, player: Player, turn_index: int) -> list[Event]:
        state = self.state
        if state.rules.no_trades or state.game_over or player.bankrupt:
            return []
        decision = self.bots[player.player_id].decide(
            state,
            {"type": "trade_offer", "player_id": player.player_id},
        )
        if decision.get("action") != "offer":
            return []
        offer_raw = decision.get("offer", {})
        offer, error = self._normalize_trade_offer(offer_raw, player.player_id)
        if error:
            return [
                Event(
                    type="TRADE_INVALID",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} попытался предложить некорректную сделку ({error})",
                    payload={"reason": error},
                )
            ]
        valid, reason = self._validate_trade_offer(offer)
        if not valid:
            return [
                Event(
                    type="TRADE_INVALID",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} предложил недопустимую сделку ({reason})",
                    payload={"reason": reason, "offer": offer},
                )
            ]

        from_id = offer["from_player"]
        to_id = offer["to_player"]
        from_player = state.players[from_id]
        to_player = state.players[to_id]
        offer_hash = self._trade_offer_hash(offer)

        events: list[Event] = [
            Event(
                type="TRADE_OFFER",
                turn_index=turn_index,
                player_id=from_id,
                msg_ru=f"{from_player.name} предлагает сделку {to_player.name}",
                payload={"offer": offer},
            )
        ]

        decision_to = self.bots[to_id].decide(
            state,
            {"type": "trade_accept", "player_id": to_id, "offer": offer},
        )
        score = float(decision_to.get("score", 0.0))
        valid_decision = bool(decision_to.get("valid", True))
        accept = decision_to.get("action") == "accept" and valid_decision and score > 0.0
        policy_to = str(decision_to.get("mortgage_policy") or "keep")

        if not self._trade_offer_improved(from_id, to_id, score):
            events.append(
                Event(
                    type="TRADE_REJECT",
                    turn_index=turn_index,
                    player_id=to_id,
                    msg_ru=f"{to_player.name} отклонил сделку от {from_player.name} (оффер не лучше предыдущего)",
                    payload={"reason": "offer_not_improved", "score": score},
                )
            )
            return events

        if not accept:
            events.append(
                Event(
                    type="TRADE_REJECT",
                    turn_index=turn_index,
                    player_id=to_id,
                    msg_ru=f"{to_player.name} отклонил сделку от {from_player.name}",
                    payload={"reason": "reject", "score": score},
                )
            )
            self._record_trade_reject(from_id, to_id, score, offer_hash, turn_index)
            return events

        score_from, policy_from, valid_from, _cash_after = evaluate_trade_for_player(
            state, from_id, offer, self.bots[from_id].params
        )
        if not valid_from:
            events.append(
                Event(
                    type="TRADE_REJECT",
                    turn_index=turn_index,
                    player_id=from_id,
                    msg_ru=f"{from_player.name} не может завершить сделку",
                    payload={"reason": "invalid_from", "score": float(score_from)},
                )
            )
            self._record_trade_reject(from_id, to_id, score, offer_hash, turn_index)
            return events

        policy_from = str(policy_from or "keep")

        events.append(
            Event(
                type="TRADE_ACCEPT",
                turn_index=turn_index,
                player_id=to_id,
                msg_ru=f"{to_player.name} принял сделку от {from_player.name}",
                payload={"score": score},
            )
        )
        events.extend(self._apply_trade(offer, policy_from, policy_to, turn_index))
        if (from_id, to_id) in state.trade_history:
            del state.trade_history[(from_id, to_id)]
        return events

    def _normalize_trade_offer(
        self, offer_raw: dict[str, object] | None, proposer_id: int
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(offer_raw, dict):
            return None, "offer_not_dict"
        def _coerce_props(value: object) -> list[int]:
            if value is None:
                return []
            if isinstance(value, (list, tuple, set)):
                items: list[int] = []
                for item in value:
                    try:
                        items.append(int(item))
                    except (TypeError, ValueError):
                        continue
                return sorted(set(items))
            try:
                return [int(value)]
            except (TypeError, ValueError):
                return []
        try:
            from_id = int(offer_raw.get("from_player", proposer_id))
            to_id = int(offer_raw.get("to_player", -1))
        except (TypeError, ValueError):
            return None, "invalid_players"
        if from_id != proposer_id:
            return None, "from_mismatch"
        give_props = _coerce_props(offer_raw.get("give_props", []))
        receive_props = _coerce_props(offer_raw.get("receive_props", []))
        give_cash = int(offer_raw.get("give_cash", 0) or 0)
        receive_cash = int(offer_raw.get("receive_cash", 0) or 0)
        give_cards = int(offer_raw.get("give_cards", 0) or 0)
        receive_cards = int(offer_raw.get("receive_cards", 0) or 0)
        if give_cash < 0 or receive_cash < 0 or give_cards < 0 or receive_cards < 0:
            return None, "negative_values"
        offer = {
            "from_player": from_id,
            "to_player": to_id,
            "give_props": give_props,
            "receive_props": receive_props,
            "give_cash": give_cash,
            "receive_cash": receive_cash,
            "give_cards": give_cards,
            "receive_cards": receive_cards,
        }
        return offer, None

    def _validate_trade_offer(self, offer: dict[str, Any]) -> tuple[bool, str]:
        state = self.state
        from_id = int(offer.get("from_player", -1))
        to_id = int(offer.get("to_player", -1))
        if from_id == to_id:
            return False, "same_player"
        if from_id < 0 or to_id < 0:
            return False, "invalid_player"
        if from_id >= len(state.players) or to_id >= len(state.players):
            return False, "invalid_player"
        if state.players[from_id].bankrupt or state.players[to_id].bankrupt:
            return False, "bankrupt"
        give_props = list(offer.get("give_props", []))
        receive_props = list(offer.get("receive_props", []))
        if set(give_props) & set(receive_props):
            return False, "duplicate_props"
        if not give_props and not receive_props and not offer.get("give_cash") and not offer.get("receive_cash"):
            if not offer.get("give_cards") and not offer.get("receive_cards"):
                return False, "empty_offer"
        if offer.get("give_cash", 0) > state.players[from_id].money:
            return False, "insufficient_cash_from"
        if offer.get("receive_cash", 0) > state.players[to_id].money:
            return False, "insufficient_cash_to"
        if offer.get("give_cards", 0) > len(state.players[from_id].get_out_of_jail_cards):
            return False, "insufficient_cards_from"
        if offer.get("receive_cards", 0) > len(state.players[to_id].get_out_of_jail_cards):
            return False, "insufficient_cards_to"
        for idx in give_props:
            if idx < 0 or idx >= len(state.board):
                return False, "invalid_property"
            cell = state.board[idx]
            if cell.owner_id != from_id:
                return False, "wrong_owner"
            if not self._tradeable_cell(cell):
                return False, "not_tradeable"
        for idx in receive_props:
            if idx < 0 or idx >= len(state.board):
                return False, "invalid_property"
            cell = state.board[idx]
            if cell.owner_id != to_id:
                return False, "wrong_owner"
            if not self._tradeable_cell(cell):
                return False, "not_tradeable"
        cash_after_from = (
            state.players[from_id].money
            - int(offer.get("give_cash", 0))
            + int(offer.get("receive_cash", 0))
        )
        cash_after_to = (
            state.players[to_id].money
            - int(offer.get("receive_cash", 0))
            + int(offer.get("give_cash", 0))
        )
        fee_from = self._trade_interest_fee(receive_props)
        fee_to = self._trade_interest_fee(give_props)
        if cash_after_from < fee_from:
            return False, "interest_fee_from"
        if cash_after_to < fee_to:
            return False, "interest_fee_to"
        return True, "ok"

    def _trade_offer_hash(self, offer: dict[str, Any]) -> str:
        payload = json.dumps(offer, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _trade_interest_fee(self, props: list[int]) -> int:
        interest = float(self.state.rules.interest_rate)
        total = 0
        for idx in props:
            cell = self.state.board[idx]
            if cell.mortgaged:
                total += int((cell.mortgage_value or 0) * interest)
        return total

    def _tradeable_cell(self, cell: Cell) -> bool:
        if cell.cell_type not in {"property", "railroad", "utility"}:
            return False
        if cell.houses > 0 or cell.hotels > 0:
            return False
        if cell.cell_type == "property" and cell.group:
            for other in self.state.board:
                if other.group == cell.group and (other.houses > 0 or other.hotels > 0):
                    return False
        return True

    def _trade_offer_improved(self, from_id: int, to_id: int, score: float) -> bool:
        history = self.state.trade_history.get((from_id, to_id))
        if history is None or history.last_reject_turn < 0:
            return True
        if self._trade_history_reset_needed(history, from_id, to_id):
            self.state.trade_history.pop((from_id, to_id), None)
            return True
        threshold = history.last_reject_score + abs(history.last_reject_score) * 0.02
        return score > threshold

    def _trade_history_reset_needed(self, history: TradeHistory, from_id: int, to_id: int) -> bool:
        current_from_props = self._player_property_snapshot(from_id)
        current_to_props = self._player_property_snapshot(to_id)
        if history.from_props and history.from_props != current_from_props:
            return True
        if history.to_props and history.to_props != current_to_props:
            return True
        current_from_worth = self._player_net_worth(from_id)
        current_to_worth = self._player_net_worth(to_id)
        if history.from_worth > 0:
            if abs(current_from_worth - history.from_worth) / history.from_worth > 0.3:
                return True
        elif current_from_worth != history.from_worth:
            return True
        if history.to_worth > 0:
            if abs(current_to_worth - history.to_worth) / history.to_worth > 0.3:
                return True
        elif current_to_worth != history.to_worth:
            return True
        return False

    def _record_trade_reject(
        self, from_id: int, to_id: int, score: float, offer_hash: str, turn_index: int
    ) -> None:
        history = self.state.trade_history.get((from_id, to_id))
        if history is None:
            history = TradeHistory()
            self.state.trade_history[(from_id, to_id)] = history
        history.last_reject_score = float(score)
        history.last_offer_hash = offer_hash
        history.last_reject_turn = turn_index
        history.from_props = self._player_property_snapshot(from_id)
        history.to_props = self._player_property_snapshot(to_id)
        history.from_worth = self._player_net_worth(from_id)
        history.to_worth = self._player_net_worth(to_id)

    def _player_property_snapshot(self, player_id: int) -> tuple[int, ...]:
        owned = [cell.index for cell in self.state.board if cell.owner_id == player_id]
        return tuple(sorted(owned))

    def _player_net_worth(self, player_id: int) -> float:
        player = self.state.players[player_id]
        total = float(player.money)
        for cell in self.state.board:
            if cell.owner_id != player_id:
                continue
            if cell.mortgaged:
                total += float(cell.mortgage_value or 0)
            else:
                total += float(cell.price or 0)
            total += float((cell.houses + cell.hotels) * (cell.house_cost or 0))
        return total

    def _apply_trade(
        self, offer: dict[str, Any], policy_from: str, policy_to: str, turn_index: int
    ) -> list[Event]:
        state = self.state
        from_id = int(offer["from_player"])
        to_id = int(offer["to_player"])
        from_player = state.players[from_id]
        to_player = state.players[to_id]
        give_cash = int(offer.get("give_cash", 0))
        receive_cash = int(offer.get("receive_cash", 0))

        from_player.money -= give_cash
        from_player.money += receive_cash
        to_player.money -= receive_cash
        to_player.money += give_cash

        self._transfer_cards(from_player, to_player, int(offer.get("give_cards", 0)))
        self._transfer_cards(to_player, from_player, int(offer.get("receive_cards", 0)))

        for idx in offer.get("give_props", []):
            self._transfer_property(int(idx), from_id, to_id)
        for idx in offer.get("receive_props", []):
            self._transfer_property(int(idx), to_id, from_id)

        fee_from, unmortgaged_from = self._apply_trade_mortgage(
            offer.get("receive_props", []), from_player, policy_from
        )
        fee_to, unmortgaged_to = self._apply_trade_mortgage(
            offer.get("give_props", []), to_player, policy_to
        )

        payload = {
            "from_player": from_id,
            "to_player": to_id,
            "give_props": list(offer.get("give_props", [])),
            "receive_props": list(offer.get("receive_props", [])),
            "give_cash": give_cash,
            "receive_cash": receive_cash,
            "give_cards": int(offer.get("give_cards", 0)),
            "receive_cards": int(offer.get("receive_cards", 0)),
            "from_policy": policy_from,
            "to_policy": policy_to,
            "from_fee": fee_from,
            "to_fee": fee_to,
            "from_unmortgaged": unmortgaged_from,
            "to_unmortgaged": unmortgaged_to,
        }
        return [
            Event(
                type="TRADE_EXECUTE",
                turn_index=turn_index,
                player_id=from_id,
                msg_ru=f"Сделка выполнена: {from_player.name} ↔ {to_player.name}",
                payload=payload,
            )
        ]

    def _transfer_property(self, cell_index: int, from_id: int, to_id: int) -> None:
        cell = self.state.board[cell_index]
        if cell.owner_id != from_id:
            return
        cell.owner_id = to_id
        from_player = self.state.players[from_id]
        to_player = self.state.players[to_id]
        from_player.properties = [idx for idx in from_player.properties if idx != cell_index]
        if cell_index not in to_player.properties:
            to_player.properties.append(cell_index)

    def _transfer_cards(self, from_player: Player, to_player: Player, count: int) -> None:
        if count <= 0:
            return
        for _ in range(min(count, len(from_player.get_out_of_jail_cards))):
            card = from_player.get_out_of_jail_cards.pop()
            to_player.get_out_of_jail_cards.append(card)

    def _apply_trade_mortgage(
        self, props: list[int], player: Player, policy: str
    ) -> tuple[int, list[int]]:
        fee_total = 0
        unmortgaged: list[int] = []
        for idx in props:
            cell = self.state.board[int(idx)]
            if not cell.mortgaged:
                continue
            if policy == "unmortgage":
                cost = self._unmortgage_cost(cell)
                if player.money >= cost:
                    player.money -= cost
                    cell.mortgaged = False
                    fee_total += cost
                    unmortgaged.append(cell.index)
                    continue
            fee = int((cell.mortgage_value or 0) * float(self.state.rules.interest_rate))
            player.money -= fee
            fee_total += fee
        return fee_total, unmortgaged

    def _run_auction(self, cell: Cell, turn_index: int) -> list[Event]:
        state = self.state
        events: list[Event] = []
        participants = [p.player_id for p in state.players if not p.bankrupt]
        active = participants[:]
        current_price = 0
        increments = getattr(state.rules, "auction_increments", None)
        if not increments:
            increments = [1]
        increments = sorted({int(value) for value in increments if int(value) > 0})
        min_increment = increments[0] if increments else 1
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
                current_price = normalize_auction_price(current_price, increments)
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
                    increment = bid - current_price
                    if (
                        bid <= current_price
                        or bid > player.money
                        or bid < current_price + min_increment
                        or increment not in increments
                        or bid % min_increment != 0
                    ):
                        events.append(
                            Event(
                                type="AUCTION_PASS",
                                turn_index=turn_index,
                                player_id=player_id,
                                msg_ru=f"{player.name} пас (некорректная ставка)",
                                payload={
                                    "bid": bid,
                                    "current_price": current_price,
                                    "min_increment": min_increment,
                                },
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
                                msg_ru=f"{player.name} повышает до {bid} (+{increment})",
                                payload={"bid": bid, "increment": increment},
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
    num_players: int,
    seed: int,
    data_dir: Path | None = None,
    bot_params: BotParams | list[BotParams] | None = None,
) -> Engine:
    state = create_game(num_players=num_players, seed=seed, data_dir=data_dir)
    bots = create_bots(num_players, bot_params)
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

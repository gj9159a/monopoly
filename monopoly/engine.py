from __future__ import annotations

import random
from pathlib import Path

from .bots import BaseBot, create_bots
from .data_loader import load_board, load_rules
from .models import Cell, Event, GameState, Player


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
            state.event_log.extend(events)
            state.turn_index += 1
            if advance_player:
                self._advance_player()
            return events

        roll_events, extra_turn, went_to_jail = self._roll_and_move(
            player, turn_index, allow_extra_turn=True, count_doubles=True
        )
        events.extend(roll_events)

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
            state, {"type": "jail_decision", "player_id": player.player_id}
        )
        action = decision.get("action")

        if action == "pay" and player.money >= fine:
            player.money -= fine
            player.in_jail = False
            player.jail_turns = 0
            player.doubles_count = 0
            events.append(
                Event(
                    type="JAIL_PAY",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} заплатил {fine} и вышел из тюрьмы",
                    payload={"amount": fine},
                )
            )
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
            player.money -= fine
            player.in_jail = False
            player.jail_turns = 0
            events.append(
                Event(
                    type="JAIL_PAY_FORCED",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} заплатил {fine} после 3 попыток и вышел из тюрьмы",
                    payload={"amount": fine},
                )
            )
            cell, move_events = self._move_player(player, total, turn_index)
            events.extend(move_events)
            events.extend(self._handle_landing(player, cell, turn_index, total))
        return events, True

    def _handle_landing(self, player: Player, cell: Cell, turn_index: int, dice_total: int | None) -> list[Event]:
        events: list[Event] = []
        if cell.cell_type == "go_to_jail":
            events.extend(self._send_to_jail(player, turn_index, reason="клетка 'В тюрьму'"))
            return events
        if cell.cell_type == "tax":
            amount = cell.tax_amount or 0
            player.money -= amount
            events.append(
                Event(
                    type="PAY_TAX",
                    turn_index=turn_index,
                    player_id=player.player_id,
                    msg_ru=f"{player.name} заплатил налог {amount}",
                    payload={"amount": amount, "cell_index": cell.index},
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
                player.money -= rent
                owner.money += rent
                events.append(
                    Event(
                        type="PAY_RENT",
                        turn_index=turn_index,
                        player_id=player.player_id,
                        msg_ru=f"{player.name} заплатил ренту {rent} владельцу {owner.name}{reason}",
                        payload={
                            "amount": rent,
                            "owner_id": owner.player_id,
                            "cell_index": cell.index,
                        },
                    )
                )
        return events

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

    return GameState(
        seed=seed,
        rng=rng,
        rules=rules,
        board=board,
        players=players,
        turn_index=0,
        current_player=0,
        event_log=[],
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

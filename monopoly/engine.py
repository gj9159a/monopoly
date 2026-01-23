from __future__ import annotations

import random
from pathlib import Path

from .bots import BaseBot, create_bots
from .data_loader import load_board, load_rules
from .models import Cell, Event, GameState, Player, Rules


class Engine:
    def __init__(self, state: GameState, bots: list[BaseBot]) -> None:
        if len(bots) != len(state.players):
            raise ValueError("Число ботов должно совпадать с числом игроков")
        self.state = state
        self.bots = bots

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

        die1, die2 = _roll_dice(state.rng)
        total = die1 + die2
        events.append(
            Event(
                type="DICE_ROLL",
                turn_index=turn_index,
                player_id=player.player_id,
                msg_ru=f"{player.name} бросил {die1} и {die2}",
                payload={"dice": [die1, die2], "total": total},
            )
        )

        old_pos = player.position
        new_pos = (old_pos + total) % len(state.board)
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
                payload={"from": old_pos, "to": new_pos, "steps": total},
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

        events.extend(self._handle_landing(player, cell, turn_index))

        state.event_log.extend(events)
        state.turn_index += 1
        self._advance_player()
        return events

    def _advance_player(self) -> None:
        self.state.current_player = (self.state.current_player + 1) % len(self.state.players)

    def _handle_landing(self, player: Player, cell: Cell, turn_index: int) -> list[Event]:
        events: list[Event] = []
        if cell.cell_type in {"property", "railroad", "utility"}:
            if cell.owner_id is None and self.state.rules.hr1_always_auction:
                events.extend(self._run_auction(cell, turn_index))
        return events

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

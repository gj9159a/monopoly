from __future__ import annotations

import random
from pathlib import Path

from .data_loader import load_board, load_rules
from .models import Cell, Event, GameState, Player, Rules


class Engine:
    def __init__(self, state: GameState) -> None:
        self.state = state

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

        state.event_log.extend(events)
        state.turn_index += 1
        self._advance_player()
        return events

    def _advance_player(self) -> None:
        self.state.current_player = (self.state.current_player + 1) % len(self.state.players)


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


def _roll_dice(rng: random.Random) -> tuple[int, int]:
    return rng.randint(1, 6), rng.randint(1, 6)

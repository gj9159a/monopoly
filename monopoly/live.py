from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bots import Bot
from .engine import Engine, create_game
from .io_utils import write_json_atomic
from .models import GameState
from .params import BotParams, decide_auction_bid, decide_build_actions
from .train import score_player


def _parse_workers(value: str) -> int:
    value = value.strip().lower()
    if value == "auto":
        return max(1, os.cpu_count() or 1)
    return max(1, int(value))


def _clone_state(state: GameState, seed: int | None = None) -> GameState:
    cloned = copy.deepcopy(state)
    rng = random.Random()
    if seed is None:
        rng.setstate(state.rng.getstate())
    else:
        rng.seed(seed)
    cloned.rng = rng
    return cloned


class FixedActionBot:
    def __init__(self, base_bot: Bot, player_id: int, decision_type: str, action: dict[str, Any]) -> None:
        self.base_bot = base_bot
        self.player_id = player_id
        self.decision_type = decision_type
        self.action = action
        self.used = False

    def decide(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        if (
            not self.used
            and context.get("type") == self.decision_type
            and int(context.get("player_id", -1)) == self.player_id
        ):
            self.used = True
            return self.action
        return self.base_bot.decide(state, context)


@dataclass
class PlannerProgress:
    thinking: bool
    decision_context: str
    rollouts_done: int
    rollouts_target: int
    time_budget_sec: float
    time_left_sec: float


class LiveWriter:
    def __init__(self, path: Path, events_tail: int = 200) -> None:
        self.path = path
        self.events_tail = events_tail
        self.snapshot_index = 0

    def write(self, state: GameState, progress: PlannerProgress | None = None) -> None:
        self.snapshot_index += 1
        events = state.event_log[-self.events_tail :]
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "snapshot_index": self.snapshot_index,
            "turn_index": state.turn_index,
            "current_player": state.current_player,
            "game_over": state.game_over,
            "winner_id": state.winner_id,
            "players": [
                {
                    "player_id": player.player_id,
                    "name": player.name,
                    "position": player.position,
                    "money": player.money,
                    "in_jail": player.in_jail,
                    "bankrupt": player.bankrupt,
                    "properties": list(player.properties),
                    "get_out_of_jail_cards": len(player.get_out_of_jail_cards),
                }
                for player in state.players
            ],
            "board": [
                {
                    "index": cell.index,
                    "name": cell.name,
                    "cell_type": cell.cell_type,
                    "group": cell.group,
                    "owner_id": cell.owner_id,
                    "houses": cell.houses,
                    "hotels": cell.hotels,
                    "mortgaged": cell.mortgaged,
                }
                for cell in state.board
            ],
            "event_log": [
                {
                    "type": event.type,
                    "turn_index": event.turn_index,
                    "player_id": event.player_id,
                    "msg_ru": event.msg_ru,
                }
                for event in events
            ],
        }
        if progress is None:
            payload.update(
                {
                    "thinking": False,
                    "decision_context": "",
                    "rollouts_done": 0,
                    "rollouts_target": 0,
                    "time_budget_sec": 0.0,
                    "time_left_sec": 0.0,
                }
            )
        else:
            payload.update(
                {
                    "thinking": progress.thinking,
                    "decision_context": progress.decision_context,
                    "rollouts_done": progress.rollouts_done,
                    "rollouts_target": progress.rollouts_target,
                    "time_budget_sec": progress.time_budget_sec,
                    "time_left_sec": progress.time_left_sec,
                }
            )
        write_json_atomic(self.path, payload)


class DeepPlannerBot:
    def __init__(
        self,
        params: BotParams,
        player_id: int,
        writer: LiveWriter,
        time_budget_sec: float,
        horizon_turns: int,
    ) -> None:
        self.params = params
        self.player_id = player_id
        self.base_bot = Bot(params)
        self.writer = writer
        self.time_budget_sec = max(0.0, time_budget_sec)
        self.horizon_turns = max(1, horizon_turns)
        self.decision_index = 0
        self._last_progress_ts = 0.0

    def decide(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        decision_type = context.get("type", "")
        if decision_type not in {"auction_bid", "jail_decision", "economy_phase"}:
            return self.base_bot.decide(state, context)

        candidates = self._candidate_actions(state, context)
        if len(candidates) <= 1 or self.time_budget_sec <= 0:
            return candidates[0] if candidates else self.base_bot.decide(state, context)

        self.decision_index += 1
        start = time.perf_counter()
        scores = [0.0] * len(candidates)
        counts = [0] * len(candidates)
        rollouts_done = 0

        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= self.time_budget_sec:
                break
            for idx, action in enumerate(candidates):
                if elapsed >= self.time_budget_sec:
                    break
                rollout_seed = state.seed + self.decision_index * 10007 + idx * 97 + rollouts_done * 13
                score = self._rollout_score(state, context, action, rollout_seed)
                scores[idx] += score
                counts[idx] += 1
                rollouts_done += 1
                elapsed = time.perf_counter() - start
                if elapsed >= self.time_budget_sec:
                    break
                self._maybe_report_progress(
                    state,
                    decision_type,
                    rollouts_done,
                    self.time_budget_sec,
                    start,
                )

        self._report_progress(state, decision_type, rollouts_done, self.time_budget_sec, final=True)

        best_idx = 0
        best_value = float("-inf")
        for idx, total in enumerate(scores):
            if counts[idx] == 0:
                continue
            avg = total / counts[idx]
            if avg > best_value:
                best_value = avg
                best_idx = idx
        return candidates[best_idx]

    def _candidate_actions(self, state: GameState, context: dict[str, Any]) -> list[dict[str, Any]]:
        decision_type = context.get("type")
        if decision_type == "auction_bid":
            player_id = int(context["player_id"])
            player = state.players[player_id]
            cell = context["cell"]
            current_price = int(context["current_price"])
            min_increment = int(context["min_increment"])
            base_bid = decide_auction_bid(state, player, cell, current_price, min_increment, self.params)
            options = [{"action": "pass"}]
            next_bid = current_price + min_increment
            if next_bid <= player.money:
                options.append({"action": "bid", "bid": next_bid})
            if base_bid > 0:
                options.append({"action": "bid", "bid": base_bid})
            options = self._dedupe_actions(options)
            return options
        if decision_type == "jail_decision":
            player_id = int(context["player_id"])
            player = state.players[player_id]
            actions = [{"action": "roll"}]
            if player.money >= state.rules.jail_fine:
                actions.append({"action": "pay"})
            if player.get_out_of_jail_cards:
                actions.append({"action": "use_card"})
            return actions
        if decision_type == "economy_phase":
            player_id = int(context["player_id"])
            player = state.players[player_id]
            base_actions = decide_build_actions(state, player, self.params)
            return self._dedupe_actions(
                [
                    {"actions": []},
                    {"actions": base_actions},
                ]
            )
        return [self.base_bot.decide(state, context)]

    def _dedupe_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for action in actions:
            key = json.dumps(action, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            unique.append(action)
        return unique

    def _rollout_score(
        self,
        state: GameState,
        context: dict[str, Any],
        action: dict[str, Any],
        rollout_seed: int,
    ) -> float:
        cloned = _clone_state(state, seed=rollout_seed)
        bots: list[Bot] = []
        decision_type = str(context.get("type"))
        for pid in range(len(cloned.players)):
            base_bot = Bot(self.params)
            if pid == self.player_id:
                bots.append(FixedActionBot(base_bot, pid, decision_type, action))
            else:
                bots.append(base_bot)
        engine = Engine(cloned, bots)
        first_bankrupt_id = None
        steps = 0
        while steps < self.horizon_turns and not engine.state.game_over:
            events = engine.step()
            for event in events:
                if event.type == "BANKRUPTCY" and first_bankrupt_id is None:
                    first_bankrupt_id = event.player_id
            steps += 1
        return score_player(engine.state, self.player_id, first_bankrupt_id)

    def _maybe_report_progress(
        self,
        state: GameState,
        decision_type: str,
        rollouts_done: int,
        time_budget_sec: float,
        start_time: float,
    ) -> None:
        now = time.perf_counter()
        if now - self._last_progress_ts < 0.2:
            return
        self._last_progress_ts = now
        elapsed = max(0.0, now - start_time)
        time_left = max(0.0, time_budget_sec - elapsed)
        self.writer.write(
            state,
            PlannerProgress(
                thinking=True,
                decision_context=decision_type,
                rollouts_done=rollouts_done,
                rollouts_target=0,
                time_budget_sec=time_budget_sec,
                time_left_sec=time_left,
            ),
        )

    def _report_progress(
        self,
        state: GameState,
        decision_type: str,
        rollouts_done: int,
        time_budget_sec: float,
        final: bool = False,
    ) -> None:
        time_left = 0.0 if final else time_budget_sec
        self.writer.write(
            state,
            PlannerProgress(
                thinking=not final,
                decision_context=decision_type if not final else "",
                rollouts_done=rollouts_done,
                rollouts_target=0,
                time_budget_sec=time_budget_sec,
                time_left_sec=time_left,
            ),
        )


def run_live(
    params_path: Path | None,
    players: int,
    seed: int,
    mode: str,
    time_budget_sec: float,
    horizon_turns: int,
    out_path: Path,
    max_steps: int,
    workers: int,
) -> None:
    params = BotParams()
    if params_path is not None:
        from .params import load_params

        params = load_params(params_path)

    state = create_game(players, seed)
    writer = LiveWriter(out_path)

    bots: list[Any] = []
    if mode == "deep":
        for pid in range(players):
            bots.append(DeepPlannerBot(params, pid, writer, time_budget_sec, horizon_turns))
    else:
        bots = [Bot(params) for _ in range(players)]

    engine = Engine(state, bots)  # type: ignore[arg-type]
    writer.write(engine.state)

    steps = 0
    while steps < max_steps and not engine.state.game_over:
        engine.step()
        writer.write(engine.state)
        steps += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live-симуляция с прогрессом планировщика")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--mode", type=str, choices=["deep", "fast"], default="deep")
    parser.add_argument("--workers", type=_parse_workers, default=_parse_workers("auto"))
    parser.add_argument("--time-per-decision-sec", type=float, default=3.0)
    parser.add_argument("--horizon-turns", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("runs/live_state.json"))
    parser.add_argument("--max-steps", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_live(
        params_path=args.params,
        players=args.players,
        seed=args.seed,
        mode=args.mode,
        time_budget_sec=args.time_per_decision_sec,
        horizon_turns=args.horizon_turns,
        out_path=args.out,
        max_steps=args.max_steps,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

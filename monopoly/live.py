from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bots import Bot
from .engine import Engine, create_game
from .io_utils import write_json_atomic
from .models import GameState
from .params import BotParams, ThinkingConfig
from .thinking import choose_action, fast_decide


def _parse_workers(value: str) -> int:
    value = value.strip().lower()
    if value == "auto":
        return max(1, os.cpu_count() or 1)
    return max(1, int(value))


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


class ThinkingLiveBot:
    def __init__(
        self,
        params: BotParams,
        player_id: int,
        writer: LiveWriter,
        config: ThinkingConfig,
    ) -> None:
        self.params = params.with_thinking(config)
        self.player_id = player_id
        self.writer = writer
        self.config = config
        self.cache: dict[str, float] = {}
        self._last_progress_ts = 0.0

    def decide(self, state: GameState, context: dict[str, Any]) -> dict[str, Any]:
        decision_type = context.get("type", "")
        if decision_type not in {"auction_bid", "jail_decision", "economy_phase", "liquidation", "liquidity"}:
            return fast_decide(state, context, self.params)

        start = time.perf_counter()
        time_budget_sec = max(0.0, self.config.time_budget_ms / 1000.0)

        def progress_cb(rollouts_done: int, candidates: int) -> None:
            now = time.perf_counter()
            if now - self._last_progress_ts < 0.2:
                return
            self._last_progress_ts = now
            elapsed = max(0.0, now - start)
            time_left = max(0.0, time_budget_sec - elapsed)
            self.writer.write(
                state,
                PlannerProgress(
                    thinking=True,
                    decision_context=str(decision_type),
                    rollouts_done=rollouts_done,
                    rollouts_target=candidates,
                    time_budget_sec=time_budget_sec,
                    time_left_sec=time_left,
                ),
            )

        cache = self.cache if self.config.cache_enabled else None
        if cache is not None:
            cache_size = max(0, int(self.config.cache_size))
            if cache_size and len(cache) > cache_size:
                cache.clear()

        action, stats = choose_action(
            state,
            context,
            self.params,
            self.config,
            cache=cache,
            progress_cb=progress_cb,
        )
        self.writer.write(
            state,
            PlannerProgress(
                thinking=False,
                decision_context="",
                rollouts_done=stats.rollouts,
                rollouts_target=stats.candidates,
                time_budget_sec=time_budget_sec,
                time_left_sec=0.0,
            ),
        )
        return action


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
    rollouts_per_action: int,
    cache_enabled: bool,
    cache_size: int,
    thinking_enabled: bool,
) -> None:
    params = BotParams()
    if params_path is not None:
        from .params import load_params

        params = load_params(params_path)

    state = create_game(players, seed)
    writer = LiveWriter(out_path)

    bots: list[Any] = []
    if thinking_enabled or mode == "deep":
        rollouts_value = rollouts_per_action if rollouts_per_action > 0 else 12
        config = ThinkingConfig(
            enabled=True,
            horizon_turns=max(1, horizon_turns),
            rollouts_per_action=rollouts_value,
            time_budget_ms=int(max(0.0, time_budget_sec) * 1000),
            workers=max(1, workers),
            cache_enabled=cache_enabled,
            cache_size=max(0, cache_size),
        )
        for pid in range(players):
            bots.append(ThinkingLiveBot(params, pid, writer, config))
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
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--mode", type=str, choices=["deep", "fast"], default="fast")
    parser.add_argument("--workers", type=_parse_workers, default=_parse_workers("auto"))
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--time-per-decision-sec", type=float, default=3.0)
    parser.add_argument("--horizon-turns", type=int, default=60)
    parser.add_argument("--rollouts-per-action", type=int, default=0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cache-size", type=int, default=4096)
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
        rollouts_per_action=int(args.rollouts_per_action),
        cache_enabled=bool(args.cache),
        cache_size=int(args.cache_size),
        thinking_enabled=bool(args.thinking),
    )


if __name__ == "__main__":
    main()

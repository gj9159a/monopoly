from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .models import GameState, Player, Cell
from .params import (
    BotParams,
    ThinkingConfig,
    choose_auction_bid,
    compute_cash_buffer,
    decide_auction_bid,
    decide_build_actions,
    decide_jail_exit,
    decide_liquidation,
    estimate_asset_value,
    _group_cells,
    _owns_group,
    _property_level,
)


@dataclass
class ThinkingStats:
    decision_type: str
    ms: float
    candidates: int
    rollouts: int
    best_score: float


DEFAULT_WEIGHTS = {
    "win": 1.0,
    "net": 0.002,
    "survive": 0.3,
}


def fast_decide(state: GameState, context: dict[str, Any], params: BotParams) -> dict[str, Any]:
    decision_type = context.get("type")
    if decision_type == "auction_bid":
        player_id = int(context["player_id"])
        player = state.players[player_id]
        cell = context["cell"]
        current_price = int(context["current_price"])
        min_increment = int(context["min_increment"])
        increments = getattr(state.rules, "auction_increments", None)
        if not increments:
            increments = [min_increment]
        target_max = decide_auction_bid(state, player, cell, current_price, min_increment, params)
        bid = choose_auction_bid(int(target_max), current_price, list(increments))
        if bid <= 0:
            return {"action": "pass"}
        return {"action": "bid", "bid": bid}
    if decision_type == "jail_decision":
        player_id = int(context["player_id"])
        player = state.players[player_id]
        action = decide_jail_exit(state, player, params)
        if action == "use_card" and not player.get_out_of_jail_cards:
            action = "roll"
        return {"action": action}
    if decision_type == "economy_phase":
        player_id = int(context["player_id"])
        player = state.players[player_id]
        actions = decide_build_actions(state, player, params)
        return {"actions": actions}
    if decision_type == "liquidation":
        player_id = int(context["player_id"])
        player = state.players[player_id]
        debt = int(context.get("debt", 0))
        actions = decide_liquidation(state, player, debt, params)
        return {"actions": actions}
    if decision_type == "trade_offer":
        return {"action": "pass"}
    if decision_type == "trade_accept":
        return {"action": "reject", "score": 0.0, "valid": True}
    raise ValueError(f"Неизвестный тип контекста: {decision_type}")


class FixedActionBot:
    def __init__(self, params: BotParams, player_id: int, decision_type: str, action: dict[str, Any]) -> None:
        self.params = params
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
            return dict(self.action)
        return fast_decide(state, context, self.params)


def _clone_state(state: GameState, seed: int) -> GameState:
    cloned = copy.deepcopy(state)
    cloned.rng.seed(seed)
    return cloned


def _state_signature(state: GameState, decision_type: str, player_id: int) -> str:
    payload = {
        "decision": decision_type,
        "player": player_id,
        "turn": state.turn_index,
        "current": state.current_player,
        "players": [
            {
                "id": p.player_id,
                "pos": p.position,
                "money": p.money,
                "jail": p.in_jail,
                "jt": p.jail_turns,
                "dbl": p.doubles_count,
                "bank": p.bankrupt,
                "cards": len(p.get_out_of_jail_cards),
            }
            for p in state.players
        ],
        "board": [
            {
                "idx": c.index,
                "owner": c.owner_id,
                "h": c.houses,
                "ht": c.hotels,
                "m": c.mortgaged,
            }
            for c in state.board
        ],
    }
    return json.dumps(payload, sort_keys=True)


def _action_signature(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, default=str)


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _apply_mortgage_action(state: GameState, player_id: int, cell_index: int) -> None:
    cell = state.board[cell_index]
    if cell.owner_id != player_id:
        return
    if cell.mortgaged:
        return
    if cell.houses > 0 or cell.hotels > 0:
        return
    mortgage_value = int(cell.mortgage_value or 0)
    if mortgage_value <= 0:
        return
    cell.mortgaged = True
    state.players[player_id].money += mortgage_value


def _apply_sell_building(state: GameState, player_id: int, cell_index: int) -> None:
    cell = state.board[cell_index]
    if cell.owner_id != player_id:
        return
    if cell.house_cost is None:
        return
    if cell.hotels <= 0 and cell.houses <= 0:
        return
    refund = int(cell.house_cost / 2)
    if cell.hotels > 0:
        cell.hotels = 0
        cell.houses = 4
    else:
        cell.houses = max(0, cell.houses - 1)
    state.players[player_id].money += refund


def _apply_liquidation_actions(state: GameState, player_id: int, actions: list[dict[str, Any]]) -> None:
    for action in actions:
        kind = action.get("action")
        cell_index = int(action.get("cell_index", -1))
        if cell_index < 0 or cell_index >= len(state.board):
            continue
        if kind == "sell_building":
            _apply_sell_building(state, player_id, cell_index)
        elif kind == "mortgage":
            _apply_mortgage_action(state, player_id, cell_index)


def _net_worth(state: GameState, player_id: int) -> float:
    player = state.players[player_id]
    total = player.money
    for cell in state.board:
        if cell.owner_id != player_id:
            continue
        total += cell.mortgage_value or 0 if cell.mortgaged else (cell.price or 0)
        total += (cell.houses + cell.hotels) * (cell.house_cost or 0)
    return float(total)


def _rollout_task(task: tuple[GameState, BotParams, int, dict[str, Any], dict[str, Any], int, int]) -> float:
    state, params, player_id, context, action, seed, horizon_turns = task
    from .engine import Engine

    cloned = _clone_state(state, seed)
    decision_type = context.get("type", "")
    if action.get("action_type") == "mortgage":
        _apply_mortgage_action(cloned, player_id, int(action.get("cell_index", -1)))
        action_to_apply = None
    elif action.get("action_type") == "mortgage_order":
        order = action.get("order", [])
        apply_count = int(action.get("apply_count", 1))
        if isinstance(order, list):
            for idx in order[:apply_count]:
                _apply_mortgage_action(cloned, player_id, int(idx))
        action_to_apply = None
    elif action.get("action_type") == "liquidation":
        actions = action.get("actions", [])
        if isinstance(actions, list):
            _apply_liquidation_actions(cloned, player_id, actions)
        action_to_apply = None
    else:
        action_to_apply = action

    bots = []
    for pid in range(len(cloned.players)):
        if pid == player_id and action_to_apply is not None:
            bots.append(FixedActionBot(params, pid, decision_type, action_to_apply))
        else:
            bots.append(FixedActionBot(params, pid, "", {}))
    engine = Engine(cloned, bots)  # type: ignore[arg-type]

    first_bankrupt_id = None
    steps = 0
    while steps < horizon_turns and not engine.state.game_over:
        events = engine.step()
        for event in events:
            if event.type == "BANKRUPTCY" and first_bankrupt_id is None:
                first_bankrupt_id = event.player_id
        steps += 1

    win = 1.0 if engine.state.winner_id == player_id else 0.0
    survive = 0.0 if engine.state.players[player_id].bankrupt else 1.0
    net_norm = _net_worth(engine.state, player_id) / max(1.0, engine.state.rules.starting_cash)

    return (
        DEFAULT_WEIGHTS["win"] * win
        + DEFAULT_WEIGHTS["survive"] * survive
        + DEFAULT_WEIGHTS["net"] * net_norm
    )


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for action in actions:
        key = _action_signature(action)
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return unique


def _auction_candidates(
    state: GameState,
    player: Player,
    cell: Cell,
    params: BotParams,
    current_price: int,
    min_increment: int,
) -> list[dict[str, Any]]:
    increments = getattr(state.rules, "auction_increments", None)
    if not increments:
        increments = [min_increment]
    target_max = decide_auction_bid(state, player, cell, current_price, min_increment, params)
    buffer = compute_cash_buffer(state, player, params)
    max_cash = max(0, player.money - buffer)
    max_bid = min(player.money, int(max_cash * params.max_bid_fraction), int(target_max))
    min_bid = current_price + min(increments) if increments else current_price + min_increment
    if max_bid < min_bid:
        return [{"action": "pass"}]

    options = [{"action": "pass"}]
    for inc in sorted({int(value) for value in increments if int(value) > 0}):
        bid = current_price + inc
        if bid <= 0 or bid > max_bid:
            continue
        options.append({"action": "bid", "bid": int(bid)})
    return _dedupe_actions(options)


def _can_build_local(state: GameState, player_id: int, cell: Cell, houses: dict[int, int], hotels: dict[int, int]) -> bool:
    if cell.owner_id != player_id or cell.cell_type != "property":
        return False
    if cell.group is None or not _owns_group(state, player_id, cell.group):
        return False
    group_cells = _group_cells(state, cell.group)
    if any(other.mortgaged for other in group_cells):
        return False
    if cell.house_cost is None:
        return False
    levels = [
        _property_level(houses[other.index], hotels[other.index])
        for other in group_cells
    ]
    min_level = min(levels)
    if _property_level(houses[cell.index], hotels[cell.index]) != min_level:
        return False
    if _property_level(houses[cell.index], hotels[cell.index]) >= 5:
        return False
    total_houses = sum(houses.values())
    total_hotels = sum(hotels.values())
    if _property_level(houses[cell.index], hotels[cell.index]) == 4 and total_hotels >= state.rules.bank_hotels:
        return False
    if _property_level(houses[cell.index], hotels[cell.index]) < 4 and total_houses >= state.rules.bank_houses:
        return False
    return True


def _plan_to_level(state: GameState, player: Player, group: str, target: int, max_actions: int = 8) -> list[dict[str, Any]]:
    houses = {cell.index: cell.houses for cell in state.board}
    hotels = {cell.index: cell.hotels for cell in state.board}
    plan: list[dict[str, Any]] = []
    available_cash = player.money

    group_cells = [cell for cell in state.board if cell.group == group and cell.owner_id == player.player_id]
    while len(plan) < max_actions:
        candidates: list[Cell] = []
        for cell in group_cells:
            level = _property_level(houses[cell.index], hotels[cell.index])
            if level >= target:
                continue
            if not _can_build_local(state, player.player_id, cell, houses, hotels):
                continue
            candidates.append(cell)
        if not candidates:
            break
        candidates.sort(key=lambda c: (_property_level(houses[c.index], hotels[c.index]), c.index))
        cell = candidates[0]
        cost = int(cell.house_cost or 0)
        if available_cash - cost < 0:
            break
        plan.append({"action": "build", "cell_index": cell.index})
        available_cash -= cost
        if _property_level(houses[cell.index], hotels[cell.index]) < 4:
            houses[cell.index] += 1
        else:
            hotels[cell.index] = 1
            houses[cell.index] = 0
    return plan


def _build_candidates(state: GameState, player: Player, params: BotParams) -> list[dict[str, Any]]:
    base_actions = decide_build_actions(state, player, params)
    candidates: list[list[dict[str, Any]]] = []
    candidates.append([])
    if base_actions:
        candidates.append(base_actions)
        for cut in (1, 2, 3):
            if len(base_actions) >= cut:
                candidates.append(base_actions[:cut])

    groups = [
        cell.group
        for cell in state.board
        if cell.cell_type == "property" and cell.group and cell.owner_id == player.player_id
    ]
    unique_groups = list(dict.fromkeys(groups))
    unique_groups = [g for g in unique_groups if _owns_group(state, player.player_id, g)]

    def _group_score(group: str) -> float:
        group_cells = _group_cells(state, group)
        if not group_cells:
            return 0.0
        return estimate_asset_value(state, player, group_cells[0], params)

    scored_groups = sorted(unique_groups, key=_group_score, reverse=True)
    for group in scored_groups[:2]:
        for target in (1, 2, 3):
            plan = _plan_to_level(state, player, group, target)
            if plan:
                candidates.append(plan)

    actions = [
        {"actions": plan}
        for plan in candidates
    ]
    return _dedupe_actions(actions)


def _jail_candidates(state: GameState, player: Player) -> list[dict[str, Any]]:
    actions = [{"action": "roll"}]
    if player.money >= state.rules.jail_fine:
        actions.append({"action": "pay"})
    if player.get_out_of_jail_cards:
        actions.append({"action": "use_card"})
    return actions


def _mortgage_candidates(
    state: GameState,
    player: Player,
    params: BotParams,
    cells: list[Cell],
) -> list[dict[str, Any]]:
    if not cells:
        return []

    def asset_value(cell: Cell) -> float:
        return estimate_asset_value(state, player, cell, params)

    def breaks_monopoly(cell: Cell) -> int:
        if cell.cell_type != "property":
            return 0
        if cell.group and _owns_group(state, player.player_id, cell.group):
            return 1
        return 0

    orderings = []
    orderings.append(sorted(cells, key=lambda c: (asset_value(c), c.index)))
    orderings.append(sorted(cells, key=lambda c: (-asset_value(c), c.index)))
    orderings.append(sorted(cells, key=lambda c: (breaks_monopoly(c), asset_value(c), c.index)))
    orderings.append(sorted(cells, key=lambda c: (c.mortgage_value or 0, c.index)))

    actions = []
    for ordering in orderings:
        order = [cell.index for cell in ordering]
        if not order:
            continue
        actions.append(
            {
                "action_type": "mortgage_order",
                "order": order,
                "apply_count": min(2, len(order)),
            }
        )
    return _dedupe_actions(actions)


def choose_action(
    state: GameState,
    context: dict[str, Any],
    params: BotParams,
    config: ThinkingConfig,
    cache: dict[str, float] | None = None,
    workers: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], ThinkingStats]:
    decision_type = str(context.get("type"))
    player_id = int(context.get("player_id", 0))
    player = state.players[player_id]

    if decision_type == "auction_bid":
        cell = context["cell"]
        candidates = _auction_candidates(
            state,
            player,
            cell,
            params,
            int(context["current_price"]),
            int(context["min_increment"]),
        )
    elif decision_type == "economy_phase":
        candidates = _build_candidates(state, player, params)
    elif decision_type == "jail_decision":
        candidates = _jail_candidates(state, player)
    elif decision_type == "mortgage":
        candidates = _mortgage_candidates(state, player, params, context.get("cells", []))
    elif decision_type == "liquidation":
        debt = int(context.get("debt", 0))
        base_actions = decide_liquidation(state, player, debt, params)
        candidates = [{"action_type": "liquidation", "actions": []}]
        if base_actions:
            candidates.append({"action_type": "liquidation", "actions": base_actions})
            if len(base_actions) > 1:
                candidates.append(
                    {"action_type": "liquidation", "actions": base_actions[: max(1, len(base_actions) // 2)]}
                )
    else:
        return fast_decide(state, context, params), ThinkingStats(decision_type, 0.0, 1, 0, 0.0)

    candidates = _dedupe_actions(candidates)
    if not candidates:
        return fast_decide(state, context, params), ThinkingStats(decision_type, 0.0, 0, 0, 0.0)

    use_time_budget = config.time_budget_ms > 0
    rollouts_per_action = max(1, int(config.rollouts_per_action))
    horizon_turns = max(1, int(config.horizon_turns))
    workers = workers if workers is not None else max(1, config.workers)

    start = time.perf_counter()
    best_score = float("-inf")
    best_action = candidates[0]
    total_rollouts = 0

    signature = _state_signature(state, decision_type, player_id)
    cache = cache if cache is not None else {}

    def eval_candidate(action: dict[str, Any]) -> float:
        nonlocal total_rollouts
        action_sig = _action_signature(action)
        cache_key = f"{signature}|{action_sig}|{horizon_turns}|{rollouts_per_action}"
        if not use_time_budget and cache_key in cache:
            return cache[cache_key]

        scores = []
        if not use_time_budget:
            tasks = []
            for idx in range(rollouts_per_action):
                seed = state.seed + _stable_hash(action_sig) % 100000 + idx * 31 + player_id * 17
                tasks.append((state, params, player_id, context, action, seed, horizon_turns))
            if workers > 1:
                try:
                    from multiprocessing import get_context

                    ctx = get_context("spawn")
                    with ctx.Pool(processes=workers) as pool:
                        scores = pool.map(_rollout_task, tasks)
                except (OSError, RuntimeError):
                    scores = [_rollout_task(task) for task in tasks]
            else:
                scores = [_rollout_task(task) for task in tasks]
            total_rollouts += len(scores)
            if progress_cb:
                progress_cb(total_rollouts, len(candidates))

        avg_score = sum(scores) / max(1, len(scores))
        if not use_time_budget:
            cache[cache_key] = avg_score
        return avg_score

    if use_time_budget:
        scores_by_action: dict[str, list[float]] = {}
        actions_by_sig: dict[str, dict[str, Any]] = {}
        tasks: list[tuple[dict[str, Any], int]] = []
        for idx in range(rollouts_per_action):
            for action in candidates:
                tasks.append((action, idx))

        for action, idx in tasks:
            if (time.perf_counter() - start) * 1000 >= config.time_budget_ms:
                break
            action_sig = _action_signature(action)
            actions_by_sig[action_sig] = action
            seed = state.seed + _stable_hash(action_sig) % 100000 + idx * 31 + player_id * 17
            score = _rollout_task((state, params, player_id, context, action, seed, horizon_turns))
            scores_by_action.setdefault(action_sig, []).append(score)
            total_rollouts += 1
            if progress_cb:
                progress_cb(total_rollouts, len(candidates))

        for action in candidates:
            action_sig = _action_signature(action)
            actions_by_sig[action_sig] = action
            if scores_by_action.get(action_sig):
                continue
            seed = state.seed + _stable_hash(action_sig) % 100000 + player_id * 17
            score = _rollout_task((state, params, player_id, context, action, seed, horizon_turns))
            scores_by_action.setdefault(action_sig, []).append(score)
            total_rollouts += 1

        for action_sig, scores in scores_by_action.items():
            avg_score = sum(scores) / max(1, len(scores))
            if avg_score > best_score:
                best_score = avg_score
                best_action = actions_by_sig[action_sig]
    else:
        for action in candidates:
            score = eval_candidate(action)
            if score > best_score:
                best_score = score
                best_action = action

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    stats = ThinkingStats(
        decision_type=decision_type,
        ms=elapsed_ms,
        candidates=len(candidates),
        rollouts=total_rollouts,
        best_score=best_score,
    )
    return best_action, stats

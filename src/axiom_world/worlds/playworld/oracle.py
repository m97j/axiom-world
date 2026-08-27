"""Oracle — optimal action-sequence solver (BFS over the transition engine).

Completes the neuro-symbolic loop: the simulator (engine) defines legality,
the oracle computes ground-truth optimal solutions, verifiers grade model
outputs. SFT targets are therefore oracle-derived, never LLM-derived
(tech report: "no LLM-authored ground truth").

BFS is exact for the small symbolic worlds the generator emits (<= ~10
locations, small inventories). State deduplication uses the canonical JSON
fingerprint of WorldState.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from axiom_world.core.fingerprints import canonical_json
from axiom_world.worlds.playworld.engine import TransitionEngine
from axiom_world.worlds.playworld.spec import Action, Scenario, WorldState


@dataclass(frozen=True)
class OracleSolution:
    solvable: bool
    actions: list[Action]
    final_state: WorldState | None
    states_expanded: int


def _state_key(state: WorldState) -> str:
    payload = state.model_dump()
    payload.pop("turn")  # turn is bookkeeping, not world identity
    return canonical_json(payload)


def solve(scenario: Scenario, max_states: int = 200_000) -> OracleSolution:
    """Shortest legal action sequence reaching the goal within step_limit."""
    engine = TransitionEngine(scenario.spec)
    initial = scenario.initial_state
    if scenario.goal.satisfied(initial):
        return OracleSolution(True, [], initial, 0)

    queue: deque[tuple[WorldState, list[Action]]] = deque([(initial, [])])
    seen: set[str] = {_state_key(initial)}
    expanded = 0

    while queue and expanded < max_states:
        state, path = queue.popleft()
        expanded += 1
        if len(path) >= scenario.step_limit:
            continue
        for action in engine.legal_actions(state):
            result = engine.step(state, action)
            assert result.next_state is not None
            next_state = result.next_state
            key = _state_key(next_state)
            if key in seen:
                continue
            seen.add(key)
            next_path = path + [action]
            if scenario.goal.satisfied(next_state):
                return OracleSolution(True, next_path, next_state, expanded)
            queue.append((next_state, next_path))

    return OracleSolution(False, [], None, expanded)

"""Deterministic transition engine — the sole authority on legality.

Every verifier and every reward derives from this engine. It never guesses:
an action is either legal (returns the unique next state) or illegal
(returns a reason code). Reason codes are stable strings used by the
failure taxonomy.
"""
from __future__ import annotations

from dataclasses import dataclass

from axiom_world.playworld.spec import Action, Goal, Scenario, WorldSpec, WorldState

# Stable reason codes (failure taxonomy keys).
ILLEGAL_UNKNOWN_ACTION = "illegal_unknown_action"
ILLEGAL_BAD_ARGUMENTS = "illegal_bad_arguments"
ILLEGAL_NOT_ADJACENT = "illegal_not_adjacent"
ILLEGAL_INSUFFICIENT_ENERGY = "illegal_insufficient_energy"
ILLEGAL_RESOURCE_ABSENT = "illegal_resource_absent"
ILLEGAL_INVENTORY_FULL = "illegal_inventory_full"
ILLEGAL_NOT_AT_DEPOSIT = "illegal_not_at_deposit"
ILLEGAL_NOTHING_TO_DEPOSIT = "illegal_nothing_to_deposit"
LEGAL = "legal"


@dataclass(frozen=True)
class TransitionResult:
    legal: bool
    reason_code: str
    next_state: WorldState | None


class TransitionEngine:
    def __init__(self, spec: WorldSpec) -> None:
        self.spec = spec

    # -- legality -------------------------------------------------------------

    def step(self, state: WorldState, action: Action) -> TransitionResult:
        handler = getattr(self, f"_step_{action.type.lower()}", None)
        if handler is None:
            return TransitionResult(False, ILLEGAL_UNKNOWN_ACTION, None)
        return handler(state, action)

    def legal_actions(self, state: WorldState) -> list[Action]:
        """Enumerate the full legal action set (used for eval statistics)."""
        candidates: list[Action] = [Action(type="REST"), Action(type="WAIT")]
        candidates += [Action(type="MOVE", target=loc) for loc in self.spec.neighbors(state.location)]
        candidates += [
            Action(type="COLLECT", resource=rule.resource_id)
            for rule in self.spec.resources
            if rule.location_id == state.location
        ]
        candidates += [Action(type="DEPOSIT", resource=r) for r in state.inventory]
        return [a for a in candidates if self.step(state, a).legal]

    # -- handlers -------------------------------------------------------------

    def _bump(self, state: WorldState, **updates: object) -> WorldState:
        payload = state.model_dump()
        payload.update(updates)
        payload["turn"] = state.turn + 1
        return WorldState.model_validate(payload)

    def _step_move(self, state: WorldState, action: Action) -> TransitionResult:
        if not action.target:
            return TransitionResult(False, ILLEGAL_BAD_ARGUMENTS, None)
        if action.target not in self.spec.neighbors(state.location):
            return TransitionResult(False, ILLEGAL_NOT_ADJACENT, None)
        if state.energy < self.spec.move_cost:
            return TransitionResult(False, ILLEGAL_INSUFFICIENT_ENERGY, None)
        next_state = self._bump(
            state, location=action.target, energy=state.energy - self.spec.move_cost
        )
        return TransitionResult(True, LEGAL, next_state)

    def _step_collect(self, state: WorldState, action: Action) -> TransitionResult:
        if not action.resource:
            return TransitionResult(False, ILLEGAL_BAD_ARGUMENTS, None)
        rule = next(
            (
                r
                for r in self.spec.resources
                if r.resource_id == action.resource and r.location_id == state.location
            ),
            None,
        )
        if rule is None:
            return TransitionResult(False, ILLEGAL_RESOURCE_ABSENT, None)
        if state.inventory_size() >= self.spec.inventory_capacity:
            return TransitionResult(False, ILLEGAL_INVENTORY_FULL, None)
        if state.energy < rule.collect_cost:
            return TransitionResult(False, ILLEGAL_INSUFFICIENT_ENERGY, None)
        inventory = dict(state.inventory)
        inventory[action.resource] = inventory.get(action.resource, 0) + 1
        next_state = self._bump(
            state, inventory=inventory, energy=state.energy - rule.collect_cost
        )
        return TransitionResult(True, LEGAL, next_state)

    def _step_rest(self, state: WorldState, action: Action) -> TransitionResult:
        energy = min(state.energy + self.spec.rest_gain, self.spec.max_energy)
        return TransitionResult(True, LEGAL, self._bump(state, energy=energy))

    def _step_deposit(self, state: WorldState, action: Action) -> TransitionResult:
        if not action.resource:
            return TransitionResult(False, ILLEGAL_BAD_ARGUMENTS, None)
        if self.spec.deposit_location is None or state.location != self.spec.deposit_location:
            return TransitionResult(False, ILLEGAL_NOT_AT_DEPOSIT, None)
        if state.inventory.get(action.resource, 0) <= 0:
            return TransitionResult(False, ILLEGAL_NOTHING_TO_DEPOSIT, None)
        inventory = dict(state.inventory)
        inventory[action.resource] -= 1
        if inventory[action.resource] == 0:
            del inventory[action.resource]
        deposited = dict(state.deposited)
        deposited[action.resource] = deposited.get(action.resource, 0) + 1
        return TransitionResult(
            True, LEGAL, self._bump(state, inventory=inventory, deposited=deposited)
        )

    def _step_wait(self, state: WorldState, action: Action) -> TransitionResult:
        return TransitionResult(True, LEGAL, self._bump(state))

    # -- episode replay ---------------------------------------------------------

    def replay(
        self, scenario: Scenario, actions: list[Action]
    ) -> tuple[bool, str, WorldState, list[dict[str, object]]]:
        """Replay an action sequence. Returns (goal_reached, reason, final, trace)."""
        state = scenario.initial_state
        trace: list[dict[str, object]] = []
        goal: Goal = scenario.goal
        for index, action in enumerate(actions):
            if index >= scenario.step_limit:
                return False, "step_limit_exceeded", state, trace
            result = self.step(state, action)
            trace.append(
                {
                    "index": index,
                    "action": action.model_dump(),
                    "legal": result.legal,
                    "reason_code": result.reason_code,
                }
            )
            if not result.legal:
                return False, result.reason_code, state, trace
            assert result.next_state is not None
            state = result.next_state
            if goal.satisfied(state):
                return True, "goal_reached", state, trace
        return goal.satisfied(state), "episode_ended", state, trace

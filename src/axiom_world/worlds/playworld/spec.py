"""PlayWorld symbolic world specification (protocol §4.3).

A PlayWorld task is fully symbolic and deterministic:

- ``WorldSpec``  : rule primitives — locations, resources, action costs,
                   capacity limits. A *scenario family* is identified by the
                   composition of rule primitives it uses (family_id).
- ``WorldState`` : agent location, energy, inventory, turn counter.
- ``Action``     : typed action with arguments; grammar is closed.
- ``Goal``       : predicate over the final state.
- ``Scenario``   : (spec, initial state, goal, step limit) + family metadata.

Determinism guarantee: given (spec, state, action) the transition result is
unique. This is what makes verifier rewards exact rather than judged.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ActionType = Literal["MOVE", "COLLECT", "REST", "DEPOSIT", "WAIT"]

ACTION_GRAMMAR: dict[str, dict[str, str]] = {
    "MOVE": {"target": "location_id"},
    "COLLECT": {"resource": "resource_id"},
    "REST": {},
    "DEPOSIT": {"resource": "resource_id"},
    "WAIT": {},
}


class Action(StrictModel):
    type: ActionType
    target: str | None = None
    resource: str | None = None


class ResourceRule(StrictModel):
    resource_id: str
    location_id: str
    collect_cost: int = 1
    quantity: int = 1


class WorldSpec(StrictModel):
    spec_id: str
    family_id: str = Field(description="Scenario family = ruleset x template x goal type (§4.3).")
    rule_primitives: list[str] = Field(
        description="Named rule primitives composed in this spec; drives OOD splits."
    )
    locations: list[str]
    edges: list[tuple[str, str]] = Field(description="Undirected passable edges.")
    resources: list[ResourceRule] = Field(default_factory=list)
    move_cost: int = 1
    rest_gain: int = 2
    max_energy: int = 10
    inventory_capacity: int = 3
    deposit_location: str | None = None

    def neighbors(self, location: str) -> set[str]:
        result: set[str] = set()
        for a, b in self.edges:
            if a == location:
                result.add(b)
            elif b == location:
                result.add(a)
        return result


class WorldState(StrictModel):
    location: str
    energy: int
    inventory: dict[str, int] = Field(default_factory=dict)
    deposited: dict[str, int] = Field(default_factory=dict)
    turn: int = 0

    def inventory_size(self) -> int:
        return sum(self.inventory.values())


class Goal(StrictModel):
    kind: Literal["reach_location", "deposit_resources", "collect_resources"]
    location: str | None = None
    resources: dict[str, int] = Field(default_factory=dict)

    def satisfied(self, state: WorldState) -> bool:
        if self.kind == "reach_location":
            return state.location == self.location
        if self.kind == "deposit_resources":
            return all(state.deposited.get(r, 0) >= n for r, n in self.resources.items())
        return all(state.inventory.get(r, 0) >= n for r, n in self.resources.items())


class Scenario(StrictModel):
    scenario_id: str
    spec: WorldSpec
    initial_state: WorldState
    goal: Goal
    step_limit: int = 12
    split_hint: str | None = Field(
        default=None, description="Filled by the split assigner; informational only."
    )


class Episode(StrictModel):
    """A recorded interaction: scenario + the model's action sequence."""

    scenario_id: str
    actions: list[Action]
    final_state: WorldState | None = None
    goal_reached: bool | None = None

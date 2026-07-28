"""Deterministic scenario generation and family-level splits (protocol §4.3).

- Scenarios are generated from a seeded RNG; identical (seed, params) yields
  byte-identical scenarios — the eval-suite freeze (Gate G3) hashes them.
- Splits are assigned at the FAMILY level. ``split_families`` guarantees that
  no family_id appears in more than one split, which is the leakage-resistance
  property the protocol pre-registers.
"""
from __future__ import annotations

import random

from axiom_world.core.errors import AxiomError
from axiom_world.playworld.spec import (
    Goal,
    ResourceRule,
    Scenario,
    WorldSpec,
    WorldState,
)

RULE_PRIMITIVES = ("movement", "energy_budget", "collection", "capacity", "deposit")


class ScenarioGenerator:
    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self.seed = seed

    def _grid_world(self, spec_id: str, family_id: str, primitives: list[str], size: int) -> WorldSpec:
        locations = [f"loc_{i}" for i in range(size)]
        edges = [(locations[i], locations[i + 1]) for i in range(size - 1)]
        # a few random shortcuts for branching factor
        for _ in range(max(1, size // 3)):
            a, b = self._rng.sample(locations, 2)
            if (a, b) not in edges and (b, a) not in edges:
                edges.append((a, b))
        resources = []
        if "collection" in primitives:
            for i in range(2):
                resources.append(
                    ResourceRule(
                        resource_id=f"res_{i}",
                        location_id=self._rng.choice(locations),
                        collect_cost=self._rng.randint(1, 2),
                    )
                )
        return WorldSpec(
            spec_id=spec_id,
            family_id=family_id,
            rule_primitives=primitives,
            locations=locations,
            edges=edges,
            resources=resources,
            deposit_location=locations[0] if "deposit" in primitives else None,
            inventory_capacity=2 if "capacity" in primitives else 8,
        )

    def generate(self, family_id: str, primitives: list[str], count: int, size: int = 6) -> list[Scenario]:
        unknown = set(primitives) - set(RULE_PRIMITIVES)
        if unknown:
            raise AxiomError(f"Unknown rule primitives: {sorted(unknown)}")
        scenarios: list[Scenario] = []
        for index in range(count):
            spec = self._grid_world(f"{family_id}-spec{index}", family_id, primitives, size)
            start = self._rng.choice(spec.locations)
            state = WorldState(location=start, energy=self._rng.randint(5, spec.max_energy))
            goal = self._make_goal(spec, primitives, start)
            scenarios.append(
                Scenario(
                    scenario_id=f"{family_id}-{index:04d}",
                    spec=spec,
                    initial_state=state,
                    goal=goal,
                    step_limit=4 * size,
                )
            )
        return scenarios

    def _make_goal(self, spec: WorldSpec, primitives: list[str], start: str) -> Goal:
        if "deposit" in primitives and spec.resources:
            resource = self._rng.choice(spec.resources).resource_id
            return Goal(kind="deposit_resources", resources={resource: 1})
        if "collection" in primitives and spec.resources:
            resource = self._rng.choice(spec.resources).resource_id
            return Goal(kind="collect_resources", resources={resource: 1})
        target = self._rng.choice([loc for loc in spec.locations if loc != start])
        return Goal(kind="reach_location", location=target)


def split_families(
    family_ids: list[str],
    train_fraction: float = 0.7,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Disjoint family-level split. Returns {'train': [...], 'held_out': [...]}."""
    if len(set(family_ids)) != len(family_ids):
        raise AxiomError("family_ids must be unique.")
    rng = random.Random(seed)
    shuffled = list(family_ids)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * train_fraction))
    if cut >= len(shuffled):
        cut = len(shuffled) - 1
    return {"train": sorted(shuffled[:cut]), "held_out": sorted(shuffled[cut:])}

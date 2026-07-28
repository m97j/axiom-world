from axiom_world.playworld.engine import (
    ILLEGAL_INSUFFICIENT_ENERGY,
    ILLEGAL_INVENTORY_FULL,
    ILLEGAL_NOT_ADJACENT,
    TransitionEngine,
)
from axiom_world.playworld.scenario import ScenarioGenerator, split_families
from axiom_world.playworld.spec import Action, Goal, ResourceRule, Scenario, WorldSpec, WorldState


def _spec(**overrides) -> WorldSpec:
    payload = {
        "spec_id": "s0",
        "family_id": "fam-a",
        "rule_primitives": ["movement", "collection", "capacity", "deposit"],
        "locations": ["loc_0", "loc_1", "loc_2"],
        "edges": [("loc_0", "loc_1"), ("loc_1", "loc_2")],
        "resources": [ResourceRule(resource_id="res_0", location_id="loc_1")],
        "deposit_location": "loc_0",
        "inventory_capacity": 1,
    }
    payload.update(overrides)
    return WorldSpec.model_validate(payload)


def test_move_legal_and_deterministic() -> None:
    engine = TransitionEngine(_spec())
    state = WorldState(location="loc_0", energy=5)
    result = engine.step(state, Action(type="MOVE", target="loc_1"))
    assert result.legal and result.next_state.location == "loc_1"
    assert result.next_state.energy == 4 and result.next_state.turn == 1


def test_move_not_adjacent_rejected() -> None:
    engine = TransitionEngine(_spec())
    state = WorldState(location="loc_0", energy=5)
    result = engine.step(state, Action(type="MOVE", target="loc_2"))
    assert not result.legal and result.reason_code == ILLEGAL_NOT_ADJACENT


def test_energy_gate() -> None:
    engine = TransitionEngine(_spec())
    state = WorldState(location="loc_0", energy=0)
    result = engine.step(state, Action(type="MOVE", target="loc_1"))
    assert result.reason_code == ILLEGAL_INSUFFICIENT_ENERGY


def test_capacity_gate() -> None:
    engine = TransitionEngine(_spec())
    state = WorldState(location="loc_1", energy=5, inventory={"res_0": 1})
    result = engine.step(state, Action(type="COLLECT", resource="res_0"))
    assert result.reason_code == ILLEGAL_INVENTORY_FULL


def test_full_episode_deposit_goal() -> None:
    spec = _spec()
    scenario = Scenario(
        scenario_id="ep-1",
        spec=spec,
        initial_state=WorldState(location="loc_0", energy=6),
        goal=Goal(kind="deposit_resources", resources={"res_0": 1}),
        step_limit=10,
    )
    engine = TransitionEngine(spec)
    actions = [
        Action(type="MOVE", target="loc_1"),
        Action(type="COLLECT", resource="res_0"),
        Action(type="MOVE", target="loc_0"),
        Action(type="DEPOSIT", resource="res_0"),
    ]
    reached, reason, final, trace = engine.replay(scenario, actions)
    assert reached and reason == "goal_reached"
    assert final.deposited == {"res_0": 1} and len(trace) == 4


def test_generator_determinism() -> None:
    a = ScenarioGenerator(seed=7).generate("fam-x", ["movement"], count=3)
    b = ScenarioGenerator(seed=7).generate("fam-x", ["movement"], count=3)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_family_split_disjoint() -> None:
    families = [f"fam-{i}" for i in range(10)]
    splits = split_families(families, train_fraction=0.7, seed=1)
    assert set(splits["train"]).isdisjoint(splits["held_out"])
    assert sorted(splits["train"] + splits["held_out"]) == sorted(families)


def test_legal_actions_enumeration() -> None:
    engine = TransitionEngine(_spec())
    state = WorldState(location="loc_1", energy=5)
    actions = engine.legal_actions(state)
    types = {(a.type, a.target or a.resource) for a in actions}
    assert ("MOVE", "loc_0") in types and ("COLLECT", "res_0") in types
    assert all(engine.step(state, a).legal for a in actions)

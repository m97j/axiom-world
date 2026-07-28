import json

from axiom_world.playworld.oracle import solve
from axiom_world.playworld.scenario import ScenarioGenerator
from axiom_world.verifiers.hybrid import default_playworld_verifier
from tests.unit.test_verifiers import _scenario


def test_oracle_solves_deposit_scenario() -> None:
    scenario = _scenario()
    solution = solve(scenario)
    assert solution.solvable
    assert solution.final_state is not None
    assert scenario.goal.satisfied(solution.final_state)


def test_oracle_solution_passes_hybrid_verifier() -> None:
    """The neuro-symbolic loop closes: oracle output == verifier PASSED."""
    scenario = _scenario()
    solution = solve(scenario)
    prediction = json.dumps(
        {
            "actions": [a.model_dump(exclude_none=True) for a in solution.actions],
            "final_state": {
                "location": solution.final_state.location,
                "energy": solution.final_state.energy,
            },
        }
    )
    verdict = default_playworld_verifier().verify(prediction, {"scenario": scenario})
    assert verdict.status.value == "passed" and verdict.score == 1.0


def test_oracle_is_shortest() -> None:
    scenario = _scenario()
    solution = solve(scenario)
    # deposit goal requires at least: move, collect, move back, deposit
    assert len(solution.actions) == 4


def test_generated_scenarios_mostly_solvable() -> None:
    scenarios = ScenarioGenerator(seed=3).generate(
        "fam-orc", ["movement", "collection", "deposit"], count=10
    )
    solved = sum(solve(s).solvable for s in scenarios)
    assert solved >= 8  # generator may rarely emit unreachable goals; those are dropped

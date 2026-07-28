import json

from axiom_world.core.enums import VerificationStatus
from axiom_world.playworld.spec import Goal, ResourceRule, Scenario, WorldSpec, WorldState
from axiom_world.verifiers.base import Verdict, Verifier
from axiom_world.verifiers.hybrid import default_playworld_verifier
from axiom_world.verifiers.rule import GoalVerifier, LegalityVerifier, SchemaVerifier


def _scenario() -> Scenario:
    spec = WorldSpec(
        spec_id="s0",
        family_id="fam-a",
        rule_primitives=["movement", "collection", "deposit"],
        locations=["loc_0", "loc_1"],
        edges=[("loc_0", "loc_1")],
        resources=[ResourceRule(resource_id="res_0", location_id="loc_1")],
        deposit_location="loc_0",
    )
    return Scenario(
        scenario_id="sc-1",
        spec=spec,
        initial_state=WorldState(location="loc_0", energy=6),
        goal=Goal(kind="deposit_resources", resources={"res_0": 1}),
        step_limit=8,
    )


GOOD = json.dumps(
    {
        "actions": [
            {"type": "MOVE", "target": "loc_1"},
            {"type": "COLLECT", "resource": "res_0"},
            {"type": "MOVE", "target": "loc_0"},
            {"type": "DEPOSIT", "resource": "res_0"},
        ],
        "final_state": {"location": "loc_0", "energy": 3},
    }
)
ILLEGAL = json.dumps({"actions": [{"type": "COLLECT", "resource": "res_0"}]})
MALFORMED = "I will move to loc_1 and collect."


def test_schema_verifier() -> None:
    ctx = {"scenario": _scenario()}
    assert SchemaVerifier().verify(GOOD, ctx).status is VerificationStatus.PASSED
    assert SchemaVerifier().verify(MALFORMED, ctx).status is VerificationStatus.FAILED


def test_legality_verifier_reason_codes() -> None:
    ctx = {"scenario": _scenario()}
    verdict = LegalityVerifier().verify(ILLEGAL, ctx)
    assert verdict.status is VerificationStatus.FAILED
    assert verdict.reason_code == "illegal_resource_absent"
    assert verdict.evidence["legal_action_rate"] == 0.0


def test_goal_verifier_pass_and_fail() -> None:
    ctx = {"scenario": _scenario()}
    assert GoalVerifier().verify(GOOD, ctx).status is VerificationStatus.PASSED
    short = json.dumps({"actions": [{"type": "WAIT"}]})
    verdict = GoalVerifier().verify(short, ctx)
    assert verdict.status is VerificationStatus.FAILED


def test_hybrid_gate_blocks_malformed() -> None:
    ctx = {"scenario": _scenario()}
    verdict = default_playworld_verifier().verify(MALFORMED, ctx)
    assert verdict.status is VerificationStatus.FAILED
    assert verdict.reason_code.startswith("gate_failed:")
    assert verdict.score == 0.0


def test_hybrid_pass_on_good_episode() -> None:
    ctx = {"scenario": _scenario()}
    verdict = default_playworld_verifier().verify(GOOD, ctx)
    assert verdict.status is VerificationStatus.PASSED
    assert verdict.score == 1.0
    assert set(verdict.evidence["components"]) == {
        "schema", "legality", "goal", "state_consistency",
    }


def test_state_contradiction_detected() -> None:
    ctx = {"scenario": _scenario()}
    lying = json.loads(GOOD)
    lying["final_state"]["energy"] = 99
    verdict = default_playworld_verifier().verify(json.dumps(lying), ctx)
    assert verdict.status is VerificationStatus.FAILED
    assert verdict.reason_code == "required_component_failed" or verdict.score < 1.0


def test_infra_error_never_counts_as_model_failure() -> None:
    class Exploding(Verifier):
        name = "exploding"

        def _verify(self, prediction: str, context: dict) -> Verdict:
            raise RuntimeError("boom")

    verdict = Exploding().verify("x", {})
    assert verdict.status is VerificationStatus.INFRA_ERROR
    assert not verdict.reward_eligible

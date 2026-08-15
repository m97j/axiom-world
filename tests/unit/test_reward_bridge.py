from collections import Counter

from axiom_world.training.reward_bridge import verifier_reward_function
from axiom_world.verifiers.hybrid import default_playworld_verifier
from tests.unit.test_verifiers import GOOD, ILLEGAL, MALFORMED, _scenario


def test_reward_semantics() -> None:
    counter = Counter()
    reward_fn = verifier_reward_function(default_playworld_verifier(), counter)
    scenario_dump = _scenario().model_dump(mode="json")
    rewards = reward_fn(
        prompts=["p"] * 3,
        completions=[GOOD, ILLEGAL, MALFORMED],
        scenario=[scenario_dump] * 3,
    )
    assert rewards[0] == 1.0          # passed -> aggregate score
    assert rewards[1] == 0.0          # failed -> 0.0
    assert rewards[2] == 0.0          # malformed fails at the schema gate -> failed
    assert counter["passed"] == 1 and counter["failed"] == 2


def test_conversational_completion_extraction() -> None:
    reward_fn = verifier_reward_function(default_playworld_verifier())
    scenario_dump = _scenario().model_dump(mode="json")
    rewards = reward_fn(
        prompts=["p"],
        completions=[[{"role": "assistant", "content": GOOD}]],
        scenario=[scenario_dump],
    )
    assert rewards == [1.0]


# --- v0.6.12 regression tests: B6 None-contamination incident (2026-08-15) ---

def test_scenario_json_string_transport() -> None:
    """Fixed path: 'scenario_json' string column decodes and scores identically."""
    import json

    reward_fn = verifier_reward_function(default_playworld_verifier())
    scenario_dump = _scenario().model_dump(mode="json")
    rewards = reward_fn(
        prompts=["p"] * 2,
        completions=[GOOD, ILLEGAL],
        scenario_json=[json.dumps(scenario_dump, sort_keys=True)] * 2,
    )
    assert rewards == [1.0, 0.0]


def test_to_grpo_rows_serializes_scenario_as_string() -> None:
    import json

    from axiom_world.data.bundle import (
        DataBundle,
        build_data_bundle,  # noqa: F401 - import guard
    )
    from axiom_world.data.records import EvaluationRecord, Message, Provenance
    from axiom_world.training.adapter import to_grpo_rows

    scenario_dump = _scenario().model_dump(mode="json")
    record = EvaluationRecord(
        id="prompt-x", suite="eval_id",
        prompt=[Message(role="user", content="go")],
        scenario=scenario_dump, scenario_family_id="train-x",
        provenance=Provenance(source_type="synthetic", source_id="test"),
    )
    bundle = DataBundle(kind="evaluation", records=[record],
                        fingerprint="sha256:0", manifest={})
    rows = to_grpo_rows(bundle)
    assert set(rows[0]) == {"prompt", "scenario_json"}
    assert isinstance(rows[0]["scenario_json"], str)
    assert json.loads(rows[0]["scenario_json"]) == json.loads(
        json.dumps(scenario_dump))


def test_reward_health_guard_trips_on_degenerate_stream() -> None:
    import pytest

    from axiom_world.core.errors import RewardHealthError

    reward_fn = verifier_reward_function(
        default_playworld_verifier(), min_calls=4, max_excluded_fraction=0.5,
    )
    # scenario payload that fails Scenario validation -> INFRA_ERROR -> None
    with pytest.raises(RewardHealthError):
        reward_fn(
            prompts=["p"] * 4,
            completions=[GOOD] * 4,
            scenario=[{"broken": None}] * 4,
        )


def test_reward_health_guard_silent_below_min_calls() -> None:
    reward_fn = verifier_reward_function(
        default_playworld_verifier(), min_calls=100, max_excluded_fraction=0.5,
    )
    rewards = reward_fn(prompts=["p"], completions=[GOOD], scenario=[{"broken": None}])
    assert rewards == [None]

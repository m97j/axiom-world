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

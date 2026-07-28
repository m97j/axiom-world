from axiom_world.data.records import Message
from axiom_world.generation.backend import GenerationRequest
from axiom_world.generation.mock_backend import DeterministicMockBackend
from axiom_world.generation.pair_mining import PairMiningPolicy, mine_preference_pairs
from axiom_world.verifiers.hybrid import default_playworld_verifier
from tests.unit.test_verifiers import GOOD, ILLEGAL, MALFORMED, _scenario


def _candidates(texts: list[str]):
    backend = DeterministicMockBackend(lambda req, i: texts[i % len(texts)])
    request = GenerationRequest(
        request_id="req-1", source_record_id="src-1", prompt="p", n=len(texts)
    )
    return {"src-1": backend.generate([request])}


PROMPTS = {"src-1": [Message(role="user", content="solve the scenario")]}


def _contexts():
    return {"src-1": {"scenario": _scenario(), "scenario_family_id": "fam-a"}}


def test_verifier_rank_mines_pair() -> None:
    records, decisions = mine_preference_pairs(
        _candidates([GOOD, ILLEGAL, MALFORMED]),
        default_playworld_verifier(),
        _contexts(),
        PROMPTS,
    )
    assert decisions["accepted"] == 1 and len(records) == 1
    record = records[0]
    assert record.chosen == GOOD
    assert record.chosen_verification.status.value == "passed"
    assert record.score_margin and record.score_margin >= 0.10
    assert record.scenario_family_id == "fam-a"


def test_no_passed_candidate_yields_no_pair() -> None:
    records, decisions = mine_preference_pairs(
        _candidates([ILLEGAL, MALFORMED]),
        default_playworld_verifier(),
        _contexts(),
        PROMPTS,
    )
    assert not records and decisions["no_passed_candidate"] == 1


def test_margin_gate() -> None:
    records, decisions = mine_preference_pairs(
        _candidates([GOOD, GOOD + " "]),  # near-identical high scorers
        default_playworld_verifier(),
        _contexts(),
        PROMPTS,
        PairMiningPolicy(minimum_margin=0.10),
    )
    assert not records
    assert decisions["insufficient_margin"] == 1 or decisions["identical_texts"] == 1


def test_random_pairing_control_equal_count() -> None:
    records, decisions = mine_preference_pairs(
        _candidates([GOOD, ILLEGAL, MALFORMED]),
        default_playworld_verifier(),
        _contexts(),
        PROMPTS,
        PairMiningPolicy(selection_method="random_pairing", seed=7),
    )
    assert decisions["accepted"] == 1
    assert records[0].selection_method == "random_pairing"


def test_determinism_across_calls() -> None:
    args = (
        _candidates([GOOD, ILLEGAL, MALFORMED]),
        default_playworld_verifier(),
        _contexts(),
        PROMPTS,
    )
    first, _ = mine_preference_pairs(*args)
    second, _ = mine_preference_pairs(*args)
    assert [r.model_dump() for r in first] == [r.model_dump() for r in second]

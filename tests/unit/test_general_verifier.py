from axiom_world.core.enums import VerificationStatus
from axiom_world.verifiers.general import ExactAnswerVerifier, extract_final_answer


def test_extracts_gsm8k_style_answer() -> None:
    assert extract_final_answer("... so the total is 42.\n#### 42") == "42"


def test_extracts_boxed() -> None:
    assert extract_final_answer(r"Thus \boxed{3.5} is the result.") == "3.5"


def test_falls_back_to_last_number() -> None:
    assert extract_final_answer("First 3 apples, then 7, total 10 apples.") == "10"


def test_numeric_match_ignores_formatting() -> None:
    verdict = ExactAnswerVerifier().verify("The answer is $1,200.", {"answer": "1200"})
    assert verdict.status is VerificationStatus.PASSED


def test_mismatch_fails() -> None:
    verdict = ExactAnswerVerifier().verify("#### 41", {"answer": "42"})
    assert verdict.status is VerificationStatus.FAILED
    assert verdict.reason_code == "answer_mismatch"


def test_missing_gold_is_skipped_not_failed() -> None:
    verdict = ExactAnswerVerifier().verify("#### 42", {})
    assert verdict.status is VerificationStatus.SKIPPED
    assert not verdict.reward_eligible

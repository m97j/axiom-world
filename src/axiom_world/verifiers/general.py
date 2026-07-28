"""Phase-1 general-reasoning verifiers (protocol §5.1 B1–B3).

Phase 1 uses PUBLIC datasets with pre-existing verified answers (math word
problems, short-form QA). Verification is deterministic normalization +
exact/numeric match against the dataset's gold answer — no LLM judge, and
no code-execution sandbox.

Design decision (report §Design Decisions): a subprocess/code sandbox is
deliberately out of scope for Phase 1. Phase 1's role in the pre-registered
protocol is a *controlled upstream intervention* for RQ1 (transfer), not a
frontier general-reasoning result. Math/QA exact-match gives verifiable
rewards with zero infrastructure risk on Colab; code-execution tasks would
add sandbox engineering without serving RQ1-RQ3.
"""
from __future__ import annotations

import re
from typing import Any

from axiom_world.core.enums import VerificationStatus
from axiom_world.verifiers.base import Verdict, Verifier

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_MARKERS = ("####", "answer:", "final answer:", "정답:")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_final_answer(text: str) -> str | None:
    """Extract the model's final answer: \\boxed{}, '####' / 'Answer:' markers,
    else the last number in the text."""
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    lowered = text.lower()
    for marker in _FINAL_MARKERS:
        if marker in lowered:
            tail = text[lowered.rindex(marker) + len(marker):]
            first_line = tail.strip().splitlines()[0] if tail.strip() else ""
            if first_line:
                return first_line.strip()
    numbers = _NUMBER.findall(text)
    return numbers[-1] if numbers else None


def _to_number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


class ExactAnswerVerifier(Verifier):
    """Gold-answer match: numeric comparison when both sides parse as numbers,
    else normalized string equality. Context contract: {'answer': <gold>}."""

    name = "exact_answer"
    version = "1.0"

    def _verify(self, prediction: str, context: dict[str, Any]) -> Verdict:
        gold = context.get("answer")
        if gold is None:
            return self._make(VerificationStatus.SKIPPED, "no_gold_answer")
        predicted = extract_final_answer(prediction)
        if predicted is None:
            return self._make(VerificationStatus.FAILED, "no_final_answer", score=0.0)
        gold_text = str(gold)
        pred_num, gold_num = _to_number(predicted), _to_number(gold_text)
        if pred_num is not None and gold_num is not None:
            correct = abs(pred_num - gold_num) < 1e-6
        else:
            correct = normalize_text(predicted) == normalize_text(gold_text)
        return self._make(
            VerificationStatus.PASSED if correct else VerificationStatus.FAILED,
            "answer_match" if correct else "answer_mismatch",
            score=1.0 if correct else 0.0,
            predicted=predicted,
            gold=gold_text,
        )

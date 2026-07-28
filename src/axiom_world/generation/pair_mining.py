"""Verifier-guided preference pair mining (protocol §5.2, RQ2).

Two pre-registered selection methods:
- hybrid_verifier_rank : chosen = best PASSED candidate, rejected = worst
  reward-eligible candidate, margin gate applied. Powers B3/B5.
- random_pairing       : the E-RANDPAIR control — random chosen/rejected among
  distinct candidates at EQUAL pair count, ignoring verifier rank (but pairs
  where both texts are identical are still skipped).

Status semantics: only reward-eligible verdicts (passed/failed) participate.
skipped/indeterminate/timeout/infra_error candidates never form pairs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from axiom_world.core.enums import VerificationStatus
from axiom_world.data.records import (
    Message,
    PreferenceRecord,
    Provenance,
    VerificationEvidence,
)
from axiom_world.generation.backend import Candidate
from axiom_world.verifiers.base import Verdict, Verifier


@dataclass(frozen=True)
class PairMiningPolicy:
    selection_method: Literal["hybrid_verifier_rank", "random_pairing"] = "hybrid_verifier_rank"
    minimum_margin: float = 0.10
    prefer_shorter_on_tie: bool = True
    seed: int = 42


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    verdict: Verdict

    @property
    def score(self) -> float:
        return self.verdict.score if self.verdict.score is not None else 0.0


def _rank_key(scored: ScoredCandidate, prefer_shorter: bool) -> tuple:
    length_term = len(scored.candidate.text) if prefer_shorter else 0
    return (-scored.score, length_term, scored.candidate.index)


def mine_preference_pairs(
    grouped_candidates: dict[str, list[Candidate]],
    verifier: Verifier,
    contexts: dict[str, dict[str, Any]],
    prompts: dict[str, list[Message]],
    policy: PairMiningPolicy = PairMiningPolicy(),
    provenance_source_id: str = "pair-mining-v1",
) -> tuple[list[PreferenceRecord], dict[str, int]]:
    """Mine pairs per source record. Returns (records, decision_counts)."""
    rng = random.Random(policy.seed)
    records: list[PreferenceRecord] = []
    decisions: dict[str, int] = {
        "accepted": 0,
        "no_passed_candidate": 0,
        "insufficient_margin": 0,
        "insufficient_candidates": 0,
        "identical_texts": 0,
    }

    for source_id in sorted(grouped_candidates):
        candidates = grouped_candidates[source_id]
        context = contexts.get(source_id, {})
        scored = [
            ScoredCandidate(candidate=c, verdict=verifier.verify(c.text, context))
            for c in candidates
        ]
        eligible = [s for s in scored if s.verdict.reward_eligible]
        if len(eligible) < 2:
            decisions["insufficient_candidates"] += 1
            continue

        if policy.selection_method == "random_pairing":
            pool = list(eligible)
            rng.shuffle(pool)
            chosen, rejected = pool[0], pool[1]
            if chosen.candidate.text.strip() == rejected.candidate.text.strip():
                decisions["identical_texts"] += 1
                continue
            # PreferenceRecord requires PASSED chosen for verifier-ranked pairs
            # only; random_pairing is exempt by schema design.
        else:
            passed = [s for s in eligible if s.verdict.status is VerificationStatus.PASSED]
            if not passed:
                decisions["no_passed_candidate"] += 1
                continue
            ranked = sorted(eligible, key=lambda s: _rank_key(s, policy.prefer_shorter_on_tie))
            chosen = min(passed, key=lambda s: _rank_key(s, policy.prefer_shorter_on_tie))
            rejected = ranked[-1]
            if chosen.candidate.candidate_id == rejected.candidate.candidate_id:
                decisions["insufficient_candidates"] += 1
                continue
            if chosen.candidate.text.strip() == rejected.candidate.text.strip():
                decisions["identical_texts"] += 1
                continue
            margin = chosen.score - rejected.score
            if margin < policy.minimum_margin:
                decisions["insufficient_margin"] += 1
                continue

        margin_value = round(chosen.score - rejected.score, 6)
        records.append(
            PreferenceRecord(
                id=f"pair-{source_id}",
                prompt=prompts[source_id],
                chosen=chosen.candidate.text,
                rejected=rejected.candidate.text,
                chosen_verification=VerificationEvidence(
                    status=chosen.verdict.status,
                    score=chosen.verdict.score,
                    verifier_version=chosen.verdict.verifier_version,
                ),
                rejected_verification=VerificationEvidence(
                    status=rejected.verdict.status,
                    score=rejected.verdict.score,
                    verifier_version=rejected.verdict.verifier_version,
                ),
                selection_method=policy.selection_method,
                score_margin=margin_value,
                provenance=Provenance(source_type="generated", source_id=provenance_source_id),
                scenario_family_id=context.get("scenario_family_id"),
            )
        )
        decisions["accepted"] += 1

    return records, decisions

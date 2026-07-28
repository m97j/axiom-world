"""Hybrid tiered aggregation (protocol §7.4).

Aggregation policy:
1. Tier-0 gate: if schema FAILED -> overall FAILED (score 0). No later tier
   can rescue a malformed output.
2. If any component is INFRA_ERROR/TIMEOUT -> overall inherits that status
   (excluded from rewards; reported as infrastructure evidence).
3. Otherwise weighted mean over components with numeric scores; PASSED iff
   all required components passed and score >= pass_threshold.

The LLM judge is deliberately absent: it is audit-only and lives outside
the reward path (protocol §7.4).
"""
from __future__ import annotations

from typing import Any

from axiom_world.core.enums import VerificationStatus
from axiom_world.verifiers.base import Verdict, Verifier

_INFRA = (VerificationStatus.INFRA_ERROR, VerificationStatus.TIMEOUT)


class HybridVerifier(Verifier):
    name = "hybrid"
    version = "1.0"

    def __init__(
        self,
        gate: Verifier,
        components: dict[str, tuple[Verifier, float]],
        required: tuple[str, ...] = (),
        pass_threshold: float = 0.8,
    ) -> None:
        if not components:
            raise ValueError("HybridVerifier requires at least one component.")
        total = sum(weight for _, weight in components.values())
        if total <= 0:
            raise ValueError("Component weights must sum to a positive value.")
        self.gate = gate
        self.components = components
        self.required = required
        self.pass_threshold = pass_threshold
        self._total_weight = total

    def _verify(self, prediction: str, context: dict[str, Any]) -> Verdict:
        gate_verdict = self.gate.verify(prediction, context)
        evidence: dict[str, Any] = {"components": {self.gate.name: gate_verdict.model_dump()}}
        if gate_verdict.status in _INFRA:
            return self._make(gate_verdict.status, f"gate_{gate_verdict.status.value}", **evidence)
        if gate_verdict.status is not VerificationStatus.PASSED:
            return self._make(
                VerificationStatus.FAILED, f"gate_failed:{gate_verdict.reason_code}",
                score=0.0, **evidence
            )

        weighted = 0.0
        used_weight = 0.0
        all_required_passed = True
        for key, (verifier, weight) in self.components.items():
            verdict = verifier.verify(prediction, context)
            evidence["components"][key] = verdict.model_dump()
            if verdict.status in _INFRA:
                return self._make(verdict.status, f"component_{verdict.status.value}:{key}",
                                  **evidence)
            if verdict.status is VerificationStatus.SKIPPED or verdict.score is None:
                continue
            weighted += verdict.score * weight
            used_weight += weight
            if key in self.required and verdict.status is not VerificationStatus.PASSED:
                all_required_passed = False

        if used_weight == 0.0:
            return self._make(VerificationStatus.INDETERMINATE, "no_scoring_evidence", **evidence)
        score = weighted / used_weight
        passed = all_required_passed and score >= self.pass_threshold
        status = VerificationStatus.PASSED if passed else VerificationStatus.FAILED
        reason = "weighted_aggregation" if passed else (
            "required_component_failed" if not all_required_passed else "below_threshold"
        )
        return self._make(status, reason, score=round(score, 6), **evidence)


def default_playworld_verifier() -> HybridVerifier:
    """The pre-registered PlayWorld reward stack (weights fixed pre-experiment)."""
    from axiom_world.verifiers.rule import (
        GoalVerifier,
        LegalityVerifier,
        SchemaVerifier,
        StateConsistencyVerifier,
    )

    return HybridVerifier(
        gate=SchemaVerifier(),
        components={
            "legality": (LegalityVerifier(), 0.35),
            "goal": (GoalVerifier(), 0.45),
            "state_consistency": (StateConsistencyVerifier(), 0.20),
        },
        # state_consistency is required: a contradicted claimed state is a
        # hard failure even when actions succeed (protocol §7.1 metric intent).
        required=("legality", "goal", "state_consistency"),
        pass_threshold=0.8,
    )

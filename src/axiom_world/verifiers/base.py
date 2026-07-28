"""Verifier protocol (docs/verifier-contract.md, protocol §7.4).

Status semantics are the load-bearing part:
- passed / failed          -> model evidence (rewards, pair mining)
- skipped / indeterminate  -> excluded from rewards, reported separately
- timeout / infra_error    -> infrastructure evidence, never model failure
"""
from __future__ import annotations

import abc
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from axiom_world.core.enums import VerificationStatus


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verifier_name: str
    verifier_version: str
    status: VerificationStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason_code: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def reward_eligible(self) -> bool:
        return self.status in (VerificationStatus.PASSED, VerificationStatus.FAILED)


class Verifier(abc.ABC):
    """One verifier = one deterministic check over (prediction, context)."""

    name: str = "verifier"
    version: str = "1.0"

    @abc.abstractmethod
    def _verify(self, prediction: str, context: dict[str, Any]) -> Verdict:
        ...

    def verify(self, prediction: str, context: dict[str, Any]) -> Verdict:
        start = time.perf_counter()
        try:
            verdict = self._verify(prediction, context)
        except Exception as exc:
            verdict = Verdict(
                verifier_name=self.name,
                verifier_version=self.version,
                status=VerificationStatus.INFRA_ERROR,
                score=None,
                reason_code="verifier_exception",
                evidence={"error": f"{type(exc).__name__}: {exc}"},
            )
        verdict.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return verdict

    def _make(
        self,
        status: VerificationStatus,
        reason_code: str,
        score: float | None = None,
        **evidence: Any,
    ) -> Verdict:
        return Verdict(
            verifier_name=self.name,
            verifier_version=self.version,
            status=status,
            score=score,
            reason_code=reason_code,
            evidence=dict(evidence),
        )

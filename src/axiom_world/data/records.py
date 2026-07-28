"""Canonical dataset record schemas (docs/data-contract.md).

Three record types cover the whole study:
- SFTRecord        : chat messages, last message is the assistant target.
- PreferenceRecord : prompt + chosen/rejected with verifier evidence.
- EvaluationRecord : a frozen PlayWorld scenario episode prompt.

Every record carries Provenance. Trainers consume THESE names via the
adapter layer; raw external formats never reach a trainer.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from axiom_world.core.enums import VerificationStatus

SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictModel):
    source_type: Literal["synthetic", "public", "human_curated", "generated"]
    source_id: str
    source_revision: str | None = None
    generator_model: str | None = None
    transformation_version: str = "v1"
    created_at: str | None = None


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @model_validator(mode="after")
    def _non_empty(self) -> Message:
        if not self.content.strip():
            raise ValueError("Message content must be non-empty.")
        return self


class SFTRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    messages: list[Message] = Field(min_length=2)
    provenance: Provenance
    task_family: str = "playworld"
    scenario_family_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _chat_contract(self) -> SFTRecord:
        if self.messages[-1].role != "assistant":
            raise ValueError("Last message must be the assistant target.")
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("At least one user message is required.")
        return self


class VerificationEvidence(StrictModel):
    status: VerificationStatus
    score: float | None = None
    verifier_version: str


class PreferenceRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    prompt: list[Message] = Field(min_length=1)
    chosen: str
    rejected: str
    chosen_verification: VerificationEvidence
    rejected_verification: VerificationEvidence
    selection_method: Literal["hybrid_verifier_rank", "random_pairing"] = "hybrid_verifier_rank"
    score_margin: float | None = None
    provenance: Provenance
    scenario_family_id: str | None = None

    @model_validator(mode="after")
    def _pair_contract(self) -> PreferenceRecord:
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen and rejected must differ.")
        if (
            self.selection_method == "hybrid_verifier_rank"
            and self.chosen_verification.status is not VerificationStatus.PASSED
        ):
            raise ValueError(
                "verifier-ranked pairs require a PASSED chosen candidate (protocol §5.2)."
            )
        return self


class EvaluationRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    suite: Literal[
        "eval_id", "eval_template_ood", "eval_comp_ood", "eval_rule_ood", "eval_adversarial"
    ]
    scenario: dict[str, Any] = Field(description="Serialized playworld Scenario.")
    prompt: list[Message] = Field(min_length=1)
    scenario_family_id: str
    provenance: Provenance

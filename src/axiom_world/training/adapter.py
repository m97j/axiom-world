"""Canonical records -> trainer-boundary rows.

Raw dataset formats never reach a trainer; canonical records never leak TRL
field names. This module is the only translation point.
"""
from __future__ import annotations

from typing import Any

from axiom_world.core.errors import AxiomError
from axiom_world.data.bundle import DataBundle


def _require(bundle: DataBundle, kind: str) -> None:
    if bundle.kind != kind:
        raise AxiomError(f"Expected a {kind!r} bundle, got {bundle.kind!r}.")


def to_sft_rows(bundle: DataBundle) -> list[dict[str, Any]]:
    """SFTRecord -> {'messages': [...]} (TRL SFTTrainer conversational format)."""
    _require(bundle, "sft")
    return [
        {"messages": [m.model_dump() for m in record.messages]}
        for record in bundle.records
    ]


def to_dpo_rows(bundle: DataBundle) -> list[dict[str, Any]]:
    """PreferenceRecord -> {'prompt','chosen','rejected'} string rows."""
    _require(bundle, "preference")
    rows: list[dict[str, Any]] = []
    for record in bundle.records:
        prompt_text = "\n".join(m.content for m in record.prompt)
        rows.append(
            {"prompt": prompt_text, "chosen": record.chosen, "rejected": record.rejected}
        )
    return rows


def to_grpo_rows(bundle: DataBundle) -> list[dict[str, Any]]:
    """EvaluationRecord/prompt bundle -> {'prompt', 'scenario'} rows for GRPO rollouts."""
    _require(bundle, "evaluation")
    rows: list[dict[str, Any]] = []
    for record in bundle.records:
        prompt_text = "\n".join(m.content for m in record.prompt)
        rows.append({"prompt": prompt_text, "scenario": record.scenario})
    return rows

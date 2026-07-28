"""Verifier -> TRL GRPO reward bridge (protocol §7.4 reward semantics).

Reward policy (pre-registered):
- passed  -> aggregate score (0..1)
- failed  -> 0.0
- skipped / indeterminate / timeout / infra_error -> None (TRL >= 1.x treats
  None as 'exclude this completion from reward normalization'); counts are
  tallied for the run's infrastructure telemetry.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from axiom_world.core.enums import VerificationStatus
from axiom_world.verifiers.base import Verifier

_EXCLUDED = (
    VerificationStatus.SKIPPED,
    VerificationStatus.INDETERMINATE,
    VerificationStatus.TIMEOUT,
    VerificationStatus.INFRA_ERROR,
)


def verifier_reward_function(
    verifier: Verifier,
    status_counter: Counter | None = None,
) -> Callable[..., list[float | None]]:
    """Build a TRL-compatible reward callable.

    TRL GRPOTrainer calls reward funcs as fn(prompts=..., completions=...,
    **dataset_columns). The 'scenario' dataset column becomes the verifier
    context, which is how PlayWorld rewards stay exact.
    """
    counter = status_counter if status_counter is not None else Counter()

    def reward(
        prompts: list[str],
        completions: list[Any],
        scenario: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[float | None]:
        rewards: list[float | None] = []
        for index, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else _extract_text(completion)
            context: dict[str, Any] = {}
            if scenario is not None:
                context["scenario"] = scenario[index]
            verdict = verifier.verify(text, context)
            counter[verdict.status.value] += 1
            if verdict.status is VerificationStatus.PASSED:
                rewards.append(verdict.score if verdict.score is not None else 1.0)
            elif verdict.status is VerificationStatus.FAILED:
                rewards.append(0.0)
            elif verdict.status in _EXCLUDED:
                rewards.append(None)
            else:  # pragma: no cover - enum is closed
                rewards.append(None)
        return rewards

    reward.__name__ = f"verifier_reward_{verifier.name}"
    return reward


def _extract_text(completion: Any) -> str:
    """Handle TRL conversational completions: [{'role','content'}, ...]."""
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return str(completion)

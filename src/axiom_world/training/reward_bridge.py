"""Verifier -> TRL GRPO reward bridge (protocol §7.4 reward semantics).

Reward policy (pre-registered):
- passed  -> aggregate score (0..1)
- failed  -> 0.0
- skipped / indeterminate / timeout / infra_error -> None (TRL >= 1.x treats
  None as 'exclude this completion from reward normalization'); counts are
  tallied for the run's infrastructure telemetry.

Two hardening rules added after the 2026-08-15 B6 incident (Arrow struct
unification injected None into nested 'scenario' dicts -> every completion
INFRA_ERROR -> 17h of training at grad_norm 0):

1. Scenario transport is a JSON STRING column ('scenario_json'); this module
   decodes it. Strings are opaque scalars to Arrow, so the payload
   round-trips byte-exactly (see training.adapter.to_grpo_rows).
2. Fail-fast health guard: once at least ``min_calls`` completions have been
   scored, if the excluded (None-reward) fraction exceeds
   ``max_excluded_fraction`` the bridge raises RewardHealthError instead of
   letting the run silently produce zero gradients.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import Any

from axiom_world.core.enums import VerificationStatus
from axiom_world.core.errors import RewardHealthError
from axiom_world.verifiers.base import Verifier

_EXCLUDED = (
    VerificationStatus.SKIPPED,
    VerificationStatus.INDETERMINATE,
    VerificationStatus.TIMEOUT,
    VerificationStatus.INFRA_ERROR,
)

_EXCLUDED_VALUES = tuple(status.value for status in _EXCLUDED)

# Pre-registered guard defaults (protocol §7.4 amendment, 2026-08-15):
# 256 completions = 32 optimizer micro-batches at num_generations=8 — enough
# to rule out a transient blip; >50% excluded means the reward stream is
# structurally broken, not occasionally noisy.
DEFAULT_MIN_CALLS = 256
DEFAULT_MAX_EXCLUDED_FRACTION = 0.5


def _decode_scenario(raw: Any) -> Any:
    """Accept both the canonical JSON-string transport and legacy dicts."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def verifier_reward_function(
    verifier: Verifier,
    status_counter: Counter | None = None,
    min_calls: int = DEFAULT_MIN_CALLS,
    max_excluded_fraction: float = DEFAULT_MAX_EXCLUDED_FRACTION,
) -> Callable[..., list[float | None]]:
    """Build a TRL-compatible reward callable.

    TRL GRPOTrainer calls reward funcs as fn(prompts=..., completions=...,
    **dataset_columns). The 'scenario_json' dataset column (JSON string) is
    decoded into the verifier context, which is how PlayWorld rewards stay
    exact AND Arrow-proof. The legacy 'scenario' dict column is still
    accepted for backward compatibility with pre-v0.6.12 datasets.
    """
    counter = status_counter if status_counter is not None else Counter()

    def _check_health() -> None:
        total = sum(counter.values())
        if total < min_calls:
            return
        excluded = sum(counter[value] for value in _EXCLUDED_VALUES)
        fraction = excluded / total
        if fraction > max_excluded_fraction:
            raise RewardHealthError(
                f"Reward stream degenerate: {excluded}/{total} completions "
                f"({fraction:.1%}) returned excluded statuses "
                f"{dict((k, counter[k]) for k in _EXCLUDED_VALUES if counter[k])} "   # noqa: C402
                f"(threshold {max_excluded_fraction:.0%} after {min_calls} calls). "
                "Aborting instead of training with zero advantages. Most likely "
                "causes: corrupted scenario transport (run scripts/"
                "x17_grpo_scenario_audit.py) or a verifier/environment fault."
            )

    def reward(
        prompts: list[str],
        completions: list[Any],
        scenario_json: list[str] | None = None,
        scenario: list[dict[str, Any]] | None = None,
        **kwargs: Any,  # noqa: ARG001 - TRL passes extra columns
    ) -> list[float | None]:
        payloads = scenario_json if scenario_json is not None else scenario
        rewards: list[float | None] = []
        for index, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else _extract_text(completion)
            context: dict[str, Any] = {}
            if payloads is not None:
                context["scenario"] = _decode_scenario(payloads[index])
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
        _check_health()
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

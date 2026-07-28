"""Evaluation metrics with uncertainty (protocol §7.1, §8).

Pure-Python (random module) bootstrap: no numpy dependency in the contract
layer, deterministic under a fixed seed, adequate at suite sizes ~300.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from axiom_world.core.enums import VerificationStatus

BOOTSTRAP_RESAMPLES = 10_000


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    if not values:
        raise ValueError("bootstrap_ci requires at least one value.")
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    low_index = int((alpha / 2) * resamples)
    high_index = min(resamples - 1, int((1 - alpha / 2) * resamples))
    return mean, means[low_index], means[high_index]


def paired_bootstrap_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Paired bootstrap of mean(A) - mean(B) over shared episodes (§8)."""
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("Paired bootstrap requires equal-length, non-empty sequences.")
    rng = random.Random(seed)
    n = len(values_a)
    diffs = [a - b for a, b in zip(values_a, values_b, strict=False)]
    point = sum(diffs) / n
    resampled = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    low = resampled[int((alpha / 2) * resamples)]
    high = resampled[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return {
        "delta": point,
        "ci_low": low,
        "ci_high": high,
        "significant": not (low <= 0.0 <= high),
    }


def summarize_suite(traces: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    """Aggregate one suite's evaluation traces into the §7.1 metric block.

    Each trace: {"verdict": Verdict.model_dump(), "prediction": str, ...}.
    """
    if not traces:
        raise ValueError("Cannot summarize an empty suite.")
    eligible_scores: list[float] = []
    pass_flags: list[float] = []
    failure_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    legal_rates: list[float] = []

    for trace in traces:
        verdict = trace["verdict"]
        status = verdict["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in (VerificationStatus.PASSED.value, VerificationStatus.FAILED.value):
            score = verdict.get("score")
            eligible_scores.append(score if score is not None else 0.0)
            pass_flags.append(1.0 if status == VerificationStatus.PASSED.value else 0.0)
            if status == VerificationStatus.FAILED.value:
                code = verdict.get("reason_code", "unknown")
                failure_counts[code] = failure_counts.get(code, 0) + 1
        components = verdict.get("evidence", {}).get("components", {})
        legality = components.get("legality", {})
        rate = legality.get("evidence", {}).get("legal_action_rate")
        if rate is not None:
            legal_rates.append(float(rate))

    summary: dict[str, Any] = {
        "episodes": len(traces),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_taxonomy": dict(sorted(failure_counts.items(), key=lambda kv: -kv[1])),
    }
    if pass_flags:
        mean, low, high = bootstrap_ci(pass_flags, seed=seed)
        summary["pass_rate"] = {"mean": round(mean, 4), "ci95": [round(low, 4), round(high, 4)]}
        mean_s, low_s, high_s = bootstrap_ci(eligible_scores, seed=seed)
        summary["mean_score"] = {
            "mean": round(mean_s, 4),
            "ci95": [round(low_s, 4), round(high_s, 4)],
        }
    if legal_rates:
        mean_l, low_l, high_l = bootstrap_ci(legal_rates, seed=seed)
        summary["legal_action_rate"] = {
            "mean": round(mean_l, 4),
            "ci95": [round(low_l, 4), round(high_l, 4)],
        }
    return summary

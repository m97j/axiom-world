"""Analysis layer: paired run comparison with uncertainty (protocol §8).

Compares two completed evaluation runs (e.g. B03 A1-adapter vs B04 base)
episode-by-episode on the frozen suites. Episodes are paired by their frozen
`id`, so both runs MUST have been executed against the same freeze
fingerprint; the loader enforces exact id-set equality per suite.

Statistics, all pure-Python and deterministic under a fixed seed:
- per-suite pass_rate and mean_score with percentile-bootstrap 95% CIs
  (reuses evaluation.metrics.bootstrap_ci);
- paired bootstrap of the A-B delta (evaluation.metrics.paired_bootstrap_diff);
- paired sign-flip permutation test: under H0 (no difference) each paired
  delta is symmetric around zero, so flipping signs uniformly at random
  yields the null distribution of the mean delta. Two-sided p-value with
  add-one smoothing: p = (1 + #{|perm| >= |observed|}) / (1 + resamples).
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from axiom_world.evaluation.metrics import bootstrap_ci, paired_bootstrap_diff

PERMUTATION_RESAMPLES = 10_000


def paired_permutation_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    resamples: int = PERMUTATION_RESAMPLES,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired sign-flip permutation test on mean(A - B)."""
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("Permutation test requires equal-length, non-empty sequences.")
    diffs = [a - b for a, b in zip(values_a, values_b, strict=True)]
    n = len(diffs)
    observed = sum(diffs) / n
    rng = random.Random(seed)
    hits = 0
    for _ in range(resamples):
        permuted = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(permuted) >= abs(observed):
            hits += 1
    return {
        "observed_delta": observed,
        "p_value": (1 + hits) / (1 + resamples),
        "resamples": resamples,
    }


def load_episode_scores(run_dir: Path, suite: str) -> dict[str, dict[str, float]]:
    """Load {episode_id: {score, passed}} from a run's evaluation_<suite>.jsonl."""
    path = run_dir / f"evaluation_{suite}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing evaluation output: {path}")
    episodes: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        verdict = record["verdict"]
        episodes[record["id"]] = {
            "score": float(verdict["score"]),
            "passed": 1.0 if verdict["status"] == "passed" else 0.0,
        }
    if not episodes:
        raise ValueError(f"no episodes parsed from {path}")
    return episodes


def _paired_metric(
    episodes_a: dict[str, dict[str, float]],
    episodes_b: dict[str, dict[str, float]],
    key: str,
    seed: int,
) -> dict:
    ids = sorted(episodes_a)
    values_a = [episodes_a[i][key] for i in ids]
    values_b = [episodes_b[i][key] for i in ids]
    mean_a, low_a, high_a = bootstrap_ci(values_a, seed=seed)
    mean_b, low_b, high_b = bootstrap_ci(values_b, seed=seed)
    return {
        "a": {"mean": round(mean_a, 4), "ci95": [round(low_a, 4), round(high_a, 4)]},
        "b": {"mean": round(mean_b, 4), "ci95": [round(low_b, 4), round(high_b, 4)]},
        "paired_bootstrap": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in paired_bootstrap_diff(values_a, values_b, seed=seed).items()
        },
        "permutation": {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in paired_permutation_test(values_a, values_b, seed=seed).items()
        },
    }


def compare_runs(
    run_dir_a: Path,
    run_dir_b: Path,
    suites: Sequence[str],
    label_a: str = "candidate",
    label_b: str = "baseline",
    seed: int = 42,
) -> dict:
    """Full paired comparison across suites. 'a' is the candidate run."""
    report: dict = {
        "labels": {"a": label_a, "b": label_b},
        "run_dir_a": str(run_dir_a),
        "run_dir_b": str(run_dir_b),
        "seed": seed,
        "suites": {},
    }
    for suite in suites:
        episodes_a = load_episode_scores(run_dir_a, suite)
        episodes_b = load_episode_scores(run_dir_b, suite)
        if set(episodes_a) != set(episodes_b):
            only_a = len(set(episodes_a) - set(episodes_b))
            only_b = len(set(episodes_b) - set(episodes_a))
            raise ValueError(
                f"suite '{suite}': episode id sets differ (only-in-a={only_a}, "
                f"only-in-b={only_b}); runs must share the same freeze fingerprint."
            )
        report["suites"][suite] = {
            "episodes": len(episodes_a),
            "pass_rate": _paired_metric(episodes_a, episodes_b, "passed", seed),
            "mean_score": _paired_metric(episodes_a, episodes_b, "score", seed),
        }
    return report

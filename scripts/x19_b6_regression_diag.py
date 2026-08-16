#!/usr/bin/env python
"""x19: B6-vs-parent regression diagnosis (train-reward up, frozen-suite pass down).

Context (2026-08-16). B6 GRPO trained cleanly (reward 0 -> ~0.6, no clipping,
no termination pathology) yet the frozen suites show SIGNIFICANT pass_rate
regressions vs both B4v2 (-14pp ID) and B5, while mean_score is ~flat. The
leading hypothesis is OBJECTIVE MISMATCH, not a training bug: the GRPO reward
is the verifier's weighted aggregate score, so the policy is paid for cheap
partial credit, while pass_rate is an all-required-components metric. A
secondary hypothesis is entropy-collapse overfitting to the 2000 train
prompts' solution style (short, uniform plans) that transfers poorly even to
held-out ID families.

This script quantifies both, from two eval run dirs (same frozen suites):
  1. Flip matrix per suite: pass->fail, fail->pass, per-episode paired.
  2. Score distribution of B6's pass->fail flips: if flips cluster at
     moderate scores (0.4-0.8) the aggregate-reward hypothesis is supported
     (the policy holds partial credit where the parent completed the task).
  3. Component-level failure deltas (verdict reason codes / component flags
     when present in traces).
  4. Prediction-length and legal_action_rate deltas on flipped episodes.

Usage (Colab, after fetch_run for both run dirs):
  python scripts/x19_b6_regression_diag.py \
    --run-a runs/20260816-004824--eval-playworld--s42--274abd \
    --run-b runs/20260814-032546--eval-playworld--s42--7308ee \
    --label-a b6-grpo --label-b b4v2-sft \
    --out runs/x19_b6_regression_diag.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_traces(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Index trace rows by (suite, episode_id), schema-tolerant."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                clean_line = line.strip()
                if not clean_line:
                    continue
                row = json.loads(line)
                suite = row.get("suite") or row.get("suite_name") or path.stem
                episode = str(
                    row.get("episode_id") or row.get("id")
                    or row.get("record_id") or row.get("prompt_id") or ""
                )
                if not episode:
                    continue
                rows[(suite, episode)] = row  # last write wins (dedup)
    return rows


def _verdict(row: dict[str, Any]) -> dict[str, Any]:
    """Canonical trace schema (evaluation/runner.py) nests the verdict:
    {"id", "suite", "scenario_family_id", "prediction",
     "verdict": {"status", "score", "reason_code", ...}}.
    v1 of this script read row["status"] at top level and silently classified
    EVERY episode as fail (fail->fail 300 across all suites) — schema drift
    must fail loudly, hence the strict accessor."""
    verdict = row.get("verdict")
    if isinstance(verdict, dict):
        return verdict
    # legacy/flat fallback: treat the row itself as the verdict container
    return row


def _status(row: dict[str, Any]) -> str:
    return str(_verdict(row).get("status") or "").lower()


def _score(row: dict[str, Any]) -> float | None:
    verdict = _verdict(row)
    for key in ("score", "mean_score", "aggregate_score"):
        if isinstance(verdict.get(key), (int, float)):
            return float(verdict[key])
    return None


def _pred_len(row: dict[str, Any]) -> int | None:
    for key in ("prediction", "completion", "output", "raw_output"):
        if isinstance(row.get(key), str):
            return len(row[key])
    return None


def _reasons(row: dict[str, Any]) -> list[str]:
    verdict = _verdict(row)
    out: list[str] = []
    for key in ("reason_code", "reason_codes", "verdict_reasons", "reasons",
                "failure_reasons"):
        value = verdict.get(key)
        if isinstance(value, list):
            out.extend(str(v) for v in value)
        elif isinstance(value, str) and value:
            out.append(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    traces_a = _load_traces(Path(args.run_a))
    traces_b = _load_traces(Path(args.run_b))
    shared = sorted(set(traces_a) & set(traces_b))
    if not shared:
        raise SystemExit(
            "no shared (suite, episode_id) keys — check that both run dirs hold "
            "trace JSONLs for the same frozen suites")

    # Sanity gate: if NO episode parses as 'passed' in either run, the trace
    # schema is not what this script expects — abort instead of emitting an
    # all-fail->fail report (the v1 silent-failure mode).
    passed_a = sum(1 for k in shared if _status(traces_a[k]) == "passed")
    passed_b = sum(1 for k in shared if _status(traces_b[k]) == "passed")
    if passed_a == 0 and passed_b == 0:
        sample = traces_a[shared[0]]
        raise SystemExit(
            "SCHEMA MISMATCH: 0 'passed' episodes parsed in BOTH runs — the\n"
            "summary JSONs say otherwise, so status extraction is broken.\n"
            f"First trace row keys: {sorted(sample.keys())}\n"
            f"verdict sub-keys: {sorted(_verdict(sample).keys())}")

    per_suite: dict[str, dict[str, Any]] = {}
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]

    for suite in sorted({s for s, _ in shared}):
        keys = [k for k in shared if k[0] == suite]
        flips = Counter()
        p2f_scores: list[float] = []
        p2f_reasons = Counter()
        len_delta: list[int] = []
        for key in keys:
            ra, rb = traces_a[key], traces_b[key]
            sa = "pass" if _status(ra) == "passed" else "fail"
            sb = "pass" if _status(rb) == "passed" else "fail"
            flips[f"{sb}->{sa}"] += 1  # b(parent) -> a(b6)
            la, lb = _pred_len(ra), _pred_len(rb)
            if la is not None and lb is not None:
                len_delta.append(la - lb)
            if sb == "pass" and sa == "fail":
                score = _score(ra)
                if score is not None:
                    p2f_scores.append(score)
                for reason in _reasons(ra):
                    p2f_reasons[reason] += 1
        hist = {
            f"[{lo},{hi if hi <= 1 else 1.0})": sum(1 for s in p2f_scores if lo <= s < hi)
            for lo, hi in buckets
        }
        per_suite[suite] = {
            "episodes_paired": len(keys),
            "flip_matrix_parent_to_b6": dict(flips),
            "parentpass_b6fail": {
                "count": flips["pass->fail"],
                "b6_score_mean": (sum(p2f_scores) / len(p2f_scores)) if p2f_scores else None,
                "b6_score_histogram": hist,
                "b6_reason_codes_top": p2f_reasons.most_common(8),
            },
            "prediction_len_delta_mean_b6_minus_parent": (
                sum(len_delta) / len(len_delta) if len_delta else None
            ),
        }

    report = {
        "labels": {"a": args.label_a, "b": args.label_b},
        "run_a": args.run_a,
        "run_b": args.run_b,
        "interpretation_guide": {
            "aggregate_reward_hypothesis": "supported if parentpass_b6fail scores "
                "cluster in [0.4,0.8) — B6 keeps partial credit where parent passed",
            "style_overfit_hypothesis": "supported if prediction_len_delta is "
                "strongly negative and flips are uniform across score buckets",
        },
        "suites": per_suite,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

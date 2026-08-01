#!/usr/bin/env python
"""Paired statistical comparison of two evaluation runs (protocol §8).

Given two eval-run directories (candidate vs baseline) containing
evaluation_<suite>.jsonl files over the SAME frozen suites, writes
analysis_summary.json with per-suite pass_rate / mean_score, bootstrap 95%
CIs, paired-bootstrap deltas, and sign-flip permutation p-values.

Usage (Colab, project root — CPU cell, seconds):
  python scripts/run_analysis.py \
      --run-a runs/<a1-eval-run> --label-a a1-sft \
      --run-b runs/<base-eval-run> --label-b qwen3-8b-base \
      --output runs/<a1-eval-run>/analysis_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SUITES = [
    "eval_id",
    "eval_template_ood",
    "eval_comp_ood",
    "eval_rule_ood",
    "eval_adversarial",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, help="candidate eval run dir")
    parser.add_argument("--run-b", required=True, help="baseline eval run dir")
    parser.add_argument("--label-a", default="candidate")
    parser.add_argument("--label-b", default="baseline")
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from axiom_world.analysis import compare_runs

    report = compare_runs(
        Path(args.run_a), Path(args.run_b), args.suites,
        label_a=args.label_a, label_b=args.label_b, seed=args.seed,
    )

    print("=" * 78)
    print(f"{'suite':<20} {'metric':<11} {args.label_a:>10} {args.label_b:>10} "
          f"{'delta':>8} {'95% CI':>18} {'p-perm':>9} sig")
    print("-" * 78)
    for suite, stats in report["suites"].items():
        for metric in ("pass_rate", "mean_score"):
            m = stats[metric]
            delta = m["paired_bootstrap"]
            ci = f"[{delta['ci_low']:+.3f},{delta['ci_high']:+.3f}]"
            sig = "YES" if delta["significant"] else "no"
            print(f"{suite:<20} {metric:<11} {m['a']['mean']:>10.4f} "
                  f"{m['b']['mean']:>10.4f} {delta['delta']:>+8.4f} {ci:>18} "
                  f"{m['permutation']['p_value']:>9.5f} {sig}")
    print("=" * 78)

    output = Path(args.output) if args.output else Path(args.run_a) / "analysis_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"analysis summary -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

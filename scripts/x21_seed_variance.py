#!/usr/bin/env python
"""x21 — Seed-variance aggregation for the 3-seed final (Gate G6, Amendment v1.4).

Aggregates per-suite pass_rate / mean_score across (model, seed) eval runs and
emits mean ± sd tables plus the pre-registered G6 verdict:

  PASS  <=> for every frozen suite, sign(pass_rate[b4v2] - pass_rate[a2v2])
            is identical across all provided seeds (ties count as inconsistent).

Input: repeated (--model NAME --eval-run DIR --seed N) triples. Each eval-run
dir must contain artifacts/evaluation_summary.json (or evaluation_summary.json
at its root) as produced by scripts/run_evaluation.py.

Usage:
  python scripts/x21_seed_variance.py \
      --model b4v2 --eval-run runs/<b4v2-s42-eval> --seed 42 \
      --model a2v2 --eval-run runs/<a2v2-s42-eval> --seed 42 \
      ... \
      --output runs/seed_variance_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SUITES = [
    "eval_id",
    "eval_template_ood",
    "eval_comp_ood",
    "eval_rule_ood",
    "eval_adversarial",
]
METRICS = ["pass_rate", "mean_score"]


def _load_summary(run_dir: Path) -> dict:
    for cand in (run_dir / "artifacts" / "evaluation_summary.json",
                 run_dir / "evaluation_summary.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    raise FileNotFoundError(f"evaluation_summary.json not found under {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--eval-run", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--suites", nargs="+", default=SUITES)
    parser.add_argument("--output", default="runs/seed_variance_report.json")
    args = parser.parse_args()

    if not (len(args.model) == len(args.eval_run) == len(args.seed)):
        parser.error("--model/--eval-run/--seed must be repeated in triples")

    # records[model][seed][suite][metric]
    records: dict[str, dict[int, dict]] = {}
    for model, run, seed in zip(args.model, args.eval_run, args.seed, strict=True):
        summary = _load_summary(Path(run))
        suites = summary.get("suites", summary)
        records.setdefault(model, {})[seed] = {
            s: {m: suites[s].get(m) for m in METRICS} for s in args.suites
        }

    report: dict = {"suites": {}, "models": sorted(records), "verdict": {}}
    for suite in args.suites:
        report["suites"][suite] = {}
        for model, by_seed in records.items():
            for metric in METRICS:
                vals = [by_seed[s][suite][metric] for s in sorted(by_seed)]
                report["suites"][suite].setdefault(model, {})[metric] = {
                    "per_seed": {str(s): by_seed[s][suite][metric]
                                 for s in sorted(by_seed)},
                    "mean": statistics.mean(vals),
                    "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                }

    # G6 sign-consistency verdict (b4v2 vs a2v2 on pass_rate)
    verdict = {"criterion": "sign(b4v2-a2v2) pass_rate identical across seeds",
               "per_suite": {}, "pass": True}
    if {"b4v2", "a2v2"} <= records.keys():
        common = sorted(set(records["b4v2"]) & set(records["a2v2"]))
        verdict["seeds"] = common
        for suite in args.suites:
            deltas = {s: records["b4v2"][s][suite]["pass_rate"]
                         - records["a2v2"][s][suite]["pass_rate"]
                      for s in common}
            signs = {(-1 if d < 0 else (1 if d > 0 else 0)) for d in deltas.values()}
            consistent = len(signs) == 1 and 0 not in signs
            verdict["per_suite"][suite] = {
                "delta_per_seed": {str(s): round(d, 4) for s, d in deltas.items()},
                "sign_consistent": consistent,
            }
            verdict["pass"] &= consistent
    else:
        verdict["pass"] = False
        verdict["error"] = "need both b4v2 and a2v2 records"
    report["verdict"] = verdict

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"{'suite':<20} {'model':<6} {'pass mean±sd':>16} {'score mean±sd':>16}")
    print("-" * 62)
    for suite in args.suites:
        for model in sorted(records):
            pr = report["suites"][suite][model]["pass_rate"]
            ms = report["suites"][suite][model]["mean_score"]
            print(f"{suite:<20} {model:<6} "
                  f"{pr['mean']:.3f} ± {pr['sd']:.3f}      "
                  f"{ms['mean']:.3f} ± {ms['sd']:.3f}")
    print(f"\nG6 verdict: {'PASS' if verdict['pass'] else 'FAIL'} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

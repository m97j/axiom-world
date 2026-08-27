#!/usr/bin/env python
"""x20: eval identity audit — did two eval runs score the SAME weights?

Context (2026-08-16). analysis_b6r_vs_b6 reported delta 0.0000 / p=1.0 on
EVERY suite and metric, and the x19 flip histograms matched the earlier B6
diagnosis digit-for-digit. Two independently trained adapters (different
reward shapes) cannot produce byte-identical predictions on 1,500 episodes;
the plausible failure is that the "B6-R eval" actually fetched the B6-era
weights (the hub layout stores the training adapter at the repo ROOT
artifacts/, shared and overwritten across runs, and the B6-R final sync
failed on quota, so the root may still hold B6).

This audit settles it from the eval traces themselves, without re-running
anything:

  1. pairs every (suite, episode_id) across the two eval run dirs and
     computes the exact-match rate of the raw `prediction` strings;
  2. reports per-suite identical/total and a global verdict:
       identical > 99%  -> SAME_WEIGHTS (the newer eval is INVALID — discard)
       identical < 90%  -> DIFFERENT_WEIGHTS (evals are genuinely distinct)
       otherwise        -> AMBIGUOUS (inspect diffs manually)
  3. if --repo is given, also fetches the CURRENT hub root artifacts/
     lineage.json and prints its run_id + adapter sha256, so you can see
     which training run the shared root slot holds right now.

Usage:
  python scripts/x20_eval_identity_audit.py \
    --run-a runs/<newer-eval-run> --run-b runs/<older-eval-run> \
    --repo m97j/aw-runs-b6 --out runs/x20_eval_identity_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_traces(run_dir: Path) -> dict[tuple[str, str], str]:
    rows: dict[tuple[str, str], str] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                clean_line = line.strip()
                if not clean_line:
                    continue
                row = json.loads(clean_line)
                suite = row.get("suite") or path.stem
                episode = str(row.get("id") or row.get("episode_id") or "")
                if not episode:
                    continue
                rows[(suite, episode)] = str(row.get("prediction", ""))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, help="newer eval run dir")
    parser.add_argument("--run-b", required=True, help="older eval run dir")
    parser.add_argument("--repo", default=None,
                        help="optional: also report which training run the hub "
                             "ROOT artifacts/ slot currently holds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    a = _load_traces(Path(args.run_a))
    b = _load_traces(Path(args.run_b))
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no shared (suite, episode) keys between the two runs")

    per_suite: dict[str, dict[str, Any]] = {}
    total = same = 0
    first_diffs: list[dict[str, str]] = []
    for key in shared:
        suite = key[0]
        stats = per_suite.setdefault(suite, {"episodes": 0, "identical": 0})
        stats["episodes"] += 1
        total += 1
        if a[key] == b[key]:
            stats["identical"] += 1
            same += 1
        elif len(first_diffs) < 3:
            first_diffs.append({"suite": suite, "episode": key[1],
                                "a_head": a[key][:120], "b_head": b[key][:120]})
    for stats in per_suite.values():
        stats["identical_rate"] = round(stats["identical"] / stats["episodes"], 4)

    rate = same / total
    verdict = ("SAME_WEIGHTS" if rate > 0.99
               else "DIFFERENT_WEIGHTS" if rate < 0.90 else "AMBIGUOUS")

    report: dict[str, Any] = {
        "run_a": args.run_a,
        "run_b": args.run_b,
        "episodes_paired": total,
        "identical_predictions": same,
        "identical_rate": round(rate, 4),
        "verdict": verdict,
        "verdict_meaning": {
            "SAME_WEIGHTS": "the newer eval scored the SAME adapter as the older "
                            "one — discard it as invalid and re-fetch/re-eval the "
                            "intended weights",
            "DIFFERENT_WEIGHTS": "evals are genuinely distinct adapters",
            "AMBIGUOUS": "partial overlap — inspect sample_diffs",
        }[verdict],
        "per_suite": per_suite,
        "sample_diffs": first_diffs,
    }

    if args.repo:
        try:
            from huggingface_hub import hf_hub_download

            lineage_path = hf_hub_download(
                repo_id=args.repo, filename="artifacts/lineage.json")
            lineage = json.loads(Path(lineage_path).read_text())
            report["hub_root_artifacts"] = {
                "run_id": lineage.get("run_id"),
                "output_adapter_sha256": lineage.get("output_adapter_sha256"),
                "parent_run_id": lineage.get("parent_run_id"),
            }
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            report["hub_root_artifacts"] = f"fetch failed: {exc!r}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

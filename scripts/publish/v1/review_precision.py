"""Review both precision variants without changing either release or approving upload."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from compare_champion import read, summarize_pairs, write


def paired_interval(rows_a, rows_b, suite, resamples=10000):
    deltas = [(b["verdict"]["status"] == "passed") - (a["verdict"]["status"] == "passed")
              for a, b in zip(rows_a, rows_b, strict=True) if a["suite"] == suite]
    rng = random.Random(42)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(resamples))
    return {"delta": sum(deltas) / len(deltas), "ci95_low": samples[int(0.025 * resamples)], "ci95_high": samples[int(0.975 * resamples) - 1],
            "method": f"paired episode bootstrap, {resamples} draws, seed 42; exploratory, unadjusted"}


def load_comparison(folder):
    report = read(folder / "comparison.json")
    if report["status"] != "completed_requires_review":
        raise ValueError("Comparison incomplete")
    rows = {arm: [json.loads(line) for line in (folder / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
            for arm in ("reference", "candidate")}
    if any(len(r) != 1500 for r in rows.values()):
        raise ValueError("Expected complete 1500-row comparisons")
    if summarize_pairs(rows["reference"], rows["candidate"]) != report["suites"]:
        raise ValueError("Report differs from per-episode evidence")
    return report, rows


def review(fp32_folder, bf16_folder, *, independent_controls=False):
    fp, fp_rows = load_comparison(fp32_folder)
    bf, bf_rows = load_comparison(bf16_folder)
    for key in ("legacy_code", "data_revision", "freeze", "max_new_tokens", "attention", "seed"):
        if fp["request"][key] != bf["request"][key]:
            raise ValueError(f"Comparison conditions differ: {key}")
    batches_match = fp["request"]["batch_size"] == bf["request"]["batch_size"]
    if not batches_match and not independent_controls:
        raise ValueError("Comparison conditions differ: batch_size; use --independent-controls for descriptive within-run review only")
    if bf["request"].get("candidate_dtype") != "bfloat16":
        raise ValueError("Need a BF16 task comparison")
    if bf["request"].get("source_fp32_receipt_sha256") != fp["request"]["release_receipt_sha256"]:
        raise ValueError("BF16 candidate does not derive from the compared FP32 release")
    same_reference = fp_rows["reference"] == bf_rows["reference"]
    rows = {}
    for name in fp["suites"]:
        rows[name] = {"fp32": fp["suites"][name], "bf16": bf["suites"][name],
                      "fp32_vs_adapter_ci": paired_interval(fp_rows["reference"], fp_rows["candidate"], name),
                      "bf16_vs_adapter_ci": paired_interval(bf_rows["reference"], bf_rows["candidate"], name)}
    checks = {name: {
        "no_observed_pass_count_loss": r["candidate_pass"] >= r["reference_pass"],
        "no_increase_in_format_errors": r["candidate_schema_failed"] <= r["reference_schema_failed"],
        "no_increase_in_truncation": r["candidate_truncated"] <= r["reference_truncated"],
    } for name, r in bf["suites"].items()}
    matched = same_reference and batches_match and not independent_controls
    bf16_preferred = matched and all(all(c.values()) for c in checks.values())
    result = {"status": "completed_requires_review", "reference_repeated_exactly": same_reference,
              "suites": rows, "publication_approved": False,
              "independent_controls": independent_controls,
              "batch_sizes": {"fp32": fp["request"]["batch_size"], "bf16": bf["request"]["batch_size"]},
              "cross_precision_comparison_valid": matched,
              "fp32_source_receipt_sha256": fp["request"]["release_receipt_sha256"],
              "interpretation": "Each variant is compared only with its own paired adapter control. Unmatched runs do not establish a precision-only effect or justify pooled controls.",
              "deployment_checks": checks,
              "suggested_precision": ("bfloat16" if bf16_preferred else "float32") if matched else "review_required",
              "decision_policy": "Conservative observed-suite non-degradation rule defined before BF16 task results; not statistical equivalence. FP32 fallback retains its disclosed compositional-OOD regression.",
              "scope": "Deployment precision audit, not new champion selection or proof of equivalence",
              "note": "Review every suite and discordant episodes; do not use historical 3-seed means as acceptance bounds"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-comparison", type=Path, required=True)
    parser.add_argument("--bf16-comparison", type=Path, required=True)
    parser.add_argument("--independent-controls", action="store_true", help="Describe unmatched runs separately; never issue automatic precision selection")
    args = parser.parse_args()
    destination = args.bf16_comparison / "precision_review.json"
    if destination.exists():
        raise SystemExit("Review already exists; preserve existing evidence")
    result = review(args.fp32_comparison, args.bf16_comparison, independent_controls=args.independent_controls)
    write(destination, result)
    print(json.dumps(result, indent=2))

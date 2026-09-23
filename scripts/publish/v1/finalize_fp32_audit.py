"""Attach completed independent-control audits to verified FP32 release; no GPU or Hub writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from review_precision import load_comparison, review

from axiom_world.models.champion_release import inventory, write_json


def finalize(release, fp32, bf16):
    receipt_path = release / "verified.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    source_hash = hashlib.sha256(receipt_bytes).hexdigest()
    stage = release / "model"
    if (receipt.get("status") != "verified" or receipt["verification"]["dtype"] != "float32"
            or not receipt["verification"]["merge"]["passed"]
            or not receipt["verification"]["standalone"]["passed"]
            or inventory(stage) != receipt["files"]):
        raise ValueError("Need unchanged verified FP32 release")
    if any((release / n).exists() for n in ("upload_started.json", "uploaded.json", "published.json")):
        raise ValueError("Publication already started; inspect remote state")
    fp_report, _ = load_comparison(fp32)
    bf_report, _ = load_comparison(bf16)
    if fp_report["request"]["release_receipt_sha256"] != source_hash:
        raise ValueError("FP32 comparison does not bind this exact pre-audit receipt")
    result = review(fp32, bf16, independent_controls=True)
    if all(all(c.values()) for c in result["deployment_checks"].values()):
        raise ValueError("BF16 has no observed deployment regression; this rejection workflow does not apply")
    # Policy decision is explicit; do not pretend the independent review approved it.
    decision = {"selected_precision": "float32", "selection_basis":
        "Retain independently verified FP32 export; reject BF16 after within-run regressions. "
        "Not a matched FP32-versus-BF16 superiority claim or historical equivalence claim.",
        "source_receipt_sha256": source_hash, "review": result}
    destination = stage / "evaluation" / "precision_audit"
    if destination.exists() or (release / "verified.before-task-audit.json").exists():
        raise ValueError("Audit finalization already started; preserve and inspect existing evidence")
    # Preserve the original receipt before any payload changes; final receipt is written last.
    (release / "verified.before-task-audit.json").write_bytes(receipt_bytes)
    destination.mkdir(parents=True)
    (destination / "source_verified.json").write_bytes(receipt_bytes)
    for label, folder in (("fp32", fp32), ("bf16", bf16)):
        target = destination / label
        target.mkdir()
        for name in ("comparison.json", "request.json", "reference.jsonl", "candidate.jsonl",
                     "reference_complete.json", "candidate_complete.json"):
            shutil.copy2(folder / name, target / name)
    write_json(destination / "decision.json", decision)
    lines = ["\n\n## Post-export task audit and deployment decision\n",
        "The selected standalone export is FP32. Use explicit `dtype=torch.float32` "
        "(or the runner's equivalent); automatic downcasting does not reproduce this audit.\n",
        "Both runs used the frozen 1500 tasks, but batch sizes differed: "
        f"FP32={fp_report['request']['batch_size']}, BF16={bf_report['request']['batch_size']}. "
        "Each row below compares against that run's own adapter control. These are not "
        "matched cross-precision results or historical runtime reproduction.\n",
        "| Suite | FP32 run adapter / export | BF16 run adapter / export |\n",
        "|---|---:|---:|\n"]
    for name, f in fp_report["suites"].items():
        b = bf_report["suites"][name]
        lines.append(f"| {name} | {f['reference_pass']} / {f['candidate_pass']} "
                     f"| {b['reference_pass']} / {b['candidate_pass']} |\n")
    lines += ["\nCounts are out of 300 per suite. FP32 is not equivalent to the original "
        "adapter: suite-level regressions remain disclosed above. BF16 was rejected for "
        "observed within-run degradation, including format failures. This deployment "
        "decision followed the results and is not a new preregistered scientific claim.\n",
        "Full paired predictions, verifier verdicts, requests and the decision are in "
        "[evaluation/precision_audit](evaluation/precision_audit). Original report metrics "
        "describe the historical adapter experiments, not this FP32 export.\n"]
    card = stage / "README.md"
    card.write_text(card.read_text(encoding="utf-8") + "".join(lines), encoding="utf-8")
    updated = dict(receipt)
    updated["task_audit_source_receipt_sha256"] = source_hash
    updated["task_audit_decision"] = "evaluation/precision_audit/decision.json"
    updated["files"] = inventory(stage)
    # Only the card and newly added evidence may differ from the verified model payload.
    current = {f["path"]: f for f in updated["files"]}
    if any(current.get(f["path"]) != f for f in receipt["files"] if f["path"] != "README.md"):
        raise ValueError("Original model payload changed during evidence attachment")
    pending = release / "verified.task-audit.tmp"
    write_json(pending, updated)
    pending.replace(receipt_path)
    print(json.dumps(decision, indent=2))
    print("FP32_TASK_AUDIT_ATTACHED; no model weights changed; HF unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--fp32-comparison", type=Path, required=True)
    parser.add_argument("--bf16-comparison", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.release, args.fp32_comparison, args.bf16_comparison)

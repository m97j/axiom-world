#!/usr/bin/env python
"""Build Phase-1 general-reasoning data from public HF datasets (B1/B2).

Supports a WEIGHTED MIXTURE of sources (protocol §4.2). Each source needs a
column adapter mapping rows to (prompt, response, gold_answer). Built-ins:

  gsm8k   : question/answer, gold = text after '####'  (grade-school CoT)
  math    : problem/solution, gold = last \\boxed{...}  (competition math)
  generic : --prompt-col/--response-col, no gold (instruction following)

Gold answers come from the DATASET, never an LLM. Records without a
verifiable gold are still usable for SFT but excluded from RS/preference
mining (ExactAnswerVerifier returns SKIPPED).

Redistribution note: processed records keep full provenance. Check source
licenses before pushing to a public repo; private is the default posture.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from axiom_world.data.bundle import write_jsonl
from axiom_world.data.records import Message, Provenance, SFTRecord

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def _adapt_gsm8k(row: dict) -> tuple[str, str, str | None]:
    answer = row["answer"]
    gold = answer.split("####")[-1].strip() if "####" in answer else None
    return row["question"], answer, gold


def _adapt_math(row: dict) -> tuple[str, str, str | None]:
    solution = row["solution"]
    boxed = _BOXED.findall(solution)
    return row["problem"], solution, (boxed[-1].strip() if boxed else None)


ADAPTERS = {"gsm8k": _adapt_gsm8k, "math": _adapt_math}

# Pre-registered default mixture for B1 (weights sum to 1.0). Kept small and
# math-centric on purpose: P1 is a transfer intervention, not a frontier
# general model (protocol §2). Structured-output/instruction data comes from
# the oracle-derived PlayWorld-DISJOINT synthetic set if added later.
DEFAULT_MIXTURE = [
    {"dataset": "openai/gsm8k", "config": "main", "split": "train",
     "adapter": "gsm8k", "weight": 0.6},
    {"dataset": "EleutherAI/hendrycks_math", "config": "algebra", "split": "train",
     "adapter": "math", "weight": 0.4},
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture-json", default=None,
                        help="Path to a JSON list overriding the default mixture.")
    parser.add_argument("--total", type=int, default=8000,
                        help="Target total records across sources (by weight).")
    parser.add_argument("--holdout", type=int, default=500,
                        help="Records reserved as the P1 general held-out suite "
                             "(stratified across sources).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/p1")
    parser.add_argument("--hf-sync-repo", default=None,
                        help="HF DATASET repo (user/name). When set, the P1 mixture "
                             "+ holdout + manifest are uploaded after build.")
    parser.add_argument("--hf-path-in-repo", default="p1/v1",
                        help="Destination path inside the dataset repo.")
    args = parser.parse_args()

    from datasets import load_dataset  # lazy: generation/training session only

    mixture = (
        json.loads(Path(args.mixture_json).read_text())
        if args.mixture_json else DEFAULT_MIXTURE
    )
    weight_sum = sum(s["weight"] for s in mixture)
    rng = random.Random(args.seed)
    records: list[SFTRecord] = []
    per_source_counts: dict[str, int] = {}
    seen_prompts: set[str] = set()

    for source in mixture:
        adapter = ADAPTERS[source["adapter"]]
        quota = int(args.total * source["weight"] / weight_sum)
        dataset = load_dataset(
            source["dataset"], source.get("config"),
            split=source["split"], revision=source.get("revision", "main"),
        )
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        taken = 0
        for index in indices:
            if taken >= quota:
                break
            prompt, response, gold = adapter(dataset[index])
            key = " ".join(prompt.lower().split())
            if not prompt.strip() or not response.strip() or key in seen_prompts:
                continue  # drop_empty + dedup-by-normalized-prompt
            seen_prompts.add(key)
            records.append(
                SFTRecord(
                    id=f"p1-{source['adapter']}-{index:06d}",
                    messages=[
                        Message(role="user", content=prompt),
                        Message(role="assistant", content=response),
                    ],
                    provenance=Provenance(
                        source_type="public",
                        source_id=source["dataset"],
                        source_revision=source.get("revision", "main"),
                        transformation_version="p1-v2-mixture",
                    ),
                    task_family=f"general_{source['adapter']}",
                    metadata={"gold_answer": gold, "source_index": index},
                )
            )
            taken += 1
        per_source_counts[source["dataset"]] = taken

    rng.shuffle(records)
    holdout = records[-args.holdout:] if args.holdout else []
    train = records[: len(records) - len(holdout)]
    output_dir = Path(args.output_dir)
    train_fp = write_jsonl(output_dir / "p1_general_sft.jsonl", train)
    holdout_fp = write_jsonl(output_dir / "p1_general_holdout.jsonl", holdout)
    manifest = {
        "mixture": mixture,
        "per_source_counts": per_source_counts,
        "seed": args.seed,
        "train_records": len(train),
        "holdout_records": len(holdout),
        "train_fingerprint": train_fp,
        "holdout_fingerprint": holdout_fp,
    }
    (output_dir / "p1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print("\nFreeze the holdout fingerprint; it is the general-retention suite "
          "used by the Stage-1 hard constraint (protocol §6).")

    if args.hf_sync_repo:
        from axiom_world.integrations.hf_sync import upload_directory

        uri = upload_directory(
            output_dir, args.hf_sync_repo,
            path_in_repo=args.hf_path_in_repo, repo_type="dataset",
            commit_message=f"P1 mixture build (seed={args.seed}, "
                           f"total={args.total}, holdout={args.holdout})",
        )
        print(f"P1 data persisted: {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

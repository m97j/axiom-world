#!/usr/bin/env python
"""Build Phase-1 general-reasoning data from a public HF dataset (B1/B2).

Canonicalizes a math word-problem dataset (default: openai/gsm8k) into
SFTRecords whose gold answers come from the DATASET, never an LLM. The gold
answer is also stored in metadata so rejection sampling (B2) and preference
mining (B3) can verify candidates with ExactAnswerVerifier.

Redistribution note: processed records keep full provenance (source dataset,
revision, original split/index). Check the source license before pushing the
processed set to a public HF repo; private is the default posture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.data.bundle import write_jsonl
from axiom_world.data.records import Message, Provenance, SFTRecord


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--config-name", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--holdout", type=int, default=500,
                        help="Last N records reserved as the P1 general held-out suite.")
    parser.add_argument("--output-dir", default="data/p1")
    args = parser.parse_args()

    from datasets import load_dataset  # lazy: generation/training session only

    dataset = load_dataset(
        args.dataset, args.config_name, split=args.split, revision=args.revision
    )
    records: list[SFTRecord] = []
    for index, row in enumerate(dataset):
        if index >= args.limit:
            break
        question, answer = row["question"], row["answer"]
        gold = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
        records.append(
            SFTRecord(
                id=f"p1-{args.dataset.replace('/', '-')}-{index:06d}",
                messages=[
                    Message(role="user", content=question),
                    Message(role="assistant", content=answer),
                ],
                provenance=Provenance(
                    source_type="public",
                    source_id=args.dataset,
                    source_revision=args.revision,
                    transformation_version="p1-v1",
                ),
                task_family="general_math",
                metadata={"gold_answer": gold, "source_index": index},
            )
        )

    holdout = records[-args.holdout:] if args.holdout else []
    train = records[: len(records) - len(holdout)]
    output_dir = Path(args.output_dir)
    train_fp = write_jsonl(output_dir / "p1_general_sft.jsonl", train)
    holdout_fp = write_jsonl(output_dir / "p1_general_holdout.jsonl", holdout)
    manifest = {
        "source": {"dataset": args.dataset, "revision": args.revision, "split": args.split},
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Freeze the C2 few-shot exemplar set (protocol §4, reference baselines).

Deterministically selects the first N records (seed-ordered as built) from
the TRAIN sft jsonl — never from eval families, preserving the G3 leakage
gate — and writes them as [{prompt, completion}] with a sha256 fingerprint
echoed for the run card. Re-running on the same input reproduces the same
file byte-for-byte.

Usage (after a02 data build):
  python scripts/build_fewshot_exemplars.py \
      --sft-file data/train/playworld_sft.jsonl \
      --output data/eval_suites/fewshot_exemplars.json \
      --count 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-file", default="data/train/playworld_sft.jsonl")
    parser.add_argument("--output", default="data/eval_suites/fewshot_exemplars.json")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    exemplars: list[dict[str, str]] = []
    for line in Path(args.sft_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]
        if not user_turns or not assistant_turns:
            continue
        exemplars.append({"prompt": user_turns[-1], "completion": assistant_turns[-1]})
        if len(exemplars) == args.count:
            break

    if len(exemplars) < args.count:
        raise SystemExit(
            f"only {len(exemplars)} usable records found; need {args.count}."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(exemplars, indent=2, sort_keys=True, ensure_ascii=False)
    output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"{len(exemplars)} exemplars -> {output} (sha256:{digest[:19]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

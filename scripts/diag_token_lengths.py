#!/usr/bin/env python
"""Diagnostic e03: token-length distribution of SFT records vs truncation caps.

Quantifies hypothesis C for the A1 collapse: TRL 1.x SFTConfig.max_length
defaults to 1024 with right-truncation, so any record whose full
chat-template rendering exceeds the cap loses (part of) its assistant
completion from the loss. A model trained on such data learns to CONTINUE
prompts rather than answer them — which matches the observed degeneration
(no stop, repetition) despite 95.5% teacher-forced token accuracy.

For every record in the SFT jsonl this script tokenizes:
  - full   : full conversation rendering (what SFTTrainer packs),
  - prompt : user-side rendering + generation prompt (eval-time prefix),
and reports, for each cap in --caps (default: 1024 legacy, 4096 new):
  - #records whose FULL length exceeds the cap (any truncation),
  - #records whose PROMPT alone reaches the cap (completion FULLY deleted),
  - #records with partial completion loss (prompt < cap < full).

Usage (Colab, project root):
  python scripts/diag_token_lengths.py \
      --config configs/experiments/eval_playworld.yaml \
      --prompt-file data/train/playworld_sft.jsonl \
      --caps 1024 4096

Exit code 0 always; read the VERDICT block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-file", default="data/train/playworld_sft.jsonl")
    parser.add_argument("--caps", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--limit", type=int, default=0, help="0 = all records")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config, _, _ = resolve(args.config, [])
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_repo_id or config.model.repo_id,
        revision=config.model.revision,
    )

    lines = Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[: args.limit]

    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        non_assistant = [m for m in messages if m["role"] != "assistant"]
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        prompt_ids = tokenizer.apply_chat_template(
            non_assistant, tokenize=True, add_generation_prompt=True
        )
        full_lengths.append(len(full_ids))
        prompt_lengths.append(len(prompt_ids))

    n = len(full_lengths)
    sorted_full = sorted(full_lengths)
    print("=" * 70)
    print(f"records tokenized: {n}  (file: {args.prompt_file})")
    print(
        "FULL length  min/p50/p90/p99/max: "
        f"{sorted_full[0]}/{_percentile(sorted_full, 0.5)}/"
        f"{_percentile(sorted_full, 0.9)}/{_percentile(sorted_full, 0.99)}/{sorted_full[-1]}"
    )
    sorted_prompt = sorted(prompt_lengths)
    print(
        "PROMPT length min/p50/p90/p99/max: "
        f"{sorted_prompt[0]}/{_percentile(sorted_prompt, 0.5)}/"
        f"{_percentile(sorted_prompt, 0.9)}/{_percentile(sorted_prompt, 0.99)}/{sorted_prompt[-1]}"
    )
    print("=" * 70)

    worst_cap_hit = 0.0
    for cap in args.caps:
        truncated = sum(1 for length in full_lengths if length > cap)
        fully_lost = sum(1 for p in prompt_lengths if p >= cap)
        partial = sum(
            1 for p, f in zip(prompt_lengths, full_lengths, strict=True) if p < cap < f
        )
        ratio = truncated / n if n else 0.0
        if cap == min(args.caps):
            worst_cap_hit = ratio
        print(f"--- cap={cap} ---")
        print(f"  truncated at all      : {truncated}/{n} ({100 * ratio:.1f}%)")
        print(f"  completion FULLY lost : {fully_lost}/{n} ({100 * fully_lost / n:.1f}%)")
        print(f"  completion partly lost: {partial}/{n} ({100 * partial / n:.1f}%)")

    print("=" * 70)
    if worst_cap_hit > 0.05:
        print(
            "VERDICT: hypothesis C CONFIRMED — a material fraction of records "
            f"({100 * worst_cap_hit:.1f}%) was truncated at the legacy cap. "
            "Retrain A1 with training.max_length >= the reported p99 FULL length."
        )
    else:
        print(
            "VERDICT: hypothesis C NOT supported — truncation at the legacy cap "
            "is negligible; investigate loss masking / label construction next."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

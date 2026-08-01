#!/usr/bin/env python
"""Diagnostic e06: reproduce the EXACT text TRL trained on (hypothesis G).

Context. e05 localized the pathology precisely:
  - boundary tokenizations match (hypothesis E rejected);
  - the training TARGET opens with an empty think block
    '<think>\\n\\n</think>\\n\\n' injected by the Qwen3 chat template;
  - the adapter's distribution at the boundary and inside the think block
    is FLAT GARBAGE (top-1 '.Aggressive' at p=0.013), yet from k=4 (just
    past '</think>\\n\\n') it produces perfect PlayWorld JSON.
  => The adapter genuinely learned the JSON but was NEVER trained on the
     think-block opener tokens. That happens if TRL's train-time rendering
     of the conversation differs from tokenizer.apply_chat_template(messages)
     exactly at the assistant opener (hypothesis G) — e.g. TRL renders via
     maybe_apply_chat_template with different template kwargs, so training
     completions start directly at '{"'.

This script needs NO GPU and NO model download. It compares, per record:
  1. ours  : tokenizer.apply_chat_template(messages, tokenize=False)
             (what e01/e04/e05 and run_evaluation.py assume),
  2. trl   : trl.data_utils.maybe_apply_chat_template({'messages': ...})
             (what SFTTrainer actually trains on for conversational rows),
  3. eval  : apply_chat_template(non_assistant, add_generation_prompt=True)
and reports the first divergence plus whether each rendering contains a
'<think>' block.

Usage (Colab, project root — CPU cell is fine):
  python scripts/diag_trl_rendering.py \
      --config configs/experiments/eval_playworld.yaml \
      --prompt-file data/train/playworld_sft.jsonl \
      --num-samples 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve


def _first_divergence(a: str, b: str) -> int:
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), min(len(a), len(b))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-file", default="data/train/playworld_sft.jsonl")
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config, _, _ = resolve(args.config, [])
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_repo_id or config.model.repo_id,
        revision=config.model.revision,
    )

    try:
        from trl.data_utils import maybe_apply_chat_template
    except ImportError as exc:  # pragma: no cover - env specific
        raise SystemExit(f"TRL is required for this diagnostic: {exc}")

    records = [
        json.loads(line)
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.num_samples]

    mismatches = 0
    for idx, record in enumerate(records):
        messages = record["messages"]
        non_assistant = [m for m in messages if m["role"] != "assistant"]

        ours = tokenizer.apply_chat_template(messages, tokenize=False)
        trl_row = maybe_apply_chat_template({"messages": messages}, tokenizer)
        trl_text = trl_row.get("text") or trl_row.get("messages")
        if not isinstance(trl_text, str):
            raise SystemExit(
                "maybe_apply_chat_template did not return a text row; TRL keeps "
                f"conversational format here (keys={list(trl_row)}). Inspect "
                "SFTTrainer's internal template call for this TRL version."
            )
        eval_prompt = tokenizer.apply_chat_template(
            non_assistant, tokenize=False, add_generation_prompt=True
        )

        same = ours == trl_text
        if not same:
            mismatches += 1
        if idx == 0 or not same:
            print("=" * 70)
            print(f"record {idx}: ours == trl-rendered : {same}")
            print(f"  '<think>' in ours rendering : {'<think>' in ours}")
            print(f"  '<think>' in trl rendering  : {'<think>' in trl_text}")
            print(f"  '<think>' in eval prompt    : {'<think>' in eval_prompt}")
            print(f"  eval prompt tail            : {eval_prompt[-80:]!r}")
            if not same:
                i = _first_divergence(ours, trl_text)
                print(f"  FIRST DIVERGENCE at char {i}:")
                print(f"    ours: {ours[max(0, i - 80): i + 120]!r}")
                print(f"    trl : {trl_text[max(0, i - 80): i + 120]!r}")
            else:
                boundary = ours.find(messages[-1]["content"][:20])
                print(f"  assistant opener region     : "
                      f"{ours[max(0, boundary - 120): boundary + 40]!r}")

    print("=" * 70)
    if mismatches:
        print(
            f"VERDICT: hypothesis G CONFIRMED ({mismatches}/{len(records)} rows "
            "diverge) — TRL trained on a different rendering than evaluation "
            "assumes. Fix: make run_evaluation.py build prompts from the SAME "
            "rendering TRL used (align template kwargs), no retrain needed if "
            "only the opener differs."
        )
    else:
        print(
            "VERDICT: renderings identical — hypothesis G rejected. The adapter "
            "was trained on the think-block opener yet failed to learn it: "
            "audit the training loop next (packing, Liger/collator label "
            "shifts, or optimizer instability on rare boundary tokens)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

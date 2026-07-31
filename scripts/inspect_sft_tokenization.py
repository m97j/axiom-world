#!/usr/bin/env python
"""Diagnostic: show the EXACT text/token forms used at train vs eval time.

Prints, for the first SFT record:
  1. the chat-template rendering TRL SFTTrainer trains on (full conversation),
  2. the chat-template rendering run_evaluation.py feeds at inference
     (user turn + generation prompt),
  3. token ids around the assistant boundary, and which stop tokens exist.

If (2) is not a prefix of (1), training and evaluation are format-mismatched
and free-running generation will derail even when teacher-forced token
accuracy is high.

Usage:
  python scripts/inspect_sft_tokenization.py \
      --config configs/experiments/eval_playworld.yaml \
      [--prompt-file data/train/playworld_sft.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-file", default="data/train/playworld_sft.jsonl")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config, _, _ = resolve(args.config, [])
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_repo_id or config.model.repo_id,
        revision=config.model.revision,
    )

    record = json.loads(Path(args.prompt_file).read_text(encoding="utf-8").splitlines()[0])
    messages = record["messages"]
    user_only = [m for m in messages if m["role"] == "user"]

    train_text = tokenizer.apply_chat_template(messages, tokenize=False)
    eval_text = tokenizer.apply_chat_template(
        user_only, tokenize=False, add_generation_prompt=True
    )

    print("=" * 70)
    print("chat_template defined:", tokenizer.chat_template is not None)
    print("eos:", repr(tokenizer.eos_token), tokenizer.eos_token_id)
    print("pad:", repr(tokenizer.pad_token), tokenizer.pad_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print("<|im_end|> id:", im_end)
    print("=" * 70)
    print("--- TRAIN-TIME RENDERING (first 600 chars) ---")
    print(train_text[:600])
    print("--- TRAIN-TIME RENDERING (last 300 chars) ---")
    print(train_text[-300:])
    print("=" * 70)
    print("--- EVAL-TIME RENDERING (last 300 chars, incl. generation prompt) ---")
    print(eval_text[-300:])
    print("=" * 70)
    is_prefix = train_text.startswith(eval_text)
    print("eval prompt is a prefix of train text:", is_prefix)
    if not is_prefix:
        # locate first divergence
        i = next((k for k, (a, b) in enumerate(zip(train_text, eval_text)) if a != b),
                 min(len(train_text), len(eval_text)))
        print(f"FIRST DIVERGENCE at char {i}:")
        print("  train:", repr(train_text[max(0, i - 60): i + 60]))
        print("  eval :", repr(eval_text[max(0, i - 60): i + 60]))
        print("VERDICT: FORMAT MISMATCH — fix eval prompt construction "
              "(or trainer template) before drawing any conclusion from eval runs.")
    else:
        print("VERDICT: formats consistent — investigate decoding/stop tokens instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

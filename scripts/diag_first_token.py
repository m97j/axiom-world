#!/usr/bin/env python
"""Diagnostic e05: dissect the assistant-boundary first token.

Context. e04 showed a paradox: the adapter's teacher-forced completion-span
accuracy is 96.2%, yet free-run generation diverges at completion token 0
and degenerates (".Aggressive ..." loops). Base (84.3% acc) free-runs to
valid JSON. Two remaining mechanisms can produce exactly this signature:

  (E) Boundary tokenization mismatch: the eval-time prompt token sequence
      (encode(prompt_text)) differs from the training-time token sequence
      at the same boundary (full-text tokens sliced at prompt_len) — e.g.
      the trailing "assistant\n" newline merges differently with the first
      content token. The adapter then sees an out-of-distribution boundary
      it never trained on, while teacher-forced scoring (which uses the
      full-text tokenization) still looks perfect.
  (F) Genuine first-token misprediction + exposure bias: the adapter is
      simply wrong at position 0 (within its 4% error budget) and greedy
      decoding never recovers.

This script decides between them, per record:
  1. boundary check: full_ids[:prompt_len] vs encode(prompt_text) — any
     mismatch confirms (E);
  2. top-5 logits at the boundary for base and adapter vs the true first
     target tokens;
  3. forced-prefix generation: prepend the first k target tokens
     (k in {1, 2, 4, 8}) and greedy-decode — if the adapter recovers valid
     structure once past the boundary, the pathology is localized there.

Usage (Colab, project root):
  python scripts/diag_first_token.py \
      --config configs/experiments/eval_playworld.yaml \
      --adapter-dir runs/<run_id>/artifacts/final_adapter \
      --prompt-file data/train/playworld_sft.jsonl \
      --num-samples 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompt-file", default="data/train/playworld_sft.jsonl")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    import torch

    config, _, _ = resolve(args.config, [])

    from axiom_world.models.builder import build_for_inference

    model, tokenizer = build_for_inference(config, adapter_dir=None)

    records = [
        json.loads(line)
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.num_samples]

    def render(messages: list[dict]) -> tuple[list[int], list[int], int]:
        non_assistant = [m for m in messages if m["role"] != "assistant"]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_text = tokenizer.apply_chat_template(
            non_assistant, tokenize=False, add_generation_prompt=True
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        return full_ids, prompt_ids, len(prompt_ids)

    @torch.no_grad()
    def top5_at_boundary(m, context_ids: list[int]) -> list[tuple[str, float]]:
        ids = torch.tensor([context_ids], device=m.device)
        logits = m(ids).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        values, indices = probs.topk(5)
        return [
            (tokenizer.decode([int(i)]), float(v)) for v, i in zip(values, indices, strict=True)
        ]

    @torch.no_grad()
    def forced_prefix_run(m, context_ids: list[int], target_ids: list[int], k: int) -> str:
        ids = torch.tensor([context_ids + target_ids[:k]], device=m.device)
        out = m.generate(
            ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0][len(context_ids):].tolist()
        return tokenizer.decode(out, skip_special_tokens=False)

    def report(m, label: str) -> None:
        boundary_mismatches = 0
        for idx, record in enumerate(records):
            full_ids, prompt_ids, prompt_len = render(record["messages"])
            sliced = full_ids[:prompt_len]
            if sliced != prompt_ids:
                boundary_mismatches += 1
                first_bad = next(
                    (i for i, (a, b) in enumerate(zip(sliced, prompt_ids, strict=True)) if a != b),
                    min(len(sliced), len(prompt_ids)),
                )
                print(f"[{label}] record {idx}: BOUNDARY TOKEN MISMATCH at index "
                      f"{first_bad}: train-slice={sliced[first_bad:first_bad + 3]} "
                      f"eval-encode={prompt_ids[first_bad:first_bad + 3]}")
            target_ids = full_ids[prompt_len:]
            if idx == 0:
                print(f"[{label}] target first tokens: "
                      f"{[tokenizer.decode([t]) for t in target_ids[:8]]!r}")
                print(f"[{label}] top-5 @ TRAIN-slice boundary : "
                      f"{top5_at_boundary(m, sliced)}")
                print(f"[{label}] top-5 @ EVAL-encode boundary : "
                      f"{top5_at_boundary(m, prompt_ids)}")
                for k in (1, 2, 4, 8):
                    text = forced_prefix_run(m, sliced, target_ids, k)
                    print(f"[{label}] forced k={k}: {text[:160]!r}")
        print(f"[{label}] boundary mismatches: {boundary_mismatches}/{len(records)}")

    print("=" * 70)
    report(model, "BASE")
    print("-" * 70)

    from peft import PeftModel

    adapter_model = PeftModel.from_pretrained(model, args.adapter_dir)
    adapter_model.eval()
    report(adapter_model, "ADAPTER")
    print("=" * 70)
    print(
        "READ-OUT: any 'BOUNDARY TOKEN MISMATCH' line confirms hypothesis E "
        "(fix eval prompt encoding to reuse the training-time tokenization). "
        "If boundaries match everywhere but the adapter's top-1 at the boundary "
        "is not the target's first token while forced k>=1 runs recover valid "
        "structure, hypothesis F holds (first-token brittleness): constrain "
        "decoding or retrain with label smoothing / more epochs on the opener."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

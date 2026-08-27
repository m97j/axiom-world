#!/usr/bin/env python
"""Diagnostic e04: did the adapter actually LEARN the completion span?

Context. e01 (template) and e02 (adapter effect) cleared hypotheses A and B;
e03 (v0.3.4) cleared hypothesis C (truncation: 0%, max FULL length 516).
Remaining contradiction: 95.5% training token accuracy vs free-run collapse
WORSE than base. But TRL's mean_token_accuracy is a FULL-SEQUENCE metric and
PlayWorld prompts are ~78% of each sequence (PROMPT p50 305 / FULL p50 391),
so high accuracy may be prompt-copying, not completion learning.

This script measures, for N training records, teacher-forced next-token
argmax accuracy SEPARATELY on the prompt span and the completion span, for
base vs base+adapter, and locates the first free-run divergence from the
training target.

Interpretation:
  - adapter completion-accuracy HIGH (>0.9), free-run still collapses
      -> exposure bias / decoding-side issue: first divergent token matters.
  - adapter completion-accuracy LOW, prompt-accuracy HIGH
      -> hypothesis D CONFIRMED: full-sequence loss let the adapter optimize
         prompt continuation; fix = assistant-only loss (label masking) and
         retrain A1.
  - adapter accuracy << base accuracy on BOTH spans
      -> adapter is corrupted/mis-scaled: audit save/load + alpha/r.

Usage (Colab, project root):
  python scripts/diag_completion_learning.py \
      --config configs/experiments/eval_playworld.yaml \
      --adapter-dir runs/<run_id>/artifacts/final_adapter \
      --prompt-file data/train/playworld_sft.jsonl \
      --num-samples 8
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
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    import torch

    config, _, _ = resolve(args.config, [])

    from axiom_world.models.builder import build_for_inference

    base_model, tokenizer = build_for_inference(config, adapter_dir=None)

    records = [
        json.loads(line)
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.num_samples]

    def render(messages: list[dict]) -> tuple[list[int], int]:
        """Return (full token ids, prompt span length) for one record."""
        non_assistant = [m for m in messages if m["role"] != "assistant"]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_text = tokenizer.apply_chat_template(
            non_assistant, tokenize=False, add_generation_prompt=True
        )
        if not full_text.startswith(prompt_text):
            raise RuntimeError("prompt rendering is not a prefix of full rendering")
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        return full_ids, len(prompt_ids)

    @torch.no_grad()
    def span_accuracy(model, full_ids: list[int], prompt_len: int) -> tuple[float, float]:
        ids = torch.tensor([full_ids], device=model.device)
        logits = model(ids).logits[0]
        preds = logits[:-1].argmax(dim=-1)          # predicts token t+1 at pos t
        targets = ids[0][1:]
        correct = (preds == targets)
        prompt_span = correct[: prompt_len - 1]
        completion_span = correct[prompt_len - 1:]
        return (
            float(prompt_span.float().mean()) if len(prompt_span) else float("nan"),
            float(completion_span.float().mean()) if len(completion_span) else float("nan"),
        )

    @torch.no_grad()
    def free_run(model, full_ids: list[int], prompt_len: int) -> tuple[str, int]:
        ids = torch.tensor([full_ids[:prompt_len]], device=model.device)
        out = model.generate(
            ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0][prompt_len:].tolist()
        target = full_ids[prompt_len:]
        diverge = next(
            (i for i, (a, b) in enumerate(zip(out, target, strict=True)) if a != b),
            min(len(out), len(target)),
        )
        return tokenizer.decode(out, skip_special_tokens=False), diverge

    def evaluate(model, label: str) -> tuple[float, float]:
        prompt_accs, completion_accs = [], []
        for i, record in enumerate(records):
            full_ids, prompt_len = render(record["messages"])
            pa, ca = span_accuracy(model, full_ids, prompt_len)
            prompt_accs.append(pa)
            completion_accs.append(ca)
            if i == 0:
                text, diverge = free_run(model, full_ids, prompt_len)
                target_len = len(full_ids) - prompt_len
                print(f"[{label}] record 0: free-run diverges from target at "
                      f"completion token {diverge}/{target_len}")
                print(f"[{label}] free-run head: {text[:300]!r}")
        mean_p = sum(prompt_accs) / len(prompt_accs)
        mean_c = sum(completion_accs) / len(completion_accs)
        print(f"[{label}] teacher-forced accuracy  prompt-span={mean_p:.4f}  "
              f"completion-span={mean_c:.4f}  (n={len(records)})")
        return mean_p, mean_c

    print("=" * 70)
    _, base_c = evaluate(base_model, "BASE")
    print("-" * 70)

    from peft import PeftModel

    adapter_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    adapter_model.eval()
    _, adapter_c = evaluate(adapter_model, "ADAPTER")
    print("=" * 70)

    if adapter_c >= 0.9:
        print("VERDICT: adapter DID learn the completion span — the collapse is "
              "decoding-side (exposure bias / stop handling); inspect the first "
              "divergent token above.")
    elif adapter_c < base_c:
        print("VERDICT: adapter completion accuracy is BELOW base — adapter is "
              "corrupted or mis-scaled; audit save/load path and alpha/r scaling.")
    else:
        print("VERDICT: hypothesis D CONFIRMED — adapter under-learned the "
              "completion span (training accuracy was prompt-dominated). Enable "
              "assistant-only label masking and retrain A1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Diagnostic: prove whether a trained adapter changes model behaviour.

Loads the base model ONCE, generates on a probe prompt, then attaches the
trained adapter to the SAME model instance and regenerates. Prints both
outputs plus the first-step logit delta. If the two outputs are identical
and the logit delta is ~0, the adapter is not being applied (or is a
no-op); if they differ, the adapter is active and any eval failure is a
format/decoding issue instead.

Usage:
  python scripts/diag_adapter_effect.py \
      --config configs/experiments/eval_playworld.yaml \
      --adapter-dir runs/<run_id>/artifacts/final_adapter \
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
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompt-file", default="data/train/playworld_sft.jsonl",
                        help="jsonl with SFT records; the FIRST record's user "
                             "message is used as the probe prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    import torch

    config, _, _ = resolve(args.config, [])
    record = json.loads(Path(args.prompt_file).read_text(encoding="utf-8").splitlines()[0])
    messages = record["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    target = next((m["content"] for m in messages if m["role"] == "assistant"), None)

    from axiom_world.models.builder import build_for_inference

    model, tokenizer = build_for_inference(config, adapter_dir=None)
    tokenizer.padding_side = "left"
    inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": user_msg["content"]}]],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    @torch.no_grad()
    def gen(m):
        out = m.generate(**inputs, max_new_tokens=args.max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=False)

    @torch.no_grad()
    def first_logits(m):
        return m(**inputs).logits[0, -1, :].float().cpu()

    base_logits = first_logits(model)
    base_text = gen(model)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    adapter_logits = first_logits(model)
    adapter_text = gen(model)

    delta = (base_logits - adapter_logits).abs()
    print("=" * 70)
    print(f"first-step logit delta: mean={delta.mean():.6f} max={delta.max():.6f}")
    print(f"outputs identical: {base_text == adapter_text}")
    print("=" * 70)
    print("--- BASE OUTPUT (first 800 chars) ---")
    print(base_text[:800])
    print("--- ADAPTER OUTPUT (first 800 chars) ---")
    print(adapter_text[:800])
    if target:
        print("--- TRAINING TARGET (first 400 chars) ---")
        print(target[:400])
    print("=" * 70)
    print("VERDICT:",
          "adapter INACTIVE or no-op — investigate loading path"
          if delta.max() < 1e-4 else
          "adapter ACTIVE — failures are format/decoding, not loading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

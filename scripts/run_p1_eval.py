#!/usr/bin/env python
"""Evaluate a model on the frozen P1 general held-out suite (protocol §7.1).

Produces the GENERAL RETENTION metric: greedy decoding on the held-out
prompts, scored by ExactAnswerVerifier against dataset gold answers, with a
bootstrap 95% CI. Run it for the base model (reference point) and for every
P1/P2 candidate; the Stage-1 hard constraint is drop <= 3 pts absolute vs
base (protocol §6).

Usage:
  python scripts/run_p1_eval.py \
      --config configs/experiments/b1_general_sft.yaml \
      [--adapter-dir runs/<run>/artifacts/final_adapter] \
      --holdout data/p1/p1_general_holdout.jsonl \
      --output runs/p1_eval_<label>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--holdout", default="data/p1/p1_general_holdout.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--label", default="model")
    parser.add_argument("--output", default=None)
    parser.add_argument("--hf-sync-repo", default=None,
                        help="HF MODEL repo; uploads under p1_eval/.")
    args = parser.parse_args()

    import torch

    from axiom_world.evaluation.metrics import bootstrap_ci
    from axiom_world.models.builder import build_for_inference
    from axiom_world.verifiers.base import VerificationStatus
    from axiom_world.verifiers.general import ExactAnswerVerifier

    config, _, _ = resolve(args.config, [])
    model, tokenizer = build_for_inference(config, adapter_dir=args.adapter_dir)
    opener_seed = config.evaluation.opener_seed
    verifier = ExactAnswerVerifier()

    records = [
        json.loads(line)
        for line in Path(args.holdout).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [r for r in records if r.get("metadata", {}).get("gold_answer")]

    stop_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.eos_token_id:
        stop_ids.append(im_end)

    @torch.no_grad()
    def generate(prompts: list[str]) -> list[str]:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True,
            ) + opener_seed
            for p in prompts
        ]
        inputs = tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids,
        )
        return tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

    from tqdm import tqdm

    outcomes: list[float] = []
    per_family: dict[str, list[float]] = {}
    for start in tqdm(range(0, len(records), args.batch_size), desc="p1-eval(batched)"):
        batch = records[start : start + args.batch_size]
        outputs = generate([r["messages"][0]["content"] for r in batch])
        for record, text in zip(batch, outputs, strict=True):
            verdict = verifier.verify(text.strip(), {"answer": record["metadata"]["gold_answer"]})
            value = 1.0 if verdict.status is VerificationStatus.PASSED else 0.0
            outcomes.append(value)
            per_family.setdefault(record.get("task_family", "unknown"), []).append(value)

    mean, low, high = bootstrap_ci(outcomes)
    summary = {
        "label": args.label,
        "adapter_dir": args.adapter_dir,
        "holdout": args.holdout,
        "episodes": len(outcomes),
        "accuracy": {"mean": round(mean, 4), "ci95": [round(low, 4), round(high, 4)]},
        "per_family": {
            family: round(sum(v) / len(v), 4) for family, v in sorted(per_family.items())
        },
        "conditioning": {"opener_seed": opener_seed},
        "decoding": {"max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size},
    }
    output = Path(args.output or f"runs/p1_eval_{args.label}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.hf_sync_repo:
        from axiom_world.integrations.hf_sync import upload_directory

        destination = upload_directory(
            output.parent, args.hf_sync_repo, path_in_repo="p1_eval",
            commit_message=f"p1 eval: {args.label}",
        )
        print(f"p1 eval synced -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

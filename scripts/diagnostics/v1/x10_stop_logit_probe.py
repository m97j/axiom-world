#!/usr/bin/env python
"""x10: stop-token logit probe — discriminate hypothesis U1 vs U2.

Context. x09 (label audit) REJECTED hypothesis T: the terminal <|im_end|>
label is LIVE in the collated batch (collated_terminal_masked_count = 0,
pad_id == eos_id == 151643 notwithstanding). The SFT models were therefore
trained on the stop token, yet probe evals show 100% truncation. Two
hypotheses remain:

  U1 (under-training / convergence): the adapter never pushed the <|im_end|>
     logit high enough at termination positions — the token is trained but
     loses to content tokens at generation time.
  U2 (rendering mismatch): the adapter DID learn to stop under the training
     rendering, but the eval-time prompt (template + opener seed) puts the
     model in a different context where termination mass never accumulates.

Method (teacher-forcing, one forward pass per sample, no generation):
  1. Render each SFT record EXACTLY like training (apply_chat_template on the
     full messages, matching diag_trl_rendering/e06 ground truth).
  2. Forward the full sequence; at the position PRECEDING the gold <|im_end|>,
     read P(<|im_end|>), its rank, and the top-5 tokens.
  3. Optionally repeat with the EVAL rendering (user turn only +
     add_generation_prompt + opener seed + the gold response appended) to
     compare the same measurement under eval conditioning.

Readout:
  - adapter P(<|im_end|>) high (>0.5, rank 1) under TRAIN rendering but low
    under EVAL rendering  => U2 (rendering mismatch).
  - adapter P(<|im_end|>) low under BOTH                => U1 (under-training).
  - base vs adapter delta shows whether SFT moved the stop logit at all.

Usage (repo root; GPU recommended, CPU works for few samples):
  python scripts/x10_stop_logit_probe.py \
      --config configs/experiments/b1_general_sft.yaml \
      --adapter-dir runs/<b1-run>/artifacts/final_adapter \
      --sft-jsonl data/p1/p1_general_sft.jsonl \
      --num-samples 8 --eval-rendering \
      --out runs/x10_stop_logit_b1.json
Run once WITHOUT --adapter-dir for the base-model reference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def render_train(tokenizer, messages: list[dict]) -> str:
    """Training-identical rendering (full conversation, no generation prompt)."""
    return tokenizer.apply_chat_template(messages, tokenize=False)


def render_eval(tokenizer, messages: list[dict], opener_seed: str) -> str:
    """Eval-conditioned rendering: prompt as run_p1_eval builds it, then the
    gold assistant response appended verbatim so the gold <|im_end|> position
    exists to probe."""
    user_only = [m for m in messages if m["role"] == "user"][:1]
    prompt = tokenizer.apply_chat_template(
        user_only, tokenize=False, add_generation_prompt=True
    ) + (opener_seed or "")
    assistant = next(m for m in messages if m["role"] == "assistant")
    return prompt + assistant["content"] + "<|im_end|>"


def probe_positions(model, tokenizer, text: str, stop_id: int, top_k: int = 5) -> dict:
    import torch

    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(model.device)
    positions = (ids[0] == stop_id).nonzero(as_tuple=True)[0].tolist()
    if not positions:
        return {"error": "gold <|im_end|> not found in rendered text"}
    pos = positions[-1]  # terminal stop token of the (last) assistant turn
    with torch.no_grad():
        logits = model(ids).logits[0, pos - 1]  # distribution PREDICTING position pos
    probs = torch.softmax(logits.float(), dim=-1)
    p_stop = probs[stop_id].item()
    rank = int((probs > probs[stop_id]).sum().item()) + 1
    top = torch.topk(probs, top_k)
    return {
        "gold_stop_position": pos,
        "p_im_end": round(p_stop, 6),
        "rank_im_end": rank,
        "top_tokens": [
            {"token": tokenizer.convert_ids_to_tokens([i])[0], "p": round(p, 4)}
            for i, p in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default=None,
                        help="omit for the base-model reference run")
    parser.add_argument("--sft-jsonl", default="data/p1/p1_general_sft.jsonl")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--eval-rendering", action="store_true",
                        help="also probe under eval prompt conditioning (U2 test)")
    parser.add_argument("--out", default="runs/x10_stop_logit_probe.json")
    args = parser.parse_args()

    from axiom_world.core.config_loader import resolve
    from axiom_world.models.builder import build_for_inference

    config, _, _ = resolve(args.config, [])
    model, tokenizer = build_for_inference(config, adapter_dir=args.adapter_dir)
    model.eval()
    stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    opener_seed = getattr(config.evaluation, "opener_seed", "") or ""

    rows = []
    with open(args.sft_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= args.num_samples:
                break
            clean_line = line.strip()
            if clean_line:
                rows.append(json.loads(clean_line))

    report = {
        "config": args.config,
        "adapter_dir": args.adapter_dir,
        "arm": "adapter" if args.adapter_dir else "base",
        "opener_seed": opener_seed,
        "im_end_id": stop_id,
        "samples": [],
    }
    p_train, p_eval = [], []
    for i, row in enumerate(rows):
        messages = row["messages"]
        entry: dict = {"idx": i}
        entry["train_rendering"] = probe_positions(
            model, tokenizer, render_train(tokenizer, messages), stop_id
        )
        if "p_im_end" in entry["train_rendering"]:
            p_train.append(entry["train_rendering"]["p_im_end"])
        if args.eval_rendering:
            entry["eval_rendering"] = probe_positions(
                model, tokenizer, render_eval(tokenizer, messages, opener_seed), stop_id
            )
            if "p_im_end" in entry["eval_rendering"]:
                p_eval.append(entry["eval_rendering"]["p_im_end"])
        report["samples"].append(entry)

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 6) if xs else None

    report["mean_p_im_end"] = {"train_rendering": _mean(p_train)}
    if args.eval_rendering:
        report["mean_p_im_end"]["eval_rendering"] = _mean(p_eval)

    mt, me = _mean(p_train), _mean(p_eval)
    if mt is None:
        report["verdict"] = "no probe positions found — check jsonl schema"
    elif mt >= 0.5 and args.eval_rendering and me is not None and me < 0.1:
        report["verdict"] = (
            "U2 (RENDERING MISMATCH): stop token is confidently predicted under "
            "the training rendering but not under eval conditioning — align "
            "run_p1_eval/run_evaluation prompt construction with the training "
            "template (opener seed / system turn / template kwargs)."
        )
    elif mt < 0.1:
        report["verdict"] = (
            "U1 (UNDER-TRAINING): stop token mass is low even under the exact "
            "training rendering — the adapter never converged on termination. "
            "Inspect diag_training_dynamics (loss on the stop-token position), "
            "consider more epochs / higher LR on the tail, or verify the LoRA "
            "targets can express the change (lm_head/embeddings excluded?)."
        )
    else:
        report["verdict"] = (
            f"AMBIGUOUS (mean p_train={mt}, p_eval={me}): compare against the "
            "base reference run; a small base->adapter delta still indicates U1."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

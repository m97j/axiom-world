#!/usr/bin/env python
"""x13: NLL profile on the EXACT trained token ids — localize the B1 pathology.

Context (x12, attested). The x10 anomaly is closed: both x10 reports were B1
(stale dir); the fresh B2 probe differs. The REAL picture per arm, on the
apply_chat_template rendering:

  B2: peaked, healthy-looking distribution at the pre-stop position — it just
      puts ~0.9-0.99 mass on '\n' (or ' The') instead of <|im_end|>. Clean U1
      (termination never learned), model otherwise intact.
  B1: NEAR-UNIFORM garbage (top-1 ~0.001-0.006, gibberish tokens) at the same
      positions — a distribution collapse, despite HEALTHY training metrics
      (loss 1.22->0.42, entropy ~0.45, token-acc 0.91).

A model cannot have entropy 0.45 during training and ~7+ nats on the same
data unless the probe context differs from the trained context. Hypothesis W:
the probe/eval rendering (tokenizer.apply_chat_template) does NOT match the
rendering TRL trained on (trainer factory processed dataset), and B1 —
trained on curated text whose format diverges more from the template — is
fully off-distribution under the probe rendering, while B2 (RS corpus =
base-model-generated text) is robust to it.

Method: rebuild the trainer's processed dataset via the REAL factory (tiny
model, x09b pattern) to obtain the EXACT input_ids each arm trained on; then
run the real Qwen3-8B(+adapter) over those ids and report:
  - per-position NLL summary over assistant tokens (mean/p50/p90),
  - NLL + P(token) at the terminal <|im_end|> position,
  - the same for the base model (reference),
  - a decoded snippet of the trained rendering (to eyeball template drift
    vs apply_chat_template).

Readout:
  - B1 NLL low on trained ids  => hypothesis W confirmed: rendering mismatch;
    fix eval/probe rendering to the trained one (code fix, no retrain needed
    for the probe; eval conditioning amendment).
  - B1 NLL high on trained ids => the adapter is genuinely degenerate despite
    the logged metrics; audit checkpoint saving/optimizer state.

Usage (repo root, GPU):
  python scripts/x13_trained_ids_nll.py \
      --config configs/experiments/b1_general_sft.yaml \
      --adapter-dir runs/<b1-run>/artifacts/final_adapter \
      --sft-jsonl data/p1/p1_general_sft.jsonl \
      --num-samples 4 --out runs/x13_trained_ids_nll_b1.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def build_processed_examples(config, sft_jsonl: str, n: int, tokenizer):
    """EXACT trained token ids via the real trainer factory (tiny model)."""
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM

    from axiom_world.training.factory import build_trainer

    repo = config.model.tokenizer_repo_id or config.model.repo_id
    tiny_config = AutoConfig.from_pretrained(repo, revision=config.model.revision)
    for name, value in {
        "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 32,
    }.items():
        if hasattr(tiny_config, name):
            setattr(tiny_config, name, value)
    tiny_model = AutoModelForCausalLM.from_config(tiny_config)

    rows = []
    with open(sft_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= n:
                break
            clean_line = line.strip()
            if clean_line:
                rows.append({"messages": json.loads(clean_line)["messages"]})

    with tempfile.TemporaryDirectory() as tmp:
        trainer = build_trainer(
            config, tiny_model, tokenizer, Dataset.from_list(rows), output_dir=tmp
        )
        processed = trainer.train_dataset
        return [processed[i] for i in range(min(n, len(processed)))]


def nll_profile(model, tokenizer, input_ids: list[int], labels: list[int]) -> dict:
    import torch

    ids = torch.tensor([input_ids], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nlls, stop_entry = [], None
    for pos in range(1, len(input_ids)):
        if labels[pos] == -100:
            continue
        token_lp = float(logprobs[pos - 1, input_ids[pos]])
        nlls.append(-token_lp)
        if input_ids[pos] == im_end_id:
            probs = torch.softmax(logits[pos - 1], dim=-1)
            stop_entry = {
                "position": pos,
                "nll": round(-token_lp, 4),
                "p_im_end": round(float(probs[im_end_id]), 6),
                "rank_im_end": int((probs > probs[im_end_id]).sum()) + 1,
            }
    nlls_sorted = sorted(nlls)
    q = lambda f: round(nlls_sorted[int(f * (len(nlls_sorted) - 1))], 4)  # noqa: E731
    return {
        "supervised_tokens": len(nlls),
        "nll_mean": round(sum(nlls) / len(nlls), 4) if nlls else None,
        "nll_p50": q(0.5) if nlls else None,
        "nll_p90": q(0.9) if nlls else None,
        "terminal_stop": stop_entry,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--sft-jsonl", default="data/p1/p1_general_sft.jsonl")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--out", default="runs/x13_trained_ids_nll.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from axiom_world.core.config_loader import resolve
    from axiom_world.models.builder import build_for_inference

    config, _, _ = resolve(args.config, [])
    repo = config.model.tokenizer_repo_id or config.model.repo_id
    proc_tokenizer = AutoTokenizer.from_pretrained(repo, revision=config.model.revision)
    if proc_tokenizer.pad_token is None:
        proc_tokenizer.pad_token = proc_tokenizer.eos_token

    examples = build_processed_examples(
        config, args.sft_jsonl, args.num_samples, proc_tokenizer
    )

    model, tokenizer = build_for_inference(config, adapter_dir=args.adapter_dir)
    model.eval()

    report: dict = {
        "config": args.config,
        "adapter_dir": args.adapter_dir,
        "arm": "adapter" if args.adapter_dir else "base",
        "samples": [],
    }
    for i, example in enumerate(examples):
        ids, labels = example["input_ids"], example.get("labels") or example["input_ids"]
        entry = {"idx": i, **nll_profile(model, tokenizer, ids, labels)}
        if i == 0:
            entry["trained_rendering_snippet"] = tokenizer.decode(ids[:64])
            entry["trained_rendering_tail"] = tokenizer.decode(ids[-24:])
        report["samples"].append(entry)

    means = [s["nll_mean"] for s in report["samples"] if s["nll_mean"] is not None]
    stops = [s["terminal_stop"]["p_im_end"] for s in report["samples"] if s["terminal_stop"]]
    report["summary"] = {
        "mean_nll_over_samples": round(sum(means) / len(means), 4) if means else None,
        "mean_p_im_end_at_terminal": round(sum(stops) / len(stops), 6) if stops else None,
    }
    m = report["summary"]["mean_nll_over_samples"]
    report["verdict"] = (
        "no data" if m is None else
        "LOW NLL on trained ids => adapter fine on ITS OWN rendering; the flat "
        "distribution seen under apply_chat_template is a RENDERING MISMATCH "
        "(hypothesis W) — align probe/eval prompt construction with the trained "
        "rendering." if m < 1.5 else
        "HIGH NLL on trained ids => adapter degenerate on its own training "
        "data; audit checkpoint save/load and optimizer state."
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

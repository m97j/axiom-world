#!/usr/bin/env python
"""Diagnostic e07: inspect the EXACT (input_ids, labels) SFTTrainer trains on.

Context. e06 rejected hypothesis G: the training rendering DOES contain the
'<think>\\n\\n</think>\\n\\n' opener. Yet e05 shows the adapter's distribution
is flat garbage precisely on those opener positions while being near-perfect
afterwards. An identically repeated 4-token opener present in all 2000
records with live loss would be learned in a few steps — so the remaining
explanation is hypothesis H: the trainer's PROCESSED tensors differ from the
rendered text, i.e. the opener tokens are either (a) absent from input_ids
after TRL's internal tokenization, or (b) present but label-masked (-100),
or (c) packing/collator shifts corrupt the assistant boundary.

This script builds the REAL trainer via axiom_world.training.factory with a
TINY random model sharing the Qwen3 tokenizer/vocab (no 8B download, CPU
cell is fine), then dumps the first processed example and one collated
batch: token ids, decoded pieces, and label state around the assistant
boundary.

Usage (Colab, project root):
  python scripts/diag_trainer_labels.py \
      --config configs/experiments/a1_playworld_sft.yaml \
      --prompt-file data/train/playworld_sft.jsonl \
      --num-samples 4
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from axiom_world.core.config_loader import resolve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-file", default="data/train/playworld_sft.jsonl")
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config, _, _ = resolve(args.config, [])
    repo = config.model.tokenizer_repo_id or config.model.repo_id
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=config.model.revision)

    # Tiny random model with the SAME vocab: dataset preprocessing depends on
    # the tokenizer, not on model weights, so this is sufficient and cheap.
    tiny_config = AutoConfig.from_pretrained(repo, revision=config.model.revision)
    for name, value in {
        "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 32,
    }.items():
        if hasattr(tiny_config, name):
            setattr(tiny_config, name, value)
    tiny_model = AutoModelForCausalLM.from_config(tiny_config)

    rows = [
        {"messages": json.loads(line)["messages"]}
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.num_samples]
    dataset = Dataset.from_list(rows)

    from axiom_world.training.factory import build_trainer

    with tempfile.TemporaryDirectory() as tmp:
        trainer = build_trainer(config, tiny_model, tokenizer, dataset, output_dir=tmp)

        processed = trainer.train_dataset
        print("=" * 70)
        print(f"processed dataset columns: {processed.column_names}")
        example = processed[0]

        think_open = tokenizer.encode("<think>", add_special_tokens=False)
        think_close = tokenizer.encode("</think>", add_special_tokens=False)
        print(f"'<think>' token id(s): {think_open}   '</think>': {think_close}")

        input_ids = example.get("input_ids")
        if input_ids is None:
            print("!! processed example has NO input_ids — trainer tokenizes at "
                  f"collation time. example keys: {list(example)}")
        else:
            labels = example.get("labels")
            print(f"example length: {len(input_ids)}   has labels: {labels is not None}")
            _dump_boundary(tokenizer, input_ids, labels, think_open, "PROCESSED")

        # One collated batch — this is literally what reaches the forward pass.
        collator = trainer.data_collator
        batch = collator([processed[i] for i in range(min(2, len(processed)))])
        batch_ids = batch["input_ids"][0].tolist()
        batch_labels = batch["labels"][0].tolist() if "labels" in batch else None
        print("-" * 70)
        print(f"collated batch keys: {list(batch.keys())}")
        _dump_boundary(tokenizer, batch_ids, batch_labels, think_open, "COLLATED")

        if batch_labels is not None:
            live = sum(1 for label in batch_labels if label != -100)
            print(f"[COLLATED] live-label tokens: {live}/{len(batch_labels)} "
                  f"({100 * live / len(batch_labels):.1f}%)")
    print("=" * 70)
    print(
        "READ-OUT (hypothesis H): opener tokens ABSENT from input_ids -> TRL "
        "re-tokenized without the think block (fix: align eval to that, no "
        "retrain). Opener PRESENT but labels=-100 on it -> masking bug (fix "
        "masking, retrain A1). Opener present with live labels -> pathology "
        "is optimizer-side; escalate to training-dynamics audit."
    )
    return 0


def _dump_boundary(tokenizer, input_ids, labels, think_open, tag: str) -> None:
    try:
        pos = input_ids.index(think_open[0])
    except ValueError:
        print(f"[{tag}] !! '<think>' token NOT FOUND in input_ids — trainer "
              "tokenization dropped the think block entirely.")
        # show the assistant boundary instead
        im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        anchors = [i for i, t in enumerate(input_ids) if t == im_start]
        pos = anchors[-1] if anchors else max(0, len(input_ids) - 24)
    lo, hi = max(0, pos - 4), min(len(input_ids), pos + 12)
    print(f"[{tag}] window around position {pos}:")
    for i in range(lo, hi):
        piece = tokenizer.decode([input_ids[i]])
        label = labels[i] if labels is not None else "n/a"
        state = "LIVE" if isinstance(label, int) and label != -100 else (
            "MASK" if label == -100 else str(label))
        print(f"  pos {i:5d}  id {input_ids[i]:>7}  {piece!r:20}  label={state}")


if __name__ == "__main__":
    raise SystemExit(main())

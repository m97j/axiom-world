#!/usr/bin/env python
"""Diagnostic e08: split-brain audit — pipeline overfit test + checkpoint audit.

Context. e07 rejected hypothesis H: the trainer's processed tensors contain
the '<think>\\n\\n</think>\\n\\n' opener with LIVE labels (98.6% live overall).
We now face a hard contradiction: an opener repeated identically in every
record, under live loss for 2 epochs x 2000 records, is trivially learnable —
yet the saved adapter emits a FLAT GARBAGE distribution exactly there
(e05: top-1 '.Aggressive' p=0.013) while emitting perfect JSON afterwards.

Only two families of explanations remain:
  (I) Pipeline-dynamics bug: something in OUR training loop (loss/collator/
      precision) prevents learning at those positions, reproducibly.
  (J) Checkpoint-side fault: the training was fine but the SAVED/LOADED
      adapter is not the trained state — mis-scaled (alpha/r double
      application), partially saved, corrupted, or from the wrong step.

This script decides between them with two independent probes:

  PROBE 1 — overfit test (tiny same-vocab random model, CPU-ok):
      Build the REAL trainer via training.factory on 4 records and train
      ~60 steps. Then teacher-force the opener positions. A healthy
      pipeline overfits 4 records trivially; if opener accuracy stays at 0
      while JSON accuracy rises, hypothesis I is confirmed IN OUR CODE.

  PROBE 2 — checkpoint audit (no 8B download):
      Read adapter_config.json + safetensors weight stats from the run's
      final_adapter dir: r/alpha/target_modules vs experiment config,
      per-module norm distribution, NaN/Inf scan, and the implied scaling
      factor. Gross norm anomalies or config drift confirm hypothesis J.

Usage (Colab, project root):
  python scripts/diag_training_dynamics.py \
      --config configs/experiments/a1_playworld_sft.yaml \
      --prompt-file data/train/playworld_sft.jsonl \
      --adapter-dir runs/<run_id>/artifacts/final_adapter \
      --steps 60
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from axiom_world.core.config_loader import resolve


def probe_overfit(config, tokenizer, records, steps: int) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM

    repo = config.model.tokenizer_repo_id or config.model.repo_id
    tiny_config = AutoConfig.from_pretrained(repo, revision=config.model.revision)
    for name, value in {
        "num_hidden_layers": 2, "hidden_size": 128, "intermediate_size": 256,
        "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 64,
    }.items():
        if hasattr(tiny_config, name):
            setattr(tiny_config, name, value)
    # Transformers v5 strict config validation: layer_types must match
    # num_hidden_layers, and the checkpoint-save path validates the config
    # (this crashed e08's first run at the end-of-training save).
    layer_types = getattr(tiny_config, "layer_types", None)
    if layer_types:
        tiny_config.layer_types = list(layer_types)[: tiny_config.num_hidden_layers]
    model = AutoModelForCausalLM.from_config(tiny_config)

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in records])

    from axiom_world.training.factory import build_trainer

    # Force a short, fast overfit run without touching the recipe on disk.
    config = config.model_copy(deep=True)
    config.training.max_steps = steps
    config.training.num_train_epochs = None
    config.training.learning_rate = 5.0e-4
    config.training.per_device_batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.save_steps = 10_000_000

    with tempfile.TemporaryDirectory() as tmp:
        trainer = build_trainer(config, model, tokenizer, dataset, output_dir=tmp)
        # Belt-and-braces: disable checkpointing entirely for the probe; we
        # only need in-memory weights, and saving is where v5 validation bites.
        trainer.args.save_strategy = "no"
        if hasattr(trainer, "control"):
            trainer.control.should_save = False
        trainer.train()
        processed = trainer.train_dataset

        think_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        model.eval()
        opener_hits, opener_total, json_hits, json_total = 0, 0, 0, 0
        with torch.no_grad():
            for i in range(len(processed)):
                ids = processed[i]["input_ids"]
                pos = ids.index(think_id)
                tensor = torch.tensor([ids], device=model.device)
                preds = model(tensor).logits[0, :-1].argmax(dim=-1).tolist()
                targets = ids[1:]
                # opener = 4 tokens starting at <think>
                for j in range(pos - 1, pos + 3):
                    opener_total += 1
                    opener_hits += int(preds[j] == targets[j])
                for j in range(pos + 3, len(targets)):
                    json_total += 1
                    json_hits += int(preds[j] == targets[j])
        print(f"[PROBE1] after {steps} steps on {len(processed)} records:")
        print(f"[PROBE1]   opener accuracy : {opener_hits}/{opener_total} "
              f"({100 * opener_hits / max(1, opener_total):.1f}%)")
        print(f"[PROBE1]   json   accuracy : {json_hits}/{json_total} "
              f"({100 * json_hits / max(1, json_total):.1f}%)")
        if opener_total and opener_hits / opener_total < 0.5 <= json_hits / max(1, json_total):
            print("[PROBE1] => hypothesis I SUPPORTED: our pipeline fails to "
                  "learn the opener even in a trivial overfit.")
        else:
            print("[PROBE1] => pipeline learns the opener; hypothesis I "
                  "rejected for the training loop itself.")


def probe_checkpoint(config, adapter_dir: Path) -> None:
    import torch
    from safetensors.torch import load_file

    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        print(f"[PROBE2] !! {cfg_path} missing — cannot audit checkpoint.")
        return
    adapter_config = json.loads(cfg_path.read_text())
    r = adapter_config.get("r")
    alpha = adapter_config.get("lora_alpha")
    print(f"[PROBE2] adapter_config: r={r} alpha={alpha} "
          f"scaling={alpha / r if r else '?'} "
          f"target_modules={sorted(adapter_config.get('target_modules', []))}")
    expected = config.adapter
    drift = (r != expected.r) or (alpha != expected.alpha) or (
        sorted(adapter_config.get("target_modules", []))
        != sorted(expected.target_modules)
    )
    print(f"[PROBE2] config drift vs experiment yaml: {drift}")

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        print(f"[PROBE2] !! {weights_path} missing.")
        return
    state = load_file(str(weights_path))
    norms = []
    bad = 0
    for name, tensor in state.items():
        t = tensor.float()
        if not torch.isfinite(t).all():
            bad += 1
            print(f"[PROBE2] !! non-finite values in {name}")
        norms.append((float(t.norm()), name))
    norms.sort(reverse=True)
    values = [n for n, _ in norms]
    mean_norm = sum(values) / len(values)
    print(f"[PROBE2] tensors={len(values)}  non-finite={bad}  "
          f"norm mean={mean_norm:.3f}  max={values[0]:.3f}  min={values[-1]:.3f}")
    print("[PROBE2] top-5 norms:")
    for norm, name in norms[:5]:
        ratio = norm / mean_norm if mean_norm else 0
        flag = "  <-- OUTLIER" if ratio > 10 else ""
        print(f"    {norm:10.3f}  ({ratio:5.1f}x mean)  {name}{flag}")
    zero_b = sum(1 for n, name in norms if "lora_B" in name and n == 0.0)
    total_b = sum(1 for _, name in norms if "lora_B" in name)
    print(f"[PROBE2] all-zero lora_B tensors: {zero_b}/{total_b} "
          "(all zero would mean an UNTRAINED adapter saved by mistake)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-file", default="data/train/playworld_sft.jsonl")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config, _, _ = resolve(args.config, [])
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_repo_id or config.model.repo_id,
        revision=config.model.revision,
    )
    records = [
        json.loads(line)
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.num_samples]

    print("=" * 70)
    probe_overfit(config, tokenizer, records, args.steps)
    print("-" * 70)
    probe_checkpoint(config, Path(args.adapter_dir))
    print("=" * 70)
    print(
        "READ-OUT: PROBE1 failing to learn the opener -> hypothesis I "
        "(pipeline dynamics) — fix loop, retrain A1. PROBE1 healthy but "
        "PROBE2 showing drift/outliers/zeros -> hypothesis J (checkpoint "
        "fault) — fix save/load or scaling; retraining may be unnecessary "
        "if a good checkpoint exists. Both healthy -> the historical A1 run "
        "diverges from current code; re-run A1 under v0.3.x as the cleanest "
        "resolution."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""x12: attested in-process stop-logit probe — resolve the x10 vs x11 conflict.

Conflict to resolve.
  x10 (separate processes, one per arm): B1 and B2 adapter reports were
  BIT-IDENTICAL (every p_im_end, rank, top-5 equal).
  x11 (single process): the same two adapter dirs hold DIFFERENT weights
  (0/504 identical tensors) and produce DIFFERENT forward logits
  (l2_vs_previous_adapter = 672.5). Both cannot be true of the same disk
  state, so the x10 session must have probed something other than what x11
  later hashed — prime suspect: a STALE local runs/<id>/artifacts dir in the
  x10 Colab session (fetch_run skipping an existing dir), or a loading-path
  quirk. x12 removes every such ambiguity by doing, in ONE process, for each
  arm: (1) sha256 of adapter_model.safetensors ON DISK at probe time,
  (2) sha256 + L2 norm of the LoRA tensors ACTUALLY RESIDENT in the loaded
  PEFT model, (3) the exact x10 probe on the same SFT samples, then
  (4) cross-arm comparison of probe outputs.

Readout:
  - runtime hashes differ AND probe outputs differ -> x10 anomaly was a stale
    -dir/wiring artifact of that session; adopt x12 numbers and judge U1 on
    them.
  - runtime hashes differ BUT probe outputs identical -> loading-path bug
    (V3) inside build_for_inference / PEFT application; escalate to src/.
  - runtime hashes identical despite different disk files -> loader reads
    weights from somewhere other than the given dir (config/hub leakage).

Usage (repo root, GPU recommended):
  python scripts/x12_attested_probe.py \
      --config configs/experiments/b1_general_sft.yaml \
      --adapter-dirs runs/<b1-run>/artifacts/final_adapter \
                     runs/<b2-run>/artifacts/final_adapter \
      --labels b1 b2 \
      --sft-jsonl data/p1/p1_general_sft.jsonl \
      --num-samples 4 \
      --out runs/x12_attested_probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from x10_stop_logit_probe import probe_positions, render_train


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def live_lora_attestation(model) -> dict:
    """Hash + norm the LoRA tensors resident in the loaded model."""

    digest = hashlib.sha256()
    total_sq = 0.0
    n_tensors = 0
    for name, param in sorted(model.named_parameters()):
        if "lora_" in name:
            data = param.detach().float().cpu().contiguous()
            digest.update(name.encode())
            digest.update(data.numpy().tobytes())
            total_sq += float((data ** 2).sum())
            n_tensors += 1
    return {
        "live_lora_tensors": n_tensors,
        "live_lora_sha256": digest.hexdigest() if n_tensors else None,
        "live_lora_global_l2": round(total_sq ** 0.5, 6),
        "note_if_zero_tensors": None
        if n_tensors
        else "NO lora_ parameters found in loaded model — adapter NOT applied",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--sft-jsonl", default="data/p1/p1_general_sft.jsonl")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--out", default="runs/x12_attested_probe.json")
    args = parser.parse_args()

    import torch

    from axiom_world.core.config_loader import resolve
    from axiom_world.models.builder import build_for_inference

    labels = args.labels or [f"arm{i}" for i in range(len(args.adapter_dirs))]
    if len(labels) != len(args.adapter_dirs):
        parser.error("--labels must match --adapter-dirs")

    config, _, _ = resolve(args.config, [])

    rows = []
    with open(args.sft_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= args.num_samples:
                break
            clean_line = line.strip()
            if clean_line:
                rows.append(json.loads(clean_line))

    report: dict = {"config": args.config, "arms": []}
    probe_matrix: list[list[dict]] = []

    for label, adapter_dir in zip(labels, args.adapter_dirs, strict=True):
        dir_path = Path(adapter_dir)
        safetensors = list(dir_path.glob("*.safetensors"))
        arm: dict = {
            "label": label,
            "adapter_dir": adapter_dir,
            "disk_safetensors_sha256": sha256_file(safetensors[0]) if safetensors else None,
        }
        model, tokenizer = build_for_inference(config, adapter_dir=adapter_dir)
        model.eval()
        arm.update(live_lora_attestation(model))

        stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        probes = []
        for i, row in enumerate(rows):
            result = probe_positions(
                model, tokenizer, render_train(tokenizer, row["messages"]), stop_id
            )
            result["idx"] = i
            probes.append(result)
        arm["probes"] = probes
        probe_matrix.append(probes)
        report["arms"].append(arm)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # cross-arm comparison
    comparisons = []
    for i in range(len(report["arms"])):
        for j in range(i + 1, len(report["arms"])):
            a, b = report["arms"][i], report["arms"][j]
            probes_equal = all(
                pa.get("p_im_end") == pb.get("p_im_end")
                and pa.get("rank_im_end") == pb.get("rank_im_end")
                for pa, pb in zip(probe_matrix[i], probe_matrix[j], strict=True)
            )
            comparisons.append(
                {
                    "pair": [a["label"], b["label"]],
                    "disk_sha_equal": a["disk_safetensors_sha256"]
                    == b["disk_safetensors_sha256"],
                    "live_lora_sha_equal": a["live_lora_sha256"] == b["live_lora_sha256"],
                    "probe_outputs_equal": probes_equal,
                }
            )
    report["comparisons"] = comparisons

    verdicts = []
    for comp in comparisons:
        pair = "/".join(comp["pair"])
        if comp["live_lora_sha_equal"] and not comp["disk_sha_equal"]:
            verdicts.append(
                f"{pair}: LOADER LEAKAGE — different disk files but identical live "
                "weights; build_for_inference is not reading the given dir."
            )
        elif not comp["live_lora_sha_equal"] and comp["probe_outputs_equal"]:
            verdicts.append(
                f"{pair}: V3 CONFIRMED — different live weights yet identical probe "
                "outputs; adapter application path is degenerate."
            )
        elif not comp["live_lora_sha_equal"] and not comp["probe_outputs_equal"]:
            verdicts.append(
                f"{pair}: PROBE HEALTHY — arms differ end-to-end; the x10 identical-"
                "logits anomaly was a stale-dir/session artifact. Judge U1 on the "
                "x12 probe numbers."
            )
        else:
            verdicts.append(f"{pair}: identical on disk AND live — same adapter (V1/V2).")
    report["verdict"] = verdicts

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

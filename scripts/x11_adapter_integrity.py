#!/usr/bin/env python
"""x11: adapter integrity audit — explain the impossible B1 == B2 x10 result.

Context. x10 (stop-logit probe) returned BIT-IDENTICAL logit reports for the
B1 (curated) and B2 (rejection-sampled) adapters: every p_im_end, rank and
top-5 list matches to the last digit (mean_p_im_end = 0.000318 for both).
Two LoRA adapters trained on different corpora cannot produce identical
logits. Additionally, the adapter-arm distributions are NEAR-UNIFORM garbage
(top-1 p ~= 0.001-0.005, gibberish code/multilingual tokens) — an entropy
explosion echoing the A1 'flat garbage' signature (hypothesis-K era).

Hypothesis V (adapter identity/integrity):
  V1: the same adapter directory was loaded twice (cell wiring error), or
  V2: both training runs persisted identical adapter weights (training was a
      no-op or collapsed identically -> saved tensors are the same), or
  V3: adapter loading in build_for_inference degenerates the model the same
      way regardless of the loaded weights (e.g. dtype/scaling bug), which
      would also explain the near-uniform distributions.

Checks (CPU-only unless --forward-diff):
  A) file audit    : per-file sha256 of each adapter dir + comparison against
                     lineage.json output_adapter_sha256 when present.
  B) tensor audit  : load adapter_model.safetensors from each dir; report
                     per-tensor shapes and L2 norms of lora_A/lora_B; count
                     all-zero lora_B tensors (all-zero B => no-op adapter).
  C) pairwise diff : elementwise comparison across the given dirs — max abs
                     diff per tensor; verdict 'IDENTICAL' if every tensor is
                     bit-equal (V1/V2 confirmed).
  D) forward diff  : (--forward-diff, GPU) one fixed prompt through base and
                     each base+adapter; report logit L2 distance and entropy
                     at the last position (V3 signal: adapter entropy >> base).

Usage (repo root):
  python scripts/x11_adapter_integrity.py \
      --adapter-dirs runs/<b1-run>/artifacts/final_adapter \
                     runs/<b2-run>/artifacts/final_adapter \
      --config configs/experiments/b1_general_sft.yaml \
      --forward-diff \
      --out runs/x11_adapter_integrity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_files(adapter_dir: Path) -> dict:
    report: dict = {"adapter_dir": str(adapter_dir), "files": {}}
    if not adapter_dir.is_dir():
        report["error"] = "directory not found"
        return report
    for path in sorted(adapter_dir.rglob("*")):
        if path.is_file():
            report["files"][str(path.relative_to(adapter_dir))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    # lineage.json lives at <run>/artifacts/lineage.json (adapter dir sibling)
    lineage_path = adapter_dir.parent / "lineage.json"
    if lineage_path.is_file():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        report["lineage_output_adapter_sha256"] = lineage.get("output_adapter_sha256")
    return report


def load_tensors(adapter_dir: Path) -> dict:
    from safetensors.torch import load_file

    candidates = list(adapter_dir.glob("adapter_model.safetensors")) or list(
        adapter_dir.glob("*.safetensors")
    )
    if not candidates:
        raise FileNotFoundError(f"no .safetensors under {adapter_dir}")
    return load_file(str(candidates[0]))


def audit_tensors(tensors: dict) -> dict:
    import torch

    lora_a = [k for k in tensors if "lora_A" in k]
    lora_b = [k for k in tensors if "lora_B" in k]
    zero_b = [k for k in lora_b if not torch.any(tensors[k] != 0)]
    norms = {
        "lora_A_mean_l2": round(
            float(sum(tensors[k].float().norm() for k in lora_a) / max(len(lora_a), 1)), 6
        ),
        "lora_B_mean_l2": round(
            float(sum(tensors[k].float().norm() for k in lora_b) / max(len(lora_b), 1)), 6
        ),
    }
    return {
        "n_tensors": len(tensors),
        "n_lora_A": len(lora_a),
        "n_lora_B": len(lora_b),
        "n_lora_B_all_zero": len(zero_b),
        "norms": norms,
        "target_modules_sample": sorted({k.split(".lora_")[0].split(".")[-1] for k in lora_a})[:12],
    }


def pairwise_diff(t1: dict, t2: dict) -> dict:

    keys1, keys2 = set(t1), set(t2)
    common = sorted(keys1 & keys2)
    max_abs = 0.0
    n_identical = 0
    for key in common:
        if t1[key].shape != t2[key].shape:
            return {"verdict": "SHAPE-MISMATCH", "key": key}
        diff = float((t1[key].float() - t2[key].float()).abs().max())
        max_abs = max(max_abs, diff)
        if diff == 0.0:
            n_identical += 1
    verdict = (
        "IDENTICAL (bit-equal tensors -> same adapter weights in both dirs; "
        "hypothesis V1/V2 confirmed)"
        if common and n_identical == len(common)
        else "DIFFERENT (weights differ -> identical x10 logits must come from "
        "the probe/loading path; investigate V3 / cell wiring)"
    )
    return {
        "common_tensors": len(common),
        "only_in_first": len(keys1 - keys2),
        "only_in_second": len(keys2 - keys1),
        "identical_tensors": n_identical,
        "max_abs_diff": max_abs,
        "verdict": verdict,
    }


def forward_diff(config_path: str, adapter_dirs: list[Path]) -> dict:
    import torch

    from axiom_world.core.config_loader import resolve
    from axiom_world.models.builder import build_for_inference

    config, _, _ = resolve(config_path, [])
    prompt_messages = [{"role": "user", "content": "What is 2 + 2? Answer with a number."}]

    def last_logits(adapter_dir: Path | None):
        model, tokenizer = build_for_inference(config, adapter_dir=adapter_dir)
        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float().cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        probs = torch.softmax(logits, dim=-1)
        entropy = float(-(probs * (probs + 1e-12).log()).sum())
        return logits, entropy

    base_logits, base_entropy = last_logits(None)
    report: dict = {"base_entropy_nats": round(base_entropy, 3), "arms": []}
    prev = None
    for adapter_dir in adapter_dirs:
        logits, entropy = last_logits(adapter_dir)
        entry = {
            "adapter_dir": str(adapter_dir),
            "entropy_nats": round(entropy, 3),
            "l2_vs_base": round(float((logits - base_logits).norm()), 3),
        }
        if prev is not None:
            entry["l2_vs_previous_adapter"] = round(float((logits - prev).norm()), 3)
        prev = logits
        report["arms"].append(entry)
    report["note"] = (
        "adapter entropy >> base entropy indicates the loaded adapter flattens "
        "the distribution (V3); l2_vs_previous_adapter == 0 reproduces the "
        "identical-logits anomaly at the weight-application level."
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dirs", nargs="+", required=True)
    parser.add_argument("--config", default=None, help="required with --forward-diff")
    parser.add_argument("--forward-diff", action="store_true")
    parser.add_argument("--out", default="runs/x11_adapter_integrity.json")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.adapter_dirs]
    out: dict = {"file_audit": [audit_files(d) for d in dirs]}

    tensor_sets = []
    out["tensor_audit"] = []
    for d in dirs:
        try:
            tensors = load_tensors(d)
            tensor_sets.append(tensors)
            out["tensor_audit"].append({"adapter_dir": str(d), **audit_tensors(tensors)})
        except Exception as exc:  # noqa: BLE001
            tensor_sets.append(None)
            out["tensor_audit"].append(
                {"adapter_dir": str(d), "error": f"{type(exc).__name__}: {exc}"}
            )

    out["pairwise"] = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            if tensor_sets[i] is not None and tensor_sets[j] is not None:
                out["pairwise"].append(
                    {
                        "first": str(dirs[i]),
                        "second": str(dirs[j]),
                        **pairwise_diff(tensor_sets[i], tensor_sets[j]),
                    }
                )

    if args.forward_diff:
        if not args.config:
            parser.error("--forward-diff requires --config")
        out["forward_diff"] = forward_diff(args.config, dirs)

    # top-level summary verdict
    identical_pairs = [p for p in out["pairwise"] if p.get("verdict", "").startswith("IDENTICAL")]
    zero_b = [t for t in out["tensor_audit"] if t.get("n_lora_B_all_zero", 0) > 0]
    if identical_pairs:
        out["verdict"] = (
            "HYPOTHESIS-V CONFIRMED (identity): the audited adapter dirs contain "
            "bit-identical weights. Check b_b1_train/b_b2_train run configs and "
            "fetch_run wiring; one arm's training or persistence overwrote or "
            "duplicated the other."
        )
    elif zero_b:
        out["verdict"] = (
            "NO-OP ADAPTER: lora_B tensors are all-zero in at least one dir — the "
            "saved adapter is untrained (checkpoint saved before any step?)."
        )
    else:
        out["verdict"] = (
            "Adapters differ on disk. If x10 still reports identical logits, the "
            "duplication happened in the probe/loading path (V3) — inspect the "
            "notebook cell wiring and build_for_inference adapter application, "
            "and compare forward_diff l2_vs_previous_adapter."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

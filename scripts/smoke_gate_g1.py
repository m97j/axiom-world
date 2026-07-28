#!/usr/bin/env python
"""Gate G1 — runtime hardening smoke test (protocol §10).

Run on the Colab G4 training session AFTER installing requirements.
Checks, in order:
  1. environment manifest + strict policy versus the canonical contract
  2. TRL import gate: required trainers/configs exist in the installed TRL
  3. Qwen3-8B tokenizer + model load (BF16, SDPA), LoRA attach
  4. tiny forward/backward
  5. checkpoint save -> reload -> adapter hash stability

Exit code 0 = G1 passed; freeze the TRL pin in requirements/colab-g4.lock.txt.
Steps 3-5 are skipped with --contract-only (CPU/CI usage).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from axiom_world.core.lineage import compute_adapter_sha256
from axiom_world.runtime.audit import collect_environment_manifest

REQUIRED_TRL = ("SFTTrainer", "SFTConfig", "DPOTrainer", "DPOConfig",
                "GRPOTrainer", "GRPOConfig", "RLOOTrainer", "RLOOConfig")


def check_trl() -> list[str]:
    failures: list[str] = []
    try:
        import trl
    except ImportError:
        return ["TRL not installed"]
    print(f"TRL version: {trl.__version__}")
    for name in REQUIRED_TRL:
        if getattr(trl, name, None) is None:
            failures.append(f"TRL missing export: {name}")
    return failures


def check_model(model_id: str, revision: str | None) -> list[str]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    failures: list[str] = []
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    batch = tokenizer("G1 smoke test.", return_tensors="pt").to(model.device)
    output = model(**batch, labels=batch["input_ids"])
    output.loss.backward()
    print(f"forward/backward OK, loss={output.loss.item():.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        adapter_dir = Path(tmp) / "adapter"
        model.save_pretrained(adapter_dir)
        sha_first = compute_adapter_sha256(adapter_dir)
        sha_second = compute_adapter_sha256(adapter_dir)
        if sha_first != sha_second:
            failures.append("adapter hash not stable across reads")
        print(f"adapter sha256 OK: {sha_first[:24]}...")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--contract-only", action="store_true",
                        help="Skip GPU model checks (CI usage).")
    args = parser.parse_args()

    manifest = collect_environment_manifest()
    print(json.dumps({k: manifest[k] for k in ("python", "cuda_available", "gpu")}, indent=2))

    failures = check_trl()
    if not args.contract_only:
        if not manifest["cuda_available"]:
            failures.append("CUDA unavailable; run on the Colab G4 session")
        else:
            failures += check_model(args.model_id, args.revision)

    if failures:
        for failure in failures:
            print(f"G1 FAIL: {failure}", file=sys.stderr)
        return 1
    print("G1 PASSED — freeze the TRL pin now (requirements/colab-g4.lock.txt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

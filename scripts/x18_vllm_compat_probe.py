#!/usr/bin/env python
"""x18: vLLM <-> Colab-G4 ABI compatibility probe (2026-08-16 incident).

Context. Installing `vllm>=0.10,<1.0` (resolved to 0.27.1) into the training
runtime replaced torch 2.11.0+cu128 with torch 2.13.0+cu13x while torchaudio
stayed at 2.11.0+cu128 -> "PyTorch and TorchAudio were compiled with
different CUDA versions" -> transformers could not import Qwen3ForCausalLM.
Root cause: the lock file was a floating range and the common header
installed it blindly alongside training (contradicting the lock file's own
"never alongside training" rule).

This probe finds, WITHOUT mutating the runtime, the newest vLLM version whose
dependency resolution keeps the image-owned ABI layer (torch / torchvision /
torchaudio / triton / nvidia-*) untouched, then optionally installs exactly
that version and smoke-tests it.

Phases:
  1 (default, SAFE — no changes): snapshot the ABI baseline, then for each
    candidate vLLM version (newest first) run `pip install --dry-run
    --report` and inspect the resolver report. A candidate PASSES iff no
    ABI-critical package would be installed/changed. Prints the newest
    passing candidate -> pin THAT in requirements/vllm.lock.txt.
  2 (--install, MUTATES runtime): `pip install vllm==<newest passing>`, then
    verify baseline unchanged, `import vllm`, `import torchaudio`,
    Qwen3ForCausalLM importable, and (--model) a tiny LLM(...) generation
    benchmark for rollout-throughput evidence.

Usage (fresh Colab G4 runtime, AFTER colab-g4.lock.txt install):
  python scripts/x18_vllm_compat_probe.py --out runs/x18_vllm_probe.json
  python scripts/x18_vllm_compat_probe.py --install \
      --model Qwen/Qwen3-8B --out runs/x18_vllm_probe_install.json

Exit 0 iff the requested phase fully passed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

# Image-owned ABI layer (requirements/colab-g4.lock.txt rule): any resolver
# plan that touches one of these prefixes disqualifies the candidate.
ABI_PREFIXES = ("torch", "torchvision", "torchaudio", "triton", "nvidia-")


def _baseline() -> dict[str, Any]:
    snap: dict[str, Any] = {}
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["torch_cuda_build"] = torch.version.cuda
        snap["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap["gpu"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            snap["sm"] = f"{major}.{minor}"
    except Exception as exc:  # noqa: BLE001
        snap["torch_error"] = repr(exc)
    for pkg in ("torchvision", "torchaudio", "transformers", "trl", "peft"):
        try:
            snap[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            snap[pkg] = None
    return snap


def _candidate_versions(limit: int) -> list[str]:
    """Newest-first published vLLM versions, via pip's resolver error trick."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", "vllm"],
        capture_output=True, text=True, check=False,
    )
    text = proc.stdout + proc.stderr
    for line in text.splitlines():
        if "Available versions:" in line:
            versions = [v.strip() for v in line.split(":", 1)[1].split(",")]
            return versions[:limit]
    raise SystemExit(f"could not enumerate vLLM versions:\n{text[-2000:]}")


def _dry_run_report(version: str) -> tuple[list[str], str | None]:
    """Return (ABI packages the resolver would install/change, error)."""
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet",
             f"vllm=={version}", "--report", str(report_path)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0 or not report_path.exists():
            return [], (proc.stderr or proc.stdout)[-1500:]
        report = json.loads(report_path.read_text())
    touched = []
    for item in report.get("install", []):
        name = (item.get("metadata", {}).get("name") or "").lower()
        ver = item.get("metadata", {}).get("version")
        if name.startswith(ABI_PREFIXES):
            touched.append(f"{name}=={ver}")
    return touched, None


def _verify_stack(report: dict[str, Any], model: str | None,
                  gpu_mem: float, n_prompts: int) -> bool:
    ok = True
    post = _baseline()
    report["post_install_baseline"] = post
    drift = {
        k: (report["baseline"].get(k), post.get(k))
        for k in ("torch", "torch_cuda_build", "torchvision", "torchaudio")
        if report["baseline"].get(k) != post.get(k)
    }
    report["abi_drift"] = drift
    ok &= not drift

    try:
        import vllm  # noqa: F401

        report["vllm_imported"] = metadata.version("vllm")
    except Exception as exc:  # noqa: BLE001
        report["vllm_imported"] = f"FAILED: {exc!r}"
        ok = False
    try:
        import torchaudio  # noqa: F401
        from transformers import AutoConfig  # noqa: F401
        from transformers.models.qwen3 import Qwen3ForCausalLM  # noqa: F401

        report["qwen3_import"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["qwen3_import"] = f"FAILED: {exc!r}"
        ok = False

    if model and ok:
        from vllm import LLM, SamplingParams

        t0 = time.time()
        llm = LLM(model=model, gpu_memory_utilization=gpu_mem,
                  max_model_len=4096)
        load_s = time.time() - t0
        prompts = ["You are in a grid world. Plan actions to reach loc-3."] * n_prompts
        t0 = time.time()
        outs = llm.generate(prompts, SamplingParams(max_tokens=256, temperature=1.0))
        gen_s = time.time() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        report["benchmark"] = {
            "model": model, "load_seconds": round(load_s, 1),
            "n_prompts": n_prompts, "generate_seconds": round(gen_s, 1),
            "completion_tokens": toks,
            "tokens_per_second": round(toks / max(gen_s, 1e-9), 1),
        }
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-candidates", type=int, default=12,
                        help="Newest-first vLLM versions to dry-run in phase 1.")
    parser.add_argument("--install", action="store_true",
                        help="Phase 2: actually install the newest passing "
                             "candidate and smoke-test (MUTATES the runtime).")
    parser.add_argument("--pin", default=None,
                        help="Skip enumeration and probe/install exactly this version.")
    parser.add_argument("--model", default=None,
                        help="Optional model id for a tiny generation benchmark "
                             "after --install (e.g. Qwen/Qwen3-8B).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--bench-prompts", type=int, default=16)
    args = parser.parse_args()

    report: dict[str, Any] = {"phase": "install" if args.install else "dry_run",
                              "baseline": _baseline(), "candidates": []}
    candidates = [args.pin] if args.pin else _candidate_versions(args.max_candidates)

    chosen: str | None = None
    for version in candidates:
        touched, err = _dry_run_report(version)
        entry = {"version": version, "abi_packages_touched": touched,
                 "resolver_error": err}
        report["candidates"].append(entry)
        print(f"vllm=={version}: "
              f"{'RESOLVER-FAIL' if err else ('ABI-TOUCH ' + str(touched) if touched else 'CLEAN')}")
        if not err and not touched:
            chosen = version
            break  # newest-first: first clean candidate is the answer

    report["newest_abi_clean_candidate"] = chosen
    ok = chosen is not None

    if ok and args.install:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", f"vllm=={chosen}"],
            check=False,
        )
        report["install_returncode"] = proc.returncode
        ok = proc.returncode == 0 and _verify_stack(
            report, args.model, args.gpu_memory_utilization, args.bench_prompts)

    report["verdict"] = "PASS" if ok else "FAIL"
    if chosen and ok:
        report["lock_line"] = f"vllm=={chosen}"
        print(f"\n=> pin in requirements/vllm.lock.txt:  vllm=={chosen}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("phase", "newest_abi_clean_candidate", "verdict")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a BF16 evaluation candidate from a verified FP32 release; never publish."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from axiom_world.models.champion_release import inventory, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def derive(source: Path, output: Path, device: str):
    source, output = source.resolve(), output.resolve()
    receipt = read(source / "verified.json")
    original = source / "model"
    if (receipt.get("status") != "verified" or receipt["verification"]["dtype"] != "float32"
            or not receipt["verification"]["merge"]["passed"]
            or not receipt["verification"]["standalone"]["passed"]
            or inventory(original) != receipt["files"]):
        raise ValueError("Need unchanged verified FP32 source before creating any output")
    source_request = read(source / "verification/request.json")
    probes_path = original / "provenance/merge_probes.json"
    if probes_path.is_file():
        probes = read(probes_path)
        if probes != source_request["probes"]:
            raise ValueError("Source probes differ from verified payload")
    else:
        raise ValueError("Verified payload lacks bound probes")
    output.mkdir(parents=True, exist_ok=False)
    stage, scratch = output / "model", output / "verification"
    stage.mkdir()
    scratch.mkdir()
    for path in original.iterdir():
        if path.name in ("config.json", "generation_config.json", "README.md", "model.safetensors.index.json"):
            continue
        if path.suffix == ".safetensors":
            continue
        if path.is_dir():
            shutil.copytree(path, stage / path.name)
        else:
            shutil.copy2(path, stage / path.name)
    write_json(stage / "provenance/source_fp32_manifest.json", read(original / "provenance/manifest.json"))
    (stage / "README.md").write_text(
        "# BF16 derived evaluation candidate\n\nConverted from verified FP32 merged weights.\n"
        "Precision equivalence is NOT established. Task comparison and release review are required.\n",
        encoding="utf-8")
    request = {"source_model": str(original), "stage": str(stage), "scratch": str(scratch),
               "adapter": str(stage / "adapter"), "probes": probes, "dtype": "bfloat16", "device": device}
    write_json(scratch / "request.json", request)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    for mode in ("cast", "reload"):
        subprocess.run([sys.executable, "-m", "axiom_world.models.champion_release", mode,
                        str(scratch / "request.json")], check=True, env=env)
    if inventory(original) != receipt["files"]:
        raise ValueError("FP32 source changed during conversion")
    verification = {"dtype": "bfloat16", "standalone": read(scratch / "standalone.json"),
                    "fp32_vs_bf16": read(scratch / "cast_drift.json"),
                    "scope": "Serialization correctness only; task comparison pending"}
    manifest = {"method": "Verified FP32 merged weights cast to BF16 with native non-persistent buffers preserved; not a new LoRA merge",
                "source_receipt_sha256": hashlib.sha256((source / "verified.json").read_bytes()).hexdigest(),
                "source_inventory": receipt["files"], "verification": verification}
    write_json(stage / "provenance/manifest.json", manifest)
    write_json(output / "verified.json", {"status": "verified_candidate", "verification": verification,
               "source_receipt_sha256": manifest["source_receipt_sha256"], "files": inventory(stage)})
    print(f"BF16_CANDIDATE={output}; not publication-ready; FP32 source preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    derive(args.source, args.output, args.device)

#!/usr/bin/env python
"""Prepare, verify, then explicitly publish the protocol-v1 standalone champion.

No flag / --dry-run only checks source identity. --prepare writes a NEW local
release directory; --publish consumes its hash-bound receipt without remerging.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from axiom_world.core.fingerprints import fingerprint_file
from axiom_world.core.lineage import compute_adapter_sha256
from axiom_world.integrations import hf_sync
from axiom_world.models.champion_release import inventory, run_workers, write_json

CHAMPION_RUN = "20260814-023603--b4v2-playworld-sft-from-p1--s42--c56ed2"
CHAMPION_SHA = "sha256:d4fcacddf21f758cdab904845ebdfee1eefde309c0edb6205bac64d5f07c76c8"
BASE = "Qwen/Qwen3-8B-Base"
BASE_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
SOURCE_REPO = "m97j/aw-runs-b4"
SOURCE_REVISION = "ef9a32095d5c4ed10fe4ab9b5914b320a89dd024"
TARGET_REPO = "m97j/aw-qwen3-8b-v1"
LEGACY_REVISION = "f33d2d16125e89eb38d4b668a2c20a6929ad3784"
IDENTITY_FILES = ("adapter_config.json", "adapter_model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
PROVENANCE_FILES = ("lineage.json", "resolved_config.yaml", "run_card.json")


def validate_source(workspace: Path) -> Path:
    artifacts = workspace / "runs" / CHAMPION_RUN / "artifacts"
    adapter = artifacts / "final_adapter"
    for name in (*IDENTITY_FILES, *TOKENIZER_FILES):
        if not (adapter / name).is_file() or (adapter / name).is_symlink():
            raise ValueError(f"Missing/nonregular champion file: {adapter / name}; run fetch_run first")
    for name in PROVENANCE_FILES:
        if not (artifacts / name).is_file():
            raise ValueError(f"Missing provenance: {name}")
    lineage = json.loads((artifacts / "lineage.json").read_text(encoding="utf-8"))
    expected = {"run_id": CHAMPION_RUN, "base_model_repo_id": BASE,
                "base_model_revision": BASE_REVISION, "output_adapter_sha256": CHAMPION_SHA}
    if any(lineage.get(k) != v for k, v in expected.items()):
        raise ValueError("Champion lineage differs from the v1 record")
    if compute_adapter_sha256(adapter) != CHAMPION_SHA:
        raise ValueError("Champion adapter identity mismatch")
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if (config.get("base_model_name_or_path") != BASE or config.get("peft_type") != "LORA"
            or set(config.get("modules_to_save") or []) != {"lm_head", "embed_tokens"}
            or config.get("revision") not in (None, BASE_REVISION)):
        raise ValueError("Unexpected champion adapter contract")
    return artifacts


def remote_plan(adapter: Path) -> dict:
    head = hf_sync.repository_revision(TARGET_REPO)
    if head != LEGACY_REVISION:
        raise ValueError("Target changed since reviewed adapter-only release; review before migrating")
    remote = hf_sync.release_inventory(TARGET_REPO, head)
    for name in (*IDENTITY_FILES, *TOKENIZER_FILES):
        info = remote.get(name)
        local = adapter / name
        if info is None or info["size"] != local.stat().st_size:
            raise ValueError(f"Published adapter/tokenizer size mismatch: {name}")
        digest = info["sha256"]
        if digest is None:
            digest = fingerprint_file(hf_sync.download_file(TARGET_REPO, name, revision=head))
        if fingerprint_file(local) != digest:
            raise ValueError(f"Published adapter/tokenizer digest mismatch: {name}")
    if any(name.startswith("model") and name.endswith(".safetensors") for name in remote):
        raise ValueError("Target already has standalone weights")
    return {"target_repo": TARGET_REPO, "expected_head": head,
            "legacy_revision": LEGACY_REVISION, "legacy_tag": "protocol-v1-adapter",
            "delete_paths": list(IDENTITY_FILES), "history_rewrite": False}


def prepare(args) -> None:
    artifacts = validate_source(args.workspace)
    adapter = artifacts / "final_adapter"
    plan = remote_plan(adapter)
    source = hf_sync.release_inventory(SOURCE_REPO, SOURCE_REVISION)
    for local in [*(adapter / n for n in (*IDENTITY_FILES, *TOKENIZER_FILES)),
                  *(artifacts / n for n in PROVENANCE_FILES)]:
        remote_path = "artifacts/" + local.relative_to(artifacts).as_posix()
        info = source.get(remote_path)
        if info is None or info["size"] != local.stat().st_size:
            raise ValueError(f"Pinned source differs: {remote_path}")
        digest = info["sha256"] or fingerprint_file(hf_sync.download_file(
            SOURCE_REPO, remote_path, revision=SOURCE_REVISION))
        if fingerprint_file(local) != digest:
            raise ValueError(f"Pinned source digest differs: {remote_path}")
    if not args.card.is_file():
        raise ValueError(f"Missing model card: {args.card}")
    probes = json.loads(args.probes.read_text(encoding="utf-8"))
    if not isinstance(probes, list) or len(probes) < 32:
        raise ValueError("At least 32 fixed probes required")
    if args.output.exists():
        raise ValueError("Output exists; preserve it and choose a NEW --output")
    args.output.mkdir(parents=True)
    stage, scratch = args.output / "model", args.output / "verification"
    stage.mkdir()
    (stage / "adapter").mkdir()
    (stage / "provenance").mkdir()
    for name in (*IDENTITY_FILES, *TOKENIZER_FILES):
        shutil.copy2(adapter / name, stage / "adapter" / name)
    for name in TOKENIZER_FILES:
        shutil.copy2(adapter / name, stage / name)
    for name in PROVENANCE_FILES:
        shutil.copy2(artifacts / name, stage / "provenance" / name)
    base_files = hf_sync.release_inventory(BASE, BASE_REVISION)
    shutil.copy2(Path(__file__).resolve().parents[3] / "licenses/Apache-2.0.txt", stage / "LICENSE")
    shutil.copy2(hf_sync.download_file(BASE, "README.md", revision=BASE_REVISION),
                 stage / "provenance" / "base_model_card.md")
    for name in ("NOTICE",):
        if name in base_files:
            shutil.copy2(hf_sync.download_file(BASE, name, revision=BASE_REVISION), stage / name)
    shutil.copy2(args.card, stage / "README.md")
    (stage / "provenance" / "MODIFICATIONS.md").write_text(
        "Derived from Qwen/Qwen3-8B-Base (Qwen Team), Apache-2.0.\n"
        "Modified by Axiom-World Protocol v1 LoRA training, including saved embeddings\n"
        "and lm_head; this release merges that adapter into BF16 base weights.\n"
        "See manifest.json for exact sources and verification.\n", encoding="utf-8")
    (stage / "adapter" / "README.md").write_text(
        "# Original v1 champion adapter\n\nExact training artifact; use the pinned base revision "
        "in ../provenance/manifest.json and PEFT subfolder='adapter'.\n", encoding="utf-8")
    write_json(args.output / "plan.json", plan)
    request = {"base": BASE, "revision": BASE_REVISION, "adapter": str(adapter.resolve()),
               "stage": str(stage.resolve()), "scratch": str(scratch.resolve()),
               "device": args.device, "probes": probes}
    verification = run_workers(request)
    # Bind source bytes again after workers, before issuing a successful receipt.
    validate_source(args.workspace)
    if compute_adapter_sha256(stage / "adapter") != CHAMPION_SHA:
        raise ValueError("Staged adapter changed")
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "transformers", "peft", "accelerate", "safetensors", "huggingface_hub"]}
    manifest = {"protocol": "v1", "champion_run": CHAMPION_RUN,
                "adapter_identity_sha256": CHAMPION_SHA, "base_model": BASE,
                "base_revision": BASE_REVISION, "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION, "legacy_adapter_revision": LEGACY_REVISION,
                "method": "peft.merge_and_unload(safe_merge=True)", "dtype": "bfloat16",
                "max_shard_size": "5GB", "versions": versions, "verification": verification,
                "release_code_sha256": {
                    "publish_champion.py": fingerprint_file(Path(__file__)),
                    "champion_release.py": fingerprint_file(Path(run_workers.__code__.co_filename)),
                    "hf_sync.py": fingerprint_file(Path(hf_sync.__file__))},
                "probe_sha256": fingerprint_file(args.probes),
                "scope": "Engineering export check; original benchmark scores belong to adapter runs",
                "generation": {"do_sample": False, "eos": ["<|endoftext|>", "<|im_end|>"]}}
    write_json(stage / "provenance" / "manifest.json", manifest)
    shutil.copy2(args.probes, stage / "provenance" / "merge_probes.json")
    receipt = {"status": "verified", "plan": plan, "verification": verification,
               "files": inventory(stage)}
    write_json(args.output / "verified.json", receipt)
    print(f"VERIFIED_LOCAL_RELEASE={args.output}; HF has not been changed")


def publish(output: Path, *, execute: bool) -> None:
    receipt = json.loads((output / "verified.json").read_text(encoding="utf-8"))
    stage = output / "model"
    if (receipt.get("status") != "verified" or inventory(stage) != receipt["files"]
            or not receipt["verification"]["merge"]["passed"]
            or not receipt["verification"]["standalone"]["passed"]):
        raise ValueError("Unverified or changed release payload")
    expected_plan = remote_plan(stage / "adapter")
    if receipt["plan"] != expected_plan:
        raise ValueError("Migration plan changed")
    print(json.dumps(expected_plan, indent=2))
    if not execute:
        print("Dry run: verified payload; no remote writes")
        return
    marker = output / "upload_started.json"
    # An uncertain network response must never silently trigger another mutation.
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(expected_plan, stream)
    revision = hf_sync.commit_model_release(
        repo_id=TARGET_REPO, stage=stage, expected_head=expected_plan["expected_head"],
        legacy_revision=LEGACY_REVISION, delete_paths=expected_plan["delete_paths"],
        legacy_tag="protocol-v1-adapter",
        commit_message="Publish verified BF16 v1 champion and preserve adapter")
    write_json(output / "uploaded.json", {"revision": revision, "repo": TARGET_REPO})
    remote = hf_sync.release_inventory(TARGET_REPO, revision)
    for file in receipt["files"]:
        info = remote.get(file["path"])
        if info is None or info["size"] != file["size"]:
            raise ValueError(f"Remote file missing/size mismatch: {file['path']}")
        digest = info["sha256"] or fingerprint_file(hf_sync.download_file(
            TARGET_REPO, file["path"], revision=revision))
        if digest != file["sha256"]:
            raise ValueError(f"Remote digest mismatch: {file['path']}")
    if any(name in remote for name in IDENTITY_FILES):
        raise ValueError("Legacy root adapter files survived publication")
    write_json(output / "published.json", {"repo": TARGET_REPO, "revision": revision,
               "verification": "Hub metadata plus small-file downloads; local offline reload passed"})
    print(f"PUBLISHED=https://huggingface.co/{TARGET_REPO}/tree/{revision}")


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/champion-v1-release"))
    parser.add_argument("--card", type=Path, default=root / "hf_cards/v1/model_card_aw-qwen3-8b-v1.md")
    parser.add_argument("--probes", type=Path, default=root / "configs/releases/v1_merge_probes.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    elif args.publish or (args.output / "verified.json").is_file():
        publish(args.output, execute=args.publish)
    else:
        artifacts = validate_source(args.workspace)
        print(f"Champion identity verified: {CHAMPION_SHA}")
        print(json.dumps(remote_plan(artifacts / "final_adapter"), indent=2))
        print("Dry run: no model allocation, merge, tag, deletion or upload")
    return 0


if __name__ == "__main__":
    sys.exit(main())

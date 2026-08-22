#!/usr/bin/env python
"""Publish the protocol-v1 champion adapter to the curated public repo.

Fetches the champion run-of-record (sha-verified via fetch_run), verifies the
adapter hash against lineage, then uploads a curated set of files to
`m97j/aw-qwen3-8b-v1` together with the model card and provenance artifacts.

Usage:
  python scripts/fetch_run.py --repo m97j/aw-runs-b4 \
      --run-id 20260814-023603--b4v2-playworld-sft-from-p1--s42--c56ed2
  python scripts/publish_champion.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from axiom_world.core.lineage import compute_adapter_sha256

CHAMPION_RUN = "20260814-023603--b4v2-playworld-sft-from-p1--s42--c56ed2"
CHAMPION_EVAL = "20260814-032546--eval-playworld--s42--7308ee"
TARGET_REPO = "m97j/aw-qwen3-8b-v1"

ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
]
PROVENANCE_FILES = ["lineage.json", "resolved_config.yaml", "run_card.json"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--card", default="hf_cards/model_card_aw-qwen3-8b-v1.md")
    args = parser.parse_args()

    artifacts = Path("runs") / CHAMPION_RUN / "artifacts"
    adapter = artifacts / "final_adapter"
    assert adapter.is_dir(), f"champion not materialized — run fetch_run first ({adapter})"

    lineage = json.loads((artifacts / "lineage.json").read_text())
    expected = lineage["output_adapter_sha256"]
    actual = compute_adapter_sha256(adapter)
    actual = actual if actual.startswith("sha256:") else f"sha256:{actual}"
    assert actual == expected, f"ADAPTER INTEGRITY FAILURE: {actual} != {expected}"
    print(f"adapter verified: {actual}")

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for name in ADAPTER_FILES:
            src = adapter / name
            assert src.is_file(), f"missing adapter file: {src}"
            shutil.copy2(src, stage / name)
        prov = stage / "provenance"
        prov.mkdir()
        for name in PROVENANCE_FILES:
            src = artifacts / name
            if src.is_file():
                shutil.copy2(src, prov / name)
        eval_summary = Path("runs") / CHAMPION_EVAL / "artifacts" / "evaluation_summary.json"
        if eval_summary.is_file():
            shutil.copy2(eval_summary, prov / "evaluation_summary_s42.json")
        card = Path(args.card)
        assert card.is_file(), f"model card not found: {card}"
        shutil.copy2(card, stage / "README.md")

        staged = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
        print("staged files:", *staged, sep="\n  ")
        if args.dry_run:
            print("dry run — nothing uploaded")
            return 0

        api = HfApi()
        api.create_repo(TARGET_REPO, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=TARGET_REPO,
            repo_type="model",
            folder_path=str(stage),
            commit_message=(
                f"Publish protocol-v1 champion (B4v2) — run {CHAMPION_RUN}, "
                f"adapter {expected}"
            ),
        )
        print(f"uploaded -> hf://{TARGET_REPO} (created private; flip to public in settings "
              f"after review)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

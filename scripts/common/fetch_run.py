#!/usr/bin/env python
"""Materialize a persisted run from the Hub into the local workspace.

Read-path counterpart of the write-path sync in run_experiment.py. Colab
sessions are ephemeral; when a later stage (evaluation, Track-B lineage)
needs an earlier run's final adapter, this script restores it and — unlike a
bare `hf download` — verifies the adapter's sha256 against the persisted
lineage.json so the provenance chain survives session boundaries.

Resolution order (standard lazy-materialization pattern):
  1. local runs/<run_id>/artifacts/final_adapter  -> reuse (verify hash)
  2. hf://<repo>/artifacts                        -> download (verify hash)
  3. neither                                      -> fail loudly with guidance

Usage:
  python scripts/common/fetch_run.py --repo m97j/aw-runs-a1 \
      --run-id 20260729-145835--a1-playworld-sft--s42--1eb4c7
Prints ADAPTER_DIR=<path> on success for notebook consumption.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from axiom_world.core.lineage import compute_adapter_sha256


def _verify(adapter_dir: Path, lineage_path: Path, run_id: str, expected_sha256: str | None = None) -> str:
    lineage = json.loads(lineage_path.read_text())
    if lineage.get("run_id") != run_id:
        raise SystemExit("REPO/RUN MISMATCH: local or fetched lineage has a different run_id")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / name).is_file() or (adapter_dir / name).is_symlink():
            raise SystemExit(f"Missing/nonregular adapter identity file: {name}")
    expected = lineage["output_adapter_sha256"]
    if expected_sha256 is not None and expected != expected_sha256:
        raise SystemExit("Requested champion SHA differs from lineage")
    actual = compute_adapter_sha256(adapter_dir)
    if actual != expected:
        raise SystemExit(
            f"INTEGRITY FAILURE: adapter sha256 mismatch for {adapter_dir}\n"
            f"  expected (lineage.json): {expected}\n"
            f"  actual   (local files) : {actual}\n"
            "Do not evaluate this adapter; re-fetch or re-run the experiment."
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HF model repo the run was synced to (user/name).")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--kind", choices=["adapter", "eval"], default="adapter",
        help="'adapter': training run persisted at the repo ROOT under "
        "artifacts/ (verified against lineage.json). 'eval': evaluation run "
        "persisted under runs/<run_id>/ by run_evaluation.py --hf-sync-repo.",
    )
    parser.add_argument("--revision", help="Exact HF commit recommended; a ref is resolved once before download")
    parser.add_argument("--expected-sha256", help="Expected sha256: adapter identity digest")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id) is None:
        parser.error("run-id must be a single safe path component")
    if args.expected_sha256 and re.fullmatch(r"sha256:[0-9a-f]{64}", args.expected_sha256) is None:
        parser.error("expected-sha256 must be sha256: followed by 64 hex characters")
    if args.kind == "eval" and args.expected_sha256:
        parser.error("expected-sha256 is an adapter identity check, not an eval digest")

    if args.kind == "eval":
        return _fetch_eval_run(args)

    run_dir = Path(args.workspace) / "runs" / args.run_id / "artifacts"
    adapter_dir = run_dir / "final_adapter"
    lineage_path = run_dir / "lineage.json"

    if adapter_dir.is_dir() and lineage_path.is_file():
        sha = _verify(adapter_dir, lineage_path, args.run_id, args.expected_sha256)
        print(f"local artifacts reused (sha256 verified: {sha[:19]}...)")
        print(f"ADAPTER_DIR={adapter_dir}")
        return 0

    from axiom_world.integrations.hf_sync import download_snapshot, repository_revision

    if run_dir.exists():
        raise SystemExit(f"Partial artifacts already exist: {run_dir}; preserve and inspect them first")
    revision = repository_revision(args.repo, args.revision or "main")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    # Download to a separate tree: a failed fetch cannot masquerade as a local run.
    with tempfile.TemporaryDirectory(prefix="fetch-", dir=run_dir.parent) as td:
        fetched = Path(td) / "artifacts"
        download_snapshot(repo_id=args.repo, repo_type="model", revision=revision,
                          allow_patterns=["artifacts/*"], local_dir=td)
        sha = _verify(fetched / "final_adapter", fetched / "lineage.json",
                      args.run_id, args.expected_sha256)
        shutil.move(str(fetched), str(run_dir))
    (run_dir.parent / "fetch_receipt.json").write_text(json.dumps(
        {"repo": args.repo, "revision": revision, "run_id": args.run_id,
         "adapter_sha256": sha}, indent=2) + "\n", encoding="utf-8")
    print(f"fetched from hf://{args.repo} (sha256 verified: {sha[:19]}...)")
    print(f"ADAPTER_DIR={adapter_dir}")
    return 0


def _fetch_eval_run(args: argparse.Namespace) -> int:
    """Materialize runs/<run_id>/ (evaluation_*.jsonl + summary) locally."""
    run_dir = Path(args.workspace) / "runs" / args.run_id
    summary_path = run_dir / "evaluation_summary.json"

    if not summary_path.is_file():
        from axiom_world.integrations.hf_sync import download_run_directory

        try:
            download_run_directory(args.repo, args.run_id, args.workspace, revision=args.revision)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Eval run {args.run_id} not found locally and fetch from "
                f"hf://{args.repo} failed ({exc}).\n"
                "This stage requires a completed evaluation persisted with "
                "run_evaluation.py --hf-sync-repo."
            ) from exc

    if not summary_path.is_file():
        raise SystemExit(
            f"hf://{args.repo} did not contain runs/{args.run_id}/"
            "evaluation_summary.json — the eval run may not have completed."
        )
    suites = sorted(p.name for p in run_dir.glob("evaluation_*.jsonl"))
    if not suites:
        raise SystemExit(
            f"{run_dir} has a summary but no evaluation_*.jsonl files — "
            "per-episode outputs are required for paired analysis."
        )
    summary = json.loads(summary_path.read_text())
    print(f"eval run materialized: {len(suites)} suite files, "
          f"freeze_fingerprint={summary.get('freeze_fingerprint', '?')[:23]}...")
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

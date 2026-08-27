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
  python scripts/fetch_run.py --repo m97j/aw-runs-a1 \
      --run-id 20260729-145835--a1-playworld-sft--s42--1eb4c7
Prints ADAPTER_DIR=<path> on success for notebook consumption.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.lineage import compute_adapter_sha256


def _verify(adapter_dir: Path, lineage_path: Path) -> str:
    lineage = json.loads(lineage_path.read_text())
    expected = lineage["output_adapter_sha256"]
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
    args = parser.parse_args()

    if args.kind == "eval":
        return _fetch_eval_run(args)

    run_dir = Path(args.workspace) / "runs" / args.run_id / "artifacts"
    adapter_dir = run_dir / "final_adapter"
    lineage_path = run_dir / "lineage.json"

    if adapter_dir.is_dir() and lineage_path.is_file():
        sha = _verify(adapter_dir, lineage_path)
        print(f"local artifacts reused (sha256 verified: {sha[:19]}...)")
        print(f"ADAPTER_DIR={adapter_dir}")
        return 0

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=args.repo, repo_type="model",
            allow_patterns=["artifacts/*"],
            local_dir=run_dir.parent,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Run {args.run_id} not found locally and fetch from "
            f"hf://{args.repo} failed ({exc}).\n"
            "This stage requires a completed training run persisted with "
            "--hf-sync-repo. Run scripts/run_experiment.py first."
        ) from exc

    if not (adapter_dir.is_dir() and lineage_path.is_file()):
        raise SystemExit(
            f"hf://{args.repo} did not contain artifacts/final_adapter + "
            "lineage.json — the source run may not have completed."
        )
    lineage = json.loads(lineage_path.read_text())
    if lineage.get("run_id") != args.run_id:
        raise SystemExit(
            f"REPO/RUN MISMATCH: hf://{args.repo} holds run "
            f"{lineage.get('run_id')!r}, not requested {args.run_id!r}."
        )
    sha = _verify(adapter_dir, lineage_path)
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
            download_run_directory(args.repo, args.run_id, args.workspace)
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

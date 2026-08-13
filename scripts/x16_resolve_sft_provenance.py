#!/usr/bin/env python
"""x16: resolve which frozen PlayWorld SFT artifact each run ACTUALLY trained on.

Context (2026-08-13). x15 returned CONTENT DIVERGED: the PlayWorld SFT builder
is deterministic only PER CODE COMMIT — 53/2000 records flip the oracle's
target among equally-optimal BFS paths across commits (prompt_fingerprint is
stable; only the demonstration path tie-break drifts). Multiple sft artifacts
therefore exist (HF uploads at ~15d/10d/1d ago + unfrozen in-session builds).

Ground truth is NOT memory: every training run's lineage.json records
dataset_fingerprints.sft, computed by fingerprint_payload over the raw JSONL
lines. This tool recomputes that same lineage-style fingerprint for each
candidate HF artifact and matches them against the fingerprints recorded in
the runs' lineage files, so the canonical artifact can be pinned by evidence.

Usage (CPU, after fetch_run has materialized the runs locally):
  python scripts/x16_resolve_sft_provenance.py \
    --candidate m97j/aw-posttrain:playworld_sft.jsonl \
    --candidate m97j/aw-playworld:preference_train/v1/playworld_sft.jsonl \
    --candidate m97j/aw-playworld:train/v1/playworld_sft.jsonl \
    --lineage runs/<a1-run>/artifacts/lineage.json \
    --lineage runs/<b4-run>/artifacts/lineage.json \
    --out runs/x16_sft_provenance.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from axiom_world.core.fingerprints import fingerprint_payload


def _lineage_style_fingerprint(path: Path) -> str:
    payloads = [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return fingerprint_payload(payloads)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True,
                        help="repo_id:path_in_repo of a frozen sft artifact "
                             "(HF dataset repo). Repeatable.")
    parser.add_argument("--lineage", action="append", default=[],
                        help="Path to a run's artifacts/lineage.json. Repeatable.")
    parser.add_argument("--out", default="runs/x16_sft_provenance.json")
    args = parser.parse_args()

    candidates = []
    for spec in args.candidate:
        repo, _, repo_path = spec.partition(":")
        try:
            local = Path(hf_hub_download(repo_id=repo, filename=repo_path,
                                         repo_type="dataset"))
            fp = _lineage_style_fingerprint(local)
            candidates.append({"repo": repo, "path": repo_path,
                               "lineage_style_fingerprint": fp})
        except Exception as exc:  # noqa: BLE001
            candidates.append({"repo": repo, "path": repo_path,
                               "error": str(exc)})

    runs = []
    for lineage_path in args.lineage:
        data = json.loads(Path(lineage_path).read_text(encoding="utf-8"))
        fp = data.get("dataset_fingerprints", {}).get("sft")
        match = next((c for c in candidates
                      if c.get("lineage_style_fingerprint") == fp), None)
        runs.append({"lineage": lineage_path, "run_id": data.get("run_id"),
                     "sft_fingerprint": fp,
                     "matched_artifact": (f"{match['repo']}:{match['path']}"
                                          if match else None)})

    report = {"candidates": candidates, "runs": runs,
              "readout": "For each run, matched_artifact names the frozen HF "
                         "copy whose lineage-style fingerprint equals the "
                         "fingerprint the trainer recorded. A null match means "
                         "that run trained on an UNFROZEN in-session build — "
                         "its exact bytes are unrecoverable; pin the canonical "
                         "artifact to the A1 run's match."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

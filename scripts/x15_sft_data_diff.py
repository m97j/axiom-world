#!/usr/bin/env python
"""x15: PlayWorld SFT data parity audit (B4 fingerprint mismatch, 2026-08-13).

Context. The single-variable RQ1 design requires B4 to train on the SAME
PlayWorld SFT data as A1. aw_07 a_b4_data rebuilt with the pinned generator
(seed 1042, 5x400) and got sft_fingerprint sha256:2764e797..., while the
aw_05/aw_06 probe builds (identical CLI, earlier code commit) produced
sha256:050f94b2.... prompt_fingerprint (cc2aef0d...) matched exactly, so the
scenario stream is identical — the divergence is confined to the SFT
serialization (assistant/oracle text or record layout), i.e. a byte-level,
possibly semantics-preserving drift.

This tool decides which of two verdicts holds, WITHOUT a model:
  A) SEMANTICALLY IDENTICAL — same record multiset under canonical
     normalization (json.loads -> sort_keys dump), only byte order/formatting
     changed => B4 vs A1 comparison stands; note the fingerprint caveat in the
     report and pin the frozen artifact for B5/B6.
  B) CONTENT DIVERGED — records differ (count or content). The B4 result is
     then confounded by a data delta and B4 must be retrained on the frozen
     A1-era artifact fetched from HF.

Usage (CPU):
  python scripts/fetch_dataset.py --repo m97j/aw_playworld \
      --path preference_train/v1/playworld_sft.jsonl \
      --output data/frozen/playworld_sft_a1era.jsonl
  python scripts/x15_sft_data_diff.py \
      --file-a data/frozen/playworld_sft_a1era.jsonl --label-a a1-era-frozen \
      --file-b data/train/playworld_sft.jsonl --label-b b4-rebuild \
      --out runs/x15_sft_data_diff.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _canonical_records(path: str) -> tuple[list[str], Counter]:
    keys: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            clean = line.strip()
            if not clean:
                continue
            record = json.loads(clean)
            keys.append(json.dumps(record, sort_keys=True, ensure_ascii=False))
    return keys, Counter(keys)


def _digest(items: list[str]) -> str:
    h = hashlib.sha256()
    for item in sorted(items):
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-a", required=True)
    parser.add_argument("--file-b", required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--out", default="runs/x15_sft_data_diff.json")
    args = parser.parse_args()

    keys_a, counts_a = _canonical_records(args.file_a)
    keys_b, counts_b = _canonical_records(args.file_b)

    only_a = counts_a - counts_b
    only_b = counts_b - counts_a
    order_identical = keys_a == keys_b
    multiset_identical = not only_a and not only_b

    def _samples(counter: Counter, limit: int = 3) -> list[str]:
        return [key[:400] for key, _ in counter.most_common(limit)]

    report = {
        "file_a": {"label": args.label_a, "path": args.file_a,
                   "records": len(keys_a), "canonical_set_sha256": _digest(keys_a)},
        "file_b": {"label": args.label_b, "path": args.file_b,
                   "records": len(keys_b), "canonical_set_sha256": _digest(keys_b)},
        "order_identical": order_identical,
        "multiset_identical": multiset_identical,
        "records_only_in_a": sum(only_a.values()),
        "records_only_in_b": sum(only_b.values()),
        "sample_only_in_a": _samples(only_a),
        "sample_only_in_b": _samples(only_b),
        "verdict": (
            "SEMANTICALLY IDENTICAL — byte-order/formatting drift only; B4 vs A1 "
            "comparison stands. Record the fingerprint caveat and pin the frozen "
            "artifact (fetch_dataset) for all later PlayWorld stages."
            if multiset_identical else
            "CONTENT DIVERGED — B4 trained on different data than A1; the RQ1 "
            "comparison is confounded. Retrain B4 from the frozen A1-era artifact "
            "before proceeding to B5/B6."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Inspect legacy run-checkpoint storage (read-only).

A path deletion commit preserves history and is suitable for release layout
migration. Permanent LFS deletion destroys historical bytes and is NOT a
champion migration mechanism. Historical checkpoint aliases can share an OID
with adapters or retained checkpoints; this path-only tool cannot prove their
safety. Execution is retired; use an independently reviewed object-aware plan.
Visible size, historical storage and physical deduplication are different; this
listing does not establish billable quota or promise reclaimed bytes.

Usage: python scripts/common/hf_prune_checkpoints.py --repo m97j/aw-runs-b4
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

from axiom_world.integrations.hf_sync import (
    list_repository_files,
    list_storage_objects,
)

# Two hub layouts exist: root-level "checkpoint-N/..." (hf_sync on_save uses
# path_in_repo=checkpoint-N, successive runs OVERWRITE the same paths) and the
# namespaced "<run>/checkpoints/checkpoint-N/...". Match both; root-level blobs
# are grouped under the pseudo-run "<root>".
_CKPT = re.compile(
    r"^(?:(?P<run>[^/]+)/checkpoints/)?checkpoint-(?P<step>\d+)/")


def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--keep-last", type=int, default=1,
                        help="newest checkpoints to keep per run (default 1; "
                             "0 = delete all checkpoints of eligible runs)")
    parser.add_argument("--also-incomplete", nargs="*", default=[],
                        help="run ids prunable even without artifacts/lineage.json at HEAD")
    parser.add_argument("--execute", action="store_true",
                        help="PERMANENTLY delete LFS blobs (default: dry-run)")
    args = parser.parse_args()
    if args.keep_last < 0:
        parser.error("keep-last must be nonnegative")
    if args.execute:
        parser.error("Permanent deletion disabled: path-only selection cannot protect shared objects/history")
    if not re.fullmatch(r"[^/]+/aw-runs-[^/]+", args.repo):
        parser.error("This diagnostic only accepts aw-runs-* repositories, never champion repositories")

    head_files = set(list_repository_files(args.repo))
    completed = {f.split("/", 1)[0] for f in head_files if "/artifacts/lineage.json" in f}

    lfs_files = list_storage_objects(args.repo)
    # group checkpoint LFS blobs: run -> step -> [LFSFileInfo]
    ckpts: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    other_bytes = 0
    for info in lfs_files:
        m = _CKPT.match(info.filename)
        if m:
            ckpts[m["run"] or "<root>"][int(m["step"])].append(info)
        else:
            other_bytes += info.size

    to_delete = []
    for run, steps in sorted(ckpts.items()):
        # "<root>" (shared, overwritten layout) is prunable when ANY run in the
        # repo has completed artifacts at HEAD, or when explicitly allowed.
        eligible = (run in completed or run in args.also_incomplete
                    or (run == "<root>" and (bool(completed) or "<root>" in args.also_incomplete)))
        ordered = sorted(steps)
        keep = set(ordered[len(ordered) - args.keep_last:]) if args.keep_last else set()
        drop = [s for s in ordered if s not in keep]
        drop_infos = [i for s in drop for i in steps[s]]
        drop_bytes = sum(i.size for i in drop_infos)
        total_bytes = sum(i.size for ss in steps.values() for i in ss)
        status = "PRUNE" if eligible and drop else ("SKIP(incomplete)" if not eligible else "SKIP(nothing)")
        print(f"{status:18s} {run}: {len(ordered)} ckpts / {_fmt(total_bytes)} LFS; "
              f"dropping {len(drop)} ckpts / {_fmt(drop_bytes)}; keeping steps {sorted(keep)}")
        if eligible:
            to_delete.extend(drop_infos)

    freed = sum(i.size for i in to_delete)
    print(f"\nnon-checkpoint LFS in repo: {_fmt(other_bytes)}")
    print(f"Path-selected LFS candidates (NOT authorized for deletion): {len(to_delete)} files, ~{_fmt(freed)} "
          f"(execute={args.execute})")
    print("Read-only candidates, not a safe deletion plan or billable storage estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

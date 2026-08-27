#!/usr/bin/env python
"""Reclaim HF storage quota from aw-runs-* repos (2026-08-16 quota incident).

WHY QUOTA != VISIBLE SIZE. HF repos are git(-LFS/Xet) backed: a commit that
deletes or overwrites a path only moves HEAD — the underlying LFS blobs of
EVERY historical revision remain on the server (that is what makes
`revision=` fetches and restores possible) and **all of them count against
the account storage quota**. This is why the B6 repo "shows" ~300GB of files
but bills ~600GB after the B6-R rerun overwrote the same paths: the B6-era
blobs are still there, one commit behind. A `delete_files(...)` commit alone
therefore frees NOTHING.

WHAT ACTUALLY FREES SPACE: `HfApi.permanently_delete_lfs_files(...)`, which
removes the LFS blobs themselves (with history rewrite). This script:

  1. lists all LFS files in the repo,
  2. selects checkpoint blobs (`<run>/checkpoints/checkpoint-N/...`),
     keeping the newest --keep-last N checkpoints per run,
  3. by default only touches runs whose artifacts/lineage.json exists at HEAD
     (completed & synced) — pass --also-incomplete <run-id> for others,
  4. PERMANENTLY deletes the rest (dry-run by default; --execute to act).

Durable evidence (runs/<id>/artifacts/, run_card, wandb) is never touched.
Intermediate checkpoints are transient resume aids per protocol; deleting
them after a completed run destroys no pre-registered evidence.

Usage:
  python scripts/hf_prune_checkpoints.py --repo m97j/aw-runs-b6                  # dry-run
  python scripts/hf_prune_checkpoints.py --repo m97j/aw-runs-b6 \
      --also-incomplete <b6r-run-id> --keep-last 1 --execute
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

from huggingface_hub import HfApi

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

    api = HfApi()
    head_files = set(api.list_repo_files(args.repo))
    completed = {f.split("/", 1)[0] for f in head_files if "/artifacts/lineage.json" in f}

    lfs_files = list(api.list_lfs_files(args.repo))
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
    print(f"LFS blobs to PERMANENTLY delete: {len(to_delete)} files, ~{_fmt(freed)} "
          f"(execute={args.execute})")
    print("NOTE: quota counts ALL revisions' blobs; this permanent deletion "
          "(rewrite_history=True) is the only path that actually frees space.")
    if args.execute and to_delete:
        api.permanently_delete_lfs_files(args.repo, to_delete, rewrite_history=True)
        print("permanently deleted. Storage accounting may take a while to refresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

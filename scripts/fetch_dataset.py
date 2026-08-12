#!/usr/bin/env python
"""Materialize a persisted dataset artifact from a Hugging Face Dataset repo.

Read-path counterpart to the dataset uploads performed by build_p1_data.py /
mine_p1_pairs.py. This script deliberately does NOT rebuild or regenerate
datasets: it materializes an already-persisted artifact so later stages consume
the exact bytes that were previously frozen.

Prints:
    DATASET_PATH=<local path>
    DATASET_SHA256=<sha256>
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_expected_sha256(path: Path, expected: str | None) -> str:
    actual = _sha256_file(path)
    if expected is not None:
        normalized = expected.removeprefix("sha256:")
        if actual != normalized:
            raise SystemExit(
                "INTEGRITY FAILURE: dataset sha256 mismatch.\n"
                f"  path:     {path}\n"
                f"  expected: {normalized}\n"
                f"  actual:   {actual}\n"
                "Do not use this dataset; re-fetch the artifact or verify "
                "the expected digest."
            )
    return actual


def _materialize(*, repo: str, repo_path: str, revision: str, output: Path,
                 expected_sha256: str | None, force: bool) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        if not output.is_file():
            raise SystemExit(f"Output path exists but is not a regular file: {output}")
        sha = _verify_expected_sha256(output, expected_sha256)
        print(f"local dataset reused: {output}")
        print(f"dataset sha256: {sha}")
        return sha

    try:
        cached_path = hf_hub_download(
            repo_id=repo, filename=repo_path, repo_type="dataset", revision=revision,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Failed to fetch dataset artifact.\n"
            f"  repo:     {repo}\n  revision: {revision}\n"
            f"  path:     {repo_path}\n  error:    {exc}\n"
        ) from exc

    cached = Path(cached_path)
    if not cached.is_file():
        raise SystemExit(f"Hugging Face returned a path that is not a file: {cached}")

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    shutil.copyfile(cached, tmp_output)
    try:
        sha = _verify_expected_sha256(tmp_output, expected_sha256)
        tmp_output.replace(output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    print(f"fetched dataset: hf://{repo}/{repo_path}")
    print(f"revision: {revision}")
    print(f"materialized: {output}")
    print(f"dataset sha256: {sha}")
    return sha


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a persisted dataset artifact from a HF Dataset repo."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--path", required=True,
                        help="Artifact path inside the HF repo, e.g. p1/v1/p1_general_sft.jsonl.")
    parser.add_argument("--revision", default="main",
                        help="Prefer an exact commit hash for canonical research runs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha256", default=None,
                        help="Raw 64-hex digest or 'sha256:<digest>' form.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_path = args.path.strip("/")
    if not repo_path:
        raise SystemExit("--path must not be empty.")
    sha = _materialize(repo=args.repo, repo_path=repo_path, revision=args.revision,
                       output=Path(args.output), expected_sha256=args.expected_sha256,
                       force=args.force)
    print(f"DATASET_PATH={args.output}")
    print(f"DATASET_SHA256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

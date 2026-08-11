#!/usr/bin/env python
"""Materialize a persisted dataset artifact from a Hugging Face Dataset repo.

This is the dataset read-path counterpart to the dataset upload performed by
scripts/build_p1_data.py.

The script deliberately does NOT rebuild or regenerate datasets. It materializes
an already-persisted dataset artifact into the local workspace so later stages
can consume the exact same bytes that were previously persisted.

Usage:

    python scripts/fetch_dataset.py \
        --repo m97j/axiom-general-posttrain \
        --path p1/v1/p1_general_sft.jsonl \
        --output data/p1/p1_general_sft.jsonl

For a frozen revision:

    python scripts/fetch_dataset.py \
        --repo m97j/axiom-general-posttrain \
        --revision <commit-hash> \
        --path p1/v1/p1_general_sft.jsonl \
        --output data/p1/p1_general_sft.jsonl

The script prints:

    DATASET_PATH=<local path>
    DATASET_SHA256=<sha256>

The downloaded bytes are never regenerated locally. If the destination
already exists, it is reused by default after integrity verification when
--expected-sha256 is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of one local file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_expected_sha256(path: Path, expected: str | None) -> str:
    """Compute and optionally verify a file SHA-256 digest."""
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


def _materialize(
    *,
    repo: str,
    repo_path: str,
    revision: str,
    output: Path,
    expected_sha256: str | None,
    force: bool,
) -> str:
    """Download one HF dataset file and materialize it at ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not force:
        if not output.is_file():
            raise SystemExit(
                f"Output path exists but is not a regular file: {output}"
            )

        sha = _verify_expected_sha256(output, expected_sha256)

        print(f"local dataset reused: {output}")
        print(f"dataset sha256: {sha}")
        return sha

    try:
        cached_path = hf_hub_download(
            repo_id=repo,
            filename=repo_path,
            repo_type="dataset",
            revision=revision,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to fetch dataset artifact.\n"
            f"  repo:     {repo}\n"
            f"  revision: {revision}\n"
            f"  path:     {repo_path}\n"
            f"  error:    {exc}\n"
        ) from exc

    cached = Path(cached_path)

    if not cached.is_file():
        raise SystemExit(
            f"Hugging Face returned a path that is not a file: {cached}"
        )

    # Copy into the project workspace rather than exposing the HF cache path.
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
        description=(
            "Materialize a persisted dataset artifact from a Hugging Face "
            "Dataset repository."
        )
    )

    parser.add_argument(
        "--repo",
        required=True,
        help="HF Dataset repo, e.g. m97j/axiom-general-posttrain.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help=(
            "Path to the dataset artifact inside the HF repo, e.g. "
            "p1/v1/p1_general_sft.jsonl."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help=(
            "HF revision to materialize. Prefer an exact commit hash for "
            "canonical research runs. Default: main."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Local destination path.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help=(
            "Optional expected SHA-256 digest. Accepts either the raw 64-hex "
            "digest or the 'sha256:<digest>' form."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the local destination already exists.",
    )

    args = parser.parse_args()

    repo_path = args.path.strip("/")
    output = Path(args.output)

    if not repo_path:
        raise SystemExit("--path must not be empty.")

    sha = _materialize(
        repo=args.repo,
        repo_path=repo_path,
        revision=args.revision,
        output=output,
        expected_sha256=args.expected_sha256,
        force=args.force,
    )

    print(f"DATASET_PATH={output}")
    print(f"DATASET_SHA256={sha}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

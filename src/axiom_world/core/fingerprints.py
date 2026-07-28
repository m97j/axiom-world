"""Deterministic fingerprints for configs, files, and datasets.

Every canonical run records fingerprints so that the tech report can claim
"identical config/data" with a hash, not a sentence (protocol §11).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def canonical_json(payload: Any) -> str:
    """Stable JSON encoding: sorted keys, no whitespace drift."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint_payload(payload: Any) -> str:
    """SHA-256 of a JSON-serializable payload, prefixed for self-description."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def fingerprint_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def fingerprint_directory(path: Path, pattern: str = "**/*") -> str:
    """Order-independent SHA-256 over (relative path, file hash) pairs.

    Used to fingerprint adapter directories for the parent-lineage check.
    """
    entries: list[tuple[str, str]] = []
    for item in sorted(path.glob(pattern)):
        if item.is_file():
            entries.append((item.relative_to(path).as_posix(), fingerprint_file(item)))
    return fingerprint_payload(entries)

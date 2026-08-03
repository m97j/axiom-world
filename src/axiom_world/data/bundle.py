"""build_data_bundle — THE single data entrypoint.

Replaces the previous snapshot's split personality (build_data_bundle vs
DataModuleFactory). Every runner obtains data exclusively through this
function; it returns validated records plus a fingerprinted manifest, and
enforces the family-level leakage gate at load time.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from axiom_world.core.errors import AxiomError
from axiom_world.core.fingerprints import fingerprint_payload
from axiom_world.data.records import EvaluationRecord, PreferenceRecord, SFTRecord

_RECORD_TYPES = {
    "sft": SFTRecord,
    "preference": PreferenceRecord,
    "evaluation": EvaluationRecord,
}


class DataContractError(AxiomError):
    pass


@dataclass
class DataBundle:
    kind: str
    records: list[Any]
    fingerprint: str
    manifest: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            clean_line = line.strip()
            if not clean_line:
                continue
            try:
                payload = json.loads(clean_line)
            except json.JSONDecodeError as exc:
                raise DataContractError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(payload, dict):
                raise DataContractError(f"{path}:{line_no}: record must be an object")
            yield line_no, payload


def build_data_bundle(
    path: Path | str,
    kind: str,
    expected_fingerprint: str | None = None,
    forbidden_family_ids: set[str] | None = None,
) -> DataBundle:
    """Load, validate, fingerprint, and leakage-gate a JSONL dataset.

    Args:
        path: JSONL file, one record per line.
        kind: 'sft' | 'preference' | 'evaluation'.
        expected_fingerprint: when set (canonical runs), a mismatch is a hard
            failure — the dataset changed since it was frozen.
        forbidden_family_ids: scenario families that must NOT appear (e.g.
            eval families when loading training data). Protocol §4.3 gate.
    """
    if kind not in _RECORD_TYPES:
        raise DataContractError(f"Unknown bundle kind {kind!r}; expected {sorted(_RECORD_TYPES)}")
    model = _RECORD_TYPES[kind]
    source = Path(path)
    if not source.is_file():
        raise DataContractError(f"Dataset file not found: {source}")

    records: list[Any] = []
    raw_payloads: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    family_counts: dict[str, int] = {}
    forbidden = forbidden_family_ids or set()

    for line_no, payload in _read_jsonl(source):
        try:
            record = model.model_validate(payload)
        except ValidationError as exc:
            raise DataContractError(f"{source}:{line_no}: schema violation\n{exc}") from exc
        if record.id in seen_ids:
            raise DataContractError(f"{source}:{line_no}: duplicate record id {record.id!r}")
        seen_ids.add(record.id)
        family = getattr(record, "scenario_family_id", None)
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
            if family in forbidden:
                raise DataContractError(
                    f"{source}:{line_no}: leakage gate — record from forbidden "
                    f"scenario family {family!r} (protocol §4.3)."
                )
        records.append(record)
        raw_payloads.append(payload)

    if not records:
        raise DataContractError(f"Dataset is empty: {source}")

    fingerprint = fingerprint_payload(raw_payloads)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise DataContractError(
            "Dataset fingerprint mismatch (frozen dataset changed).\n"
            f"  expected: {expected_fingerprint}\n  actual:   {fingerprint}\n  file: {source}"
        )

    manifest = {
        "path": str(source),
        "kind": kind,
        "record_count": len(records),
        "fingerprint": fingerprint,
        "family_counts": dict(sorted(family_counts.items())),
    }
    return DataBundle(kind=kind, records=records, fingerprint=fingerprint, manifest=manifest)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read JSONL splitting on '\n' ONLY.

    Never use str.splitlines() for JSONL: records serialized with
    ensure_ascii=False may contain literal U+2028/U+2029 (present in e.g.
    MATH LaTeX solutions), which splitlines() treats as line breaks, cutting
    records in half (JSONDecodeError: unterminated string).
    """
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def write_jsonl(path: Path | str, records: Iterable[Any]) -> str:
    """Serialize records to JSONL and return the bundle fingerprint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payloads = []
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
            payloads.append(payload)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return fingerprint_payload(payloads)

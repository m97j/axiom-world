"""Lineage contract enforcement (protocol §11).

The single most important guarantee of Axiom-World's two-stage design:

    A Phase-2 Track-B run CANNOT train unless the parent adapter it loaded
    is byte-identical to the one recorded in its lineage reference.

``verify_parent_adapter`` is called by the training layer AFTER downloading /
locating the parent adapter directory and BEFORE any training step. A
mismatch raises ``LineageError`` — never a warning.
"""
from __future__ import annotations

from pathlib import Path

from axiom_world.core.enums import InitializationMode
from axiom_world.core.errors import LineageError
from axiom_world.core.fingerprints import fingerprint_directory
from axiom_world.core.schemas import ExperimentConfig, LineageRecord, ParentAdapterRef

# Files that define adapter identity. Tokenizer/config text files are
# excluded on purpose: identity is the trained weights + adapter config.
ADAPTER_IDENTITY_PATTERNS: tuple[str, ...] = (
    "adapter_model.safetensors",
    "adapter_config.json",
)


def compute_adapter_sha256(adapter_dir: Path) -> str:
    """Order-independent digest over the adapter identity files."""
    files = []
    for pattern in ADAPTER_IDENTITY_PATTERNS:
        found = sorted(adapter_dir.glob(pattern))
        files.extend(found)
    if not files:
        raise LineageError(
            f"No adapter identity files {ADAPTER_IDENTITY_PATTERNS} in {adapter_dir}"
        )
    from axiom_world.core.fingerprints import fingerprint_file, fingerprint_payload

    entries = [(f.name, fingerprint_file(f)) for f in files]
    return fingerprint_payload(entries)


def verify_parent_adapter(ref: ParentAdapterRef, adapter_dir: Path) -> str:
    """Hard gate: local adapter bytes must match the recorded sha256."""
    actual = compute_adapter_sha256(adapter_dir)
    if actual != ref.sha256:
        raise LineageError(
            "Parent adapter hash mismatch (protocol §11 hard failure).\n"
            f"  expected: {ref.sha256}\n"
            f"  actual:   {actual}\n"
            f"  repo_id:  {ref.repo_id}@{ref.revision}\n"
            f"  local:    {adapter_dir}"
        )
    return actual


def build_lineage_record(
    run_id: str,
    config: ExperimentConfig,
    config_fingerprint: str,
    dataset_fingerprints: dict[str, str] | None = None,
    code_commit: str | None = None,
) -> LineageRecord:
    return LineageRecord(
        run_id=run_id,
        phase=config.phase,
        base_model_repo_id=config.model.repo_id,
        base_model_revision=config.model.revision,
        initialization_mode=config.lineage.initialization_mode,
        parent_adapter=config.lineage.parent_adapter,
        parent_run_id=config.lineage.parent_run_id,
        dataset_fingerprints=dataset_fingerprints or {},
        config_fingerprint=config_fingerprint,
        code_commit=code_commit,
    )


def assert_lineage_executable(config: ExperimentConfig, adapter_dir: Path | None) -> None:
    """Called by runners before training. Combines mode + hash checks."""
    mode = config.lineage.initialization_mode
    if mode is InitializationMode.FROM_BASE:
        if config.lineage.parent_adapter is not None:
            raise LineageError("from_base run must not carry a parent_adapter reference.")
        return
    ref = config.lineage.parent_adapter
    if ref is None:
        raise LineageError("continue_training_existing_adapter requires parent_adapter.")
    if adapter_dir is None:
        raise LineageError("Parent adapter directory was not provided to the runner.")
    verify_parent_adapter(ref, adapter_dir)


__all__ = [
    "ADAPTER_IDENTITY_PATTERNS",
    "assert_lineage_executable",
    "build_lineage_record",
    "compute_adapter_sha256",
    "fingerprint_directory",
    "verify_parent_adapter",
]

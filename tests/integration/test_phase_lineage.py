"""Protocol §11 hard gate: parent adapter bytes must match lineage sha256."""
from pathlib import Path

import pytest

from axiom_world.core.enums import InitializationMode
from axiom_world.core.errors import LineageError
from axiom_world.core.lineage import (
    assert_lineage_executable,
    compute_adapter_sha256,
    verify_parent_adapter,
)
from axiom_world.core.schemas import ExperimentConfig, ParentAdapterRef

BASE_CONFIG = {
    "project": {"name": "axiom-world", "protocol_version": "v1.0"},
    "experiment_name": "lineage-test",
    "track": "track_b_two_stage",
    "phase": "phase2_playworld",
    "objective": "sft",
    "model": {"repo_id": "Qwen/Qwen3-8B-Base", "revision": "abc123"},
}


def _fake_adapter(tmp_path: Path, weight_bytes: bytes = b"weights-v1") -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(weight_bytes)
    (adapter / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    return adapter


def test_matching_hash_passes(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path)
    sha = compute_adapter_sha256(adapter)
    ref = ParentAdapterRef(repo_id="x/p1", revision="deadbeef", sha256=sha)
    assert verify_parent_adapter(ref, adapter) == sha


def test_tampered_adapter_rejected(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path)
    sha = compute_adapter_sha256(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-TAMPERED")
    ref = ParentAdapterRef(repo_id="x/p1", revision="deadbeef", sha256=sha)
    with pytest.raises(LineageError, match="hash mismatch"):
        verify_parent_adapter(ref, adapter)


def test_continue_mode_end_to_end(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path)
    sha = compute_adapter_sha256(adapter)
    config = ExperimentConfig.model_validate(
        dict(
            BASE_CONFIG,
            lineage={
                "initialization_mode": "continue_training_existing_adapter",
                "parent_adapter": {"repo_id": "x/p1", "revision": "deadbeef", "sha256": sha},
            },
        )
    )
    assert config.validate_canonical() == []
    assert_lineage_executable(config, adapter)


def test_continue_mode_without_adapter_dir_fails(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path)
    sha = compute_adapter_sha256(adapter)
    config = ExperimentConfig.model_validate(
        dict(
            BASE_CONFIG,
            lineage={
                "initialization_mode": "continue_training_existing_adapter",
                "parent_adapter": {"repo_id": "x/p1", "revision": "deadbeef", "sha256": sha},
            },
        )
    )
    with pytest.raises(LineageError):
        assert_lineage_executable(config, None)


def test_from_base_with_parent_is_ambiguous(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path)
    sha = compute_adapter_sha256(adapter)
    config = ExperimentConfig.model_validate(
        dict(
            BASE_CONFIG,
            track="track_a_direct",
            lineage={
                "initialization_mode": "from_base",
                "parent_adapter": {"repo_id": "x/p1", "revision": "deadbeef", "sha256": sha},
            },
        )
    )
    assert any("ambiguous" in v.lower() for v in config.validate_canonical())
    with pytest.raises(LineageError):
        assert_lineage_executable(config, adapter)
    assert config.lineage.initialization_mode is InitializationMode.FROM_BASE

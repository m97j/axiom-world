from pathlib import Path

import pytest

from axiom_world.core.context import ExperimentContext
from axiom_world.core.enums import ArtifactKind, RunStatus
from axiom_world.core.errors import RunContractError
from axiom_world.core.paths import REQUIRED_ARTIFACTS
from axiom_world.core.schemas import ExperimentConfig

CONFIG = {
    "project": {"name": "axiom-world", "protocol_version": "v1.0"},
    "experiment_name": "ctx-test",
    "track": "track_a_direct",
    "phase": "phase2_playworld",
    "objective": "sft",
    "model": {"repo_id": "Qwen/Qwen3-8B-Base", "revision": "abc123"},
}


def _context(tmp_path: Path) -> ExperimentContext:
    config = ExperimentConfig.model_validate(CONFIG)
    ctx = ExperimentContext(config, "sha256:" + "0" * 64, tmp_path)
    ctx.initialize(CONFIG)
    return ctx


def test_initialize_writes_manifest_and_config(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    assert ctx.paths.manifest.is_file()
    assert ctx.paths.artifact("resolved_config.yaml").is_file()


def test_illegal_transition_rejected(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    with pytest.raises(RunContractError):
        ctx.transition(RunStatus.COMPLETED)  # pending -> completed forbidden


def test_completion_requires_all_artifacts(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.transition(RunStatus.RUNNING)
    with pytest.raises(RunContractError, match="missing required artifacts"):
        ctx.transition(RunStatus.COMPLETED)
    for name in REQUIRED_ARTIFACTS:
        if name == "resolved_config.yaml":
            continue
        ctx.write_json_artifact(name, {"ok": True}, ArtifactKind.MANIFEST)
    ctx.transition(RunStatus.COMPLETED)
    assert ctx.status is RunStatus.COMPLETED


def test_terminal_state_is_final(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.transition(RunStatus.RUNNING)
    ctx.transition(RunStatus.FAILED)
    with pytest.raises(RunContractError):
        ctx.transition(RunStatus.RUNNING)

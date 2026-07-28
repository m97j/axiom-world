"""Trainer-boundary tests that run WITHOUT TRL/torch installed."""
from pathlib import Path

import pytest

from axiom_world.core.errors import AxiomError, LineageError
from axiom_world.core.schemas import ExperimentConfig
from axiom_world.data.bundle import build_data_bundle, write_jsonl
from axiom_world.training.adapter import to_dpo_rows, to_sft_rows
from axiom_world.training.factory import UnsupportedTRLAPIError, build_trainer
from tests.unit.test_data_bundle import _sft

BASE = {
    "project": {"name": "axiom-world", "protocol_version": "v1.0"},
    "experiment_name": "tf-test",
    "track": "track_a_direct",
    "phase": "phase2_playworld",
    "objective": "sft",
    "model": {"repo_id": "Qwen/Qwen3-8B-Base", "revision": "abc123"},
}


def test_grpo_requires_reward_funcs(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(dict(BASE, objective="grpo"))
    with pytest.raises(AxiomError, match="reward function"):
        build_trainer(config, None, None, None, tmp_path)


def test_lineage_gate_runs_before_trl_import(tmp_path: Path) -> None:
    """A Track-B run without its parent adapter dir fails on lineage,
    NOT on a missing TRL import — proving gate ordering."""
    config = ExperimentConfig.model_validate(
        dict(
            BASE,
            track="track_b_two_stage",
            lineage={
                "initialization_mode": "continue_training_existing_adapter",
                "parent_adapter": {
                    "repo_id": "x/p1",
                    "revision": "deadbeef",
                    "sha256": "sha256:" + "0" * 64,
                },
            },
        )
    )
    with pytest.raises(LineageError):
        build_trainer(config, None, None, None, tmp_path)


def test_sft_without_trl_raises_unsupported(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(BASE)
    with pytest.raises(UnsupportedTRLAPIError, match="TRL is not installed"):
        build_trainer(config, None, None, None, tmp_path)


def test_adapter_rows(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [_sft("r1"), _sft("r2")])
    bundle = build_data_bundle(path, "sft")
    rows = to_sft_rows(bundle)
    assert rows[0]["messages"][-1]["role"] == "assistant"
    with pytest.raises(AxiomError, match="preference"):
        to_dpo_rows(bundle)

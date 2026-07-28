from pathlib import Path

import pytest

from axiom_world.core.context import ExperimentContext
from axiom_world.core.schemas import ExperimentConfig
from axiom_world.data.bundle import build_data_bundle, write_jsonl
from axiom_world.data.records import EvaluationRecord, Message, Provenance
from axiom_world.evaluation.metrics import bootstrap_ci, paired_bootstrap_diff
from axiom_world.evaluation.runner import EvaluationRunner
from axiom_world.verifiers.hybrid import default_playworld_verifier
from tests.unit.test_verifiers import GOOD, _scenario

CONFIG = {
    "project": {"name": "axiom-world", "protocol_version": "v1.0"},
    "experiment_name": "eval-test",
    "track": "track_c_reference",
    "phase": "phase2_playworld",
    "objective": "eval_only",
    "model": {"repo_id": "Qwen/Qwen3-8B", "revision": "abc123"},
}


def _eval_records(count: int = 4) -> list[EvaluationRecord]:
    scenario = _scenario().model_dump(mode="json")
    return [
        EvaluationRecord(
            id=f"ev-{i}",
            suite="eval_id",
            scenario=scenario,
            prompt=[Message(role="user", content="solve")],
            scenario_family_id="fam-a",
            provenance=Provenance(source_type="synthetic", source_id="gen-v1"),
        )
        for i in range(count)
    ]


def test_runner_writes_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, _eval_records())
    bundle = build_data_bundle(path, "evaluation")
    config = ExperimentConfig.model_validate(CONFIG)
    ctx = ExperimentContext(config, "sha256:" + "0" * 64, tmp_path)
    ctx.initialize(CONFIG)
    result = EvaluationRunner(default_playworld_verifier(), lambda p: GOOD).run(bundle, ctx)
    suite = result["summary"]["suites"]["eval_id"]
    assert suite["pass_rate"]["mean"] == 1.0
    assert ctx.paths.artifact("evaluation.jsonl").is_file()
    assert ctx.paths.artifact("evaluation_summary.json").is_file()


def test_failure_taxonomy_in_summary(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, _eval_records())
    bundle = build_data_bundle(path, "evaluation")
    result = EvaluationRunner(
        default_playworld_verifier(), lambda p: "not json at all"
    ).run(bundle)
    suite = result["summary"]["suites"]["eval_id"]
    assert suite["pass_rate"]["mean"] == 0.0
    assert any(code.startswith("gate_failed") for code in suite["failure_taxonomy"])


def test_bootstrap_ci_contains_mean() -> None:
    mean, low, high = bootstrap_ci([0.0, 1.0, 1.0, 1.0], resamples=2000, seed=1)
    assert low <= mean <= high
    assert mean == pytest.approx(0.75)


def test_paired_bootstrap_detects_difference() -> None:
    a = [1.0] * 30
    b = [0.0] * 30
    result = paired_bootstrap_diff(a, b, resamples=2000, seed=1)
    assert result["significant"] is True and result["delta"] == 1.0
    same = paired_bootstrap_diff([0.5, 0.4, 0.6] * 10, [0.5, 0.6, 0.4] * 10, resamples=2000, seed=1)
    assert same["significant"] is False

"""Unit tests for the analysis layer (paired comparison, permutation test)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_world.analysis import compare_runs, load_episode_scores, paired_permutation_test


def _write_eval(run_dir: Path, suite: str, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"evaluation_{suite}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _row(i: int, passed: bool, score: float) -> dict:
    return {
        "id": f"ep-{i:04d}",
        "verdict": {"status": "passed" if passed else "failed", "score": score},
    }


def test_permutation_detects_clear_difference() -> None:
    a = [1.0] * 40 + [0.0] * 10
    b = [0.0] * 40 + [1.0] * 10
    result = paired_permutation_test(a, b, resamples=2000, seed=7)
    assert result["observed_delta"] == pytest.approx(0.6)
    assert result["p_value"] < 0.01


def test_permutation_null_is_insignificant() -> None:
    a = [float(i % 2) for i in range(60)]
    b = [float((i + 1) % 2) for i in range(60)]
    result = paired_permutation_test(a, b, resamples=2000, seed=7)
    assert result["p_value"] > 0.05


def test_permutation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        paired_permutation_test([1.0], [1.0, 0.0])


def test_compare_runs_end_to_end(tmp_path: Path) -> None:
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    rows_a = [_row(i, passed=i < 30, score=1.0 if i < 30 else 0.2) for i in range(50)]
    rows_b = [_row(i, passed=i < 5, score=1.0 if i < 5 else 0.1) for i in range(50)]
    _write_eval(run_a, "eval_id", rows_a)
    _write_eval(run_b, "eval_id", rows_b)

    report = compare_runs(run_a, run_b, ["eval_id"], label_a="x", label_b="y", seed=3)
    suite = report["suites"]["eval_id"]
    assert suite["episodes"] == 50
    assert suite["pass_rate"]["a"]["mean"] == pytest.approx(0.6)
    assert suite["pass_rate"]["b"]["mean"] == pytest.approx(0.1)
    assert suite["pass_rate"]["paired_bootstrap"]["significant"] is True
    assert suite["pass_rate"]["permutation"]["p_value"] < 0.01


def test_compare_runs_rejects_id_mismatch(tmp_path: Path) -> None:
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    _write_eval(run_a, "eval_id", [_row(i, True, 1.0) for i in range(3)])
    _write_eval(run_b, "eval_id", [_row(i + 1, True, 1.0) for i in range(3)])
    with pytest.raises(ValueError):
        compare_runs(run_a, run_b, ["eval_id"])


def test_load_episode_scores_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_episode_scores(tmp_path, "eval_id")

"""Pure paired-report boundaries; no GPU or Hub mutations."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("compare_champion",
    Path(__file__).resolve().parents[2] / "scripts/publish/v1/compare_champion.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(identity, passed, *, prompt="same", reason="weighted_aggregation", truncated=False):
    return {"id": identity, "suite": "eval_id", "prompt_sha256": prompt,
            "prediction": str(passed), "truncated": truncated,
            "verdict": {"status": "passed" if passed else "failed", "reason_code": reason}}


def test_paired_report_preserves_regressions_and_format_failures():
    a = [row("a", True), row("b", False), row("c", True)]
    b = [row("a", False, reason="gate_failed:invalid_json", truncated=True),
         row("b", True), row("c", True)]
    r = module.summarize_pairs(a, b)["eval_id"]
    assert r["regressions"] == r["improvements"] == 1
    assert r["pass_rate_delta"] == 0
    assert r["identical_text"] == 1
    assert r["candidate_schema_failed"] == r["candidate_truncated"] == 1


@pytest.mark.parametrize("candidate", [[row("other", True)], [row("a", True, prompt="different")], []])
def test_pairing_rejects_identity_prompt_and_missing_rows(candidate):
    with pytest.raises(ValueError):
        module.summarize_pairs([row("a", True)], candidate)


def test_infra_error_is_not_a_task_failure():
    b = row("a", False)
    b["verdict"]["status"] = "infra_error"
    with pytest.raises(ValueError, match="Infrastructure"):
        module.summarize_pairs([row("a", True)], [b])


def test_precision_review_preserves_regression_and_rejects_changed_source(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "compare_champion", module)
    spec = importlib.util.spec_from_file_location("review_precision",
        Path(__file__).resolve().parents[2] / "scripts/publish/v1/review_precision.py")
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    reference = [row("a", True), row("b", False)]
    candidate = [row("a", False), row("b", False)]
    request = dict.fromkeys(("legacy_code", "data_revision", "freeze", "max_new_tokens",
                            "attention", "seed", "batch_size"), "same")
    fp = {"request": {**request, "release_receipt_sha256": "bound"},
          "suites": module.summarize_pairs(reference, reference)}
    bf = {"request": {**request, "candidate_dtype": "bfloat16", "source_fp32_receipt_sha256": "bound"},
          "suites": module.summarize_pairs(reference, candidate)}
    monkeypatch.setattr(review, "load_comparison", lambda folder:
        (fp, {"reference": reference, "candidate": reference}) if folder == "fp"
        else (bf, {"reference": reference, "candidate": candidate}))
    result = review.review("fp", "bf")
    assert result["suggested_precision"] == "float32"
    assert not result["publication_approved"]
    assert result["suites"]["eval_id"]["bf16_vs_adapter_ci"]["delta"] == -0.5
    bf["request"]["source_fp32_receipt_sha256"] = "changed"
    with pytest.raises(ValueError, match="does not derive"):
        review.review("fp", "bf")

def test_unmatched_batches_require_explicit_descriptive_mode(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "compare_champion", module)
    spec = importlib.util.spec_from_file_location("review_precision",
        Path(__file__).resolve().parents[2] / "scripts/publish/v1/review_precision.py")
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    a, b = [row("a", True)], [row("a", False)]
    request = dict.fromkeys(("legacy_code", "data_revision", "freeze", "max_new_tokens", "attention", "seed"), "same")
    fp = {"request": {**request, "batch_size": 4, "release_receipt_sha256": "bound"},
          "suites": module.summarize_pairs(a, a)}
    bf = {"request": {**request, "batch_size": 256, "candidate_dtype": "bfloat16", "source_fp32_receipt_sha256": "bound"},
          "suites": module.summarize_pairs(a, b)}
    monkeypatch.setattr(review, "load_comparison", lambda p: (fp if p == "fp" else bf,
        {"reference": a, "candidate": a if p == "fp" else b}))
    with pytest.raises(ValueError, match="batch_size"):
        review.review("fp", "bf")
    result = review.review("fp", "bf", independent_controls=True)
    assert not result["cross_precision_comparison_valid"]
    assert result["suggested_precision"] == "review_required"
    assert not result["publication_approved"]
    assert result["suites"]["eval_id"]["bf16"]["regressions"] == 1


def test_audit_attachment_preserves_weights_and_binds_source(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/publish/v1"))
    monkeypatch.setitem(sys.modules, "compare_champion", module)
    import finalize_fp32_audit as finalizer

    from axiom_world.models.champion_release import inventory

    stage = tmp_path / "model"
    stage.mkdir()
    (stage / "README.md").write_text("original card")
    (stage / "model.safetensors").write_bytes(b"unchanged fixture")
    receipt = {"status": "verified", "verification": {"dtype": "float32", "merge": {"passed": True},
        "standalone": {"passed": True}}, "files": inventory(stage)}
    original = json.dumps(receipt).encode()
    (tmp_path / "verified.json").write_bytes(original)
    source_hash = hashlib.sha256(original).hexdigest()
    summaries = module.summarize_pairs([row("a", True)], [row("a", False)])
    fp, bf = tmp_path / "fp", tmp_path / "bf"
    for folder in (fp, bf):
        folder.mkdir()
        for name in ("comparison.json", "request.json", "reference.jsonl", "candidate.jsonl",
                     "reference_complete.json", "candidate_complete.json"):
            (folder / name).write_text("{}")
    monkeypatch.setattr(finalizer, "load_comparison", lambda p: ({"request": {
        "release_receipt_sha256": source_hash, "batch_size": 4 if p == fp else 256}, "suites": summaries}, {}))
    monkeypatch.setattr(finalizer, "review", lambda *a, **k: {"deployment_checks": {"eval_id": {"no_loss": False}}})
    finalizer.finalize(tmp_path, fp, bf)
    assert (stage / "model.safetensors").read_bytes() == b"unchanged fixture"
    assert (tmp_path / "verified.before-task-audit.json").read_bytes() == original
    updated = json.loads((tmp_path / "verified.json").read_text())
    assert updated["files"] == inventory(stage)
    assert updated["task_audit_source_receipt_sha256"] == source_hash
    assert "batch sizes differed" in (stage / "README.md").read_text()
    with pytest.raises(ValueError):
        finalizer.finalize(tmp_path, fp, bf)

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

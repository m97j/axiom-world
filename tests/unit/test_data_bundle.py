from pathlib import Path

import pytest

from axiom_world.data.bundle import DataContractError, build_data_bundle, write_jsonl
from axiom_world.data.records import Message, PreferenceRecord, Provenance, SFTRecord


def _sft(record_id: str, family: str = "fam-a") -> SFTRecord:
    return SFTRecord(
        id=record_id,
        messages=[
            Message(role="user", content="state + goal prompt"),
            Message(role="assistant", content='{"actions":[{"type":"WAIT"}]}'),
        ],
        provenance=Provenance(source_type="synthetic", source_id="gen-v1"),
        scenario_family_id=family,
    )


def test_roundtrip_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    fp_written = write_jsonl(path, [_sft("r1"), _sft("r2")])
    bundle = build_data_bundle(path, "sft")
    assert len(bundle) == 2
    assert bundle.fingerprint == fp_written
    assert bundle.manifest["family_counts"] == {"fam-a": 2}


def test_frozen_fingerprint_mismatch_hard_fails(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [_sft("r1")])
    good = build_data_bundle(path, "sft").fingerprint
    write_jsonl(path, [_sft("r1"), _sft("r2")])  # dataset changed after freeze
    with pytest.raises(DataContractError, match="fingerprint mismatch"):
        build_data_bundle(path, "sft", expected_fingerprint=good)


def test_leakage_gate_blocks_eval_families(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [_sft("r1", family="fam-eval")])
    with pytest.raises(DataContractError, match="leakage gate"):
        build_data_bundle(path, "sft", forbidden_family_ids={"fam-eval"})


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [_sft("r1"), _sft("r1")])
    with pytest.raises(DataContractError, match="duplicate record id"):
        build_data_bundle(path, "sft")


def test_sft_chat_contract() -> None:
    with pytest.raises(ValueError):
        SFTRecord(
            id="bad",
            messages=[
                Message(role="assistant", content="a"),
                Message(role="user", content="b"),
            ],
            provenance=Provenance(source_type="synthetic", source_id="x"),
        )


def test_preference_requires_passed_chosen() -> None:
    with pytest.raises(ValueError, match="PASSED chosen"):
        PreferenceRecord(
            id="p1",
            prompt=[Message(role="user", content="q")],
            chosen="a",
            rejected="b",
            chosen_verification={"status": "failed", "score": 0.0, "verifier_version": "1.0"},
            rejected_verification={"status": "failed", "score": 0.0, "verifier_version": "1.0"},
            provenance=Provenance(source_type="generated", source_id="mine-v1"),
        )

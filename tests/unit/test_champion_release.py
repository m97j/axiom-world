"""Release boundary regressions: no network, no pretrained weights, no GPU."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_world.core.lineage import compute_adapter_sha256
from axiom_world.integrations import hf_sync
from axiom_world.models.champion_release import compare_logits, inventory

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/common/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapter_fixture(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"test-only")
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"run_id": "run-1", "output_adapter_sha256": compute_adapter_sha256(adapter)}))
    return adapter, lineage


def test_reused_local_run_cannot_impersonate_requested_run(tmp_path):
    adapter, lineage = adapter_fixture(tmp_path)
    fetch = script("fetch_run")
    assert fetch._verify(adapter, lineage, "run-1") == compute_adapter_sha256(adapter)
    with pytest.raises(SystemExit, match="MISMATCH"):
        fetch._verify(adapter, lineage, "run-2")
    with pytest.raises(SystemExit, match="SHA"):
        fetch._verify(adapter, lineage, "run-1", "sha256:" + "0" * 64)
    (adapter / "adapter_config.json").unlink()
    with pytest.raises(SystemExit, match="Missing"):
        fetch._verify(adapter, lineage, "run-1")


def test_partial_fetch_does_not_create_verified_artifacts(tmp_path, monkeypatch):
    fetch = script("fetch_run")
    monkeypatch.setattr(sys, "argv", ["fetch", "--repo", "o/r", "--run-id", "run-1", "--workspace", str(tmp_path)])
    monkeypatch.setattr(hf_sync, "repository_revision", lambda *a: "a" * 40)
    def incomplete(**kwargs):
        (Path(kwargs["local_dir"]) / "artifacts").mkdir()
    monkeypatch.setattr(hf_sync, "download_snapshot", incomplete)
    with pytest.raises(FileNotFoundError):
        fetch.main()
    assert not (tmp_path / "runs/run-1/artifacts").exists()


def test_prune_execution_and_champion_rejected_before_network(monkeypatch):
    prune = script("hf_prune_checkpoints")
    for args in [["--repo", "m97j/aw-qwen3-8b-v1"],
                 ["--repo", "m97j/aw-runs-b4", "--execute"],
                 ["--repo", "m97j/aw-runs-b4", "--keep-last", "-1"]]:
        monkeypatch.setattr(sys, "argv", ["prune", *args])
        with pytest.raises(SystemExit) as exc:
            prune.main()
        assert exc.value.code == 2


def test_numeric_gate_rejects_nan_and_large_drift():
    torch = pytest.importorskip("torch")
    ref = {"0": torch.tensor([[1.0, 2.0]])}
    assert compare_logits(ref, ref, [[1]], [[1]])["passed"]
    assert not compare_logits(ref, {"0": ref["0"] + 1}, [[1]], [[1]])["passed"]
    with pytest.raises(ValueError, match="Non-finite"):
        compare_logits(ref, {"0": torch.full((1, 2), float("nan"))}, [[1]], [[1]])


def test_changed_payload_refuses_publish_before_network(tmp_path):
    stage = tmp_path / "model"
    stage.mkdir()
    (stage / "config.json").write_text("{}")
    receipt = {"status": "verified", "files": inventory(stage)}
    (tmp_path / "verified.json").write_text(json.dumps(receipt))
    (stage / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        script("publish_champion").publish(tmp_path, execute=True)


def test_atomic_migration_preserves_history_and_has_parent_guard(monkeypatch, tmp_path):
    calls = []
    class API:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="a" * 40)
        def list_repo_refs(self, *args, **kwargs):
            return SimpleNamespace(tags=[])
        def create_tag(self, *args, **kwargs):
            calls.append(("tag", kwargs))
        def create_commit(self, **kwargs):
            calls.append(("commit", kwargs))
            return SimpleNamespace(oid="b" * 40)
    class Op:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=API, CommitOperationAdd=Op, CommitOperationDelete=Op))
    (tmp_path / "config.json").write_text("{}")
    sha = hf_sync.commit_model_release(repo_id="o/r", stage=tmp_path,
        expected_head="a" * 40, legacy_revision="a" * 40,
        delete_paths=["adapter_config.json", "adapter_model.safetensors"])
    assert sha == "b" * 40
    assert [c[0] for c in calls] == ["tag", "commit"]
    commit = calls[-1][1]
    assert commit["parent_commit"] == "a" * 40
    assert [o.path_in_repo for o in commit["operations"]] == [
        "adapter_config.json", "adapter_model.safetensors", "config.json"]
    calls.clear()
    with pytest.raises(ValueError, match="Target changed"):
        hf_sync.commit_model_release(repo_id="o/r", stage=tmp_path,
            expected_head="c" * 40, legacy_revision="a" * 40, delete_paths=[])
    assert calls == []

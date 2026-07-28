from pathlib import Path

import pytest

from axiom_world.core.config_loader import apply_overrides, deep_merge, resolve
from axiom_world.core.enums import InitializationMode, Track
from axiom_world.core.errors import ConfigError

FIXTURE = {
    "project": {"name": "axiom-world", "protocol_version": "v1.0"},
    "experiment_name": "t-exp",
    "track": "track_a_direct",
    "phase": "phase2_playworld",
    "objective": "sft",
    "model": {"repo_id": "Qwen/Qwen3-8B-Base", "revision": "abc123"},
}


def _write_recipe(tmp_path: Path, payload: dict, name: str = "recipe.yaml") -> Path:
    import yaml

    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_resolve_valid_recipe(tmp_path: Path) -> None:
    path = _write_recipe(tmp_path, FIXTURE)
    config, fingerprint, _ = resolve(path)
    assert config.track is Track.A_DIRECT
    assert fingerprint.startswith("sha256:")
    assert config.validate_canonical() == []


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = dict(FIXTURE, unknown_key=1)
    path = _write_recipe(tmp_path, bad)
    with pytest.raises(ConfigError):
        resolve(path)


def test_extends_composition(tmp_path: Path) -> None:
    import yaml

    base = {"project": {"name": "axiom-world", "protocol_version": "v1.0"}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    child = dict(FIXTURE)
    child.pop("project")
    child["extends"] = ["base.yaml"]
    path = _write_recipe(tmp_path, child)
    config, _, _ = resolve(path)
    assert config.project.protocol_version == "v1.0"


def test_circular_extends_detected(tmp_path: Path) -> None:
    import yaml

    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"extends": ["b.yaml"]}), encoding="utf-8")
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"extends": ["a.yaml"]}), encoding="utf-8")
    with pytest.raises(ConfigError):
        resolve(tmp_path / "a.yaml")


def test_override_types_coerced(tmp_path: Path) -> None:
    path = _write_recipe(tmp_path, FIXTURE)
    config, _, _ = resolve(path, ["runtime.seed=43", "training.learning_rate=2.0e-5"])
    assert config.runtime.seed == 43
    assert config.training.learning_rate == pytest.approx(2.0e-5)


def test_fingerprint_changes_with_override(tmp_path: Path) -> None:
    path = _write_recipe(tmp_path, FIXTURE)
    _, fp1, _ = resolve(path)
    _, fp2, _ = resolve(path, ["runtime.seed=43"])
    assert fp1 != fp2


def test_canonical_violation_fa2_outside_benchmark(tmp_path: Path) -> None:
    bad = dict(FIXTURE)
    bad["runtime"] = {"attention_backend": "flash_attention_2"}
    path = _write_recipe(tmp_path, bad)
    config, _, _ = resolve(path)
    violations = config.validate_canonical()
    assert any("sdpa" in v for v in violations)


def test_canonical_violation_track_b_requires_parent(tmp_path: Path) -> None:
    bad = dict(FIXTURE, track="track_b_two_stage")
    path = _write_recipe(tmp_path, bad)
    config, _, _ = resolve(path)
    violations = config.validate_canonical()
    assert any("Phase-1 champion" in v for v in violations)
    assert config.lineage.initialization_mode is InitializationMode.FROM_BASE


def test_deep_merge_and_overrides_pure() -> None:
    merged = deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}}
    out = apply_overrides({"a": {"b": 1}}, ["a.b=5"])
    assert out["a"]["b"] == 5

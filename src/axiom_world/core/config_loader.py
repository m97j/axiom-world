"""Single config composition mechanism.

Rules (replacing the pseudo-Hydra of the previous snapshot):
- A recipe YAML may declare ``extends: [relative/path.yaml, ...]``. Bases are
  deep-merged in order, recipe last (recipe wins).
- No ``defaults:`` lists, no ``${...}`` interpolation, no package aliases.
  What this loader supports is exactly what the YAML files may use.
- Dotlist overrides (``a.b.c=value``) are applied after composition.
- The composed mapping is validated into ``ExperimentConfig`` (strict).
- ``resolve()`` returns (config, fingerprint, resolved_mapping).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from axiom_world.core.errors import ConfigError
from axiom_world.core.fingerprints import fingerprint_payload
from axiom_world.core.schemas import ExperimentConfig

_EXTENDS_KEY = "extends"
_MAX_DEPTH = 10


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML must be a mapping: {path}")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _compose(path: Path, depth: int = 0, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    resolved_path = path.resolve()
    if depth > _MAX_DEPTH:
        raise ConfigError(f"extends nesting exceeds {_MAX_DEPTH}: {path}")
    if resolved_path in seen:
        raise ConfigError(f"Circular extends detected at: {path}")
    raw = _load_yaml(resolved_path)
    bases = raw.pop(_EXTENDS_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ConfigError(f"'{_EXTENDS_KEY}' must be a string or list: {path}")
    composed: dict[str, Any] = {}
    for base_rel in bases:
        base_path = (resolved_path.parent / str(base_rel)).resolve()
        composed = deep_merge(
            composed, _compose(base_path, depth + 1, seen | {resolved_path})
        )
    return deep_merge(composed, raw)


def _coerce_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(mapping: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(mapping)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must be key.path=value, got: {item!r}")
        dotted, _, raw_value = item.partition("=")
        keys = dotted.strip().split(".")
        if not all(keys):
            raise ConfigError(f"Malformed override key: {item!r}")
        cursor = result
        for key in keys[:-1]:
            node = cursor.get(key)
            if node is None:
                node = {}
                cursor[key] = node
            if not isinstance(node, dict):
                raise ConfigError(
                    f"Override path {dotted!r} collides with non-mapping value at {key!r}"
                )
            cursor = node
        cursor[keys[-1]] = _coerce_scalar(raw_value)
    return result


def resolve(
    recipe_path: Path | str,
    overrides: list[str] | None = None,
) -> tuple[ExperimentConfig, str, dict[str, Any]]:
    """Compose, override, validate. Returns (config, fingerprint, mapping)."""
    mapping = _compose(Path(recipe_path))
    if overrides:
        mapping = apply_overrides(mapping, overrides)
    try:
        config = ExperimentConfig.model_validate(mapping)
    except ValidationError as exc:
        raise ConfigError(f"Config validation failed for {recipe_path}:\n{exc}") from exc
    fingerprint = fingerprint_payload(config.model_dump(mode="json"))
    return config, fingerprint, mapping

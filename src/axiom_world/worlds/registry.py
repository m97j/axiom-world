"""Name-based resolution of world implementations.

Configs name a world as a string; nothing outside this module needs to know
which class implements it. Registration is explicit -- no import-time scanning,
no plugin discovery -- so that the set of available worlds is a fact you can read
rather than infer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

WorldFactory = Callable[..., Any]

_REGISTRY: dict[str, WorldFactory] = {}


def register_world(name: str, factory: WorldFactory) -> None:
    """Register ``factory`` under ``name``.

    Raises:
        ValueError: if ``name`` is already registered. Silent replacement would
            let a stray import change which world a config resolves to, which is
            precisely the kind of provenance failure this project audits for.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"world name must be a non-empty string, got {name!r}")
    if name in _REGISTRY:
        raise ValueError(
            f"world already registered: {name!r}; "
            "registration is deliberate and must not be overwritten"
        )
    _REGISTRY[name] = factory


def get_world(name: str, **kwargs: Any) -> Any:
    """Instantiate the world registered under ``name``.

    Raises:
        KeyError: if ``name`` is unknown. The message lists known names, because
            a config typo should not look like a missing dependency.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown world: {name!r}; known worlds: {list_worlds()}"
        ) from None
    return factory(**kwargs)


def list_worlds() -> list[str]:
    """Return the registered world names, sorted."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    """Return whether ``name`` resolves to a world."""
    return name in _REGISTRY

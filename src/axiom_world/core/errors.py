"""Canonical exception hierarchy. All framework errors derive from AxiomError."""
from __future__ import annotations


class AxiomError(Exception):
    """Base class for all Axiom-World errors."""


class ConfigError(AxiomError):
    """Config composition, override, or validation failure."""


class ArtifactError(AxiomError):
    """Artifact persistence or retrieval failure."""


class LineageError(AxiomError):
    """Lineage contract violation (protocol §11). Always a hard failure."""


class RunContractError(AxiomError):
    """A run is missing required artifacts or violates status transitions."""


class EnvironmentError_(AxiomError):
    """Canonical runtime environment contract violation (protocol §3)."""


class RewardHealthError(AxiomError):
    """Online-RL reward stream is degenerate (protocol §7.4 fail-fast guard).

    Raised when the excluded (None-reward) fraction stays above threshold
    after a minimum number of reward calls — e.g. the 2026-08-15 B6 incident
    where Arrow struct unification injected None into nested scenario dicts
    and every completion became INFRA_ERROR, silently zeroing all gradients
    for 17 hours. Failing fast converts that silent waste into a hard error.
    """

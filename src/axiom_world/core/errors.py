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

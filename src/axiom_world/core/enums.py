"""Canonical enums for Axiom-World.

This module is the SINGLE source of truth for cross-layer enums.
Historical drift (``ArtifactType`` vs ``ArtifactKind``, ``RunContext`` vs
``ExperimentContext``) is resolved here: only the names below exist.
"""
from __future__ import annotations

from enum import StrEnum


class ArtifactKind(StrEnum):
    """Every persisted run artifact declares exactly one kind."""

    CONFIG = "config"
    MANIFEST = "manifest"
    METRICS = "metrics"
    STATE = "state"
    CHECKPOINT = "checkpoint"
    DATASET = "dataset"
    EVALUATION = "evaluation"
    LOG = "log"
    REPORT = "report"


class Phase(StrEnum):
    PHASE1_GENERAL = "phase1_general"
    PHASE2_PLAYWORLD = "phase2_playworld"


class Track(StrEnum):
    """Experiment tracks pre-registered in docs/experimental-protocol.md §5."""

    A_DIRECT = "track_a_direct"
    B_TWO_STAGE = "track_b_two_stage"
    C_REFERENCE = "track_c_reference"
    ABLATION = "ablation"
    SYSTEM_BENCHMARK = "system_benchmark"
    EXPLORATORY = "exploratory"


class Objective(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    RLOO = "rloo"
    EVAL_ONLY = "eval_only"
    DATA_BUILD = "data_build"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"  # stopped by a protocol stopping rule (§10)


class InitializationMode(StrEnum):
    """How a training run initializes its trainable parameters."""

    FROM_BASE = "from_base"
    CONTINUE_PARENT_ADAPTER = "continue_training_existing_adapter"


class VerificationStatus(StrEnum):
    """Verifier status semantics (docs/verifier-contract.md)."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INDETERMINATE = "indeterminate"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"

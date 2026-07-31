"""Typed configuration and artifact schemas (Pydantic v2, strict).

Design rules:
- ``extra="forbid"`` everywhere: an unknown key is a config bug, not a warning.
- Field names here are THE contract. Trainers/adapters must consume these
  names; no layer may invent parallel field names (the failure mode of the
  previous snapshot).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from axiom_world.core.enums import (
    InitializationMode,
    Objective,
    Phase,
    RunStatus,
    Track,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)


# ---------------------------------------------------------------------------
# Configuration schema (composed YAML -> validated AxiomConfig)
# ---------------------------------------------------------------------------


class ProjectConfig(StrictModel):
    name: str = "axiom-world"
    protocol_version: str = Field(
        description="Version of docs/experimental-protocol.md this run obeys."
    )


class RuntimeConfig(StrictModel):
    seed: int = 42
    device: Literal["cuda", "cpu"] = "cuda"
    precision: Literal["bf16", "fp32"] = "bf16"
    attention_backend: Literal["sdpa", "flash_attention_2"] = "sdpa"
    environment_policy: Literal["strict", "warn"] = "strict"
    expected_gpu_name_substring: str | None = None
    expected_min_vram_gb: float | None = None

    @field_validator("attention_backend")
    @classmethod
    def _canonical_attention(cls, value: str) -> str:
        # FA2 is allowed only for system benchmarks; enforcement of "never in
        # canonical result runs" happens in validate_canonical() below where
        # track information is available.
        return value


class ModelConfig(StrictModel):
    repo_id: str
    revision: str = Field(description="Exact HF revision hash; 'main' is rejected in strict mode.")
    tokenizer_repo_id: str | None = None
    max_seq_length: int = 4096


class AdapterConfig(StrictModel):
    kind: Literal["lora", "qlora"] = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=list)


class ParentAdapterRef(StrictModel):
    """Reference to the Phase-1 champion adapter (protocol §11)."""

    repo_id: str
    revision: str
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class LineageConfig(StrictModel):
    initialization_mode: InitializationMode = InitializationMode.FROM_BASE
    parent_adapter: ParentAdapterRef | None = None
    parent_run_id: str | None = None

    @field_validator("parent_adapter")
    @classmethod
    def _mode_consistency(cls, value: ParentAdapterRef | None, info: Any) -> ParentAdapterRef | None:
        return value


class DataSourceConfig(StrictModel):
    repo_id: str | None = None
    revision: str | None = None
    split: str | None = None
    local_path: str | None = None
    fingerprint: str | None = Field(
        default=None,
        description="Expected dataset fingerprint; verified at load time when set.",
    )


class DataConfig(StrictModel):
    source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    eval_suites: dict[str, DataSourceConfig] = Field(default_factory=dict)


class TrainingConfig(StrictModel):
    learning_rate: float = 1.0e-5
    max_steps: int | None = None
    num_train_epochs: float | None = None
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    max_grad_norm: float = 1.0
    max_length: int = 4096
    """Hard cap on tokenized sequence length passed to the TRL trainer config.

    NEVER leave this to the TRL default (1024 in TRL 1.x): PlayWorld prompts
    routinely exceed 1024 tokens, and right-truncation silently deletes the
    assistant completion — the model then 'learns' to continue prompts instead
    of answering them (root cause of the A1 eval collapse, see e03 diagnostic).
    """
    assistant_only_loss: bool = False
    """Forwarded to TRL SFTConfig when True. Left False by default because
    Qwen3-Base's chat template lacks the `{% generation %}` keyword TRL needs
    for assistant token masking; enabling it without template support raises.
    """
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Objective-specific knobs (e.g. dpo.beta, grpo.num_generations). "
        "Consumed only by the matching runner; validated there.",
    )


class ExperimentConfig(StrictModel):
    """Root config. One YAML recipe == one validated ExperimentConfig."""

    project: ProjectConfig
    experiment_name: str
    track: Track
    phase: Phase
    objective: Objective
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    lineage: LineageConfig = Field(default_factory=LineageConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    def validate_canonical(self) -> list[str]:
        """Protocol-level cross-field checks. Returns a list of violations.

        These are the rules that make a run 'canonical' per the protocol:
        empty list => eligible for main tables.
        """
        violations: list[str] = []
        if self.track is not Track.SYSTEM_BENCHMARK and (
            self.runtime.attention_backend != "sdpa"
        ):
            violations.append(
                "attention_backend must be 'sdpa' outside system benchmarks (protocol §3)."
            )
        if self.track is not Track.ABLATION and self.adapter.kind == "qlora":
            violations.append("QLoRA is ablation-only (protocol §3/§5.2 E-QLORA).")
        if self.runtime.environment_policy == "strict" and self.model.revision in {"main", ""}:
            violations.append(
                "model.revision must be an exact commit hash in strict mode (protocol §4.1)."
            )
        if (
            self.lineage.initialization_mode is InitializationMode.CONTINUE_PARENT_ADAPTER
            and self.lineage.parent_adapter is None
        ):
            violations.append(
                "initialization_mode=continue_training_existing_adapter requires "
                "lineage.parent_adapter (protocol §11)."
            )
        if (
            self.lineage.initialization_mode is InitializationMode.FROM_BASE
            and self.lineage.parent_adapter is not None
        ):
            violations.append(
                "parent_adapter is set but initialization_mode is from_base; "
                "ambiguous lineage is forbidden."
            )
        if self.phase is Phase.PHASE2_PLAYWORLD and self.track is Track.B_TWO_STAGE:
            if self.lineage.initialization_mode is not InitializationMode.CONTINUE_PARENT_ADAPTER:
                violations.append(
                    "Track B Phase 2 runs must continue the Phase-1 champion adapter "
                    "(protocol §5.1 B4-B6)."
                )
        if self.phase is Phase.PHASE1_GENERAL and (
            self.lineage.initialization_mode is not InitializationMode.FROM_BASE
            and self.objective is Objective.SFT
            and self.lineage.parent_run_id is None
        ):
            violations.append(
                "Phase 1 SFT (B1/B2) must initialize from base; only B3 preference "
                "stages may continue a P1 adapter (set lineage.parent_run_id)."
            )
        return violations


# ---------------------------------------------------------------------------
# Artifact payload schemas
# ---------------------------------------------------------------------------


class GitState(StrictModel):
    commit: str
    branch: str | None = None
    dirty: bool = False


class LineageRecord(StrictModel):
    """Persisted lineage.json (protocol §11)."""

    run_id: str
    phase: Phase
    base_model_repo_id: str
    base_model_revision: str
    initialization_mode: InitializationMode
    parent_adapter: ParentAdapterRef | None = None
    parent_run_id: str | None = None
    dataset_fingerprints: dict[str, str] = Field(default_factory=dict)
    config_fingerprint: str
    code_commit: str | None = None
    output_adapter_sha256: str | None = None


class RunCard(StrictModel):
    """Persisted run_card.json — human-facing summary of one run."""

    run_id: str
    experiment_name: str
    track: Track
    phase: Phase
    objective: Objective
    protocol_version: str
    seed: int
    status: RunStatus
    config_fingerprint: str
    created_at: str
    updated_at: str
    notes: str = ""

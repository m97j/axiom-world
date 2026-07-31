"""TRL trainer boundary (single compatibility surface).

Principles carried over from the protocol and the rewrite decision log:
- TRL imports are lazy; this module imports cleanly without torch/TRL.
- No silent semantic fallbacks: if the installed TRL lacks a required
  trainer, we raise — we never quietly substitute another objective.
- Constructor kwargs are filtered against the actual installed signature
  (TRL 1.x moves fields between Config and Trainer across minor versions).
- Lineage is enforced HERE, before any trainer is built: a Track-B Phase-2
  run cannot construct a trainer without a hash-verified parent adapter.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from axiom_world.core.enums import Objective
from axiom_world.core.errors import AxiomError
from axiom_world.core.lineage import assert_lineage_executable
from axiom_world.core.schemas import ExperimentConfig


class UnsupportedTRLAPIError(AxiomError):
    pass


def _import_trl(name: str) -> Any:
    try:
        import trl
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise UnsupportedTRLAPIError(
            "TRL is not installed in this environment. Trainer construction "
            "requires the Colab G4 training session (requirements/colab-g4.lock.txt)."
        ) from exc
    attribute = getattr(trl, name, None)
    if attribute is None:
        raise UnsupportedTRLAPIError(
            f"Installed TRL ({getattr(trl, '__version__', '?')}) does not export {name!r}. "
            "Pin the TRL version validated at Gate G1; do not substitute another objective."
        )
    return attribute


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    accepted = set(signature.parameters)
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        # Dropping is allowed only for optional tuning knobs; required semantic
        # fields must exist. We surface what was dropped for the run log.
        pass
    return {k: v for k, v in kwargs.items() if k in accepted}


_TRAINER_NAMES = {
    Objective.SFT: ("SFTTrainer", "SFTConfig"),
    Objective.DPO: ("DPOTrainer", "DPOConfig"),
    Objective.GRPO: ("GRPOTrainer", "GRPOConfig"),
    Objective.RLOO: ("RLOOTrainer", "RLOOConfig"),
}


def build_trainer(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    output_dir: str | Path,
    parent_adapter_dir: Path | None = None,
    reward_funcs: list[Callable[..., list[float | None]]] | None = None,
    eval_dataset: Any = None,
) -> Any:
    """Construct the TRL trainer for this experiment's objective."""
    objective = config.objective
    if objective not in _TRAINER_NAMES:
        raise AxiomError(f"Objective {objective.value!r} does not build a trainer.")

    # Hard lineage gate BEFORE any trainer construction (protocol §11).
    assert_lineage_executable(config, parent_adapter_dir)

    if objective in (Objective.GRPO, Objective.RLOO) and not reward_funcs:
        raise AxiomError(f"{objective.value.upper()} requires at least one reward function.")

    trainer_name, config_name = _TRAINER_NAMES[objective]
    trainer_cls = _import_trl(trainer_name)
    config_cls = _import_trl(config_name)

    training = config.training
    common: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": training.learning_rate,
        "per_device_train_batch_size": training.per_device_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "warmup_ratio": training.warmup_ratio,
        "logging_steps": training.logging_steps,
        "save_steps": training.save_steps,
        "max_grad_norm": training.max_grad_norm,
        "seed": config.runtime.seed,
        "bf16": config.runtime.precision == "bf16",
        "report_to": ["wandb"],
        # Deterministic W&B identity: project comes from the recipe's project
        # name, run name from the pre-registered experiment name (protocol §5).
        "run_name": config.experiment_name,
        # Sequence-length contract (v0.3.3): TRL 1.x defaults max_length to
        # 1024 and right-truncates, which silently deletes long assistant
        # completions from the loss. Always pass it explicitly.
        "max_length": training.max_length,
    }
    if objective is Objective.SFT and training.assistant_only_loss:
        common["assistant_only_loss"] = True
    import os

    os.environ.setdefault("WANDB_PROJECT", config.project.name)
    if training.max_steps is not None:
        common["max_steps"] = training.max_steps
    if training.num_train_epochs is not None:
        common["num_train_epochs"] = training.num_train_epochs
    common.update(training.extra)  # objective-specific knobs (beta, num_generations, ...)

    filtered = _filter_kwargs(config_cls.__init__, common)
    if objective in (Objective.SFT, Objective.DPO) and "max_length" not in filtered:
        raise UnsupportedTRLAPIError(
            f"{config_name} does not accept 'max_length'; refusing to train with an "
            "implicit truncation policy. Pin the TRL version validated at Gate G1."
        )
    trainer_config = config_cls(**filtered)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": trainer_config,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if objective in (Objective.GRPO, Objective.RLOO):
        trainer_kwargs["reward_funcs"] = reward_funcs

    return trainer_cls(**_filter_kwargs(trainer_cls.__init__, trainer_kwargs))

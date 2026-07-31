"""Model builder — Qwen3-8B + BF16 LoRA canonical path (protocol §3).

torch/transformers/peft are imported lazily; this module imports cleanly on
CPU-only contract environments. QLoRA (NF4) is constructed only when the
config's adapter.kind == 'qlora' (ablation E-QLORA).

Parent-adapter continuation (Track B): the caller passes the LOCAL directory
of the hash-verified Phase-1 adapter; lineage verification happens in
training.factory BEFORE this function is reached, and is re-checked here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from axiom_world.core.enums import InitializationMode
from axiom_world.core.errors import AxiomError
from axiom_world.core.lineage import verify_parent_adapter
from axiom_world.core.schemas import ExperimentConfig


def _load_base_and_tokenizer(config: ExperimentConfig) -> tuple[Any, Any]:
    """Load the pinned base model + tokenizer (shared by training/inference)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.tokenizer_repo_id or model_cfg.repo_id,
        revision=model_cfg.revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "revision": model_cfg.revision,
        "dtype": torch.bfloat16 if config.runtime.precision == "bf16" else torch.float32,
        "attn_implementation": config.runtime.attention_backend,
        "device_map": "auto" if config.runtime.device == "cuda" else None,
    }

    if config.adapter.kind == "qlora":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_cfg.repo_id, **load_kwargs)
    return model, tokenizer


def build_for_inference(
    config: ExperimentConfig,
    adapter_dir: Path | str | None = None,
) -> tuple[Any, Any]:
    """Frozen inference model: base only, or base + a TRAINED adapter.

    Central inference entry point (no throwaway LoRA wrapper, no gradient
    checkpointing). Callers must not attach adapters themselves.
    """
    model, tokenizer = _load_base_and_tokenizer(config)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def build_model_and_tokenizer(
    config: ExperimentConfig,
    parent_adapter_dir: Path | None = None,
) -> tuple[Any, Any]:
    model, tokenizer = _load_base_and_tokenizer(config)
    model.config.use_cache = False

    mode = config.lineage.initialization_mode
    if mode is InitializationMode.CONTINUE_PARENT_ADAPTER:
        if parent_adapter_dir is None or config.lineage.parent_adapter is None:
            raise AxiomError("Parent adapter continuation requires a verified local adapter dir.")
        verify_parent_adapter(config.lineage.parent_adapter, parent_adapter_dir)
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(parent_adapter_dir), is_trainable=True
        )
    else:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=config.adapter.r,
            lora_alpha=config.adapter.alpha,
            lora_dropout=config.adapter.dropout,
            target_modules=config.adapter.target_modules or None,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

    if config.runtime.device == "cuda":
        model.gradient_checkpointing_enable()
    return model, tokenizer

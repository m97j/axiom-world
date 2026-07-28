"""Runtime environment audit (protocol §3).

Collects the environment manifest every canonical run must persist, and
enforces the strict environment policy (expected GPU / VRAM) when configured.
torch is imported lazily so the core package works on CPU-only machines
(tests, CI) — the strict gate then reports cuda_available=False.
"""
from __future__ import annotations

import logging
import platform
import sys
from typing import Any

from axiom_world.core.errors import EnvironmentError_
from axiom_world.core.schemas import RuntimeConfig

logger = logging.getLogger()

_AUDIT_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "tokenizers",
    "safetensors",
    "pydantic",
    "yaml",
)


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception as exc:
        logger.debug("runtime audit failed: %s", exc)
    return getattr(module, "__version__", "unknown")


def collect_environment_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in _AUDIT_PACKAGES},
        "cuda_available": False,
        "gpu": None,
    }
    try:
        import torch

        manifest["torch_cuda_version"] = torch.version.cuda
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        if manifest["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            manifest["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "sm_count": props.multi_processor_count,
            }
    except Exception:
        pass
    return manifest


def enforce_environment(runtime: RuntimeConfig, manifest: dict[str, Any]) -> list[str]:
    """Return violations; raise in strict mode if any exist."""
    violations: list[str] = []
    if runtime.device == "cuda" and not manifest.get("cuda_available"):
        violations.append("CUDA required but not available.")
    gpu = manifest.get("gpu") or {}
    if runtime.expected_gpu_name_substring:
        name = gpu.get("name", "")
        if runtime.expected_gpu_name_substring not in name:
            violations.append(
                f"GPU {name!r} does not match expected substring "
                f"{runtime.expected_gpu_name_substring!r}."
            )
    if runtime.expected_min_vram_gb is not None:
        vram = gpu.get("vram_gb", 0.0)
        if vram < runtime.expected_min_vram_gb:
            violations.append(
                f"VRAM {vram} GB below required {runtime.expected_min_vram_gb} GB."
            )
    if violations and runtime.environment_policy == "strict":
        raise EnvironmentError_(
            "Environment contract violations (strict policy): " + " | ".join(violations)
        )
    return violations

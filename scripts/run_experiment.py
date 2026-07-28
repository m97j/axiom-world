#!/usr/bin/env python
"""Run one pre-registered experiment recipe end-to-end (protocol §5, §11).

Flow:
  resolve config -> canonical validation -> environment enforcement
  -> init run dir -> load data bundle (fingerprint + leakage gate)
  -> build model (+ verified parent adapter for Track B)
  -> build TRL trainer -> train -> save adapter + required artifacts
  -> mark run completed (fails if any required artifact is missing)

Usage (Colab G4):
  python scripts/run_experiment.py \
    --config configs/experiments/a1_playworld_sft.yaml \
    --override model.revision=<hash> \
    --override data.source.local_path=data/train/playworld_sft.jsonl \
    --eval-freeze-manifest data/eval_suites/freeze_manifest.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from axiom_world.core.config_loader import resolve
from axiom_world.core.context import ExperimentContext
from axiom_world.core.enums import ArtifactKind, Objective, RunStatus
from axiom_world.core.lineage import build_lineage_record, compute_adapter_sha256
from axiom_world.data.bundle import build_data_bundle
from axiom_world.runtime.audit import collect_environment_manifest, enforce_environment
from axiom_world.training.adapter import to_dpo_rows, to_grpo_rows, to_sft_rows

_BUNDLE_KIND = {
    Objective.SFT: "sft",
    Objective.DPO: "preference",
    Objective.GRPO: "evaluation",
    Objective.RLOO: "evaluation",
}
_ROW_FN = {
    Objective.SFT: to_sft_rows,
    Objective.DPO: to_dpo_rows,
    Objective.GRPO: to_grpo_rows,
    Objective.RLOO: to_grpo_rows,
}


def _git_state() -> dict[str, object]:
    import subprocess

    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    return {
        "commit": _run("rev-parse", "HEAD") or "unknown",
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def main() -> int:   # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--parent-adapter-dir", default=None,
                        help="Local dir of the hash-verified parent adapter (Track B).")
    parser.add_argument("--eval-freeze-manifest", default="data/eval_suites/freeze_manifest.json")
    args = parser.parse_args()

    config, fingerprint, mapping = resolve(args.config, args.override)
    violations = config.validate_canonical()
    if violations:
        raise SystemExit("canonical violations: " + " | ".join(violations))
    manifest = collect_environment_manifest()
    enforce_environment(config.runtime, manifest)

    ctx = ExperimentContext(config, fingerprint, Path(args.workspace))
    ctx.initialize(mapping)
    ctx.write_json_artifact("environment_manifest.json", manifest, ArtifactKind.MANIFEST)
    ctx.write_json_artifact("git_state.json", _git_state(), ArtifactKind.MANIFEST)
    print(f"run_id: {ctx.run_id}")

    # --- data (fingerprint + leakage gate) ---------------------------------
    freeze_path = Path(args.eval_freeze_manifest)
    forbidden: set[str] = set()
    if freeze_path.is_file():
        forbidden = set(json.loads(freeze_path.read_text())["eval_family_ids"])
    source = config.data.source
    if not source.local_path:
        raise SystemExit("data.source.local_path is required by this runner.")
    bundle = build_data_bundle(
        source.local_path,
        _BUNDLE_KIND[config.objective],
        expected_fingerprint=source.fingerprint,
        forbidden_family_ids=forbidden,
    )
    ctx.write_json_artifact("dataset_manifest.json", bundle.manifest, ArtifactKind.DATASET)

    lineage = build_lineage_record(
        ctx.run_id, config, fingerprint,
        dataset_fingerprints={bundle.kind: bundle.fingerprint},
        code_commit=str(_git_state()["commit"]),
    )

    # --- model + trainer -----------------------------------------------------
    from datasets import Dataset

    from axiom_world.models.builder import build_model_and_tokenizer
    from axiom_world.training.factory import build_trainer

    parent_dir = Path(args.parent_adapter_dir) if args.parent_adapter_dir else None
    model, tokenizer = build_model_and_tokenizer(config, parent_dir)
    train_dataset = Dataset.from_list(_ROW_FN[config.objective](bundle))

    reward_funcs = None
    status_counter: Counter = Counter()
    if config.objective in (Objective.GRPO, Objective.RLOO):
        from axiom_world.training.reward_bridge import verifier_reward_function
        from axiom_world.verifiers.hybrid import default_playworld_verifier

        reward_funcs = [verifier_reward_function(default_playworld_verifier(), status_counter)]

    trainer = build_trainer(
        config, model, tokenizer, train_dataset,
        output_dir=ctx.paths.checkpoints_dir,
        parent_adapter_dir=parent_dir,
        reward_funcs=reward_funcs,
    )

    ctx.transition(RunStatus.RUNNING)
    try:
        result = trainer.train()
        adapter_dir = ctx.paths.artifacts_dir / "final_adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        lineage.output_adapter_sha256 = compute_adapter_sha256(adapter_dir)

        metrics = dict(result.metrics)
        if status_counter:
            metrics["verifier_status_counts"] = dict(status_counter)
        ctx.write_json_artifact("metrics.json", metrics, ArtifactKind.METRICS)
        ctx.write_json_artifact(
            "lineage.json", lineage.model_dump(mode="json"), ArtifactKind.MANIFEST
        )
        ctx.write_json_artifact(
            "checkpoint_pointer.json",
            {"final_adapter": str(adapter_dir),
             "adapter_sha256": lineage.output_adapter_sha256},
            ArtifactKind.CHECKPOINT,
        )
        ctx.write_json_artifact(
            "run_card.json", ctx.run_card().model_dump(mode="json"), ArtifactKind.MANIFEST
        )
        ctx.transition(RunStatus.COMPLETED)
        print(f"COMPLETED: {ctx.run_id}")
        print(f"final adapter sha256: {lineage.output_adapter_sha256}")
    except Exception:
        ctx.transition(RunStatus.FAILED)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

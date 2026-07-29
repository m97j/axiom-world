#!/usr/bin/env python
"""Evaluate a checkpoint/adapter on the frozen eval suites (protocol §7).

Canonical decoding profile: greedy, temperature 0, fixed max-new-tokens.
Writes evaluation.jsonl + evaluation_summary.json into a fresh eval-only run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve
from axiom_world.core.context import ExperimentContext
from axiom_world.core.enums import ArtifactKind
from axiom_world.data.bundle import build_data_bundle
from axiom_world.evaluation.runner import EvaluationRunner
from axiom_world.runtime.audit import collect_environment_manifest
from axiom_world.verifiers.hybrid import default_playworld_verifier


def make_generator(config, adapter_dir: str | None, max_new_tokens: int):
    import torch

    from axiom_world.models.builder import build_model_and_tokenizer

    model, tokenizer = build_model_and_tokenizer(config)
    if adapter_dir:
        from peft import PeftModel

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    @torch.no_grad()
    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        output = model.generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

    return generate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="eval_only recipe")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--suites-dir", default="data/eval_suites")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--hf-sync-repo",
        default=None,
        help="Private HF model repo (user/name). When set, the eval run's "
        "artifacts dir (evaluation_summary.json, manifests) is uploaded "
        "under runs/<run_id>/ after evaluation completes.",
    )
    args = parser.parse_args()

    config, fingerprint, mapping = resolve(args.config, args.override)
    ctx = ExperimentContext(config, fingerprint, Path(args.workspace))
    ctx.initialize(mapping)
    ctx.write_json_artifact(
        "environment_manifest.json", collect_environment_manifest(), ArtifactKind.MANIFEST
    )

    generator = make_generator(config, args.adapter_dir, args.max_new_tokens)
    runner = EvaluationRunner(default_playworld_verifier(), generator)

    suites_dir = Path(args.suites_dir)
    freeze = json.loads((suites_dir / "freeze_manifest.json").read_text())
    combined: dict[str, dict] = {}
    for suite_name, entry in freeze["suites"].items():
        bundle = build_data_bundle(
            suites_dir / f"{suite_name}.jsonl", "evaluation",
            expected_fingerprint=entry["fingerprint"],  # frozen-suite hard gate
        )
        result = runner.run(bundle)
        combined[suite_name] = result["summary"]["suites"][suite_name]
        print(f"{suite_name}: pass_rate={combined[suite_name].get('pass_rate')}")

    ctx.write_json_artifact(
        "evaluation_summary.json",
        {"adapter_dir": args.adapter_dir, "suites": combined,
         "freeze_fingerprint": freeze["manifest_fingerprint"]},
        ArtifactKind.EVALUATION,
    )
    print(f"eval run: {ctx.run_id}")

    if args.hf_sync_repo:
        from axiom_world.integrations.hf_sync import upload_directory

        uri = upload_directory(
            ctx.paths.artifacts_dir,
            args.hf_sync_repo,
            path_in_repo=f"runs/{ctx.run_id}",
            commit_message=f"eval artifacts: {ctx.run_id}",
        )
        print(f"eval artifacts synced -> {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

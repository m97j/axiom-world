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


def make_batch_generator(config, adapter_dir: str | None, max_new_tokens: int):
    """Batched greedy decoding (canonical profile). Returns (fn, stats).

    Left padding keeps every prompt right-aligned so batched greedy output
    is identical to batch-size-1 output. stats["truncated"] counts outputs
    that hit the max_new_tokens budget (possible malformed-JSON cause).
    """
    import torch

    from axiom_world.models.builder import build_for_inference

    model, tokenizer = build_for_inference(config, adapter_dir)
    tokenizer.padding_side = "left"
    stats = {"truncated": 0}

    # Stop on BOTH the tokenizer eos and the chat-template turn terminator.
    # Qwen templates end assistant turns with <|im_end|>, which differs from
    # the base tokenizer's eos (<|endoftext|>); without it greedy decoding
    # never stops, overruns the budget, and every output parses as malformed
    # JSON (the eval-1 failure mode: 1447/1500 truncated).
    stop_ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end is not None and im_end >= 0:
        stop_ids.add(im_end)
    stop_ids.discard(None)
    stop_ids = sorted(stop_ids)

    # Distribution-matching opener seed (v0.3.11, diagnostic e01-e08 chain).
    # Qwen3's chat template injects an EMPTY think block
    # '<think>\n\n</think>\n\n' at the start of every assistant training
    # target, but Qwen3-8B-BASE has effectively untrained <think>/</think>
    # embeddings and LoRA (attention/MLP only) cannot repair the opener:
    # the tuned adapter's distribution is flat garbage on those positions
    # (e05: top-1 p=0.014) yet emits exact PlayWorld JSON once the opener
    # is consumed (e05 forced k=4). Seeding the opener into the prompt
    # aligns eval conditioning with the training distribution; the seed is
    # NOT part of the scored completion.
    opener_seed = "<think>\n\n</think>\n\n"

    @torch.no_grad()
    def generate_batch(prompts: list[str]) -> list[str]:
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            + opener_seed
            for conversation in conversations
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )
        completions = output[:, prompt_len:]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for row in completions:
            if len(row) == max_new_tokens and not any(s in row for s in stop_ids):
                stats["truncated"] += 1
        return texts

    return generate_batch, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="eval_only recipe")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--suites-dir", default="data/eval_suites")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Prompts per generation batch (95GB VRAM: 16-32 is safe).")
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

    generator, gen_stats = make_batch_generator(config, args.adapter_dir, args.max_new_tokens)
    runner = EvaluationRunner(
        default_playworld_verifier(),
        batch_generator=generator,
        batch_size=args.batch_size,
    )

    suites_dir = Path(args.suites_dir)
    freeze = json.loads((suites_dir / "freeze_manifest.json").read_text())
    combined: dict[str, dict] = {}
    for suite_name, entry in freeze["suites"].items():
        bundle = build_data_bundle(
            suites_dir / f"{suite_name}.jsonl", "evaluation",
            expected_fingerprint=entry["fingerprint"],  # frozen-suite hard gate
        )
        result = runner.run(bundle, context=ctx)  # ctx => evaluation.jsonl traces persisted
        combined[suite_name] = result["summary"]["suites"][suite_name]
        print(f"{suite_name}: pass_rate={combined[suite_name].get('pass_rate')}")

    print(f"outputs truncated at max_new_tokens: {gen_stats['truncated']}")

    ctx.write_json_artifact(
        "evaluation_summary.json",
        {"adapter_dir": args.adapter_dir, "suites": combined,
         "decoding": {"max_new_tokens": args.max_new_tokens,
                      "batch_size": args.batch_size,
                      "truncated_outputs": gen_stats["truncated"]},
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

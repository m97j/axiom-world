#!/usr/bin/env python
"""Mine verifier-guided preference pairs from an SFT policy (protocol §5.2, RQ2).

Stage A2 (and E-RANDPAIR) data step: sample K candidates per training prompt
from the A1 adapter, score every candidate with the frozen hybrid verifier,
and mine chosen/rejected pairs via generation.pair_mining. Selection method is
pre-registered:
  - hybrid_verifier_rank : canonical A2/B5 mining (margin-gated).
  - random_pairing       : E-RANDPAIR control at EQUAL pair count.

Sampling uses the SAME conditioning as canonical eval (chat template +
config.evaluation.opener_seed) so candidates come from the policy's true
inference distribution — pairs then target exactly the failure modes the
verifier sees at eval time. Temperature sampling (default 0.8) provides
candidate diversity; the verifier, not likelihood, ranks them.

Every output record carries full verifier evidence and provenance; a mining
manifest (decision counts, fingerprints, config) is written next to the jsonl
and optionally synced to a HF dataset repo.

Usage (Colab, GPU cell):
  python scripts/mine_playworld_pairs.py \
      --config configs/experiments/eval_playworld.yaml \
      --adapter-dir runs/<a1-run>/artifacts/final_adapter \
      --prompt-file data/train/playworld_prompts.jsonl \
      --num-candidates 8 --batch-size 100 \
      --output data/train/playworld_preference.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="eval-only recipe (conditioning source)")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--prompt-file", default="data/train/playworld_prompts.jsonl")
    parser.add_argument("--output", default="data/train/playworld_preference.jsonl")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Generation sequences per batch (prompts x candidates).")
    parser.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    parser.add_argument("--selection-method", default="hybrid_verifier_rank",
                        choices=["hybrid_verifier_rank", "random_pairing"])
    parser.add_argument("--minimum-margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-sync-repo", default=None,
                        help="HF DATASET repo (user/name); uploads jsonl + manifest.")
    parser.add_argument("--hf-path-in-repo", default="preference_train/v1")
    args = parser.parse_args()

    import torch

    from axiom_world.data.bundle import write_jsonl
    from axiom_world.data.records import Message
    from axiom_world.generation.backend import Candidate
    from axiom_world.generation.pair_mining import PairMiningPolicy, mine_preference_pairs
    from axiom_world.models.builder import build_for_inference
    from axiom_world.verifiers.hybrid import default_playworld_verifier

    config, _, _ = resolve(args.config, [])
    model, tokenizer = build_for_inference(config, adapter_dir=args.adapter_dir)
    opener_seed = config.evaluation.opener_seed

    records = [
        json.loads(line)
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        records = records[: args.limit]

    stop_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.eos_token_id:
        stop_ids.append(im_end)

    @torch.no_grad()
    def sample_candidates(prompt_texts: list[str]) -> list[list[str]]:
        """Return num_candidates completions per prompt, batched."""
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True,
            ) + opener_seed
            for p in prompt_texts
        ]
        # Expand each prompt num_candidates times; generation batch respects
        # --batch-size in SEQUENCES to bound VRAM.
        expanded = [text for text in rendered for _ in range(args.num_candidates)]
        outputs: list[str] = []
        for start in range(0, len(expanded), args.batch_size):
            chunk = expanded[start : start + args.batch_size]
            inputs = tokenizer(
                chunk, return_tensors="pt", padding=True, add_special_tokens=False,
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids,
            )
            outputs.extend(
                tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            )
        return [
            outputs[i * args.num_candidates : (i + 1) * args.num_candidates]
            for i in range(len(prompt_texts))
        ]

    torch.manual_seed(args.seed)
    verifier = default_playworld_verifier()
    grouped: dict[str, list[Candidate]] = {}
    contexts: dict[str, dict] = {}
    prompts: dict[str, list[Message]] = {}

    from tqdm import tqdm

    prompt_chunk = max(1, args.batch_size // args.num_candidates)
    for start in tqdm(range(0, len(records), prompt_chunk), desc="mine(batched)"):
        batch = records[start : start + prompt_chunk]
        prompt_texts = [r["prompt"][-1]["content"] for r in batch]
        all_candidates = sample_candidates(prompt_texts)
        for record, candidate_texts in zip(batch, all_candidates, strict=True):
            source_id = record["id"]
            grouped[source_id] = [
                Candidate(
                    candidate_id=f"{source_id}-c{i}",
                    request_id=source_id,
                    source_record_id=source_id,
                    text=text.strip(),
                    index=i,
                    backend="hf-generate",
                    model_id=str(args.adapter_dir),
                )
                for i, text in enumerate(candidate_texts)
            ]
            contexts[source_id] = record
            prompts[source_id] = [Message(**m) for m in record["prompt"]]

    policy = PairMiningPolicy(
        selection_method=args.selection_method,
        minimum_margin=args.minimum_margin,
        seed=args.seed,
    )
    pairs, decisions = mine_preference_pairs(
        grouped, verifier, contexts, prompts, policy,
        provenance_source_id=f"pair-mining-{args.selection_method}-s{args.seed}",
    )

    output = Path(args.output)
    fingerprint = write_jsonl(output, pairs)
    manifest = {
        "adapter_dir": str(args.adapter_dir),
        "prompt_file": args.prompt_file,
        "prompts_processed": len(records),
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "selection_method": args.selection_method,
        "minimum_margin": args.minimum_margin,
        "seed": args.seed,
        "pairs_accepted": len(pairs),
        "decision_counts": decisions,
        "preference_fingerprint": fingerprint,
        "opener_seed": opener_seed,
    }
    manifest_path = output.with_name("preference_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.hf_sync_repo:
        from axiom_world.integrations.hf_sync import upload_directory

        destination = upload_directory(
            output.parent, args.hf_sync_repo,
            path_in_repo=args.hf_path_in_repo, repo_type="dataset",
            commit_message=f"preference pairs: {args.selection_method} s{args.seed}",
        )
        print(f"preference data synced -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

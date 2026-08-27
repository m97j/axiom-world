#!/usr/bin/env python
"""Mine exact-answer preference pairs from the P1 SFT champion (stage B3, §5.1).

B3 data step: sample K candidates per P1 training prompt from the B1v2
champion adapter, score each with the frozen ExactAnswerVerifier against the
gold answer carried in the P1 record metadata, and mine chosen/rejected pairs
via generation.pair_mining — the SAME mining engine and output schema as the
A2/B5 PlayWorld pipeline, so run_experiment's DPO path consumes it natively.

Prompt source is the FROZEN B1 mixture (data/p1/p1_general_sft.jsonl):
pairs target exactly the distribution the champion was tuned on, and
provenance mirrors the B2 RS builder (only verifiable records are used).

Usage (Colab, GPU cell):
  python scripts/mine_p1_pairs.py \
      --config configs/experiments/b3_general_dpo.yaml \
      --adapter-dir runs/<b1v2-run>/artifacts/final_adapter \
      --num-candidates 8 --batch-size 64 \
      --output data/p1/p1_general_preference.jsonl \
      --hf-sync-repo m97j/axiom-general-posttrain
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Recipe supplying model/tokenizer + eval conditioning.")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--input", default="data/p1/p1_general_sft.jsonl",
                        help="Frozen B1 mixture (prompt + gold-answer source).")
    parser.add_argument("--output", default="data/p1/p1_general_preference.jsonl")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Generation sequences per batch (prompts x candidates).")
    parser.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    parser.add_argument("--selection-method", default="hybrid_verifier_rank",
                        choices=["hybrid_verifier_rank", "random_pairing"])
    parser.add_argument("--minimum-margin", type=float, default=0.10,
                        help="Exact verifier scores are {0,1}: margin 0.10 keeps "
                             "only correct-vs-incorrect pairs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-sync-repo", default=None, help="HF DATASET repo.")
    parser.add_argument("--hf-path-in-repo", default="p1/pref-v1")
    args = parser.parse_args()

    import torch

    from axiom_world.data.bundle import read_jsonl, write_jsonl
    from axiom_world.data.records import Message
    from axiom_world.generation.backend import Candidate
    from axiom_world.generation.pair_mining import PairMiningPolicy, mine_preference_pairs
    from axiom_world.models.builder import build_for_inference
    from axiom_world.verifiers.general import ExactAnswerVerifier

    config, _, _ = resolve(args.config, [])
    model, tokenizer = build_for_inference(config, adapter_dir=args.adapter_dir)
    tokenizer.padding_side = "left"  # decoder-only: right padding corrupts outputs
    opener_seed = config.evaluation.opener_seed or ""

    records = read_jsonl(Path(args.input))
    verifiable = [
        r for r in records
        if r.get("metadata", {}).get("gold_answer") not in (None, "")
    ]
    if args.limit:
        verifiable = verifiable[: args.limit]

    stop_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.eos_token_id:
        stop_ids.append(im_end)

    @torch.no_grad()
    def sample_candidates(prompt_texts: list[str]) -> list[list[str]]:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True,
            ) + opener_seed
            for p in prompt_texts
        ]
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
    verifier = ExactAnswerVerifier()

    from tqdm import tqdm

    grouped: dict[str, list[Candidate]] = {}
    contexts: dict[str, dict] = {}
    prompts: dict[str, list[Message]] = {}
    prompt_chunk = max(1, args.batch_size // args.num_candidates)
    for start in tqdm(range(0, len(verifiable), prompt_chunk), desc="mine(batched)"):
        batch = verifiable[start : start + prompt_chunk]
        prompt_texts = [r["messages"][0]["content"] for r in batch]
        for record, candidate_texts in zip(batch, sample_candidates(prompt_texts), strict=True):
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
            # ExactAnswerVerifier context contract: {'answer': <gold>}
            contexts[source_id] = {"answer": record["metadata"]["gold_answer"],
                                   "family": record.get("metadata", {}).get("family")}
            prompts[source_id] = [Message(role="user",
                                          content=record["messages"][0]["content"])]

    policy = PairMiningPolicy(
        selection_method=args.selection_method,
        minimum_margin=args.minimum_margin,
        seed=args.seed,
    )
    pairs, decisions = mine_preference_pairs(
        grouped, verifier, contexts, prompts, policy,
        provenance_source_id=f"p1-pair-mining-{args.selection_method}-s{args.seed}",
    )

    output = Path(args.output)
    fingerprint = write_jsonl(output, pairs)
    manifest = {
        "adapter_dir": str(args.adapter_dir),
        "input": args.input,
        "prompts_total": len(records),
        "prompts_verifiable": len(verifiable),
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "selection_method": args.selection_method,
        "minimum_margin": args.minimum_margin,
        "seed": args.seed,
        "pairs_accepted": len(pairs),
        "pair_yield": round(len(pairs) / max(1, len(verifiable)), 4),
        "decision_counts": decisions,
        "preference_fingerprint": fingerprint,
        "opener_seed": opener_seed,
    }
    manifest_path = output.with_name("p1_preference_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.hf_sync_repo:
        from axiom_world.integrations.hf_sync import upload_directory

        destination = upload_directory(
            output.parent, args.hf_sync_repo,
            path_in_repo=args.hf_path_in_repo, repo_type="dataset",
            commit_message=f"p1 preference pairs: {args.selection_method} s{args.seed}",
        )
        print(f"preference data synced -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

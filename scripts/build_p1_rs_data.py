#!/usr/bin/env python
"""Build the B2 rejection-sampled P1 corpus (protocol §5.1, RQ2 upstream arm).

Takes the FROZEN B1 mixture (p1_general_sft.jsonl) as the prompt source so
B1 and B2 differ only in RESPONSE PROVENANCE at identical prompt sets:
  B1: dataset-written responses (curated corpus)
  B2: base-model responses sampled at temperature and kept ONLY when the
      ExactAnswerVerifier confirms the gold answer (rejection sampling).

Prompts without a verifiable gold answer are excluded (SKIPPED cannot gate a
rejection-sampling decision). Prompts where no candidate passes are dropped
and counted in the manifest — B2 is therefore a subset of B1's prompts; the
manifest records coverage so the B1-vs-B2 comparison can be token-matched at
training time (protocol §5.6).

Usage (Colab GPU, generation-only session):
  python scripts/build_p1_rs_data.py \
      --config configs/experiments/b1_general_sft.yaml \
      --input data/p1/p1_general_sft.jsonl \
      --output data/p1/p1_general_sft_rs.jsonl \
      --num-candidates 4 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.core.config_loader import resolve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Recipe providing the pinned base model (b1 recipe).")
    parser.add_argument("--input", default="data/p1/p1_general_sft.jsonl")
    parser.add_argument("--output", default="data/p1/p1_general_sft_rs.jsonl")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Generation sequences per batch (prompts x candidates).")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-sync-repo", default=None, help="HF DATASET repo.")
    parser.add_argument("--hf-path-in-repo", default="p1/v1")
    args = parser.parse_args()

    import torch

    from axiom_world.data.bundle import write_jsonl
    from axiom_world.data.records import Message, Provenance, SFTRecord
    from axiom_world.models.builder import build_for_inference
    from axiom_world.verifiers.base import VerificationStatus
    from axiom_world.verifiers.general import ExactAnswerVerifier

    config, _, _ = resolve(args.config, [])
    model, tokenizer = build_for_inference(config, adapter_dir=None)
    opener_seed = config.evaluation.opener_seed
    verifier = ExactAnswerVerifier()
    torch.manual_seed(args.seed)

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        records = records[: args.limit]
    verifiable = [r for r in records if r.get("metadata", {}).get("gold_answer")]

    stop_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.eos_token_id:
        stop_ids.append(im_end)

    @torch.no_grad()
    def sample(prompt_texts: list[str]) -> list[list[str]]:
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True,
            ) + opener_seed
            for p in prompt_texts
        ]
        expanded = [t for t in rendered for _ in range(args.num_candidates)]
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

    from tqdm import tqdm

    kept: list[SFTRecord] = []
    counts = {"prompts_total": len(records), "prompts_verifiable": len(verifiable),
              "accepted": 0, "no_passed_candidate": 0}
    prompt_chunk = max(1, args.batch_size // args.num_candidates)
    for start in tqdm(range(0, len(verifiable), prompt_chunk), desc="rs(batched)"):
        batch = verifiable[start : start + prompt_chunk]
        prompts = [r["messages"][0]["content"] for r in batch]
        for record, candidates in zip(batch, sample(prompts), strict=True):
            gold = record["metadata"]["gold_answer"]
            winner: str | None = None
            for text in candidates:
                verdict = verifier.verify(text.strip(), {"answer": gold})
                if verdict.status is VerificationStatus.PASSED:  # noqa: SIM102
                    # shortest passing candidate wins (length-bias control)
                    if winner is None or len(text.strip()) < len(winner):
                        winner = text.strip()
            if winner is None:
                counts["no_passed_candidate"] += 1
                continue
            counts["accepted"] += 1
            kept.append(
                SFTRecord(
                    id=record["id"].replace("p1-", "p1rs-", 1),
                    messages=[
                        Message(role="user", content=record["messages"][0]["content"]),
                        Message(role="assistant", content=winner),
                    ],
                    provenance=Provenance(
                        source_type="synthetic",
                        source_id=f"rejection-sampling:{config.model.repo_id}",
                        source_revision=config.model.revision or "main",
                        transformation_version="p1rs-v1",
                    ),
                    task_family=record["task_family"],
                    metadata={"gold_answer": gold, "parent_id": record["id"]},
                )
            )

    output = Path(args.output)
    fingerprint = write_jsonl(output, kept)
    manifest = {
        **counts,
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "opener_seed": opener_seed,
        "rs_fingerprint": fingerprint,
        "acceptance_rate": round(counts["accepted"] / max(1, counts["prompts_verifiable"]), 4),
    }
    manifest_path = output.with_name("p1_rs_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.hf_sync_repo:
        from axiom_world.integrations.hf_sync import upload_directory

        destination = upload_directory(
            output.parent, args.hf_sync_repo, path_in_repo=args.hf_path_in_repo,
            repo_type="dataset", commit_message=f"p1 RS corpus s{args.seed}",
        )
        print(f"p1 RS data synced -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

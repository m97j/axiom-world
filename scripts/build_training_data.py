#!/usr/bin/env python
"""Build oracle-derived PlayWorld training data (SFT + GRPO prompt pool).

Neuro-symbolic foundry (protocol §4.2/§4.3):
  scenario generator (symbolic) -> oracle BFS (ground-truth actions)
  -> SFT records whose targets are oracle solutions, never LLM text.

Leakage gate: training families are generated under a 'train-' namespace and
additionally cross-checked against the frozen eval freeze_manifest.json —
any overlap aborts the build.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from axiom_world.data.bundle import write_jsonl
from axiom_world.data.records import EvaluationRecord, Message, Provenance, SFTRecord
from axiom_world.playworld.oracle import solve
from axiom_world.playworld.scenario import ScenarioGenerator
from axiom_world.playworld.spec import Scenario

TRAIN_FAMILIES: dict[str, list[str]] = {
    "train-fam0": ["movement", "collection", "deposit"],
    "train-fam1": ["movement", "collection", "deposit"],
    "train-fam2": ["movement", "collection", "capacity"],
    "train-fam3": ["movement", "energy_budget", "collection", "deposit"],
    "train-fam4": ["movement", "collection"],
}

PROMPT_TEMPLATE = (
    "You control an agent in a grid world.\n"
    "World spec (JSON): {spec}\n"
    "Initial state (JSON): {state}\n"
    "Goal (JSON): {goal}\n"
    "Respond with ONLY a JSON object: "
    '{{"actions": [{{"type": "MOVE|COLLECT|REST|DEPOSIT|WAIT", ...}}], '
    '"final_state": {{"location": "...", "energy": N}}}}'
)


def scenario_prompt(scenario: Scenario) -> str:
    return PROMPT_TEMPLATE.format(
        spec=scenario.spec.model_dump_json(),
        state=scenario.initial_state.model_dump_json(),
        goal=scenario.goal.model_dump_json(),
    )


def oracle_target(scenario: Scenario) -> str | None:
    solution = solve(scenario)
    if not solution.solvable or solution.final_state is None:
        return None
    return json.dumps(
        {
            "actions": [a.model_dump(exclude_none=True) for a in solution.actions],
            "final_state": {
                "location": solution.final_state.location,
                "energy": solution.final_state.energy,
            },
        },
        sort_keys=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--scenarios-per-family", type=int, default=400)
    parser.add_argument("--output-dir", default="data/train")
    parser.add_argument("--eval-freeze-manifest", default="data/eval_suites/freeze_manifest.json")
    parser.add_argument("--hf-sync-repo", default=None,
                        help="HF DATASET repo (user/name). When set, the output dir "
                             "(jsonl + train_manifest.json) is uploaded after a "
                             "successful build.")
    parser.add_argument("--hf-path-in-repo", default="train/v1",
                        help="Destination path inside the dataset repo.")
    args = parser.parse_args()

    freeze_path = Path(args.eval_freeze_manifest)
    eval_families: set[str] = set()
    if freeze_path.is_file():
        eval_families = set(json.loads(freeze_path.read_text())["eval_family_ids"])
    overlap = eval_families & set(TRAIN_FAMILIES)
    if overlap:
        raise SystemExit(f"LEAKAGE: train families overlap eval families: {sorted(overlap)}")

    generator = ScenarioGenerator(seed=args.seed)
    sft_records: list[SFTRecord] = []
    prompt_records: list[EvaluationRecord] = []
    unsolvable = 0

    for family_id, primitives in TRAIN_FAMILIES.items():
        scenarios = generator.generate(family_id, primitives, count=args.scenarios_per_family)
        for scenario in scenarios:
            target = oracle_target(scenario)
            if target is None:
                unsolvable += 1
                continue
            prompt = scenario_prompt(scenario)
            provenance = Provenance(
                source_type="synthetic", source_id=f"oracle-bfs-seed{args.seed}"
            )
            sft_records.append(
                SFTRecord(
                    id=f"sft-{scenario.scenario_id}",
                    messages=[
                        Message(role="user", content=prompt),
                        Message(role="assistant", content=target),
                    ],
                    provenance=provenance,
                    scenario_family_id=family_id,
                )
            )
            prompt_records.append(
                EvaluationRecord(
                    id=f"prompt-{scenario.scenario_id}",
                    suite="eval_id",  # reused schema; suite label unused for training prompts
                    scenario=scenario.model_dump(mode="json"),
                    prompt=[Message(role="user", content=prompt)],
                    scenario_family_id=family_id,
                    provenance=provenance,
                )
            )

    output_dir = Path(args.output_dir)
    sft_fp = write_jsonl(output_dir / "playworld_sft.jsonl", sft_records)
    prompt_fp = write_jsonl(output_dir / "playworld_prompts.jsonl", prompt_records)
    manifest = {
        "seed": args.seed,
        "sft_records": len(sft_records),
        "prompt_records": len(prompt_records),
        "unsolvable_dropped": unsolvable,
        "sft_fingerprint": sft_fp,
        "prompt_fingerprint": prompt_fp,
        "train_families": sorted(TRAIN_FAMILIES),
        "eval_families_checked": sorted(eval_families),
    }
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))

    if args.hf_sync_repo:
        from axiom_world.integrations.hf_sync import upload_directory

        uri = upload_directory(
            output_dir, args.hf_sync_repo,
            path_in_repo=args.hf_path_in_repo, repo_type="dataset",
            commit_message=f"training data build (seed={args.seed}, "
                           f"sft={len(sft_records)}, prompts={len(prompt_records)})",
        )
        print(f"training data persisted: {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

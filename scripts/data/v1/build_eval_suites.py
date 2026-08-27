#!/usr/bin/env python
"""Gate G3 — generate and freeze the PlayWorld evaluation suites (protocol §4.3).

Deterministic under --seed. Produces one JSONL per suite plus a freeze
manifest with per-suite fingerprints. Re-running with the same arguments
reproduces byte-identical suites; any drift is a G3 violation.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

from axiom_world.core.fingerprints import fingerprint_payload
from axiom_world.data.bundle import write_jsonl
from axiom_world.data.records import EvaluationRecord, Message, Provenance
from axiom_world.worlds.playworld.scenario import ScenarioGenerator
from axiom_world.worlds.playworld.spec import Scenario

# Pre-registered suite composition (family primitives per suite).
SUITES: dict[str, dict] = {
    "eval_id":           {"primitives": ["movement", "collection", "deposit"], "families": 3},
    "eval_template_ood": {"primitives": ["movement", "collection", "deposit"], "families": 2},
    "eval_comp_ood":     {"primitives": ["movement", "collection", "capacity", "deposit"], "families": 2},
    "eval_rule_ood":     {"primitives": ["movement", "energy_budget", "collection", "capacity", "deposit"], "families": 2},
    "eval_adversarial":  {"primitives": ["movement", "capacity", "deposit"], "families": 1},
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-suite", type=int, default=300)
    parser.add_argument("--output-dir", default="data/eval_suites")
    parser.add_argument("--hf-sync-repo", default=None,
                        help="HF DATASET repo (user/name). When set, the frozen "
                             "suites + freeze_manifest.json are uploaded after build.")
    parser.add_argument("--hf-path-in-repo", default="eval_suites/v1",
                        help="Destination path inside the dataset repo.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest: dict[str, dict] = {}
    all_families: list[str] = []

    for suite_name, spec in SUITES.items():
        # zlib.crc32 is process-stable; builtin hash() is salted per process
        # and silently breaks byte-identical regeneration (caught by CI).
        generator = ScenarioGenerator(seed=args.seed + zlib.crc32(suite_name.encode()) % 1000)
        per_family = max(1, args.episodes_per_suite // spec["families"])
        records: list[EvaluationRecord] = []
        for family_index in range(spec["families"]):
            family_id = f"{suite_name}-fam{family_index}"
            all_families.append(family_id)
            scenarios = generator.generate(family_id, spec["primitives"], count=per_family)
            for scenario in scenarios:
                records.append(
                    EvaluationRecord(
                        id=f"{suite_name}-{scenario.scenario_id}",
                        suite=suite_name,
                        scenario=scenario.model_dump(mode="json"),
                        prompt=[Message(role="user", content=scenario_prompt(scenario))],
                        scenario_family_id=family_id,
                        provenance=Provenance(
                            source_type="synthetic",
                            source_id=f"scenario-generator-seed{args.seed}",
                        ),
                    )
                )
        path = output_dir / f"{suite_name}.jsonl"
        fingerprint = write_jsonl(path, records)
        manifest[suite_name] = {
            "path": str(path),
            "episodes": len(records),
            "families": spec["families"],
            "rule_primitives": spec["primitives"],
            "fingerprint": fingerprint,
        }
        print(f"{suite_name}: {len(records)} episodes -> {path} ({fingerprint[:24]}...)")

    # Fingerprint over CONTENT only: output paths are machine-specific and
    # must not affect the freeze identity (G3 reproducibility check).
    content_manifest = {
        name: {k: v for k, v in entry.items() if k != "path"}
        for name, entry in manifest.items()
    }
    freeze = {
        "seed": args.seed,
        "suites": manifest,
        "eval_family_ids": sorted(all_families),
        "manifest_fingerprint": fingerprint_payload(content_manifest),
    }
    freeze_path = output_dir / "freeze_manifest.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nG3 freeze manifest -> {freeze_path}")
    print("Commit this manifest; training loaders must pass eval_family_ids as "
          "forbidden_family_ids (leakage gate).")

    if args.hf_sync_repo:
        from axiom_world.integrations.hf_sync import upload_directory

        uri = upload_directory(
            output_dir, args.hf_sync_repo,
            path_in_repo=args.hf_path_in_repo, repo_type="dataset",
            commit_message=f"freeze eval suites (seed={args.seed}, "
                           f"{args.episodes_per_suite}/suite)",
        )
        print(f"eval suites persisted: {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

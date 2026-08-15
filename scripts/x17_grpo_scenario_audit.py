#!/usr/bin/env python
"""x17: GRPO scenario-transport audit (B6 None-contamination incident, 2026-08-15).

Context. The first B6 GRPO run logged `All reward functions returned None`
with reward 0 / grad_norm 0 for 17h. Root cause (reproduced in-sandbox):
`to_grpo_rows` shipped `scenario` as a NESTED DICT column; HF datasets'
Arrow backend unified the per-row dicts into one struct schema and injected
None for keys absent in a given row (variable-key maps such as
goal.resources). `Scenario.model_validate` then failed, the verifier's
broad-except converted the ValidationError into INFRA_ERROR, and the reward
bridge dutifully returned None for every completion.

This audit proves, for a given prompts JSONL, that the v0.6.12 transport fix
(scenario as JSON string, decoded in the reward bridge) is contamination-free:

  1. legacy path  : dict column -> Dataset.from_list -> validate each row
     (EXPECTED to show contamination on heterogeneous families).
  2. fixed path   : scenario_json string column -> Dataset.from_list ->
     json.loads -> validate + byte-exact round-trip check.
  3. reward parity: oracle completions scored via the fixed path must equal
     scores computed directly on the raw dicts (no Arrow), sample-by-sample.

Exit code 0 iff the FIXED path has zero validation failures, zero round-trip
mismatches, and full reward parity. The legacy-path contamination count is
reported as evidence, not as a failure.

Usage (Colab, before b_b6_train):
  python scripts/x17_grpo_scenario_audit.py \
    --prompts data/train/playworld_prompts.jsonl \
    --out runs/x17_grpo_scenario_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            clean_line = line.strip()
            if clean_line:
                rows.append(json.loads(clean_line))
    return rows


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True,
                        help="Frozen playworld_prompts.jsonl (v1.3 sha-pinned artifact).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reward-sample", type=int, default=64,
                        help="Rows used for the oracle reward-parity check.")
    args = parser.parse_args()

    from datasets import Dataset

    from axiom_world.playworld.oracle import solve
    from axiom_world.playworld.spec import Scenario
    from axiom_world.training.reward_bridge import verifier_reward_function
    from axiom_world.verifiers.hybrid import default_playworld_verifier

    records = _load_rows(Path(args.prompts))
    scenarios = [r["scenario"] for r in records]
    report: dict[str, Any] = {"prompts": args.prompts, "n_rows": len(records)}

    # --- 1. legacy path: nested dict column (evidence of the bug) -----------
    legacy_ds = Dataset.from_list([{"scenario": s} for s in scenarios])
    legacy_fail = 0
    legacy_mutated = 0
    for i in range(len(legacy_ds)):
        rt = legacy_ds[i]["scenario"]
        if _canonical(rt) != _canonical(scenarios[i]):
            legacy_mutated += 1
        try:
            Scenario.model_validate(rt)
        except Exception:  # noqa: BLE001 - counting, not handling
            legacy_fail += 1
    report["legacy_dict_column"] = {
        "rows_mutated_by_arrow": legacy_mutated,
        "validation_failures": legacy_fail,
    }

    # --- 2. fixed path: JSON string column ----------------------------------
    fixed_ds = Dataset.from_list([{"scenario_json": _canonical(s)} for s in scenarios])
    fixed_fail = 0
    fixed_mismatch = 0
    for i in range(len(fixed_ds)):
        raw = fixed_ds[i]["scenario_json"]
        if raw != _canonical(scenarios[i]):
            fixed_mismatch += 1
        try:
            Scenario.model_validate(json.loads(raw))
        except Exception:  # noqa: BLE001
            fixed_fail += 1
    report["fixed_json_string_column"] = {
        "roundtrip_mismatches": fixed_mismatch,
        "validation_failures": fixed_fail,
    }

    # --- 3. reward parity on oracle completions -----------------------------
    sample = list(range(min(args.reward_sample, len(records))))
    comps: list[str] = []
    kept: list[int] = []
    for i in sample:
        solution = solve(Scenario.model_validate(scenarios[i]))
        actions = getattr(solution, "actions", None) if solution else None
        if actions:
            comps.append(_canonical(
                {"actions": [a.model_dump(exclude_none=True) for a in actions]}
            ))
            kept.append(i)
    fn_direct = verifier_reward_function(default_playworld_verifier(), min_calls=10**9)
    fn_fixed = verifier_reward_function(default_playworld_verifier(), min_calls=10**9)
    r_direct = fn_direct(prompts=["p"] * len(kept), completions=comps,
                         scenario=[scenarios[i] for i in kept])
    r_fixed = fn_fixed(prompts=["p"] * len(kept), completions=comps,
                       scenario_json=[fixed_ds[i]["scenario_json"] for i in kept])
    parity = sum(1 for a, b in zip(r_direct, r_fixed, strict=True) if a == b)
    report["reward_parity"] = {
        "n_scored": len(kept),
        "n_equal": parity,
        "direct_none": sum(1 for r in r_direct if r is None),
        "fixed_none": sum(1 for r in r_fixed if r is None),
        "fixed_mean_reward": (
            sum(r for r in r_fixed if r is not None) / max(1, len([r for r in r_fixed if r is not None]))
        ),
    }

    ok = fixed_fail == 0 and fixed_mismatch == 0 and parity == len(kept)
    report["verdict"] = "PASS" if ok else "FAIL"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

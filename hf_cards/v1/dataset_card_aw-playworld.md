---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - planning
  - synthetic
  - verifiable-environment
  - preference-pairs
  - frozen-benchmark
pretty_name: "Axiom-World PlayWorld (frozen v1 artifacts)"
---

# aw-playworld — PlayWorld frozen training & evaluation artifacts (protocol v1)

Fully **synthetic** planning data for the PlayWorld environment of the
Axiom-World project: rule-parameterized worlds, goals, and structured-JSON
plans, all generated and verified by a deterministic transition engine
(no LLM-generated labels, no human data).

- **Code & environment generator:** https://github.com/m97j/axiom-world (tag `v1.0.0`)
- **Tech report:** DOI [10.5281/zenodo.22052149](https://doi.org/10.5281/zenodo.22052149)

## Contents

| path | what | records | integrity |
|---|---|---|---|
| `train/v1/playworld_sft.jsonl` | oracle-derived SFT episodes | 2,000 | canonical fingerprint `sha256:54fcb1d3…` |
| `preference_train/…/playworld_preference*.jsonl` | verifier-mined preference pairs (chosen must PASS) | — | `sha256:5856b5c5…` (a1v2 mining) |
| `eval/…` | **frozen** evaluation suites: `eval_id`, `eval_template_ood`, `eval_comp_ood`, `eval_rule_ood`, `eval_adversarial` — 300 episodes each | 1,500 | freeze fingerprint `sha256:3cdcbc30…` |

Suites are split at the **scenario-family** level (ruleset × world template ×
goal type); the adversarial suite contains hand-written traps
(illegal-but-plausible actions, reward-hacking bait). The evaluation harness
hard-fails on any fingerprint mismatch.

## Provenance & determinism

Episodes are produced by a BFS oracle over the deterministic transition
engine. The v1 SFT builder had non-deterministic tie-breaking; this artifact
is the **frozen** re-build all v2 experiments consumed (protocol amendment
logged). Consume via the repo's `scripts/fetch_dataset.py`, which verifies
fingerprints before use.

## Uses & limitations

Built for controlled recipe comparison, not as a general planning benchmark.
If you use the frozen suites, do not train on any family present in them —
the repo's data loader enforces this leakage gate for you.

## Citation

Cite the Axiom-World tech report (see `CITATION.cff` in the code repo).

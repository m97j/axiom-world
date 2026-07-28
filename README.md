# Axiom-World

> **Verifier-guided two-stage post-training for rule-constrained game-world
> interaction, on a single NVIDIA RTX PRO 6000 Blackwell GPU.**

A reproducible research framework studying whether general reasoning
warm-start (Phase 1) improves domain adaptation to a symbolic, verifiable
game world (Phase 2) — with exact, deterministic verifier rewards, a
pre-registered experimental protocol, and enforced model lineage.

[Experimental Protocol (pre-registered)](docs/experimental-protocol.md) ·
Tech Report (TBD) · HF Collection (TBD)

## Why this exists

Most post-training projects report "the score went up." This project fixes,
**before any experiment runs**:

- the research questions and their falsification conditions,
- the full experiment matrix (Tracks A/B/C + 4 ablations — nothing else),
- the champion-selection rule (hard constraints → Pareto → primary metric),
- the statistical analysis plan (paired bootstrap, permutation tests,
  Holm–Bonferroni over 6 pre-registered comparisons),
- and the artifact/lineage contract every run must satisfy.

See `docs/experimental-protocol.md` (frozen v1.0; amendments are logged, never
edited in place).

## Design guarantees (enforced by code, verified by tests)

| Guarantee | Where |
|---|---|
| A Track-B Phase-2 run cannot train unless its parent adapter is **byte-identical** (SHA-256) to the recorded Phase-1 champion | `core/lineage.py`, gate ordering proven in `tests/unit/test_trainer_factory.py` |
| A run missing any of the 8 required artifacts can never reach `completed` | `core/context.py` |
| Frozen datasets that change after freeze **hard-fail** at load (fingerprint mismatch) | `data/bundle.py` |
| Training data from an eval scenario family is rejected at load (leakage gate) | `data/bundle.py` |
| Rewards come only from the deterministic transition engine; infra errors are never counted as model failures | `verifiers/`, `training/reward_bridge.py` |
| Verifier-ranked preference pairs require a PASSED chosen candidate; the random-pairing control (E-RANDPAIR) is a first-class citizen | `data/records.py`, `generation/pair_mining.py` |
| Canonical runs are SDPA/BF16-LoRA only; FA2 and QLoRA are quarantined to benchmarks/ablations | `core/schemas.py::validate_canonical` |

## Repository layout

```text
configs/           # extends-based composition; experiments/ = pre-registered recipes
src/axiom_world/
  core/            # config, context, manifest, lineage, fingerprints (the contract)
  runtime/         # environment audit + strict policy
  playworld/       # symbolic world: spec, deterministic transition engine, scenarios
  verifiers/       # tiered deterministic verifiers + hybrid aggregation
  data/            # canonical records + build_data_bundle (single entrypoint)
  generation/      # backend abstraction (lazy vLLM), preference pair mining
  training/        # TRL boundary, dataset adapter, verifier→GRPO reward bridge
  evaluation/      # runner, bootstrap CIs, paired comparisons, failure taxonomy
scripts/           # audit_runtime, smoke_gate_g1, build_eval_suites
tests/             # 54 contract tests (CPU-only; run in CI)
docs/              # experimental-protocol.md + contracts
```

## Quick start

### Contract layer (any machine)

```bash
pip install -e ".[dev]"
pytest tests -q                      # 54 contract tests
axiom validate-config --config configs/experiments/a1_playworld_sft.yaml \
  --override model.revision=<exact-hash>
```

### Colab G4 (canonical runtime)

```bash
pip install -e . -r requirements/colab-g4.lock.txt
python scripts/audit_runtime.py       # environment manifest
python scripts/smoke_gate_g1.py       # Gate G1: TRL import gate + tiny LoRA train
python scripts/build_eval_suites.py   # Gate G3: freeze eval suites (commit the manifest)
axiom init-run --config configs/experiments/a1_playworld_sft.yaml \
  --override data.source.repo_id=<hf-dataset>
```

Session discipline (protocol §3): data preprocessing on CPU sessions,
vLLM generation in a dedicated session (`requirements/vllm.lock.txt`),
training/evaluation on the G4 session. Never co-locate vLLM with training.

## Status

`v0.1.0` — contract layer complete (config/lineage/data/verifier/eval
contracts + 54 tests). Gates G1–G3 pending on the canonical runtime;
experimental results will be added under the pre-registered protocol only.

No performance claims are made at this version, by design.

## License

MIT (see `LICENSE`).

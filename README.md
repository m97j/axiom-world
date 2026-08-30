# Axiom-World

> **A pre-registered, single-GPU study of two-stage post-training for
> rule-grounded planning in a fully verifiable toy world.**

Does general-reasoning SFT before task tuning transfer better than direct task
tuning? Protocol v1 answers this with frozen, fingerprint-pinned evaluation
suites, enforced artifact lineage, and a 3-seed final confirmation — including
honestly documented negative results (DPO ≈ null; GRPO regression with a
mechanistic post-mortem).

**[Tech Report v1.0](docs/reports/v1/axiom-world-tech-report-v1.md)** ·
[Pre-registered Protocol (v1.0 + amendment log)](docs/protocols/v1/experiment_protocol_v1.md) · [Tech report DOI: 22052148](https://doi.org/10.5281/zenodo.22052148) (Zenodo)  
[HF Collection](https://huggingface.co/collections/m97j/axiom-world) (champion adapter [`m97j/aw-qwen3-8b-v1`](https://huggingface.co/m97j/aw-qwen3-8b-v1), datasets
[`m97j/aw-playworld`](https://huggingface.co/datasets/m97j/aw-playworld), [`m97j/axiom-general-posttrain`](https://huggingface.co/datasets/m97j/axiom-general-posttrain))  

## Headline results (Qwen3-8B-Base, LoRA, seeds 42/43/44, pass rate mean ± sd)

| Suite | **B4v2: two-stage** (general SFT → task SFT) | A2v2: direct (task SFT → DPO) |
|---|---|---|
| in-distribution | **.393 ± .013** | .186 ± .002 |
| template-OOD | **.349 ± .005** | .207 ± .003 |
| compositional-OOD | **.329 ± .022** | .139 ± .004 |
| rule-OOD | **.302 ± .005** | .192 ± .005 |
| adversarial | **.851 ± .002** | .651 ± .011 |

All 15 suite×seed deltas positive (paired permutation p ≤ 0.0004 each; Gate G6
sign-consistency PASS). Further: offline DPO is ≈ null on both routes while the
two-stage advantage fully survives it (B5 ≫ A2v2, 10/10, p ≤ .0002); verifier-
rewarded GRPO *regressed* the champion under two reward designs — diagnosed as
advantage starvation + entropy collapse, not reward shape (report §6).

Claims are **comparative** (recipe deltas under a fixed small budget), not
absolute task mastery — see report §7.

## Why this exists

Most post-training projects report "the score went up." This project fixed,
**before any experiment ran**: the research questions and falsification
conditions, the full experiment matrix, the champion-selection rule, the
statistical analysis plan (paired bootstrap, permutation tests,
Holm–Bonferroni), and the artifact/lineage contract every run must satisfy.
Every deviation is a numbered amendment (`docs/experimental-protocol.md` §13,
v1.0 → v1.4); every incident is preserved in the as-run notebooks.

## Design guarantees (enforced by code, verified by tests)

| Guarantee | Where |
|---|---|
| A Phase-2 run cannot train unless its parent adapter is **byte-identical** (SHA-256) to the recorded parent champion | `core/lineage.py` |
| A run missing any required artifact can never reach `completed` | `core/context.py` |
| Frozen datasets that change after freeze **hard-fail** at load (fingerprint mismatch) | `data/bundle.py` |
| Training data from an eval scenario family is rejected at load (leakage gate) | `data/bundle.py` |
| Rewards come only from the deterministic transition engine; infra errors are never counted as model failures | `verifiers/`, `training/reward_bridge.py` |
| Evaluations are audited against stale-weights accidents (prediction-identity gate) | `scripts/x20_eval_identity_audit.py` |
| Canonical runs are SDPA/BF16-LoRA with the v2 adapter contract (`modules_to_save: [lm_head, embed_tokens]`) | `core/schemas.py`, failure analysis below |

## Failure analyses (read these first if you fine-tune base models with LoRA)

- [Chat-template termination under adapter-only LoRA](docs/experiments/v1/adapter_contract_termination.md)
  — why attention/MLP-only LoRA provided a brittle, indirect path for learning
  robust `<|im_end|>` termination in this Base-model setting, with the full
  x09–x13 audit chain and a parameterization/capacity analysis.
- [B6/B6-R GRPO arm closure](docs/experiments/v1/b6_grpo_closure.md)
  — aggregate-reward hacking, pass-gated reward control, episode-level flip
  analysis, advantage starvation + entropy collapse.
- [Gate G6 3-seed closure](docs/experiments/v1/g6_seed_closure.md)
  — reseeding design, verdict, and the (documented) baseline-plumbing incident.

## Repository layout

```text
configs/             # extends-based composition; experiments/ = pre-registered recipes
src/axiom_world/
  core/              # config, context, manifest, lineage, fingerprints (the contract)
  runtime/           # environment audit + strict policy
  playworld/         # symbolic world: spec, deterministic transition engine, scenarios
  verifiers/         # tiered deterministic verifiers + aggregation
  data/              # canonical records + build_data_bundle (single entrypoint)
  generation/        # backend abstraction, preference pair mining
  training/          # TRL boundary, dataset adapter, verifier→GRPO reward bridge
  evaluation/        # runner, bootstrap CIs, paired comparisons, failure taxonomy
scripts/             # runtime audit, suite freeze, run/eval/analysis CLIs, x01–x21 diagnostics
notebooks/protocol_v1/  # as-run campaign notebooks (aw_01–aw_11), outputs preserved
tests/               # contract tests (CPU-only)
docs/                # protocol, tech report, failure analyses
```

## Reproducing

```bash
pip install -e ".[dev]" && pytest tests -q          # contract layer, any machine
```

On the canonical runtime (Colab G4 / RTX PRO 6000 Blackwell):

```bash
pip install -e . -r requirements/colab-g4.lock.txt
python scripts/audit_runtime.py
python scripts/fetch_run.py --repo m97j/aw-runs-b4 \
  --run-id 20260814-023603--b4v2-playworld-sft-from-p1--s42--c56ed2   # champion, sha-verified
python scripts/run_evaluation.py --config configs/experiments/eval_playworld.yaml \
  --adapter-dir runs/<run>/artifacts/final_adapter                     # frozen suites, greedy
```

Every number in the tech report maps to a run id in its Appendix A; run
artifacts live in `m97j/aw-runs-b4` and `m97j/aw-runs-seeds` (public). The
as-run notebooks under `notebooks/protocol_v1/` are the chronological research
log, incidents included.

## Status

`v1.0.0` — Protocol v1 CLOSED. Champion: **B4v2** (two-stage).

Next: **Protocol v2.0-CLB — Closed-Loop Belief**, extending PlayWorld to
partial observability, explicit belief-state maintenance, closed-loop
interaction, counterfactual transition prediction, and verifier-grounded
online RL.

Phase-1/general agent-capability improvements are reserved for **v2-AP**;
world-complexity scaling is reserved for **v2-WE**.

See [`docs/roadmap.md`](docs/roadmap.md) for the full research roadmap.

## Protocols roadmap

Every quantitative claim in this repository is attributable to exactly one
pre-registered protocol, frozen in git before its first canonical run.

| Version | Status | Question | Records |
|---|---|---|---|
| **v1.4** | closed | Does staged post-training beat direct task tuning in a deterministically verifiable world? Where do DPO and verifier-rewarded RL help or fail? | [`docs/protocols/v1/`](docs/protocols/v1/) |
| **v2.0-CLB** | pre-freeze | Under partial observability, does explicit belief-state maintenance improve closed-loop success and transition-model knowledge? | [`docs/protocols/v2/`](docs/protocols/v2/) |

The broader v2 program separates closed-loop interaction (**v2-CLB**),
agent-capability scaling (**v2-AP**), and world-complexity scaling (**v2-WE**).

See [`docs/roadmap.md`](docs/roadmap.md) for the authoritative roadmap.

Records are layered by protocol version; machinery (`src/`, `tests/`,
`scripts/common/`, `scripts/audits/`) is shared. See
[`docs/protocols/README.md`](docs/protocols/README.md).

## Citation

See `CITATION.cff`. Please cite the tech report (DOI: 10.5281/zenodo.22052149) and/or
this repository at tag `v1.0.0`.

## License

MIT (see `LICENSE`). PlayWorld data is fully synthetic; the Phase-1 mixture
derives from GSM8K and MATH (both MIT-licensed), see the dataset cards.
 
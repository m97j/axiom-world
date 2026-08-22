# Axiom-World

> **A pre-registered, single-GPU study of two-stage post-training for
> rule-grounded planning in a fully verifiable toy world.**

Does general-reasoning SFT before task tuning transfer better than direct task
tuning? Protocol v1 answers this with frozen, fingerprint-pinned evaluation
suites, enforced artifact lineage, and a 3-seed final confirmation — including
honestly documented negative results (DPO ≈ null; GRPO regression with a
mechanistic post-mortem).

**[Tech Report v1.0](docs/reports/axiom-world-tech-report-v1.md)** ·
[Pre-registered Protocol (v1.0 + amendment log)](docs/experimental-protocol.md) ·
HF Collection: `axiom-world` (champion adapter `m97j/aw-qwen3-8b-v1`, datasets
`m97j/aw-playworld`, `m97j/axiom-general-posttrain`) · TechRxiv DOI: (pending)

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

- [Chat-template termination under adapter-only LoRA](docs/experiments/adapter_contract_termination.md)
  — why attention/MLP-only LoRA cannot reliably emit `<|im_end|>` on a base
  checkpoint (100 % truncation → 0 % after making token rows trainable), with
  the full x09–x13 audit chain and a capacity argument.
- [B6/B6-R GRPO arm closure](docs/experiments/b6_grpo_closure.md)
  — aggregate-reward hacking, pass-gated reward control, episode-level flip
  analysis, advantage starvation + entropy collapse.
- [Gate G6 3-seed closure](docs/experiments/g6_seed_closure.md)
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

## Status & roadmap

`v1.0.0` — protocol v1 CLOSED. Champion: **B4v2** (two-stage). Next (protocol
v2a): absolute-performance campaign on the same frozen spec — Phase-1
enrichment (harder math/logic, sandbox-verified code RL, structured
tool-calling), Phase-2 scenario-pool expansion, online RL under its
prerequisites. See report §8.

## Citation

See `CITATION.cff`. Please cite the tech report (TechRxiv DOI pending) and/or
this repository at tag `v1.0.0`.

## License

MIT (see `LICENSE`). PlayWorld data is fully synthetic; the Phase-1 mixture
derives from GSM8K and MATH (both MIT-licensed), see the dataset cards.

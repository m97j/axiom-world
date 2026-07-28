# Axiom-World — Experimental Protocol (Pre-Registered)

> **Status: FROZEN — v1.0**
> This protocol is committed **before** any main experiment is run. Any change after
> the first main-track run must be recorded in [§13 Amendment Log](#13-amendment-log)
> with a rationale and a new version number. Results produced under an amended
> protocol must cite the protocol version they were run under.

- Project: **Axiom-World — Verifier-Guided Two-Stage Post-Training for Rule-Constrained Game-World Interaction on a Single Blackwell GPU**
- Author: [Minjae Kim] (independent researcher)
- Protocol version: `v1.0`
- Date frozen: `2026-07-23` (KST)
- Code anchor: git commit `<SHA at freeze>` of `github.com/m97j/axiom-world`
- Companion documents: `docs/architecture.md`, `docs/verifier-contract.md`, `docs/data-governance.md`, `docs/reproducibility.md`

---

## 1. Purpose and Scope

This document pre-registers the research questions, experiment matrix, model
selection rules, evaluation contract, statistical procedures, and stopping
criteria for Axiom-World. Its goals are:

1. Prevent post-hoc cherry-picking of experiments, metrics, or checkpoints.
2. Fix the champion-selection rule **before** results exist.
3. Bound scope so that the study is completable on a single GPU within the
   project window (~8 weeks).
4. Serve as the canonical source for the tech report's *Experimental Setup*
   section.

**Out of scope (explicitly excluded from this study):** SimPO, ORPO, KTO, PPO,
Dr.GRPO sweeps, multi-model-family comparisons beyond the fixed reference
baseline, multimodal inputs, Unity/Unreal integration, and any claim of
general-purpose SOTA performance. Exclusions are design decisions, not
omissions; rationale is given in §5.4.

---

## 2. Research Questions and Hypotheses

### RQ1 — Two-stage transfer

> Does general reasoning/instruction warm-start (Phase 1) improve PlayWorld
> sample efficiency and compositional out-of-distribution (OOD) generalization,
> compared to direct domain adaptation from the base model?

- **H1a**: Track B (two-stage) ≥ Track A (direct) on the primary metric (§7.1) at equal Phase-2 token budget.
- **H1b**: The advantage of Track B is larger on compositional-OOD splits than on in-distribution (ID) splits.
- **Falsification**: If Track A ≥ Track B within the CI on both ID and compositional-OOD, Phase 1 warm-start is not justified at this scale; this is a reportable negative result.

### RQ2 — Verifier-guided data construction

> Does verifier-guided rejection sampling and hard-negative preference mining
> improve action legality and state consistency over heuristic pair
> construction?

- **H2**: Verifier-mined preference data (DPO) improves legal-action rate and state-consistency rate over (i) SFT-only and (ii) DPO on random-pairing controls, at equal pair count.
- **Falsification**: No significant improvement over the random-pairing control at equal pair count.

### RQ3 — Offline vs online alignment under verifiable rewards

> Under an identical verifier reward, what quality / stability / cost
> trade-off do offline DPO and online GRPO exhibit on PlayWorld tasks?

- **H3**: GRPO improves multi-step task completion over DPO but with higher compute cost per valid sample and measurable stability risks (reward hacking, collapse), which we quantify rather than assume.
- RLOO is run **once** as a controlled advantage-estimator ablation of GRPO (same prompts, reward, budget, seed). It is not a co-equal method.

RQ1 is the primary research question. RQ2 and RQ3 are secondary. The report's
headline claim is scoped to RQ1's outcome, whichever direction it falls.

---

## 3. Fixed Environment Contract

All canonical runs execute on the following runtime. Runs on any other
configuration are labeled `exploratory` and excluded from main tables.

| Layer                                       | Frozen value                                                                                                                                              |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Platform                                    | Google Colab Pro+ G4                                                                                                                                      |
| GPU                                         | NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GB VRAM, CC 12.0                                                                                      |
| OS / Python                                 | Ubuntu 22.04.5 LTS / Python 3.12.13                                                                                                                       |
| PyTorch / CUDA                              | 2.11.0+cu128 / 12.8 —**provided by the Colab image; never reinstalled**                                                                            |
| Transformers / PEFT / Accelerate / Datasets | 5.13.1 / 0.19.1 / 1.14.0 / 4.0.0 (Colab-image versions, pinned)                                                                                           |
| TRL                                         | 1.9.x — exact pin frozen after the import-gate + tiny-run smoke test (§10, Gate G1); recorded in`requirements/colab-g4.lock.txt`                      |
| Attention backend                           | **SDPA (canonical)**. FlashAttention-2 appears only in the attention benchmark (§5.3, E-ATTN) and never in canonical result runs. FA3 is not used. |
| Precision / adapter                         | **BF16 LoRA (canonical)**. QLoRA NF4 appears only in ablation E-QLORA.                                                                              |
| Generation for data construction            | vLLM in a dedicated generation session; never co-located with training                                                                                    |
| Tracking                                    | W&B + local JSONL event logs (both mandatory; local logs are authoritative)                                                                               |

Every run records `environment_manifest.json` (package versions, GPU, driver,
audit output of `scripts/audit_runtime.py`). A canonical run whose manifest
deviates from this table is invalidated.

---

## 4. Models, Data, and Splits

### 4.1 Models (frozen)

| Role                                | Model                                                     | Revision policy                                                      |
| ----------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------- |
| Primary initialization              | `Qwen/Qwen3-8B-Base`                                    | Exact HF revision hash recorded at first download; frozen thereafter |
| Strong reference baseline (Track C) | Official`Qwen/Qwen3-8B` (instruct/reasoning checkpoint) | Same policy;**evaluation only, never fine-tuned**              |

No other model family is used in canonical experiments. A single optional
transfer-validation run on a larger checkpoint (≤14B) may be added **only
after** all canonical results are complete, labeled `exploratory`.

### 4.2 Datasets

| Dataset                                                | Role                      | Freeze rule                                                                                       |
| ------------------------------------------------------ | ------------------------- | ------------------------------------------------------------------------------------------------- |
| `axiom-general-posttrain` (P1 SFT + P1 preference)   | Phase 1 training          | Fingerprinted (SHA-256 per split) before first P1 run                                             |
| `axiom-playworld` train (SFT / preference / prompts) | Phase 2 training          | Fingerprinted before first P2 run                                                                 |
| `axiom-playworld` eval suites (§7.2)                | All evaluation            | **Frozen before any training run. Never regenerated. Never used in any training pipeline.** |
| Verifier golden fixtures                               | Verifier regression tests | Frozen at Gate G2 (§10)                                                                          |

Data lineage: every training record carries `provenance` (source dataset,
revision, transformation version/hash) per `docs/data-contract.md`.
Contamination control: lexical near-duplicate filtering (Jaccard on token
shingles, threshold 0.80) of all training candidates against every eval suite,
plus scenario-family separation (§4.3). The contamination report is published
with the dataset manifest.

### 4.3 Split design (leakage resistance)

PlayWorld scenarios are generated from parameterized **scenario families**
(ruleset × world template × goal type). Splits are made at the **family**
level, not the instance level:

- `train`: families F-train
- `eval-ID`: held-out instances from F-train families
- `eval-template-OOD`: unseen world templates, seen rules
- `eval-comp-OOD`: unseen **compositions** of seen rule primitives
- `eval-rule-OOD`: at least one unseen rule primitive
- `eval-adversarial`: hand-written traps (illegal-but-plausible actions, state contradictions, reward-hacking bait)

No family appears in more than one of {train} vs {any OOD suite}.

---

## 5. Experiment Matrix (exhaustive; nothing else enters main tables)

### 5.1 Track definitions

```text
Track A — Direct adaptation (control)
  A1: Base → PlayWorld SFT
  A2: A1  → PlayWorld DPO (verifier-mined pairs)

Track B — Two-stage transfer (main path)
  B1: Base → P1 general SFT (curated)
  B2: Base → P1 general SFT (rejection-sampled corpus)      [RQ2 upstream arm]
  B3: best(B1,B2) → P1 DPO (verifier-mined pairs) = P1 champion candidate set
  B4: P1 champion → PlayWorld SFT
  B5: B4 → PlayWorld DPO
  B6: B4 → PlayWorld GRPO (hybrid verifier reward)

Track C — Off-the-shelf reference (evaluation only)
  C1: Qwen3-8B instruct, zero-shot, canonical decoding
  C2: Qwen3-8B instruct, few-shot (k=3, fixed exemplars)
```

### 5.2 Ablations (one run each, unless a gate fails)

```text
E-RLOO   : B6 with RLOO advantage estimator (same prompts/reward/budget/seed)
E-RULE   : B6 with rule-only reward (no executable tier)      [verifier composition]
E-RANDPAIR: B5 with random pairing at equal pair count        [RQ2 control]
E-QLORA  : B4 recipe under QLoRA NF4 (cost/quality delta only)
```

### 5.3 System benchmark (not a quality experiment)

```text
E-ATTN: SDPA vs FlashAttention-2 (if installable on this stack) —
        train tokens/sec, rollout tokens/sec, end-to-end step time,
        peak VRAM, cost per valid sample. Seq lengths 2K/4K/8K.
        Result changes NO canonical configuration; it is reported as
        a systems finding.
```

### 5.4 Pre-registered exclusions

SimPO/ORPO/KTO/PPO are excluded to keep the design identifiable: with one GPU
and a fixed window, adding preference-objective variants multiplies the matrix
without serving RQ1–RQ3. Dr.GRPO and KL-estimator sweeps are excluded for the
same reason. These are named in the report's Limitations section as future
work, with this protocol cited as the reason they were not run.

### 5.5 Seeds and repetition policy

- Development and selection runs: **seed 42 only**.
- Final confirmation: the **two** highest-ranked Phase-2 variants (per §6) plus
  their matched Track-A control are re-run with seeds **{42, 43, 44}**.
- No other run is repeated. Seed-variance is reported only for the 3-seed set.

### 5.6 Budget parity rules

- Track A vs Track B Phase-2 comparisons use **equal Phase-2 training-token
  budgets**; Track B's Phase-1 cost is additionally reported as total-GPU-hour
  overhead so both "marginal" and "total" comparisons are visible.
- DPO vs GRPO comparisons report quality **and** GPU-hours and
  cost-per-valid-sample; neither is compared on quality alone.
- E-RLOO matches B6 exactly in prompts, reward function, rollout count K,
  step budget, and seed.

---

## 6. Champion Selection Rule (frozen before results)

Champion selection (both the Phase-1 champion feeding B4 and the final
reported model) follows three ordered stages. No weighted composite score is
used.

### Stage 1 — Hard constraints (must all pass)

| Constraint                                                                                         | Threshold              |
| -------------------------------------------------------------------------------------------------- | ---------------------- |
| JSON/schema validity rate (eval-ID)                                                                | ≥ 0.95                |
| Legal action rate (eval-ID)                                                                        | ≥ 0.90                |
| General-capability retention (P1 general held-out vs Base)                                         | drop ≤ 3 pts absolute |
| Reward-hacking incidence (adversarial suite, verifier-audited)                                     | ≤ 2% of episodes      |
| Catastrophic failure (loss divergence, output collapse, degenerate repetition >5% of eval outputs) | none                   |

Checkpoints failing any constraint are ineligible regardless of other scores.

### Stage 2 — Pareto screening

Among eligible checkpoints, compute the Pareto frontier over:
`{eval-comp-OOD primary metric ↑, general retention ↑, mean latency ↓, GPU-hours ↓}`.
Non-frontier checkpoints are eliminated.

### Stage 3 — Primary metric decision

- **Primary metric**: `goal-valid action accuracy on eval-comp-OOD` (§7.1).
- **Tie-breaker 1**: eval-rule-OOD primary metric.
- **Tie-breaker 2**: GPU-hours (lower wins).

"Tie" means the 95% paired-bootstrap CI of the difference includes zero.

The Phase-1 champion is selected by the same three stages, with the primary
metric replaced by the **transfer proxy**: goal-valid action accuracy of a
fixed short PlayWorld SFT probe (identical probe recipe for all P1
candidates), evaluated on eval-ID.

---

## 7. Evaluation Contract

### 7.1 Metric definitions (all computed by the frozen evaluator, version-pinned)

| Metric                                         | Definition                                                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Goal-valid action accuracy** (primary) | Fraction of episodes in which every emitted action is legal under the transition engine**and** the final state satisfies the goal predicate |
| JSON validity rate                             | Parseable, schema-conformant outputs / total                                                                                                      |
| Legal action rate                              | Legal actions / total actions (per-action)                                                                                                        |
| State consistency rate                         | Steps without contradiction against tracked world state / total steps                                                                             |
| Multi-step completion                          | Goal achieved within the episode step limit                                                                                                       |
| Reward-hacking incidence                       | Verifier-passed episodes flagged by adversarial audit rules / total                                                                               |
| General retention                              | Score on the frozen P1 general held-out suite                                                                                                     |
| Mean / p95 latency, tokens per decision        | Measured under the canonical decoding profile                                                                                                     |
| Cost per valid sample                          | GPU-seconds / verifier-passed sample (training-time metric)                                                                                       |

### 7.2 Evaluation suites

`eval-ID`, `eval-template-OOD`, `eval-comp-OOD`, `eval-rule-OOD`,
`eval-adversarial` (§4.3), plus the P1 general held-out suite. Target sizes:
≥300 episodes per PlayWorld suite (≥150 for adversarial). All suites are
frozen at Gate G3 and hash-recorded.

### 7.3 Decoding profile (canonical)

Greedy decoding, fixed max-new-tokens, fixed chat template, fixed system
prompt, temperature 0. Inference-time policy experiments (self-consistency,
best-of-N + verifier rerank, adaptive budget — the LogosP study) are a
**separate reported section** and never substitute for canonical numbers.

### 7.4 Verifier authority

Deterministic tiers (schema → rules → executable) are the sole reward and
scoring authority. The LLM judge is audit-only: it scores a fixed 10% sample
of eval outputs for disagreement analysis and never contributes to rewards,
selection, or headline metrics. Verifier status semantics
(`passed / failed / skipped / indeterminate / timeout / infra_error`) follow
`docs/verifier-contract.md`; `indeterminate`, `timeout`, and `infra_error`
are excluded from reward computation and reported separately.

---

## 8. Statistical Analysis Plan

- **Point estimates** with 95% bootstrap CIs (10,000 resamples) on all suite-level metrics.
- **Pairwise model comparisons**: paired bootstrap on per-episode outcomes (models share identical eval episodes); report Δ with CI.
- **Significance**: permutation test (10,000 permutations) for headline comparisons (A2 vs B5, B5 vs B6, B-champion vs C1/C2); α = 0.05 with Holm–Bonferroni correction across the pre-registered comparison family (listed here: {A2−B5, B5−B6, B6−E-RLOO, B5−E-RANDPAIR, Bchamp−C1, Bchamp−C2} = 6 comparisons).
- **Correlation analyses** (verifier margin ↔ DPO gain, P1 score ↔ P2 transfer, length ↔ legality) use Spearman rank correlation with bootstrap CIs and are reported as **associational only** — no causal language.
- **Seed variance**: mean ± sd over 3 seeds for the final set; single-seed results are labeled as such in every table.
- Failure taxonomy counts are reported for every canonical run; no failure category is suppressed.

---

## 9. Compute Budget Allocation (planning targets)

| Area                                          | Share of GPU budget |
| --------------------------------------------- | ------------------: |
| Runtime hardening + smoke runs (Gate G1)      |                  5% |
| Phase 1 (B1–B3)                              |                 20% |
| Direct-vs-two-stage transfer (A1–A2, B4–B5) |                 20% |
| Phase 2 GRPO + ablations (B6, E-*)            |                 30% |
| 3-seed final confirmation                     |                 15% |
| Evaluation, E-ATTN, LogosP inference study    |                 10% |

If budget runs short, cut in this order: E-QLORA → E-ATTN long-seq points →
E-RULE → LogosP policy breadth. **Never** cut: the 3-seed final set, the
Track A control, or any frozen eval suite.

---

## 10. Gates and Stopping Criteria

Experiments proceed only through ordered gates. A gate failure stops forward
progress until resolved; resolutions touching this protocol require an
amendment entry.

| Gate                           | Content                                                                                                                                  | Pass condition                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **G1** Runtime hardening | TRL import-gate, Qwen3-8B load, BF16 LoRA forward/backward, tiny SFT/DPO/GRPO (≤20 steps), checkpoint save→kill→resume, HF round-trip | All smoke tests green on the canonical runtime          |
| **G2** Verifier freeze   | Golden fixture suite (pass/fail/malformed/illegal/hacking/timeout/indeterminate)                                                         | 100% expected-status agreement; verifier version tagged |
| **G3** Data freeze       | All eval suites generated, fingerprinted, contamination report clean                                                                     | Hashes committed; no further edits allowed              |
| **G4** Baselines         | C1/C2 and A1 evaluated                                                                                                                   | Results tables populated with CIs                       |
| **G5** Main tracks       | B-track complete through B6                                                                                                              | Champion selected per §6                               |
| **G6** Final             | 3-seed confirmation + ablations                                                                                                          | Report numbers frozen                                   |

**Run-level stopping rules**: a training run is aborted and marked `failed`
(not silently retried with new hyperparameters) if loss diverges (NaN/Inf),
eval JSON validity drops below 0.5 mid-training, or GRPO reward saturates at
ceiling with adversarial-suite hacking incidence >10%. Aborted runs are
reported in the failure analysis.

**Hyperparameter policy**: per (track, objective) we allow at most **3**
development configurations (seed 42), chosen before seeing OOD results
(development selection uses eval-ID only). The chosen configuration is then
frozen for final runs. All development runs are logged and disclosed.

---

## 11. Artifact and Reporting Contract

Every canonical run must produce, or it does not exist:
`resolved_config.yaml`, `environment_manifest.json`, `dataset_manifest.json`
(fingerprints), `git_state.json`, `run_card.json`, `metrics.json`,
`lineage.json` (parent adapter repo/revision/SHA-256, initialization mode),
`checkpoint_pointer.json`, event logs.

Phase-2 runs **must** verify at load time that the parent adapter's SHA-256
matches `lineage.json`; a mismatch is a hard failure
(`tests/integration/test_phase_lineage.py` enforces this contract).

Public release set: final P1 champion adapter, final P2 champion adapter(s),
all eval suites except hidden answers, verifier fixtures, aggregate metrics,
sampled failure cases, tech report. Private: intermediate checkpoints, raw
rollouts, judge prompts.

Reporting language rules: no "SOTA", no "production-ready"; negative and null
results for any pre-registered hypothesis are reported with the same
prominence as positive ones.

---

## 12. Roles and Conflicts

Single-author independent project; the author performs all roles (design,
implementation, evaluation, reporting). This protocol substitutes for external
review by making all decision rules public and time-stamped in git history.

---

## 13. Amendment Log

| Version | Date       | Section | Change         | Rationale |
| ------- | ---------- | ------- | -------------- | --------- |
| v1.0    | 2026-07-23 | —      | Initial freeze | —        |

<!-- Append amendments above. Never edit v1.0 in place. -->

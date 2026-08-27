# Axiom-World — Experimental Protocol (Pre-Registered)
# Protocol v3.0-CLB — Closed-Loop Belief Track

> **Status: FROZEN — v3.0-CLB**
> This protocol is committed **before** any main experiment of this track is run.
> Any change after the first main-track run must be recorded in
> [§14 Amendment Log](#14-amendment-log) with a rationale and a new version number.
> Results produced under an amended protocol must cite the protocol version they
> were run under.

- Project: **Axiom-World — Closed-Loop Belief-State Planning under Partial Observability with a Deterministic Simulator**
- Author: Minjae Kim (independent researcher)
- Protocol version: `v3.0-CLB`
- Parent protocol: `v1.4` — frozen 2026-07-23, published as `docs/protocols/v1/experiment_protocol_v1.md`
  (the pre-restructure path `docs/experimental-protocol.md`, cited by the v1.0
  report under DOI 10.5281/zenodo.22052149, is retained as a redirecting stub)
- Date frozen: `2026-08-27` (KST) — must be committed before run 1
- Code anchor: git commit `1ed61c4cf8b11fa1b02e456250e5ad37a7a7d52d` of `github.com/m97j/axiom-world`
  (the commit that freezes this document; see §12.1 for the two-commit
  freeze procedure)
- Repository baseline: tag `v1.0.1` — protocol-layered layout, `worlds/`
  abstraction in place, all 54 contract tests green
- Companion documents:
  `docs/architecture.md`,
  `docs/specs/verifier-contract.md`,
  `docs/specs/adapter-contract.md`,
  `docs/specs/step-contract-v3.md`,
  `docs/specs/metrics-definitions.md`,
  `docs/data-governance.md`,
  `docs/reproducibility.md`,
  `docs/experiments/v1/b6_grpo_closure.md`,
  `docs/experiments/v1/adapter_contract_termination.md`

---

## 0. Track Placement and Naming Rationale

Protocol v1's Future Work ordered the program as
**v2a** (absolute performance on the frozen spec) → **v2b** (world-spec extension)
→ **v3** (exploratory: partial observability, programmatic geometry, world-models-as-code).

This protocol **deliberately pulls the partial-observability stage of v3 ahead of
v2a/v2b**, and is therefore numbered `v3.0-CLB` rather than `v1.5` or `v2a`.
Three reasons, recorded here so the re-ordering is auditable rather than
opportunistic:

1. **v2a does not add a measurement axis.** v2a raises pass rates on the *same*
   fully-observable, open-loop spec. v1 already answered the recipe question on
   that spec decisively (RQ1, 15/15 sign-consistent). Additional absolute
   performance on an unchanged axis produces no new falsifiable claim about
   *world modeling*, which is the program's stated direction.

2. **Partial observability is the first point at which the object of study
   becomes a world model rather than a plan generator.** Under full observability
   the model maps a complete state description to a plan; the "state
   representation" is supplied, not inferred. Under partial observability the
   model must *maintain* state across steps, which is the minimal condition for
   asking whether it has a transition model at all. v1's Future Work already
   identified this as the track's real content; this protocol acts on that.

3. **Closed-loop evaluation is a prerequisite for counterfactual measurement.**
   v1's evaluation is single-shot and open-loop: one prompt, one plan, one verdict.
   Measuring intervention prediction ("if action *a* were taken instead, what
   state results?") requires a stepwise loop against a simulator. That loop is
   built once, here, and is reusable by v2a/v2b afterwards.

**v2a and v2b are not cancelled.** They are deferred, and this protocol is
designed so its frozen suites and its simulator become *additional* axes that
v2a/v2b checkpoints can be scored on. §12 records the deferral explicitly.

Naming note: `v1.x` is reserved for amendments to protocol v1 (exhausted through
v1.4) and `v2a`/`v2b` are reserved for the two deferred stages above. `v3.0-CLB`
denotes the first frozen protocol of the v3 exploratory track, Closed-Loop Belief
sub-track.

---

## 1. Purpose and Scope

This document pre-registers the research questions, environment contract,
experiment matrix, selection rules, evaluation contract, statistical procedures,
gates, and stopping criteria for the Closed-Loop Belief Track.

Goals, in order:

1. Establish a **closed-loop, action-conditioned** evaluation harness in which a
   deterministic simulator is the sole ground truth, extending v1's
   verifier-as-authority principle from single-shot plans to multi-step
   interaction.
2. Test whether **explicit belief-state emission** improves closed-loop task
   success and **counterfactual rollout accuracy** under partial observability.
3. Test whether v1's headline finding — that Phase-1 general-reasoning warm-start
   is the dominant transfer lever — **survives the move to partial observability**.
4. Establish, measure, and document a **multi-node distributed training stack**
   as a first-class systems result, not an incidental detail.
5. Remain completable, including final write-up, within a **4-week window** on a
   single-GPU baseline plus a bounded multi-node rental budget (§4.3).

**Out of scope (explicitly excluded).** Absolute mastery of PlayWorld (that is
v2a); NPC/dialogue/narrative extension (v2b); programmatic geometry / SVG
emission and world-models-as-code (later v3 sub-tracks); vision or any non-text
modality; model-family comparisons beyond the fixed base; any claim of general
world-model capability. Exclusions are design decisions; rationale in §5.5.

---

## 2. Research Questions and Hypotheses

### RQ-A — Does explicit belief-state emission help? (primary)

> Under partial observability, does a policy trained to emit an explicit,
> verifier-checkable belief state at every step outperform an otherwise identical
> policy trained to emit actions only?

- **H-A1 (primary):** Belief-emitting policies achieve higher **closed-loop task
  success** than action-only policies at matched final-stage training tokens.
- **H-A2:** Belief-emitting policies achieve lower **counterfactual rollout error**
  (§7.1) than action-only policies. This is the mechanistic claim: if the benefit
  in H-A1 is real, it should show up as a better transition model, not only as
  better task outcomes.
- **H-A3:** Belief accuracy is **positively associated** with closed-loop success
  at the episode level (associational, Spearman, no causal language).
- **Falsification:** If the action-only arm matches or beats the belief-emitting
  arm on closed-loop success *and* counterfactual error within the paired CI,
  explicit belief supervision is not justified at this scale. This is a
  reportable negative result and will be reported with the same prominence as a
  positive one.

### RQ-B — Does v1's two-stage transfer finding survive partial observability?

> Does the Phase-1 general-reasoning warm-start advantage established in v1
> (fully observable, open-loop) persist when the task becomes partially
> observable and closed-loop?

- **H-B1:** The two-stage-initialized policy beats the direct-initialized policy
  on closed-loop success, replicating v1's RQ1 direction under the new regime.
- **H-B2:** The advantage is **larger** on horizon-OOD than in-distribution,
  mirroring v1's compositional-OOD pattern.
- **Falsification:** No advantage, or a reversal. Either outcome bounds the
  external validity of v1's headline claim and is a substantive result.

### RQ-C — Is calibration learnable here, and does it track competence?

> Do belief-emitting policies produce **calibrated** uncertainty over unobserved
> world content, and does calibration improve with task competence?

- **H-C1:** Belief confidence is better than chance-calibrated (ECE below a
  pre-registered uninformative-baseline ECE).
- **H-C2 (weak, exploratory):** Calibration improves monotonically across the
  arm ordering P0 → P1 → P2 → P3.
- Framed as exploratory; H-C2 is not part of the correction family (§8).

### RQ-D — Does online RL help once its v1-diagnosed prerequisites are supplied? (secondary)

> v1 closed GRPO as a diagnosed negative result: advantage starvation (50–70%
> zero-variance groups) and entropy collapse (0.15 → 0.008) driven by a narrow
> scenario pool. Under a diversified pool with KL-to-parent, entropy bonus, and
> SFT replay, does verifier-rewarded RL improve the belief-emitting champion?

- **H-D1:** With prerequisites supplied, RL post-training improves closed-loop
  success over its SFT parent without degrading OOD suites.
- **Pre-registered guardrails (aborts the arm, §10):** zero-variance group
  fraction > 40% sustained over 50 steps, or policy entropy below 0.03.
- **Falsification / descope:** If guardrails trip, the arm is **closed and
  documented**, following the v1 B6 precedent, and no rescue attempt is made
  within this protocol. RQ-D is explicitly secondary and its failure does not
  affect RQ-A/RQ-B conclusions.

### RQ-E — Distributed-training characterization (systems, not quality)

> What are the throughput, memory, communication-overhead, and failure
> characteristics of multi-node data/model-sharded training for this workload,
> and at what point does distribution stop paying?

- Reported as a **systems finding**. It changes no canonical quality
  configuration. Explicitly disclosed: at 8B + LoRA scale, multi-node is **not
  required**; it is configured deliberately in order to measure the stack. This
  disclosure is mandatory in every artifact that reports RQ-E (§11).

**RQ-A is primary.** RQ-B is co-primary in reporting weight but subordinate in
gate ordering. RQ-C, RQ-D, RQ-E are secondary. The report's headline claim is
scoped to RQ-A's outcome, whichever direction it falls.

---

## 3. Environment: PlayWorld-PO

### 3.1 Construction

PlayWorld-PO is a **partially observable, sequential-decision variant** of
protocol-v1 PlayWorld. The rule engine, action space, and legality semantics are
**inherited unchanged** so that v1 checkpoints are directly runnable as baselines.

Changes from v1, and only these:

| Axis | v1 PlayWorld | v3.0-CLB PlayWorld-PO |
|---|---|---|
| Observation | full world description in prompt | **local observation**: cells within radius `r` of the agent, plus persistent goal text |
| Interaction | one prompt → one full plan | **stepwise loop**: observation → model output → simulator step → next observation |
| Model output per step | `{actions, final_state}` | `{belief, action, confidence}` (belief/confidence omitted in the action-only arm) |
| Termination | plan complete | goal predicate satisfied, or step budget exhausted, or illegal action under strict mode |
| Ground truth | verifier replays plan | **simulator holds true state at every step** |

`r` is frozen per suite (§7.2) and is never tuned after Gate G3.

### 3.2 Step contract (frozen)

At step *t* the harness supplies the observation and the running interaction
history (truncated to a frozen window `H`), and requires a single JSON object:

```json
{
  "belief": {
    "known": [ {"cell": "...", "content": "..."} ],
    "predicted": [ {"cell": "...", "content": "...", "p": 0.0} ]
  },
  "action": {"type": "...", "args": {}},
  "confidence": 0.0
}
```

- `belief.known` — cells the agent asserts it has directly observed.
- `belief.predicted` — cells it has **not** observed, with a probability in
  `[0,1]`. This field is what makes belief exactly scorable: the simulator knows
  the truth for every one of these cells.
- `confidence` — the agent's own probability that its chosen action is legal and
  goal-advancing. Used only for calibration metrics (RQ-C), never for reward,
  selection, or the primary metric.

Schema is versioned and hash-pinned at Gate G2. A malformed step is a **format
failure**, scored as such, and the episode continues with a no-op so that
format fragility does not silently truncate episodes (this differs from v1,
where a malformed plan ended the episode; the change is required to separate
format failure from planning failure in a loop, and is recorded here as a
deliberate contract difference).

### 3.3 Splits (leakage resistance, inherited from v1 §4.3)

Splits are made at the **scenario-family** level (ruleset × world template ×
goal type), never at the instance level. Families used for expert-trajectory
generation appear in no evaluation suite. Lexical near-duplicate filtering
(Jaccard on token shingles, threshold 0.80) is applied from every training
candidate against every eval suite, and the contamination report is published
with the dataset manifest.

---

## 4. Fixed Environment Contract

### 4.1 Single-GPU baseline (canonical for all quality claims)

| Layer | Frozen value |
|---|---|
| Platform | Google Colab Pro+ G4 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, ~95 GB VRAM |
| Precision / adapter | **BF16 LoRA**, v2 adapter contract: `modules_to_save = [lm_head, embed_tokens]` |
| Attention backend | SDPA |
| Generation | vLLM in a dedicated session; never co-located with training |
| Tracking | W&B + local JSONL event logs (local logs authoritative) |

The v2 adapter contract is **mandatory and non-negotiable** in this protocol.
Rationale: protocol v1 established (amendment v1.2, and
`docs/experiments/adapter_contract_termination.md`) that adapter-only LoRA on a
base checkpoint cannot emit chat-template terminal tokens, producing 100% output
truncation invisible at the loss level. In a **closed loop** that failure mode is
strictly worse than in v1: a truncated step corrupts every subsequent
observation. Any run not using the v2 contract is invalid, not merely degraded.

### 4.2 Multi-node configuration (RQ-E, and used for arm training)

| Layer | Value |
|---|---|
| Provider | rented GPU cloud (RunPod or equivalent), recorded per run |
| Topology | recorded exactly per run: node count, GPUs/node, interconnect, NCCL version |
| Sharding | FSDP (primary) and DeepSpeed ZeRO-3 (comparison), both measured |
| Measured quantities | tokens/s, step time, MFU, peak VRAM/rank, allreduce/allgather share of step time, checkpoint save/resume wall time, cost per training token |

**Mandatory disclosure (repeated in every RQ-E artifact and in the report):**
this workload does not require multi-node execution at 8B + LoRA scale. The
multi-node configuration exists to characterize the stack — including its failure
modes and its crossover point against single-GPU — and any reader should treat
the result as a measurement study, not as evidence of a scale requirement.

Every run records `environment_manifest.json` including the audit output of
`scripts/audit_runtime.py`. A canonical run whose manifest deviates from §4.1
(single-GPU claims) or omits §4.2 topology (multi-node claims) is invalidated.

### 4.3 Budget envelope (hard cap)

| Resource | Cap |
|---|---:|
| Rented multi-node compute | **KRW 700,000 total**, tracked per run in `cost_ledger.json` |
| Single-GPU (Colab) hours | existing subscription; no incremental cap |
| Wall-clock window | **4 weeks** from freeze date, including final write-up |

If the rental cap is reached, cut in this order:
RQ-E ZeRO-3 comparison → RQ-D (RL arm) → P3 entirely → multi-node training of
P2 (fall back to single-GPU).
**Never cut:** the P1 vs P2 contrast (RQ-A primary), the two-initialization
contrast (RQ-B), any frozen suite, or the counterfactual rollout evaluation.

---

## 5. Experiment Matrix (exhaustive; nothing else enters main tables)

### 5.1 Arms

```text
Initialization sources (both pinned by sha256 from protocol v1):
  INIT-B : B4v2 champion adapter  (two-stage: P1 general SFT -> PlayWorld SFT)
  INIT-A : A2v2 control adapter   (direct route endpoint)

Main arms (all trained on PlayWorld-PO expert trajectories, matched token budget):

  P0-B : INIT-B, no PO training  -- zero-shot transfer of the v1 champion
  P0-A : INIT-A, no PO training  -- zero-shot transfer of the v1 control
  P1-B : INIT-B -> action-only behavior cloning        (no belief field)
  P2-B : INIT-B -> belief+action behavior cloning      (belief supervised)
  P1-A : INIT-A -> action-only behavior cloning
  P2-A : INIT-A -> belief+action behavior cloning

  P3   : P2-champion -> verifier-rewarded RL post-training under v1-diagnosed
         prerequisites (diverse/curriculum pool, KL-to-parent, entropy bonus,
         SFT replay)                                   [RQ-D, may be closed]
```

- **RQ-A contrast:** `P2-B vs P1-B` (primary), replicated as `P2-A vs P1-A`.
  Replication across two initializations is what distinguishes a belief effect
  from an initialization artifact.
- **RQ-B contrast:** `P2-B vs P2-A` and `P1-B vs P1-A`.
- **RQ-D contrast:** `P3 vs its P2 parent`.
- P0 rows establish how much of any result is transfer versus PO training, and
  are evaluation-only.

### 5.2 Systems runs (RQ-E, not quality experiments)

```text
S-FSDP  : P2-B training under FSDP, multi-node
S-ZERO3 : P2-B training under DeepSpeed ZeRO-3, matched tokens/seed
S-SINGLE: P2-B training on the single-GPU baseline, matched tokens/seed
```

All three must produce **numerically comparable** final adapters within seed
noise; a quality divergence between S-SINGLE and S-FSDP is itself a reportable
systems finding (and, if found, S-SINGLE is authoritative for quality claims).

### 5.3 Budget parity rules

- All behavior-cloning arms use **equal final-stage training-token budgets**.
- The belief-emitting arms produce more output tokens per step by construction.
  Parity is therefore enforced on **training tokens**, and **episode counts are
  reported alongside** so both readings are visible. This asymmetry is disclosed
  in the report; it is an unavoidable property of the intervention.
- P3 reports quality **and** GPU-hours and cost per valid sample; never quality
  alone.

### 5.4 Seeds and repetition policy

- Development and selection runs: **seed 42 only**.
- Final confirmation: the **RQ-A primary pair** (`P2-B`, `P1-B`) and the **RQ-B
  pair** (`P2-A`) are re-run with seeds **{42, 43, 44}**.
- No other run is repeated. Seed variance is reported only for the 3-seed set.
- Headline claims require **per-suite sign consistency across all three seeds**,
  as in v1 Gate G6.

### 5.5 Pre-registered exclusions

Model-family comparison, SVG/geometry output, dialogue/NPC extension, stochastic
transition dynamics, learned (VLM/LLM) judges in any scoring role, and absolute
mastery targets are excluded. Rationale: the window is four weeks including
write-up, and each of these adds an axis without serving RQ-A/RQ-B. Stochastic
dynamics in particular are deferred deliberately — this protocol keeps the
simulator **deterministic** so that counterfactual rollout error remains exactly
computable, preserving v1's verifier-as-ground-truth principle. Named in
Limitations as future work with this protocol cited as the reason.

---

## 6. Champion Selection Rule (frozen before results)

Three ordered stages. No weighted composite score is used.

### Stage 1 — Hard constraints (must all pass)

| Constraint | Threshold |
|---|---|
| Step-schema validity rate (ID suite) | ≥ 0.95 |
| Legal action rate (ID suite) | ≥ 0.90 |
| Output truncation rate (all suites) | = 0.00 |
| General-capability retention (v1 P1 general held-out vs its own init) | drop ≤ 3 pts absolute |
| Degenerate behavior (no-op looping > 20% of steps; repeated identical action > 10 consecutive steps) | none |
| Catastrophic failure (loss divergence, output collapse) | none |

Checkpoints failing any constraint are ineligible regardless of other scores.

### Stage 2 — Pareto screening

Among eligible checkpoints, compute the Pareto frontier over
`{closed-loop success (horizon-OOD) ↑, counterfactual rollout error ↓, general retention ↑, GPU-hours ↓}`.
Non-frontier checkpoints are eliminated.

### Stage 3 — Primary metric decision

- **Primary metric:** **closed-loop task success rate on `eval_po_horizon_ood`**.
- **Tie-breaker 1:** counterfactual rollout error (lower wins).
- **Tie-breaker 2:** belief F1 on unobserved cells.
- **Tie-breaker 3:** GPU-hours (lower wins).

"Tie" means the 95% paired-bootstrap CI of the difference includes zero.

---

## 7. Evaluation Contract

### 7.1 Metric definitions (all computed by the frozen evaluator, version-pinned)

| Metric | Definition |
|---|---|
| **Closed-loop task success** (primary) | Goal predicate satisfied within the step budget, with every executed action legal under the transition engine |
| **Counterfactual rollout error** (mechanistic) | At `k` frozen probe points per episode, the harness asks the model to predict the resulting state for a **counterfactual** action `a'` it did not take; the simulator executes `a'` from the identical checkpointed state; error = normalized cell-level disagreement. Probe points, `a'` choices, and `k` are **frozen with the suite** and identical across all arms |
| **Belief F1** | Over `belief.predicted` cells (unobserved by construction): F1 of asserted content against simulator truth, thresholded at `p ≥ 0.5` |
| **Belief calibration (ECE, Brier)** | Over `belief.predicted` probabilities against simulator truth; 10 equal-mass bins for ECE |
| **Action confidence calibration** | ECE/Brier of `confidence` against realized "legal and goal-advancing" |
| **Step-schema validity rate** | Parseable, schema-conformant steps / total steps |
| **Legal action rate** | Legal actions / total emitted actions |
| **Steps to goal** | Steps used on successful episodes (efficiency, secondary) |
| **Truncation rate** | Steps hitting the token cap / total steps |
| **General retention** | Score on the frozen v1 Phase-1 general held-out suite |
| **Cost per valid sample** | GPU-seconds / simulator-accepted training sample |

**Counterfactual rollout error is the metric this track exists to produce.** It
is the only metric here that measures a transition model rather than task
outcome, and it is what makes H-A2 a mechanistic rather than a behavioral claim.

### 7.2 Evaluation suites (frozen at Gate G3, fingerprint-pinned)

| Suite | Held-out axis | Size |
|---|---|---|
| `eval_po_id` | held-out instances of training families, `r` as trained | 300 episodes |
| `eval_po_spatial_ood` | unseen world templates and layouts, `r` as trained | 300 episodes |
| `eval_po_horizon_ood` | step budget and goal depth beyond training distribution | 300 episodes |
| `eval_po_occlusion_ood` | **smaller** observation radius than trained (`r' < r`) — tests whether belief maintenance, not memorized layout, is doing the work | 300 episodes |
| `eval_po_adversarial` | traps: illegal-but-plausible actions, observations that invite unwarranted belief assertions, reward-hacking bait | 150 episodes |

`eval_po_occlusion_ood` is the discriminating suite for RQ-A. If explicit belief
emission is doing real work, its advantage should be **largest** here. This is a
pre-registered directional expectation, not a hypothesis in the correction
family.

Counterfactual probe points are generated once, with the suites, and frozen in
the same fingerprint.

### 7.3 Decoding profile (canonical)

Greedy decoding, temperature 0, fixed max-new-tokens per step, fixed system
prompt, fixed history window `H`.

Prompt conditioning follows **protocol v1 amendment v1.1**: suites, simulator,
and decoding are fixed across all arms, but prompt conditioning matches each
subject model's own training-time rendering, declared per eval recipe in the
`evaluation:` config block and recorded in `evaluation_summary.json`. The seed
opener is never part of the scored completion.

### 7.4 Simulator authority

The deterministic simulator is the **sole** ground truth for state, legality,
goal satisfaction, belief correctness, and counterfactual outcomes. No learned
judge participates in any reward, selection, or headline metric — not as a
primary signal and not as a graded auxiliary. This is a strengthening of v1 §7.4
(which permitted an audit-only LLM judge on a 10% sample); under this protocol
no learned judge is used at all, because every quantity of interest is exactly
computable from the simulator.

---

## 8. Statistical Analysis Plan

- **Point estimates** with 95% bootstrap CIs (10,000 resamples) on all
  suite-level metrics.
- **Pairwise comparisons**: paired bootstrap on per-episode outcomes (arms share
  identical eval episodes and identical probe points); report Δ with CI.
- **Significance**: sign-flip permutation test (10,000 permutations); α = 0.05
  with **Holm–Bonferroni across the pre-registered comparison family**, declared
  here and closed:

  ```
  F1: P2-B − P1-B      (RQ-A primary)
  F2: P2-A − P1-A      (RQ-A replication)
  F3: P2-B − P2-A      (RQ-B, belief-emitting route)
  F4: P1-B − P1-A      (RQ-B, action-only route)
  F5: P2-champion − P0 (same init)   (PO training effect)
  F6: P3 − P2-champion (RQ-D, reported as not-run if the arm is closed)
  ```
  Family size 6. Any comparison outside this list is exploratory and labeled as
  such in every table.

- **Counterfactual error** is compared with the same paired procedure on
  per-probe outcomes.
- **Calibration** is reported with bootstrap CIs on ECE; H-C2's monotonicity is
  descriptive only and enters no correction family.
- **Association analyses** (belief F1 ↔ closed-loop success; confidence ↔
  legality) use Spearman rank correlation with bootstrap CIs and are reported as
  **associational only** — no causal language.
- **Seed variance**: mean ± sd over 3 seeds for the final set. Single-seed
  results are labeled as such in every table.
- Failure taxonomy counts are reported for every canonical run; no failure
  category is suppressed.

---

## 9. Compute Budget Allocation (planning targets)

| Area | Share of compute budget |
|---|---:|
| Harness + simulator + verifier hardening, smoke runs (G1–G2) | 10% |
| Expert-trajectory generation (rejection-sampled against simulator) | 10% |
| P1/P2 behavior cloning, both initializations | 30% |
| RQ-E systems runs (S-FSDP / S-ZERO3 / S-SINGLE) | 15% |
| P3 (RL, RQ-D) | 15% |
| Evaluation incl. counterfactual probes | 10% |
| 3-seed final confirmation | 10% |

Cut order on overrun: §4.3.

---

## 10. Gates and Stopping Criteria

Experiments proceed only through ordered gates. A gate failure stops forward
progress until resolved; resolutions touching this protocol require an amendment
entry.

| Gate | Content | Pass condition |
|---|---|---|
| **G1** Harness | Simulator step API, episode loop, history truncation, checkpoint/restore of simulator state, no-op-on-malformed path, save→kill→resume of the loop | All smoke tests green on §4.1 runtime |
| **G2** Verifier & schema freeze | Golden fixtures for step schema (valid / malformed / illegal action / belief edge cases / probe execution), simulator determinism replay (same seed → identical trajectory, byte-equal) | 100% expected-status agreement; schema + simulator version tagged |
| **G3** Data freeze | All five PO suites generated, counterfactual probe points generated, fingerprinted; contamination report clean | Hashes committed; no further edits allowed |
| **G4** Baselines | P0-B and P0-A evaluated on all suites | Result tables populated with CIs |
| **G5** Main arms | P1/P2 complete on both initializations; champion selected per §6 | RQ-A and RQ-B contrasts computed |
| **G6** Systems | S-FSDP / S-ZERO3 / S-SINGLE complete; quality equivalence checked | RQ-E table populated; any quality divergence disclosed |
| **G7** Final | 3-seed confirmation; RQ-D resolved (improved or closed-and-documented) | Report numbers frozen |

**Run-level stopping rules.** A training run is aborted and marked `failed` (not
silently retried with new hyperparameters) if: loss diverges (NaN/Inf);
step-schema validity drops below 0.5 mid-training; truncation rate rises above
0.0 at any evaluation; or, for P3, either RQ-D guardrail trips (§2). Aborted runs
are reported in the failure analysis.

**Hyperparameter policy.** Per (arm, objective) at most **3** development
configurations (seed 42), chosen before seeing any OOD suite (development
selection uses `eval_po_id` only). The chosen configuration is then frozen for
final runs. All development runs are logged and disclosed.

**RQ-D closure precedent.** If P3's guardrails trip, the arm is closed with a
mechanistic write-up in `docs/experiments/` and the champion remains unchanged,
exactly as protocol v1 closed B6/B6-R. No rescue iteration is permitted within
this protocol; rescue attempts belong to a later protocol version.

---

## 11. Artifact and Reporting Contract

Every canonical run must produce, or it does not exist:
`resolved_config.yaml`, `environment_manifest.json`, `dataset_manifest.json`
(fingerprints), `simulator_manifest.json` (version + determinism replay hash),
`git_state.json`, `run_card.json`, `metrics.json`, `lineage.json` (parent adapter
repo/revision/SHA-256, initialization mode), `checkpoint_pointer.json`,
`cost_ledger.json`, event logs. Multi-node runs additionally produce
`topology.json` (node/GPU/interconnect/NCCL) and `throughput.json`.

Training runs **must** verify at load time that the parent adapter's SHA-256
matches `lineage.json`; a mismatch is a hard failure. Every evaluation run
**must** pass the x20-style identity audit against its declared adapter before
its numbers are admitted — v1 caught a stale-weights evaluation this way, and in
a closed loop a stale-weights run is harder to detect by inspection.

Public release set: final adapters for P1/P2 on both initializations, all PO
suites and probe sets except hidden answers, simulator and verifier fixtures,
aggregate metrics, sampled failure trajectories, systems tables, tech report.
Private: intermediate checkpoints, raw rollouts.

Reporting language rules: no "SOTA", no "production-ready", **no claim that the
model has a world model** — the reportable object is closed-loop success and
counterfactual rollout error under a specific deterministic simulator. Negative
and null results for any pre-registered hypothesis are reported with the same
prominence as positive ones. The §4.2 multi-node disclosure is mandatory wherever
RQ-E appears.

---

## 12. Relationship to Deferred Protocols

| Deferred protocol | Status | Interaction with this track |
|---|---|---|
| **v2a** — absolute performance on the v1 frozen spec | Deferred, not cancelled | v2a checkpoints can be evaluated on the PO suites frozen here, giving a second axis at no additional design cost |
| **v2b** — world-spec extension (NPC, interaction, narrative) | Deferred | The step contract defined in §3.2 is the natural carrier for v2b's typed structured output; v2b should extend it rather than replace it |
| **v3 later sub-tracks** — programmatic geometry (SVG), world-models-as-code | Not started | This protocol is their prerequisite: world-models-as-code requires a closed loop against a deterministic simulator, which is what G1–G3 build |

### 12.1 Freeze procedure (two-commit)

This protocol's `1ed61c4cf8b11fa1b02e456250e5ad37a7a7d52d` is **self-referential**: it must name the commit
that freezes this document, which cannot be known while writing it. The freeze
is therefore executed in two commits, and both are part of the record:

1. **Freeze commit.** The document is committed with `1ed61c4cf8b11fa1b02e456250e5ad37a7a7d52d` and
   `2026-08-27` still as literal placeholders. This commit's hash `S` is the
   protocol anchor. Nothing else may be included in this commit.
2. **Seal commit.** `1ed61c4cf8b11fa1b02e456250e5ad37a7a7d52d` is replaced by `S` and `2026-08-27` by the
   freeze commit's author date; the result is committed as a documentation-only
   change and tagged `protocol-v3.0-CLB`.

The freeze commit is authoritative for *when* the protocol was fixed; the seal
commit is authoritative for *what it says*. `git show S -- <this file>` reproduces
the frozen text verbatim, so the placeholder state is itself verifiable evidence
that no content was altered between freezing and sealing. Any change after the
seal commit is an amendment entry in `docs/protocols/v3/amendments.md`, never an
edit in place.

---

## 13. Roles and Conflicts

Single-author independent project; the author performs all roles (design,
implementation, evaluation, reporting). This protocol substitutes for external
review by making all decision rules public and time-stamped in git history before
any result exists.

---

## 14. Amendment Log

| Version | Date | Section | Change | Rationale |
|---|---|---|---|---|
| v3.0-CLB | `2028-08-27` | — | Initial freeze | Track opened; placement rationale in §0 |

<!-- Append amendments above. Never edit the initial freeze in place. -->

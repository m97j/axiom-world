# Axiom-World — Experimental Protocol

# Protocol v2.0-CLB — Closed-Loop Belief Track

> **Status: FROZEN**
>
> This document is the proposed final protocol specification for the first
> v2 subtrack. It has not yet been frozen.
>
> Before the first canonical experiment, the resolved configuration, dataset
> fingerprints, evaluation fingerprints, compute allocation, and protocol
> commit must be frozen.
>
> After freeze, substantive changes require an amendment and a new protocol
> version. Development and implementation changes that do not alter the frozen
> scientific specification remain governed by the repository's reproducibility
> rules.

* Project: **Axiom-World — Closed-Loop Belief-State Planning under Partial Observability with a Deterministic Simulator**
* Author: Minjae Kim (independent researcher)
* Protocol version: `v2.0-CLB`
* Parent protocol: `v1.4` — completed and closed
* Former protocol name: `v3.0-CLB`
* Former protocol status: retained unchanged in git history for provenance
* Date frozen: `2026-08-30`
* Freeze commit: `ab000929df543c77d285d98ae3cc0233300c992f`
* Repository baseline: `v1.0.1` protocol-layered layout
* Companion documents:

  * `docs/architecture.md`
  * `docs/specs/verifier-contract.md`
  * `docs/specs/adapter-contract.md`
  * `docs/specs/step-contract-v3.md`
  * `docs/specs/metrics-definitions.md`
  * `docs/data-governance.md`
  * `docs/reproducibility.md`
  * `docs/experiments/v1/b6_grpo_closure.md`
  * `docs/experiments/v1/adapter_contract_termination.md`

---

# 0. Track Placement and Scope

## 0.1 Why this protocol is v2.0-CLB

Protocol v1 established a recipe-level result on a fully observable,
single-shot PlayWorld task.

The strongest result was obtained by:

```text
Base
  ↓
Phase-1 general-reasoning SFT
  ↓
Phase-2 PlayWorld SFT
  ↓
B4v2
```

The direct route was substantially weaker, DPO was approximately null, and the
initial verifier-rewarded GRPO route regressed under the original conditions.

However, v1 did not require the model to infer unobserved world state or
maintain state across interaction steps.

The successor experiment therefore changes the **observation and interaction
regime** rather than simultaneously changing the general-purpose model.

The former `v3.0-CLB` protocol is consequently promoted to the first formal v2
subtrack.

The historical `v3.0-CLB` artifact is retained unchanged in git history.

---

## 0.2 Scientific Separation of v2 Axes

The v2 research phase contains three separable axes:

```text
v2
│
├── v2-CLB
│     observation + interaction regime
│
├── v2-AP
│     general capability + post-training
│
└── v2-WE
      world complexity
```

These axes are intentionally separated.

### v2-CLB

Changes:

* observation;
* sequential interaction;
* belief representation;
* counterfactual evaluation.

Keeps the v1 initialization fixed.

### v2-AP

Changes:

* Phase-1 general reasoning;
* general agent workflow capability;
* downstream post-training;
* RL/data/curriculum methods.

It is a future protocol and is not part of the canonical v2-CLB intervention.

### v2-WE

Changes:

* horizon;
* world size;
* state complexity;
* interaction structure;
* entities and dynamics.

It is a future protocol and is not part of the canonical v2-CLB intervention.

---

## 0.3 Phase-1 Policy

The v1 Phase-1 intervention is deliberately small and math-centric.

The v1 experiment used:

```text
GSM8K        0.6
MATH         0.4
```

with a target training set of approximately 8,000 records and a frozen
general-retention holdout.

The v1 report explicitly characterized this Phase-1 intervention as a
transfer intervention rather than a frontier general-purpose model.

### v2-CLB initialization rule

**Phase 1 is frozen in v2-CLB.**

The upstream Phase-1 checkpoint is frozen to the v1 `B1v2/P1` run of record.
However, `B1v2/P1` is not itself the canonical v2-CLB initialization.

The canonical two-stage initialization is the frozen v1 champion endpoint
`B4v2`, which already contains:

```text
Base → B1v2/P1 general-reasoning SFT → PlayWorld SFT → B4v2
```

The matched direct-route initialization is the frozen v1 `A2v2` endpoint:

```text
Base → PlayWorld SFT → PlayWorld DPO → A2v2
```

Therefore the canonical initialization mapping is:

```text
B4v2 → v2-CLB two-stage branch
A2v2 → v2-CLB direct-route control
```

No Phase-1 component upstream of `B4v2` may be replaced, retrained, enriched,
or otherwise modified within the canonical v2-CLB comparison.

No new:

* mathematics mixture;
* science data;
* coding data;
* tool/function-call data;
* API/search data;
* long-context data;
* online RL;
* reward-function modification;
* loss-function modification;
* optimizer sweep

may be introduced into the Phase-1 parent of the canonical v2-CLB comparison.

### Reason

RQ-B asks whether the v1 two-stage transfer result survives a change in
observation and interaction regime.

If Phase 1 were simultaneously improved, the experiment could no longer cleanly
distinguish:

```text
effect of partial observability / closed-loop interaction
```

from:

```text
effect of a stronger general-purpose parent model.
```

Phase-1 improvement is therefore reserved for v2-AP.

---

# 0.4 Multi-GPU Design Principle

Multi-GPU is not introduced merely to satisfy an engineering or portfolio
credential.

At the Qwen3-8B + LoRA scale, ordinary SFT and behavior cloning do not require
multi-GPU model parallelism merely to fit the model.

Therefore:

* Phase-1 SFT uses the canonical single-GPU configuration.
* Phase-2 offline behavior cloning uses the canonical single-GPU configuration.
* Small RL development and debugging runs may use single-GPU execution.
* The **canonical online-RL workload must use distributed multi-GPU execution**.
* Single-GPU RL remains a correctness, throughput, and cost reference.
* Single-GPU execution cannot replace the canonical distributed RL result.

The scientific reason is workload scaling:

```text
longer episodes
        +
larger rollout pools
        +
counterfactual probes
        +
repeated verifier evaluation
        +
repeated RL updates
        ↓
large simulator/generation workload
        ↓
parallel rollout workers
        ↓
distributed RL
```

The protocol therefore studies **verifier-grounded closed-loop learning at
increasing workload scale**, rather than performing a standalone distributed
training demonstration.

---

# 1. Purpose and Scope

This document pre-registers the research questions, environment contract,
experimental matrix, selection rules, evaluation contract, statistical
procedures, compute policy, gates, stopping criteria, and artifact contract for
v2-CLB.

## Goals

1. Establish a closed-loop evaluation harness in which the deterministic
   simulator is the sole ground truth.
2. Test whether explicit belief-state emission improves closed-loop success.
3. Test whether belief emission improves counterfactual transition prediction.
4. Test whether v1's two-stage transfer advantage survives partial observability.
5. Test whether calibrated uncertainty can be obtained over unobserved state.
6. Test verifier-grounded online RL after incorporating the v1 failure-mode
   prerequisites.
7. Measure the relationship between closed-loop RL workload and capability.
8. Establish a distributed rollout/training regime that is justified by the
   scientific workload.
9. Preserve v1.4 lineage, reproducibility, verifier, and negative-result rules.

---

# 2. Research Questions and Hypotheses

## RQ-A — Explicit belief-state emission

> Under partial observability, does a policy trained to emit an explicit,
> verifier-checkable belief state at every step outperform an otherwise matched
> policy trained to emit actions only?

### H-A1 — Closed-loop success

Belief-emitting policies achieve higher closed-loop task success.

### H-A2 — Counterfactual prediction

Belief-emitting policies achieve lower counterfactual rollout error.

### H-A3 — Belief/competence association

Belief accuracy is positively associated with closed-loop success.

This association is descriptive and is not treated as causal evidence.

### Falsification

If the action-only arm matches or exceeds the belief arm on both primary success
and counterfactual error within the registered uncertainty criteria, the belief
intervention is not supported at this scale.

---

# 3. RQ-B — Does the v1 transfer advantage survive?

> Does the v1 two-stage initialization advantage survive when the task becomes
> partially observable and closed-loop?

### H-B1

The two-stage initialization outperforms the direct initialization on closed-loop
success.

### H-B2

The two-stage advantage is preserved or strengthened on the registered
horizon-OOD condition.

### Critical control

The v1 parent checkpoints are frozen.

No Phase-1 capability improvement is allowed inside this comparison.

This is the principal control that preserves interpretability between v1 and v2.

---

# 4. RQ-C — Calibration

> Can a belief-emitting policy produce calibrated uncertainty over unobserved
> world content?

### H-C1

Belief probabilities provide better calibration than a pre-registered
uninformative baseline.

### H-C2

As an exploratory analysis, calibration is positively associated with competence.

H-C2 is not part of the primary multiplicity-controlled hypothesis family.

---

# 5. RQ-D — Verifier-Grounded Online RL

> After supplying the prerequisites identified in the v1 GRPO failure analysis,
> can verifier-grounded online RL improve the closed-loop belief-emitting policy
> without unacceptable general/OOD degradation?

The v1 GRPO post-mortem identified:

* advantage starvation;
* zero-variance reward groups;
* entropy collapse;
* narrow scenario-pool specialization.

Accordingly, canonical RL must use:

* diversified rollout pools;
* curriculum;
* KL regularization toward the SFT parent;
* entropy control;
* SFT replay;
* verifier-grounded reward;
* reward-group variance monitoring.

### H-D1

The RL policy improves closed-loop success over its SFT parent while satisfying
the registered retention and OOD constraints.

### Guardrails

The RL arm is stopped if:

```text
zero-variance reward-group fraction > 40%
for 50 consecutive update steps
```

or:

```text
policy entropy < 0.03
for 50 consecutive update steps
```

No rescue hyperparameter search is permitted after closure within this protocol.

---

# 6. RQ-E — Closed-Loop Compute Scaling

> How does policy quality change as verifier-grounded closed-loop rollout and
> interaction workload increases, and what distributed execution regime is
> required to sustain that workload?

The protocol does **not** claim that multi-GPU is intrinsically more effective.

Instead, predefined workload regimes are compared.

Required quantities:

* simulator interactions;
* valid trajectories;
* rollout throughput;
* generated tokens;
* update throughput;
* GPU-hours;
* cost;
* peak VRAM;
* communication overhead;
* policy quality per unit of compute.

No quality claim is attributed to GPU topology itself.

---

# 7. Environment — PlayWorld-PO

## 7.1 Construction

PlayWorld-PO is a partially observable sequential version of the v1 PlayWorld.

Inherited unchanged where possible:

* rule engine;
* action space;
* legality semantics;
* goal predicates;
* deterministic transition semantics;
* scenario-family split principle;
* verifier authority.

Primary intervention:

```text
full observation + open loop
            ↓
partial observation + closed loop
```

| Axis           | v1              | v2.0-CLB                      |
| -------------- | --------------- | ----------------------------- |
| Observation    | full state      | local observation             |
| Interaction    | one-shot        | sequential                    |
| Output         | plan            | belief + action               |
| Ground truth   | verifier replay | simulator state               |
| Counterfactual | unavailable     | checkpointed simulator replay |

---

# 8. Step Contract

At time `t`, the harness supplies:

* current observation;
* persistent goal;
* frozen history window `H`.

Belief arm:

```json
{
  "belief": {
    "known": [],
    "predicted": [
      {
        "cell": "...",
        "content": "...",
        "p": 0.0
      }
    ]
  },
  "action": {
    "type": "...",
    "args": {}
  },
  "confidence": 0.0
}
```

Action-only arm:

```json
{
  "action": {
    "type": "...",
    "args": {}
  }
}
```

`confidence` is used only for calibration analysis.

It is never used for:

* reward;
* checkpoint selection;
* headline scoring.

Malformed output is recorded as a format failure.

The episode continues with a no-op so that sequential format robustness can be
separated from total episode termination.

---

# 9. Scenario and Leakage Control

Splits occur at:

```text
ruleset × world template × goal type
```

rather than at instance level.

Training scenario families must not appear in OOD suites.

Lexical near-duplicate filtering uses the registered token-shingle Jaccard
threshold.

All contamination reports are retained with dataset manifests.

---

# 10. Compute Contract

## 10.1 Single-GPU Configuration

| Component           | Canonical value                                                        |
| ------------------- | ---------------------------------------------------------------------- |
| Platform            | Google Colab Pro+ G4                                                   |
| GPU                 | NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately 95 GB VRAM |
| Model               | Qwen3-8B-Base                                                          |
| Precision           | BF16                                                                   |
| Adaptation          | LoRA                                                                   |
| `modules_to_save` | `lm_head`, `embed_tokens`                                          |
| Attention           | SDPA                                                                   |
| Generation          | vLLM dedicated session                                                 |
| Tracking            | W&B + local JSONL                                                      |
| Log authority       | local JSONL                                                            |

This configuration is canonical for:

* Phase-1 SFT;
* Phase-2 offline BC;
* development;
* evaluation;
* single-GPU RL pilots.

The v2 adapter contract is mandatory.

---

# 11. Distributed Online-RL Configuration

The canonical online-RL stage uses multi-GPU distributed execution.

Minimum:

* 2 GPUs;
* one or more rollout workers;
* simulator workers;
* policy generation path;
* distributed policy update capability.

Per-run topology must record:

* provider;
* node count;
* GPUs/node;
* GPU model;
* interconnect;
* driver;
* CUDA;
* NCCL;
* distributed framework;
* rollout workers;
* training workers.

The architecture should favor rollout parallelism.

```text
policy
  │
  ├── rollout worker 1 ─ simulator
  ├── rollout worker 2 ─ simulator
  ├── rollout worker 3 ─ simulator
  └── rollout worker N ─ simulator
              │
              ▼
        verifier rewards
              │
              ▼
         policy update
```

Model parallelism must not be introduced solely to force a multi-GPU setup.

---

# 12. Registered RL Workload

Before the main RL run, the following must be frozen:

* episodes per update;
* maximum interaction horizon;
* number of updates;
* counterfactual probes;
* scenario-family diversity;
* curriculum stages;
* total simulator interactions;
* generated completion tokens;
* expected valid trajectories.

The canonical workload must materially exceed the development workload.

The workload must also be large enough to expose:

* reward variance;
* rollout diversity;
* policy collapse;
* throughput bottlenecks.

A single-GPU run may establish correctness and reference throughput.

It cannot replace the canonical distributed RL workload.

---

# 13. Distributed Framework

FSDP is the primary distributed training mechanism.

DeepSpeed ZeRO-3 may be used as a systems sensitivity comparison if budget
permits.

The primary scientific relationship is:

```text
RL workload scale
        ↓
policy / transition quality
```

not:

```text
FSDP vs ZeRO-3
        ↓
framework winner
```

---

# 14. Budget

Planning target:

| Area                              | Share |
| --------------------------------- | ----: |
| Harness/verifier/schema           |   10% |
| Trajectory generation/data freeze |   10% |
| Offline BC                        |   25% |
| RL development                    |    5% |
| Canonical distributed RL          |   25% |
| Counterfactual/evaluation         |   10% |
| Final confirmation                |   10% |
| Systems diagnostics               |    5% |

The canonical distributed RL stage is protected.

If budget becomes insufficient, cut:

1. ZeRO-3 sensitivity study;
2. additional scaling points;
3. exploratory calibration;
4. nonessential evaluation expansion.

Do not replace the canonical distributed RL experiment with a smaller single-GPU
run and report it as equivalent evidence.

---

# 15. Initialization Sources

The v1 endpoints are frozen parent sources.

```text
INIT-B
B4v2
Base → v1 Phase-1 general reasoning SFT → PlayWorld SFT

INIT-A
A2v2
Base → PlayWorld SFT → PlayWorld DPO
```

Each parent must be identified by:

* repository;
* revision;
* SHA-256;
* lineage manifest.

Parent-hash mismatch is a hard failure.

---

# 16. Offline Experimental Arms

```text
P0-B = INIT-B, no PlayWorld-PO training
P0-A = INIT-A, no PlayWorld-PO training

P1-B = INIT-B → action-only BC
P2-B = INIT-B → belief + action BC

P1-A = INIT-A → action-only BC
P2-A = INIT-A → belief + action BC
```

Primary RQ-A:

```text
P2-B vs P1-B
```

Replication:

```text
P2-A vs P1-A
```

RQ-B:

```text
P2-B vs P2-A
P1-B vs P1-A
```

---

# 17. Online RL Arm

```text
P2-B
  ↓
verifier-grounded online RL
  ↓
distributed multi-GPU rollout/training
```

The canonical RL intervention includes:

* diversified scenario families;
* curriculum;
* KL-to-parent;
* entropy control;
* SFT replay;
* verifier-grounded rewards.

The RL arm specifically tests whether the v1 failure mechanism can be avoided
under a richer closed-loop training regime.

---

# 18. RL Workload Scaling

At minimum:

```text
RL-DEV
small single-GPU
correctness/debugging

RL-CANONICAL
distributed multi-GPU
canonical scientific workload
```

An additional distributed scale point may be registered before the main result
is inspected.

No post-hoc scale point may be introduced because an intermediate result looks
interesting.

---

# 19. Training Data

All expert trajectories must be:

* simulator generated;
* verifier checked;
* fingerprinted;
* scenario-family labeled;
* leakage checked;
* frozen before canonical training.

All datasets are consumed through SHA-verified fetching.

The standing v1 data-lineage rule remains mandatory.

---

# 20. Budget Parity

## RQ-A

Primary parity is:

* same source episodes;
* same scenario-family distribution;
* same episode count;
* same interaction horizon distribution.

This is preferable to equal output tokens because belief emission intrinsically
changes the representation and token count.

Report separately:

* input tokens;
* output tokens;
* total training tokens;
* simulator interactions;
* GPU-hours.

A token-matched secondary analysis may be reported only if registered before
inspection of the result.

## RQ-B

Both initialization routes use identical PlayWorld-PO training conditions.

## RQ-D / RQ-E

Always report:

* rollout count;
* valid trajectory count;
* simulator interactions;
* generated tokens;
* GPU-hours;
* cost.

---

# 21. Seeds

Development:

```text
seed = 42
```

Final confirmation:

```text
{42, 43, 44}
```

The final seed set covers the primary registered contrasts.

No upstream v1 parent is re-seeded.

The v1 parent adapters remain fixed.

---

# 22. Exclusions

The following are outside canonical v2.0-CLB:

* model-family sweeps;
* larger base models solely to force multi-GPU;
* stochastic environments;
* multimodal input;
* SVG generation;
* programmatic geometry;
* NPC/dialogue/narrative as primary intervention;
* broad optimizer/objective sweeps;
* PPO/SimPO/ORPO/KTO sweeps;
* broad GRPO hyperparameter sweeps;
* learned judges;
* test-time optimization as a substitute for training;
* arbitrary RL rescue iterations;
* Phase-1 capability expansion.

Phase-1 expansion belongs to v2-AP.

---

# 23. Champion Selection

No weighted composite score is used.

## Stage 1 — Hard constraints

All candidate checkpoints must satisfy:

| Constraint              | Threshold |
| ----------------------- | --------: |
| ID schema validity      |   ≥ 0.95 |
| ID legal-action rate    |   ≥ 0.90 |
| truncation              |         0 |
| general retention drop  |   ≤ 3 pp |
| no-op looping           |    ≤ 20% |
| identical-action streak |     ≤ 10 |
| catastrophic failure    |      none |

## Stage 2 — Pareto screening

Dimensions:

* closed-loop success;
* counterfactual error;
* general retention;
* compute cost.

Dominated checkpoints are removed.

## Stage 3 — Primary metric

Primary metric:

> closed-loop task success on `eval_po_horizon_ood`.

Tie-breakers:

1. counterfactual rollout error;
2. belief F1;
3. general retention;
4. GPU-hours.

---

# 24. Evaluation Contract

All metrics are computed by the frozen evaluator.

| Metric                       | Definition                                     |
| ---------------------------- | ---------------------------------------------- |
| Closed-loop success          | goal reached within horizon with legal actions |
| Counterfactual rollout error | normalized state disagreement                  |
| Belief F1                    | predicted unobserved state vs simulator truth  |
| Belief ECE/Brier             | probability calibration                        |
| Action-confidence ECE/Brier  | action-confidence calibration                  |
| Schema validity              | valid steps / total steps                      |
| Legal-action rate            | legal actions / emitted actions                |
| Steps to goal                | successful episode length                      |
| Truncation                   | token-cap hits / total steps                   |
| General retention            | frozen v1 Phase-1 holdout                      |
| Valid trajectory rate        | accepted / generated                           |
| GPU-hours                    | total accelerator time                         |
| Cost/trajectory              | compute cost per accepted trajectory           |
| Cost/success                 | compute cost per successful episode            |

Counterfactual rollout error is the primary mechanistic metric.

---

# 25. Evaluation Suites

All suites are frozen before canonical training.

| Suite                     | Axis                         | Size |
| ------------------------- | ---------------------------- | ---: |
| `eval_po_id`            | held-out instances           |  300 |
| `eval_po_spatial_ood`   | unseen templates/layouts     |  300 |
| `eval_po_horizon_ood`   | longer horizons              |  300 |
| `eval_po_occlusion_ood` | smaller observation radius   |  300 |
| `eval_po_adversarial`   | false-belief/trap conditions |  150 |

`eval_po_occlusion_ood` is the principal discriminating suite for RQ-A.

Counterfactual probe points are frozen with the evaluation suites.

---

# 26. Decoding

Canonical evaluation:

* greedy decoding;
* temperature 0;
* fixed max-new-token limit;
* fixed system prompt;
* fixed history window;
* frozen chat template.

Prompt rendering must match registered training-time rendering.

---

# 27. Simulator Authority

The deterministic simulator is the sole authority for:

* state;
* legality;
* goal satisfaction;
* belief correctness;
* counterfactual outcomes.

No learned judge may participate in:

* reward;
* selection;
* primary metrics;
* auxiliary graded metrics.

---

# 28. Statistical Analysis

## Confidence intervals

Report:

* point estimates;
* 95% bootstrap CIs;
* 10,000 bootstrap resamples.

## Paired comparisons

Use identical evaluation episodes.

RQ-A:

```text
P2-B − P1-B
P2-A − P1-A
```

RQ-B:

```text
P2-B − P2-A
P1-B − P1-A
```

RQ-D:

```text
P3 − P2-B
```

RQ-E:

compare registered workload regimes.

## Multiplicity

The primary family is frozen as:

```text
F1: P2-B − P1-B
F2: P2-A − P1-A
F3: P2-B − P2-A
F4: P1-B − P1-A
F5: P2-B − P0-B
F6: P3 − P2-B
```

Sign-flip permutation tests use 10,000 permutations.

Holm–Bonferroni correction applies across the registered family.

---

# 29. Compute Scaling Analysis

Report:

```text
quality
  vs
simulator interactions
  vs
generated tokens
  vs
GPU-hours
  vs
cost
```

The protocol does not infer causal hardware effects from topology alone.

---

# 30. Gates

| Gate | Content                   | Pass                                             |
| ---- | ------------------------- | ------------------------------------------------ |
| G1   | runtime/model/BC smoke    | all green                                        |
| G2   | schema/verifier/replay    | 100% expected-status agreement                   |
| G3   | dataset/evaluation freeze | hashes committed                                 |
| G4   | zero-shot baselines       | complete                                         |
| G5   | offline arms              | results available                                |
| G6   | RL readiness              | variance/curriculum/KL/entropy/distributed smoke |
| G7   | canonical distributed RL  | completed or explicitly failed                   |
| G8   | final confirmation        | numbers frozen                                   |

A gate failure stops forward progress until the issue is resolved.

Protocol-affecting resolution requires an amendment after freeze.

---

# 31. Run-Level Hard Failures

A run is marked failed rather than silently restarted if:

* NaN/Inf loss;
* severe schema failure;
* canonical evaluation truncation;
* simulator nondeterminism;
* lineage mismatch;
* evaluation identity mismatch;
* RL guardrail trigger.

Failed runs remain in the artifact and failure record.

---

# 32. Hyperparameter Policy

For offline arms:

* maximum 3 development configurations;
* seed 42;
* selection on ID validation only;
* no OOD-based configuration selection.

For RL:

* maximum 3 development configurations;
* development uses RL-DEV;
* canonical distributed configuration frozen before execution;
* no post-hoc tuning based on canonical OOD results.

---

# 33. Artifact Contract

Every canonical run must produce:

```text
resolved_config.yaml
environment_manifest.json
dataset_manifest.json
simulator_manifest.json
git_state.json
run_card.json
metrics.json
lineage.json
checkpoint_pointer.json
cost_ledger.json
event logs
```

Distributed runs additionally produce:

```text
topology.json
throughput.json
distributed_run_manifest.json
```

---

# 34. Reporting Contract

The report must not claim:

* SOTA;
* production readiness;
* general AGI/world-model capability;
* causal superiority of a GPU topology.

Appropriate claims are scoped to:

> closed-loop task performance and counterfactual transition prediction under a
> specific deterministic partially observable simulator.

For P3/RQ-E, explicitly state:

> Multi-GPU execution was introduced because the registered closed-loop RL
> workload was scaled to a distributed rollout/training regime, not because the
> 8B LoRA model required model parallelism merely to fit in memory.

---

# 35. Relationship to v2-AP

The result of v2-CLB does not automatically become the v2-AP champion.

v2-AP must define its own:

* Phase-1 data;
* Phase-1 training procedure;
* general-retention suite;
* agent-workflow suite;
* Phase-2 transfer procedure;
* RL procedure;
* evaluation suites;
* compute budget;
* selection rule.

The v2-CLB parent remains a historical controlled initialization.

A later v2-AP checkpoint may become a new parent only after its own protocol is
frozen and its lineage is explicitly recorded.

---

# 36. Relationship to v2-WE

v2-WE should reuse the closed-loop contract where possible.

World complexity should be changed one principal axis at a time:

1. horizon;
2. state-space size;
3. spatial complexity;
4. persistent state;
5. action space;
6. entities;
7. transition complexity;
8. observability.

The original v2-CLB world remains a baseline.

---

## Cross-Track Inheritance and Control Rule

Shared infrastructure may be reused across v2 subtracks, but checkpoint or
dataset reuse does not silently redefine the scientific baseline.

### v2-CLB → v2-AP

The frozen v1/v2-CLB initialization remains a historical control. If v2-AP
introduces a stronger Phase-1 checkpoint, downstream transfer must include a
matched control using the previous frozen parent under the same downstream
training and evaluation conditions.

A v2-AP checkpoint becomes a new capability baseline only after its own protocol
has been frozen and completed, with lineage explicitly recorded.

### v2-AP → v2-WE

When world complexity is changed, agent capability is held fixed for the primary
world-complexity comparison. If both the historical v2-CLB checkpoint and a
later v2-AP checkpoint are evaluated, they are reported as separate fixed-agent
strata rather than pooled into one effect.

Training data introduced specifically for the new world must not be incorporated
into an older frozen-world comparison.

### Data inheritance

Frozen evaluation data are never promoted into training data. Training datasets
may be reused only when their exact fingerprints and scientific roles are
unchanged. Any enriched, regenerated, or reweighted dataset constitutes a new
intervention and requires an appropriate matched control when used in a causal
comparison.

These rules distinguish infrastructure reuse from intervention reuse and prevent
a later capability or world-complexity change from silently replacing the
control condition of an earlier subtrack.

---

# 37. Relationship to v3

v3 investigates executable world representations.

Potential directions include:

* programmatic geometry;
* SVG;
* structured world representations;
* executable transition functions;
* world-models-as-code.

The v2 closed-loop simulator is infrastructure for these future experiments.

---

# 38. Freeze Procedure

## Step 1 — Scientific freeze

Freeze:

* protocol text;
* evaluation definitions;
* training/evaluation dataset manifests;
* workload definitions;
* primary hypotheses;
* statistical family;
* selection rules.

## Step 2 — Freeze commit

Commit the frozen protocol without subsequent substantive modification.

The resulting commit becomes the immutable content anchor.

## Step 3 — Seal metadata

A documentation-only follow-up commit may record the freeze commit SHA.

The protocol must not claim that a commit containing its own hash is
self-authenticating.

The freeze commit is authoritative for the protocol content.

After sealing, substantive changes require an amendment and a new version.

---

# 39. Roles and Conflicts

This is a single-author independent project.

The author performs:

* protocol design;
* implementation;
* training;
* evaluation;
* analysis;
* reporting.

Independent external review is unavailable.

The protocol therefore compensates through:

* frozen decision rules;
* public lineage;
* deterministic verification;
* explicit stopping rules;
* artifact retention;
* amendment history.

---

# 40. Pre-Freeze Design History

The following decisions were made before canonical v2-CLB execution:

1. The former `v3.0-CLB` track is promoted to v2.0-CLB.
2. Former v2a becomes the future v2-AP capability axis.
3. Former v2b becomes the future v2-WE world-complexity axis.
4. Phase-1 capability expansion is explicitly reserved for v2-AP.
5. v2-CLB freezes the upstream v1 B1v2/P1 source and enters the canonical comparison from the frozen B4v2 and A2v2 endpoints.
6. Canonical online RL uses distributed multi-GPU execution.
7. Multi-GPU is justified by rollout/training workload rather than model size.
8. The freeze procedure is non-self-referential.

These decisions are part of the pre-freeze design record.

---

# 41. Amendment Policy

After the protocol is frozen:

* no substantive text is silently edited;
* any protocol-affecting change receives a new version;
* the old protocol remains reproducible;
* affected results are explicitly associated with the protocol version under
  which they were produced.

Before freeze, design changes are ordinary protocol drafting and do not require
formal amendments.

---

# 42. Final Experimental Logic

```text
v1 frozen parents
      │
      ├───────────────┐
      │               │
      ▼               ▼
action-only BC     belief+action BC
      │               │
      │               ▼
      │          counterfactual
      │             probes
      │               │
      └───────┬───────┘
              ▼
       closed-loop evaluation
              │
              ▼
       belief mechanism result
              │
              ▼
       verifier-grounded RL
              │
              ▼
     distributed rollout/training
              │
              ▼
       compute-scaling result
```

The protocol deliberately does **not** combine:

```text
new Phase-1 capability
+
new world
+
new observation regime
+
new RL recipe
```

in the same causal comparison.

That separation is necessary for the longitudinal interpretability of Axiom-World.

---

# 43. Amendment Log

This table is the protocol-level index of frozen amendments.

Detailed amendment records, including rationale, affected artifacts,
run impact, and remediation where applicable, are maintained in
`amendments.md`.

Before the initial scientific freeze, design changes are ordinary drafting
changes and are **not** entered into this log.

| Version | Date | Sections | Summary | Detailed Record |
| --- | --- | --- | --- | --- |
| v2.0 | 2026-08-30 | — | Initial scientific freeze of Protocol v2.0-CLB. | — |

<!--
Append frozen amendments below this row.

Do not rewrite or delete earlier entries.

Example:
| v2.1 | YYYY-MM-DD | §X, §Y | Short description of the frozen protocol change. | `amendments.md#v21` |
-->
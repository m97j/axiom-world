# Axiom-World Roadmap

## What this project is establishing

Axiom-World investigates whether claims about a language-model agent's competence can be made *checkable*.

The project therefore treats the world, simulator, verifier, evaluation protocol, model lineage, data lineage, and failure record as first-class research artifacts:

* the world the agent acts in is explicitly constructed;
* the simulator defines the authoritative transition dynamics;
* deterministic verification is preferred wherever the target quantity is exactly computable;
* evaluation is frozen before the corresponding training runs;
* model and dataset lineage are cryptographically recorded;
* counterfactual behavior is measured where exact replay is possible;
* failures, null results, and aborted experiments are retained and reported rather than silently discarded.

The long-term research direction is to move from **verifiable plan generation** toward **closed-loop world interaction**, and ultimately toward systems that can represent, construct, and revise executable models of the worlds in which they operate.

The roadmap is organized as research phases rather than as a strictly linear sequence of independent protocols. A phase may contain multiple subtracks with different scientific questions. A roadmap entry is not automatically an active experiment: only a subtrack with an explicitly specified and frozen protocol may produce canonical results.

---

## 1. Track Sequence

| Track                                         | Scientific question                                                                                                                                                  | Status                                                                                                                                                                       |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v1 — Verifiable Recipe Comparison**  | Which post-training recipe transfers to rule-grounded planning, and why?                                                                                             | **Closed.** Two-stage initialization wins on 5 suites × 3 seeds; DPO is approximately null; verifier-rewarded GRPO regresses and its failure mechanism is documented. |
| **v2 — Closed-Loop World Interaction** | Can a post-trained language model maintain state, act sequentially, and reason under incomplete observation in a verifiable world?                                   | **Current research phase.**                                                                                                                                            |
| **v2-CLB — Closed-Loop Belief**        | Does explicit belief-state maintenance improve closed-loop competence and counterfactual transition prediction under partial observability?                          | **Next. Protocol under final pre-freeze review.**                                                                                                                      |
| **v2-AP — Agent Capability Scaling**   | How far can general-purpose reasoning and agent-workflow capability, followed by task post-training, be improved while keeping the world/evaluation axis controlled? | **Planned. Protocol not yet frozen.**                                                                                                                                  |
| **v2-WE — World Extension**            | How does learned capability change as the world becomes longer-horizon, larger, and more interactive?                                                                | **Planned. Protocol not yet frozen.**                                                                                                                                  |
| **v3 — Executable World Models**       | Can the model represent, construct, inspect, and revise executable world dynamics whose behavior remains exactly checkable?                                          | **Exploratory / later.**                                                                                                                                               |

---

## 2. Why v2 Is a Research Phase Rather Than One Experiment

Protocol v1 established a recipe-level result on a fully observable, open-loop PlayWorld.

The strongest result was obtained by the two-stage route:

```text
Base
  ↓
Phase 1: general-reasoning SFT
  ↓
Phase 2: PlayWorld SFT
  ↓
B4v2 champion
```

The direct route remained substantially weaker, while DPO was approximately null and the initial verifier-rewarded GRPO experiments regressed.

This establishes two important facts.

First, **Phase 1 matters** for the tested task and should not be treated as disposable infrastructure.

Second, v1's environment does not establish whether the model maintains an evolving representation of partially observed world state. The world is fully supplied and each episode is evaluated from a single model output. The v1 report therefore deliberately bounds its claims to recipe transfer for rule-grounded planning rather than transition-model acquisition.

v2 therefore separates three axes:

1. **v2-CLB changes the observation and interaction regime.**
2. **v2-AP changes the capability/training regime while controlling the world.**
3. **v2-WE changes the world itself while retaining the established interaction/evaluation contract.**

These axes may share infrastructure, but their headline scientific claims must remain attributable to the variable actually changed.

---

# 3. v2-CLB — Closed-Loop Belief

## Central question

> Can a language model maintain and exploit an explicit belief about an incompletely observed world while interacting with that world sequentially?

The first v2 experiment therefore changes **observability and interaction**, not general-purpose model capability.

### Primary research questions

**RQ-A — Belief**

Does explicit belief-state emission improve closed-loop task success and counterfactual transition prediction relative to an otherwise matched action-only policy?

**RQ-B — Phase-1 transfer**

Does the two-stage transfer advantage established in v1 survive when the same initialization is evaluated in a partially observable closed-loop environment?

**RQ-C — Calibration**

Can the model express calibrated uncertainty over unobserved state?

**RQ-D — Verifier-grounded online RL**

After incorporating the failure-mode prerequisites identified by the v1 GRPO post-mortem, can verifier-grounded online RL improve the closed-loop policy without unacceptable loss of general or OOD capability?

**RQ-E — Rollout/training scaling**

How does closed-loop RL capability change as simulator interactions, rollout diversity, trajectory count, and training workload increase, and what distributed execution regime is required to sustain that workload?

### Phase-1 and initialization policy in v2-CLB

The **upstream Phase-1 source** is frozen to the v1 `B1v2/P1` run of record.
However, `B1v2/P1` is **not itself the canonical v2-CLB initialization**.

The canonical two-stage initialization entering v2-CLB is the frozen v1
champion endpoint **B4v2**, which already contains:

```text
Base
  ↓
v1 B1v2 / P1 general-reasoning SFT
  ↓
v1 PlayWorld SFT
  ↓
B4v2 frozen champion
  ↓
v2-CLB two-stage branch
```

The matched direct-route initialization is the frozen v1 endpoint **A2v2**:

```text
Base
  ↓
v1 PlayWorld SFT
  ↓
v1 PlayWorld DPO
  ↓
A2v2 frozen direct-route endpoint
  ↓
v2-CLB direct-route control
```

Thus the canonical v2-CLB comparison is:

```text
B4v2  ─┐
       ├──→ matched partial-observability training and evaluation
A2v2  ─┘
```

v2-CLB does **not** attempt to improve or replace the upstream general-purpose
Phase-1 recipe. This is deliberate: RQ-B asks whether the v1 two-stage transfer
result survives a new observation/interaction regime. Changing Phase 1 at the
same time would confound the effect of closed-loop partial observability with the
effect of a stronger upstream capability model.

Phase-1 capability expansion is reserved for **v2-AP**.

---

# 4. v2-AP — Agent Capability Scaling

v2-AP is the explicit successor to the capability-improvement portion of the former v2a plan.

Its purpose is not merely to maximize the PlayWorld score. It asks how improvements in general-purpose reasoning and agent workflow capability affect downstream closed-loop/task competence.

## Central question

> How much capability can be gained by improving the general-purpose post-training stack while keeping the downstream world and evaluation contract sufficiently controlled to attribute the improvement?

### Phase-1 capability axis

The current v1 Phase-1 intervention is deliberately small and math-centric:

```text
GSM8K       0.6
MATH        0.4
```

with approximately 8,000 training records and a frozen general-retention holdout. The v1 report explicitly characterizes this as a transfer intervention rather than a frontier general model.

v2-AP may therefore investigate:

* harder mathematical reasoning;
* broader scientific reasoning;
* formal and symbolic reasoning;
* verified code generation;
* structured JSON generation;
* function/tool calling;
* API-use workflows;
* retrieval/search workflows;
* longer-context training;
* long-context data construction;
* improved instruction-following;
* planning and decomposition;
* verifier-grounded reasoning;
* appropriate online RL;
* improved reward functions;
* improved objective/loss design;
* curriculum learning;
* replay and regularization;
* data mixture and data-quality effects.

These are **candidate research axes**, not automatically pre-registered interventions.

Each major intervention must be separately specified so that:

```text
Phase-1 capability
        ↓
Phase-2 task adaptation
        ↓
downstream competence
```

can be distinguished from:

```text
unchanged Phase-1
        ↓
better Phase-2 training
        ↓
downstream competence
```

### Required control principle

A stronger Phase-1 checkpoint must not be silently substituted for the v1 checkpoint in later experiments.

The v1 checkpoint remains the historical baseline.

A v2-AP checkpoint becomes a **new capability baseline** only after its own protocol is frozen and its provenance is recorded.

---

# 5. v2-AP Sub-axes

v2-AP should eventually separate at least three levels:

### AP-1 — General-purpose capability

Improve Phase 1 while evaluating general retention and general reasoning capability independently of PlayWorld.

### AP-2 — Agent workflow capability

Add structured outputs, tool/function calls, retrieval/API workflows, coding, and multi-step tool-use behavior.

### AP-3 — Downstream transfer

Measure whether these improvements transfer to the fixed PlayWorld / closed-loop environment.

The purpose is to prevent a PlayWorld score increase from being automatically interpreted as evidence that the general-purpose model became better.

---

# 6. v2-WE — World Extension

v2-WE addresses the environmental scaling axis.

Candidate extensions include:

* longer interaction horizons;
* larger state spaces;
* richer spatial layouts;
* additional persistent state;
* additional action types;
* NPC or environment entities;
* richer state transitions;
* more complex task dependencies;
* stronger partial observability;
* richer narrative/event structure;
* eventually more general 2-D environments.

Each major extension should be introduced as a separately frozen evaluation axis rather than modifying the original benchmark in place.

The v2-CLB simulator and step contract are intended to serve as the infrastructure for these extensions.

---

# 7. Relationship Between v2 Subtracks

| Subtrack         | Primary change                     | Main scientific question                                                                  |
| ---------------- | ---------------------------------- | ----------------------------------------------------------------------------------------- |
| **v2-CLB** | Observation + interaction regime   | Can the agent maintain and use a belief about an incompletely observed world?             |
| **v2-AP**  | General capability + post-training | How much can general reasoning and agent workflow capability be improved and transferred? |
| **v2-WE**  | World complexity                   | Does the resulting capability survive increasingly complex worlds?                        |

The tracks may share:

* simulator infrastructure;
* verifier infrastructure;
* data-generation machinery;
* checkpoint formats;
* rollout infrastructure;
* evaluation tooling.

They must not silently share **scientific interventions**.

---

## Cross-track inheritance and control rule

Shared infrastructure may be reused across v2 subtracks, but **checkpoint or
dataset reuse does not silently redefine the scientific baseline**.

### v2-CLB → v2-AP

The frozen v1/v2-CLB initialization remains a historical control. If v2-AP
introduces a stronger Phase-1 checkpoint, downstream transfer must include a
matched control using the previous frozen parent under the same downstream
training and evaluation conditions.

A v2-AP checkpoint becomes a new capability baseline only after its own protocol
has been frozen, completed, and its lineage has been recorded.

### v2-AP → v2-WE

When world complexity is changed, **agent capability is held fixed for the
primary world-complexity comparison**. If both a historical v2-CLB checkpoint
and a later v2-AP checkpoint are evaluated, they are reported as separate
fixed-agent strata rather than pooled into one effect.

Training data introduced specifically for a new world-complexity condition must
not be incorporated into an older frozen-world comparison.

### Data inheritance

Frozen evaluation data are never promoted into training data. Training datasets
may be reused only when their exact fingerprints and scientific roles are
unchanged. Any enriched, regenerated, or reweighted dataset is a new intervention
and requires an appropriate matched control when used in a causal comparison.

In short:

```text
checkpoint reuse     ≠ automatic baseline replacement
infrastructure reuse ≠ scientific-intervention reuse
dataset reuse        ≠ permission to mutate a frozen artifact
```

---

# 8. Multi-GPU Principle

Multi-GPU is not introduced merely to satisfy an engineering credential.

At the Qwen3-8B + LoRA scale, ordinary Phase-1 SFT and offline behavior cloning do not intrinsically require multi-GPU model parallelism.

Therefore:

```text
Phase 1 SFT              → single GPU baseline
Phase 2 offline SFT/BC   → single GPU baseline
small RL development     → single GPU allowed
canonical online RL      → distributed multi-GPU
```

The canonical multi-GPU requirement arises from the **scale of closed-loop RL**, including:

* many concurrent trajectories;
* longer interaction horizons;
* larger rollout pools;
* simulator concurrency;
* repeated verifier evaluation;
* counterfactual probes;
* repeated policy updates.

The objective is therefore:

```text
research workload scaling
        ↓
rollout/simulator bottleneck
        ↓
parallel execution
        ↓
distributed RL
```

rather than:

```text
artificially larger model
        ↓
forced multi-GPU
```

---

# 9. v3 — Executable World Models

v3 is reserved for a qualitatively different representation-level question:

> Can a model construct, inspect, modify, and execute representations of world dynamics whose behavior can be compared against an authoritative simulator?

Possible intermediate representations include:

* programmatic geometry;
* SVG/vector-based environments;
* structured world descriptions;
* executable transition rules;
* world-models-as-code.

The closed-loop infrastructure established in v2 is a prerequisite because executable world representations are only scientifically meaningful when their predicted dynamics can be executed and compared against an authoritative transition system.

---

# 10. Protocol Discipline

A roadmap item is not a protocol.

The project distinguishes:

1. **Roadmap** — research direction identified; experimental design open.
2. **Draft protocol** — design sufficiently specified for internal review.
3. **Pre-registered protocol** — design committed before the relevant experiment.
4. **Frozen protocol** — post-freeze changes require explicit amendments and versioning.

Current status:

```text
v1       → completed / closed
v2       → active research phase
v2-CLB   → final pre-freeze review
v2-AP    → planned / not protocolized
v2-WE    → planned / not protocolized
v3       → exploratory
```

No future subtrack should be described as having been pre-registered before its own protocol is actually frozen.

---

# 11. Longitudinal Experimental Invariants

Across future protocols:

* the deterministic simulator remains authoritative where exact computation is possible;
* learned judges do not control headline metrics;
* evaluation is frozen before the relevant training;
* dataset lineage is fingerprinted;
* model lineage is SHA-verified;
* failed and negative experiments remain part of the record;
* historical checkpoints are never silently replaced;
* a stronger model is not substituted into an old experiment without explicit protocol treatment;
* world-complexity changes are separated from training-capability changes;
* Phase-1 capability changes are separated from Phase-2 task-training changes;
* compute scaling is reported with quality and resource consumption;
* distributed systems measurements are distinguished from efficacy claims.

---

# 12. Current Recommended Sequence

```text
v1
│
│ Fully observable
│ Open-loop
│ Recipe comparison
│
▼
v2 — Closed-Loop World Interaction
│
├── v2-CLB
│     B4v2 / A2v2 frozen v1 endpoint initializations
│     Partial observability
│     Belief state
│     Counterfactual prediction
│     Closed-loop RL
│     Canonical distributed RL
│
├── v2-AP
│     Phase-1 general capability scaling
│     Agent workflow scaling
│     Phase-2 transfer
│     RL / curriculum / data scaling
│
└── v2-WE
      Longer horizons
      Larger state spaces
      Richer interactions
      NPC/environment dynamics
      Harder partial observability
│
▼
v3
│
├── structured world representations
├── programmatic geometry / SVG
├── executable dynamics
└── world-models-as-code
```

This sequence deliberately preserves causal separation:

```text
v2-CLB:
"What happens when the world becomes partially observable?"

v2-AP:
"What happens when the agent becomes more capable?"

v2-WE:
"What happens when the world becomes more complex?"

v3:
"What happens when the model itself represents the world's dynamics?"
```

The project should avoid changing all four at once.

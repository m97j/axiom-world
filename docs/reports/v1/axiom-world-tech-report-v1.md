# Axiom-World: A Pre-Registered Study of Two-Stage Post-Training for Rule-Grounded Planning in a Verifiable Toy World

**Minjae Kim**

**Technical Report v1.1 — Protocol v1.4**
Report date: 2026-08-23 (v1.0) · Revised: 2026-08-27 (v1.1)

**DOI (all versions):** [10.5281/zenodo.22052148](https://doi.org/10.5281/zenodo.22052148) · **This version (v1.1):** [10.5281/zenodo.22124790](https://doi.org/10.5281/zenodo.22124790) · **Author ID:** `m97j`
Code & artifacts: [https://github.com/m97j/axiom-world](https://github.com/m97j/axiom-world) (tag `v1.0.0`; the repository was later restructured into protocol-version layers at tag `v1.0.1` without changing any artifact or result — paths below follow the post-restructure layout, and pre-restructure paths cited by v1.0 are retained as stubs)
Protocol: `docs/protocols/v1/experiment_protocol_v1.md` v1.4 (pre-registered, amendment-logged; published at `docs/experimental-protocol.md`, retained as a stub)
Successor protocol: `docs/protocols/v3/experiment_protocol_v3.0-CLB.md` (pre-registered; see the Revision Note below and §8)

---

## Revision Note (v1.0 → v1.1)

**No result, table, number, statistic, or conclusion of v1.0 is changed by this revision.** All experimental content below is as published in v1.0 under protocol v1.4.

This revision makes two documentary changes, both confined to §8 (Conclusions and Future Work):

1. **Track re-ordering, disclosed.** v1.0’s Future Work ordered the program **v2a** (absolute performance on the frozen spec) → **v2b** (world-spec extension) → **exploratory track** (programmatic geometry, then partially observable worlds and world-models-as-code).

   The partial-observability stage of that exploratory track has been *pulled ahead* of v2a and v2b and frozen as its own pre-registered protocol, **v3.0-CLB (Closed-Loop Belief Track)**.

   Because a published report should not silently acquire a different roadmap than the one it was published with, the re-ordering and its rationale are recorded here rather than only in the new protocol.
2. **A methodological limitation of v1 made explicit.** v1’s evaluation is *single-shot and open-loop*: one prompt, one plan, one verdict. Under that design the world state is *supplied* to the model in the prompt rather than *inferred* and *maintained* by it. Consequently v1 — despite its verifiable environment — cannot distinguish a model that has learned a transition model from a model that has learned a good prompt-to-plan mapping.

   This is not a defect in v1’s answer to RQ1–RQ3, which are recipe questions. It is a bound on what *kind* of question v1’s design can ask, and it is the reason the successor protocol changes the evaluation regime rather than the training recipe. It is now stated in §7 item 7 and in §8.

Editorial: the successor protocol is cited where relevant; SVG / programmatic-geometry work is re-positioned as a *later* sub-track of the same exploratory line rather than its entry point.

## Introduction

Post-training pipelines for task-specialized LLM agents face a practical design question: **spend budget on general capability first, or tune directly on the target task?** Staged recipes are common practice, but controlled, seed-replicated comparisons on tasks with *exact* programmatic verification are rare at single-GPU scale.

- **RQ1 (headline).** Does two-stage post-training (general reasoning SFT → task SFT) transfer better to ID/OOD/adversarial planning than direct task tuning of the same base model, at matched final-stage data?
- **RQ2.** Does verifier-mined preference optimization (DPO) add measurable gains on top of either route?
- **RQ3.** Does verifier-rewarded online RL (GRPO) improve the champion further?

All experiments run on a single GPU (Colab G4 runtime), Qwen3-8B-Base, LoRA adapters, with an artifact discipline (resolved configs, canonical dataset fingerprints, adapter sha256 lineage, frozen eval suites, mandatory eval-identity audits) designed so that every reported number is re-derivable from pinned inputs.

**Answers:** RQ1 — yes, decisively and seed-robustly. RQ2 — no (≈ null). RQ3 — no; a diagnosed negative result with a mechanistic post-mortem (§6).

The study’s framing is deliberately **comparative**: it measures *recipe deltas under matched budgets*, not absolute task mastery (see §7).

## Environment and Evaluation

### PlayWorld

PlayWorld is a synthetic, rule-parameterized planning world. Each episode presents a world description (locations, resources, rules such as movement costs and preconditions) and a goal; the model must emit a **structured JSON plan** (`actions` + `final_state`).

A deterministic verifier chain replays the plan against the true rules:

schema/format gates → per-action legality → required-component satisfaction → goal achievement.

`passed` is strict; `score` ∈ [0,1] grants graded partial credit; failures carry a reason-code taxonomy.

**Observability and interaction regime (v1.1 clarification).** The world description is supplied *in full* in the prompt, and the episode is scored from a *single* model output. PlayWorld v1 is therefore a fully observable, open-loop planning task. The model is never required to infer unobserved world content, nor to maintain state across interaction steps. This is a deliberate v1 design choice — it isolates the recipe comparison from state-estimation confounds — and it bounds the class of claims v1 can support (§7 item 7).

### Frozen Evaluation Suites

Five suites × 300 episodes, generated once and frozen (`freeze_fingerprint sha256:3cdcbc30...`, hard-gated at every evaluation).

Splits are made at the **scenario-family** level (ruleset × world template × goal type).

| **Suite**       | **Held-out axis**                                                 | **Measures**           |
| :-------------------- | :---------------------------------------------------------------------- | :--------------------------- |
| `eval_id`           | held-out instances of training families                                 | in-distribution learning     |
| `eval_template_ood` | unseen world templates, seen rules                                      | surface robustness           |
| `eval_comp_ood`     | unseen compositions of seen rule primitives                             | compositional generalization |
| `eval_rule_ood`     | ≥ 1 unseen rule primitive                                              | rule extrapolation           |
| `eval_adversarial`  | hand-written traps (illegal-but-plausible actions, reward-hacking bait) | trap avoidance / alignment   |

The suites are **not difficulty-ordered on an absolute scale**: the four construction suites require building a complete multi-step plan (errors compound multiplicatively), while the adversarial suite is largely a *rejection* task.

Absolute pass rates are only comparable **within** a suite, across models.

All evaluations use greedy decoding, a fixed chat template, and a fixed empty-think opener; the decoding profile is part of the frozen protocol.

### Statistics

Per-suite paired episode-level comparisons: paired bootstrap 95% CIs and sign-flip permutation tests (10,000 resamples).

Headline comparisons were pre-registered with Holm–Bonferroni correction; the family was reduced by Amendment v1.4 (§7).

Gate **G6** required per-suite **sign consistency** of the champion-vs-control delta across 3 seeds.

## Experimental Arms and Data

Base model: **Qwen3-8B-Base** (revision `49e3418f...`), LoRA adapters (v2 adapter contract: `modules_to_save = [lm_head, embed_tokens]`).

**Phase-1 (P1) data.** A pre-registered weighted mixture of public math-reasoning corpora with **dataset-derived gold answers** (never LLM-generated):

GSM8K (`openai/gsm8k`, weight 0.6) and MATH-algebra (`EleutherAI/hendrycks_math`, weight 0.4);

target 8,000 records with prompt-level dedup, 500 held out as a frozen general-retention suite (used as a Stage-1 hard constraint).

The mixture is deliberately small and math-centric: P1 is a *transfer intervention*, not a frontier general model.

**Phase-2 data.** PlayWorld SFT: 2,000 oracle-derived episodes (canonical fingerprint `54fcb1d3...`); verifier-mined preference pairs (fingerprint `5856b5c5...`).

All artifacts frozen on HF and consumed via sha-verified fetches.

| **Arm**        | **Recipe**                              | **Role**                                    |
| :------------------- | :-------------------------------------------- | :------------------------------------------------ |
| **C1**         | Qwen3-8B-Instruct, 0-shot                     | off-the-shelf reference                           |
| **C2**         | Qwen3-8B-Instruct, 3-shot                     | prompted reference                                |
| **A1v2**       | Base → PlayWorld SFT                         | direct route, stage 1                             |
| **A2v2**       | A1v2 → PlayWorld DPO                         | **direct-route endpoint (Track-A control)** |
| **B1v2 (=P1)** | Base → general reasoning SFT (mixture above) | two-stage route, phase 1                          |
| **B4v2**       | P1 → PlayWorld SFT                           | **two-stage endpoint & final champion**     |
| **B5**         | B4v2 → PlayWorld DPO                         | RQ2 on champion                                   |
| **B6 / B6-R**  | B4v2 → GRPO (aggregate / pass-gated reward)  | RQ3                                               |

The “v2” designation marks re-runs after the training-set freeze (the original PlayWorld SFT builder was non-deterministic in oracle-path tie-breaking; v2 re-trained affected arms on a single frozen artifact — amendment-logged).

A sensitivity check found A1v2 ≈ A1v1 on all suites (all n.s.), confirming the freeze itself did not move results.

B6-R is *not* an engineering re-run: it is a protocol-permitted reward-design revision (pass-gated reward) made after diagnosing B6’s aggregate-reward mismatch, with everything else held fixed.

## Headline Result (RQ1): Two-Stage Beats Direct Tuning, Robustly Across Seeds

### Three-Seed Final (Gate G6)

Final-stage training of B4v2 and A2v2 was re-run with seeds 43 and 44 on identical sha-pinned parents:

P1: `b1-general-sft-v2-s42-e6e83b`

A1v2: `a1v2-playworld-sft-s42-269d0e`

Parent adapter hashes were cross-checked against child lineage pins and identical frozen data were used.

Pass rate, mean ± sd over seeds 42,43,44:

| **Suite**       | **B4v2 (two-stage)** | **A2v2 (direct)** | **Δ per seed (42/43/44)** | ***p*** **(perm., per seed)** |
| :-------------------- | :------------------------- | :---------------------- | :------------------------------- | :------------------------------------------ |
| `eval_id`           | **.393 ± .013**     | .186 ± .002            | +.207 / +.223 / +.193            | ≤ .0001                                    |
| `eval_template_ood` | **.349 ± .005**     | .207 ± .003            | +.140 / +.150 / +.137            | ≤ .0001                                    |
| `eval_comp_ood`     | **.329 ± .022**     | .139 ± .004            | +.160 / +.203 / +.207            | ≤ .0001                                    |
| `eval_rule_ood`     | **.302 ± .005**     | .192 ± .005            | +.100 / +.117 / +.113            | ≤ .0003                                    |
| `eval_adversarial`  | **.851 ± .002**     | .651 ± .011            | +.210 / +.203 / +.187            | ≤ .0001                                    |

**Verdict: G6 PASS (15/15 suite×seed deltas positive; all significant).**

Mean-score deltas mirror pass-rate deltas (ID +.133–.147, adversarial +.185–.207, all *p* = 0.0001; paired-bootstrap CIs exclude zero everywhere).

### Failure Analysis

**Champion seed stability (direct paired tests, s43/s44 vs the s42 run of record).**

No suite regresses in either reseed: 18 of 20 suite-metric comparisons are n.s. (|Δpass| ≤ .013, *p* ≥ .52).

The two exceptions are *small improvements* on comp-OOD (s43 +.037, perm. *p*=.063 / bootstrap CI excl. 0; s44 +.040, *p*=.041).

Combined with the mean ± sd table, the champion is stable and the headline gap (~10–22 pp) exceeds seed noise by roughly an order of magnitude.

Failure taxonomies show *why* the routes differ.

B4v2’s failures are almost purely semantic (`required_component_failed`; format-gate failures ≈ 0; 0 truncated outputs),

while A2v2 retains a format-fragility tail (e.g. s43: 57 malformed-JSON failures on adversarial, 55 on rule-OOD) and pervasive output truncation (~1,490/1,500 episodes hit the token cap — the direct-route model appends degenerate continuations after its JSON).

Phase-1 general SFT appears to buy:

(a) clean sequence termination and format discipline, and

(b) higher legal-action rates on OOD suites (≈ .69–.77 vs .66–.72),

consistent with transferable procedural competence rather than surface memorization.

**The adapter-contract prerequisite.**

The termination/format axis was itself the subject of a mid-campaign engineering finding that conditions all results above:

with a standard LoRA target set (attention/MLP projections only), fine-tuned models systematically failed to emit the chat template’s terminal tokens and think-block delimiters — outputs ran to the token cap with degenerate continuations, indistinguishable at the loss level but catastrophic at the verifier’s format gate.

A bottom-up audit chain (x09 termination audit → x10 stop-logit probe → x13 trained-token NLL) localized the failure to special/control tokens whose embeddings and output logits are frozen under adapter-only training while the surrounding distribution shifts.

Adding `lm_head` and `embed_tokens` to `modules_to_save` (the “v2 adapter contract”, applied uniformly to *all* arms) resolved termination without touching task semantics.

A detailed write-up with the audit evidence is maintained separately in

`docs/experiments/v1/adapter_contract_termination.md`.

Here we note only that the A2v2 truncation tail above is a *residual* of the same axis — the direct route, even under the v2 contract, retains weaker termination discipline than the two-stage route, suggesting Phase-1 data volume also contributes to stabilizing sequence-boundary behavior.

### Reference Rows (s42, Off-the-Shelf)

| **Suite (pass)** | **C1: Instruct 0-shot** | **C2: Instruct 3-shot** | **A1v2 (task SFT only)** | **B4v2** |
| :--------------------- | :---------------------------- | :---------------------------- | :----------------------------- | :------------- |
| `eval_id`            | .000                          | .227                          | .167                           | **.393** |
| `eval_template_ood`  | .000                          | .200                          | .190                           | **.350** |
| `eval_comp_ood`      | .000                          | .137                          | .137                           | **.303** |
| `eval_rule_ood`      | .000                          | .137                          | .183                           | **.297** |
| `eval_adversarial`   | .323                          | **.907**                | .590                           | .853           |

C1 fails at the format gate (malformed JSON on ~96% of construction episodes).

C2 fixes formatting well enough to reveal a real but weak planner — and is the **best adversarial trap-avoider** (.907), plausibly because instruct-tuned caution directly serves the rejection-style adversarial task.

This reinforces the evaluation caveat in the frozen-suites description: adversarial pass is not a general-competence proxy.

## RQ2: Preference Optimization Is ≈ Null on This Task

Per-suite pass rates at s42 and paired tests:

| **Suite (pass)** | **B4v2** | **B5 (B4v2→DPO)** | **Δ vs B4v2 (*p*)** | **A2v2** | **Δ B5−A2v2 (*p*)** |
| :--------------------- | :------------- | :----------------------- | :--------------------------- | :------------- | :---------------------------- |
| `eval_id`            | .393           | .380                     | -.013 (.59)                  | .187           | **+.193 (.0001)**       |
| `eval_template_ood`  | .350           | .337                     | -.013 (.64)                  | .210           | **+.127 (.0001)**       |
| `eval_comp_ood`      | .303           | .293                     | -.010 (.71)                  | .143           | **+.150 (.0001)**       |
| `eval_rule_ood`      | .297           | .307                     | +.010 (.73)                  | .197           | **+.110 (.0002)**       |
| `eval_adversarial`   | .853           | .817                     | **-.037 (.047)**       | .643           | **+.173 (.0001)**       |

- **On the champion:** no construction suite moves (all |Δ| ≤ .013, n.s.); the only significant effect is a small **adversarial pass decrease** (-.037, *p*=.047), the single isolated significant result in the DPO family.
- **On the direct route** (A1v2 → A2v2): no suite reaches significance (|Δpass| ≤ .017, all *p* ≥ .36).
- **The pre-registered two-stage-vs-direct contrast fully survives DPO on both sides:** B5 ≫ A2v2 on 10/10 suite-metrics (*p* ≤ .0002; mean-score deltas +.123 to +.212).

Verifier-mined pair quality was *higher* on the champion (mining yield .675 vs .447), so the null is not explained by pair scarcity.

Interpretation: with a strict programmatic verifier and an SFT stage that already fits the format and rule surface, offline preference deltas carry little additional trainable signal at this scale.

The random-pair mining control (E-RANDPAIR) was de-scoped accordingly (Amendment v1.4): a control for a ~ null effect is uninformative.

## RQ3: Verifier-Rewarded GRPO — a Diagnosed Negative Result

### Outcome

Two GRPO variants on top of B4v2, sharing frozen scenario pools and decoding.

Per-suite pass rates at s42 (B6-R rates derived exactly from the episode-paired flip matrices of the audited run of record `8791aa`; see Appendix 10):

| **Suite**       | **B4v2 pass / score** | **B6 pass (Δ, *p*)** | **B6-R pass / score (Δpass)** |
| :-------------------- | :-------------------------- | :---------------------------- | :----------------------------------- |
| `eval_id`           | .393 / .593                 | .253 (**-.140**, .0002) | .233 / .530 (-.160)                  |
| `eval_template_ood` | .350 / .559                 | .210 (**-.140**, .0001) | .213 / .518 (-.137)                  |
| `eval_comp_ood`     | .303 / .546                 | .217 (**-.087**, .024)  | .233 / .533 (-.070)                  |
| `eval_rule_ood`     | .297 / .513                 | .260 (-.037, .35)             | .240 / .513 (-.057)                  |
| `eval_adversarial`  | .853 / .951                 | .750 (**-.103**, .0001) | .800 / .929 (-.053)                  |

B6-R pass rates from the audited `8791aa` summary agree exactly with the episode-paired flip-matrix reconstruction.

Notably, B6-R’s *mean scores* sit much closer to B4v2 than its pass rates (rule-OOD score is identical at .513; legal-action rates are *higher* than B4v2 on OOD suites, .77–.99):

the GRPO policy remains a competent partial planner but loses precisely the required-component completion that strict passing demands — the score/pass wedge that the pass-gated reward was built to close at training time, visible here surviving into evaluation.

B6 is also significantly below B5 on ID/template/adversarial (e.g. ID -.127, *p*=.0008).

The GRPO arm was **closed with champion unchanged** (`docs/experiments/v1/b6_grpo_closure.md`).

### Mechanism

1. **Aggregate-reward mismatch (B6):** 40–50% of failing adversarial episodes scored >0.8 — the policy learned to *almost pass*, harvesting partial credit.

   B6-R’s pass-gated reward (passed → 0.5 + 0.5·score; failed → 0.1·score) removed this incentive.
2. **Episode-level confirmation (x19 flip analysis, B6-R vs B6):** the entire recovery is **adversarial-local** — net +15/300 flips there (fail→pass 25, pass→fail 10), every other suite within ± 6 net flips (noise level).

   100% of pass→fail regressions across all suites carry `required_component_failed`.

   A residual near-miss cluster persists under B6-R (6/10 adversarial pass→fail episodes score in [0.8,1.0)) but is reward-gated during training.
3. **The binding failure is data/regularization, not reward shape:** 50–70% of prompt groups had zero reward variance (no gradient — *advantage starvation*), and policy entropy collapsed 0.15 → 0.008.

   The policy specialized to the narrow training pool’s required-component patterns and lost ID/OOD coverage.

   Reward redesign (B6-R) was precisely the controlled test of the alternative hypothesis — and it did not move the construction suites.

### Implication

Reward-function or verifier refinement alone is unlikely to rescue online RL here.

The binding constraints are scenario-pool diversity/curriculum and regularization (KL-to-parent, entropy bonus, SFT replay), all explicitly out of protocol-v1 scope.

Advantage-estimator swaps (e.g. RLOO) share the group-relative structure exposed to the same starvation mechanism; E-RLOO was de-scoped on these grounds (Amendment v1.4) rather than run as a low-prior ablation.

**(v1.1 note.)** These prerequisites are now supplied and tested under the successor protocol v3.0-CLB, where advantage-starvation fraction and policy entropy are *pre-registered abort conditions* rather than post-hoc diagnostics — i.e. the v1 post-mortem is converted into a stopping rule before the arm is run.

## Threats to Validity and Disclosures

1. **Comparative, not absolute, claims.**

   Absolute pass rates of the champion remain modest on construction suites (.30–.39 ID/OOD at 8B/LoRA/2k-episode scale).

   Protocol v1 was designed to answer *which recipe transfers better under a fixed small budget*, and its gates (G1–G6) were defined over comparative deltas, never over an absolute mastery threshold.

   Absolute mastery of PlayWorld is future work (§8), not a claim of this report.
2. **Scale and task scope.**

   One base model (8B), LoRA adapters, one synthetic environment, 300 episodes/suite.

   External validity to real agent tasks is untested.
3. **Reseeding scope & de-scoped ablations.**

   Seeds cover the final-stage training of B4v2/A2v2 only; upstream stages (P1 SFT, A1v2 SFT, pair mining) and B5/B6/C rows are single-seed and labeled as such.

   E-RLOO, E-RANDPAIR, E-QLORA, E-ATTN, E-RULE and the LogosP breadth study were dropped under the pre-registered budget-cut order plus the §6 closure rationale (Amendment v1.4);

   the two dropped headline comparisons are disclosed as *not run*.
4. **Infrastructure incidents (all documented; none affect reported numbers):**

   - An early B6-R evaluation (`f1854f`) scored **stale hub weights** — caught by the x20 identity audit (1.0 prediction-identical rate with B6) and excluded.

     The audited B6-R run of record is `8791aa`.

     B6-R pass rates in the GRPO section are derived exactly from `8791aa`’s episode-paired flip matrices; x20 audits are mandatory in all later stages.
   - An interim G6 aggregation consumed a wrong summary for the b4v2-s42 baseline (missing fetch in a fresh runtime), producing a spurious FAIL.

     The run of record uses hard-gated per-entry loading (`adapter_dir + freeze fingerprint` asserted per entry).
   - Two champion-stability analyses initially bound a wrong baseline run; they were re-run against the verified s42 eval (`7308ee`) and the corrected versions are reported in the RQ1 section.
5. **Dataset non-determinism (fixed).**

   The v1 SFT builder broke oracle-path ties non-deterministically; discovered mid-campaign, fixed by freezing a single artifact and re-running affected arms as “v2” (A1v2 ≈ A1v1: all n.s.).
6. **Adversarial suite semantics.**

   Rejection-style scoring makes absolute adversarial pass rates incomparable to construction suites (see C2’s .907).
7. **Open-loop, fully observable design — a bound on the kind of claim this report can make (added in v1.1).**

   Every episode supplies the complete world state in the prompt and is scored from a single model output. The model therefore never infers unobserved content and never maintains state across steps.

   Consequently, this report’s results support claims about *which post-training recipe transfers better on a rule-grounded planning task*, and *not* claims about whether any arm has acquired a *transition model* of the world: a policy that maps a fully specified state to a correct plan is observationally indistinguishable, under this design, from one that predicts how the world changes under intervention.

   Distinguishing the two requires (i) partial observability, so that state must be inferred and carried, and (ii) a closed loop against a simulator, so that a counterfactual action can actually be executed and compared against the model’s prediction.

   Neither is present in protocol v1. Both are the subject of the successor protocol (§8). We state this explicitly because the phrase “verifiable world” can otherwise be over-read: v1’s verifier is exact, but what it verifies is a *plan*, not a *prediction*.

## Conclusions and Future Work

**Conclusions.**

On a verifiable planning task with frozen evaluation:

(i) a general-reasoning SFT phase before task SFT yields large, seed-robust ID *and* OOD gains over direct tuning at matched task data — the single strongest and most stable effect in the study;

(ii) offline DPO adds ≈ nothing on either route;

(iii) verifier-rewarded GRPO actively hurts without data-side diversity and entropy/KL regularization, and reward-shape fixes relocate, but do not remove, the failure.

### Future work (revised in v1.1)

v1.0 ordered the program v2a → v2b → exploratory track. That ordering is **revised**: the partial-observability stage of the exploratory track has been pulled ahead and frozen as its own pre-registered protocol. The three deferred items are unchanged in content and are listed after it.

#### Now in progress — Protocol v3.0-CLB (Closed-Loop Belief Track).

*Rationale for the re-ordering*, recorded here so that it is auditable rather than opportunistic:

1. **v2a adds no measurement axis.** It raises pass rates on the *same* fully observable, open-loop specification. v1 already answered the recipe question on that specification decisively (15/15 sign-consistent). Further absolute performance on an unchanged axis yields no new falsifiable claim about *world modeling*, which is this program’s stated direction.
2. **Partial observability is the first point at which the object of study becomes a world model** rather than a plan generator (§7 item 7). Under full observability the state representation is supplied, not inferred, so the question “does the model have a transition model” is not well posed.
3. **Closed-loop interaction is a prerequisite for counterfactual measurement**, and that loop, built once, is reusable by v2a and v2b afterwards.

*Design.* **PlayWorld-PO** inherits v1’s rule engine, action space, and legality semantics unchanged, and localizes observation to a radius *r*. At each step the model emits `belief`, `action`, `confidence`; a deterministic simulator executes the action and returns the next observation, holding true state throughout.

*Questions.* (RQ-A, primary) does explicit belief-state emission improve closed-loop task success and **counterfactual rollout error** over an action-only policy at matched training tokens? (RQ-B) does v1’s two-stage transfer advantage survive the move to partial observability and closed-loop scoring? (RQ-C) is uncertainty over unobserved content calibrated, and does calibration track competence? (RQ-D, secondary) does verifier-rewarded RL help once the prerequisites diagnosed in §6 are supplied?

*Key metric.* **Counterfactual rollout error**: at frozen probe points the model predicts the state resulting from an action it did *not* take; the simulator executes that counterfactual from the identical checkpointed state; error is normalized cell-level disagreement. Task success measures outcomes; this measures the transition model.

*Safeguards, registered before running.* A discriminating *occlusion-OOD* suite with a *smaller* observation radius than trained (if belief maintenance rather than memorized layout is doing the work, the advantage should be largest there); replication of the belief contrast across *two* initializations (v1’s two-stage champion and v1’s direct control) so that a belief effect is separable from an initialization artifact; advantage-starvation and entropy *abort* thresholds for RQ-D; and the removal of learned judges from every scoring path (v1 permitted an audit-only judge on a 10% sample; v3 uses none, because every quantity of interest is exactly computable from the simulator).

A distributed-training characterization (FSDP vs DeepSpeed ZeRO-3 vs the single-GPU baseline: throughput, MFU, communication share of step time, sharded checkpoint save/resume, cost per training token) accompanies the track as a *systems* result, with the explicit disclosure that multi-node execution is not required at 8B/LoRA scale and is configured in order to measure the stack.

#### Deferred — Protocol v2a: close the absolute-performance gap on the frozen spec.

Push the champion recipe toward mastery of the *current* PlayWorld (construction-suite pass rates from ~.30–.39 toward a pre-registered target band), holding the frozen suites fixed so progress is measurable against v1.

The levers are exactly those v1 identified as binding:

**(a) Phase-1 enrichment** — the dominant transfer lever in v1 (B5 ≫ A2v2 even after DPO) — extending the math-centric mixture with harder math/logic corpora, sandbox-executed code generation with verified rewards, and structured-JSON tool-calling, all representationally adjacent to PlayWorld’s output contract;

**(b) Phase-2 data** — scenario-pool expansion and difficulty curricula targeting the `required_component_failed` mass;

**(c) online RL retried under its prerequisites** — diverse/curriculum pools, KL-to-parent, entropy bonuses, SFT replay, with advantage-starvation and entropy metrics as pre-registered guardrails.

Item (c) is now partially absorbed into v3.0-CLB’s RQ-D, which tests the same prerequisites in the closed-loop regime; v2a retains it for the open-loop specification so that the two regimes remain separately attributable.

#### Deferred — Protocol v2b: controlled world-spec extension.

Longer horizons, NPC/environment interactions, grid-free 2-D layouts — each added as a *new frozen suite axis* so that v1/v2a checkpoints remain comparable baselines.

v3.0-CLB’s per-step contract is the intended carrier for v2b’s typed structured output (dialogue / action / state-delta / event), which should *extend* that contract rather than replace it.

#### Deferred — later exploratory sub-tracks: programmatic geometry, then world-models-as-code.

Text-native spatial world modeling via vector-graphics output: emit **SVG** as the plan/state representation — paths in grid or grid-free 2-D worlds, composable scenes — verified by a *hybrid* stack: deterministic geometry checks on the parsed SVG first; rendering + rule-based CV, and optionally VLM judges, only as secondary graded signal.

This inherits v1’s core lesson: keep a deterministic verifier as the ground truth and treat learned judges as noisy auxiliaries, since verifier-rewarded RL already showed reward-mismatch failure modes with *exact* verifiers (§6).

Known risks to de-risk first: LLMs’ precise-geometry weaknesses in SVG generation, VLM-judge exploitability, and rendering-pipeline complexity on a single-GPU budget.

**Re-positioned in v1.1:** v1.0 presented SVG as the entry point of the exploratory track and partial observability as its later stage. This is reversed. Partial observability requires no new representation or rendering machinery — it reuses v1’s JSON contract and rule engine — whereas SVG adds a representation, a parser, and a rendering pipeline simultaneously, and its known risks (precise geometry, judge exploitability) are exactly the ones a single-GPU budget is least able to absorb. Sequencing the cheaper, better-instrumented change first is the lower-variance path to the same destination.

The endpoint of the line is unchanged: *world-models-as-code*, where the model writes and revises executable dynamics whose rollouts remain exactly checkable — preserving the verifier-as-ground-truth principle throughout. v3.0-CLB’s closed loop against a deterministic simulator is its prerequisite: a model cannot be asked to author dynamics until there exists a loop in which its authored dynamics can be executed and compared.

## References

1. K. Cobbe et al. “Training Verifiers to Solve Math Word Problems.” arXiv:2110.14168, 2021. (GSM8K)
2. D. Hendrycks et al. “Measuring Mathematical Problem Solving with the MATH Dataset.” arXiv:2103.03874, 2021.
3. E. J. Hu et al. “LoRA: Low-Rank Adaptation of Large Language Models.” arXiv:2106.09685, 2021.
4. R. Rafailov et al. “Direct Preference Optimization: Your Language Model is Secretly a Reward Model.” arXiv:2305.18290, 2023.
5. Z. Shao et al. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.” arXiv:2402.03300, 2024. (GRPO)
6. Qwen Team. “Qwen3 Technical Report.” arXiv:2505.09388, 2025.
7. L. von Werra et al. “TRL: Transformer Reinforcement Learning.” [https://github.com/huggingface/trl](https://github.com/huggingface/trl), 2020–.
8. S. Mangrulkar et al. “PEFT: Parameter-Efficient Fine-Tuning.” [https://github.com/huggingface/peft](https://github.com/huggingface/peft), 2022–.
9. Y. Zhao et al. “PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.” arXiv:2304.11277, 2023.
10. S. Rajbhandari et al. “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.” arXiv:1910.02054, 2019.

---

## Appendix A. Runs of Record

| **Artifact**                     | **ID / value**                                                                                                                                                                           |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Champion training run                  | `20260814-023603-b4v2-playworld-sft-from-p1-s42-c56ed2` (repo `m97j/aw-runs-b4`) · adapter `sha256:d4fcacdd...`                                                                         |
| Champion eval (s42)                    | `20260814-032546-eval-playworld-s42-7308ee`                                                                                                                                                  |
| Control training run                   | `20260814-114257-a2v2-playworld-dpo-s42-ef1e20` (repo `m97j/aw-runs-a2`) · adapter `sha256:cc23345f...`                                                                                 |
| Control eval (s42)                     | `20260814-120733-eval-playworld-s42-61d7cd`                                                                                                                                                  |
| Parents (pinned)                       | P1`20260807-225109-b1-general-sft-v2-s42-e6e83b` (`sha256:747e5757...`) · A1v2 `20260814-022256-a1v2-playworld-sft-s42-269d0e` (`sha256:70c2ecb9...`)                                 |
| Seed runs (repo`m97j/aw-runs-seeds`) | b4v2-s43`20260817-151254-...-c04b9c` · a2v2-s43, b4v2-s44, a2v2-s44 (run cards in repo)                                                                                                     |
| Seed evals                             | s43:`20260818-100258-...-304567` (b4v2), `20260818-101023-...-44a500` (a2v2) · s44: `20260818-105059-...-082b2a` (b4v2), `20260818-105846-...-40b6f2` (a2v2)                          |
| B6 / B6-R runs                         | B6`20260815-...-b6-grpo-s42` · B6-R `20260816-...-de963e` (pass-gated rerun) · B6-R eval `20260816-201107-...-8791aa` (x20-audited) · superseded eval `f1854f` void (stale weights) |
| Frozen suites                          | `m97j/aw-playworld` — 5 suites × 300 episodes, fingerprint-pinned                                                                                                                          |
| Champion model                         | `m97j/axiom-world-b4v2` (curated)                                                                                                                                                            |

## Appendix B. Diagnostic Tooling (x-scripts)

x09 termination audit · x10 stop-logit probe · x11 adapter integrity · x12 attested probe · x13 trained-ids NLL · x15 SFT data diff · x16 SFT provenance resolver · x17 GRPO scenario audit · x19 regression/flip diagnostics · x20 eval identity audit (stale-weights guard) · x21 seed-variance aggregation.

Each incident that motivated a tool is preserved chronologically in the campaign notebooks (aw_01–aw_11).

## Appendix C. Document Revision History

| **Version** | **Date** | **Change**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| :---------------- | :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1.0              | 2026-08-23     | Initial publication under protocol v1.4. Zenodo version DOI 10.5281/zenodo.22052149 (concept DOI 10.5281/zenodo.22052148).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| v1.1              | 2026-08-27     | Documentary revision only; no result changed. (a) §7 item 7 added: the open-loop, fully observable design bounds the class of claims the report can support — an exact verifier over*plans* is not an exact verifier over *predictions*. (b) §8 Future Work re-ordered: the partial-observability stage is pulled ahead of v2a/v2b and frozen as protocol v3.0-CLB, with the rationale recorded in-text; SVG / programmatic geometry re-positioned as a later sub-track with an explicit justification for the swap. (c) §6 implication note: the v1 GRPO post-mortem is converted into pre-registered abort conditions in the successor protocol. (d) Two systems references added (FSDP, ZeRO) for the distributed characterization accompanying the successor track. (e) Documentary path updates following the repository restructure at tag `v1.0.1` (protocol-version layering); no artifact, hash, or number is affected, and the pre-restructure paths cited by v1.0 remain resolvable as stubs. |

# Axiom-World: A Pre-Registered Study of Two-Stage Post-Training for Rule-Grounded Planning in a Verifiable Toy World

**Technical Report v1.2 — Protocol v1.4**

> **Repository Markdown edition.** This file is the GitHub-facing Markdown rendering of
> Technical Report v1.2. No v1 experimental result, table, numerical statistic, run
> identifier, or conclusion is changed by v1.2. The revision synchronizes the documented
> future research program with the pre-freeze v2 roadmap. **v2.0-CLB remains pre-freeze
> at this report revision.**

**DOI (all versions):** [10.5281/zenodo.22052148](https://doi.org/10.5281/zenodo.22052148) $\cdot$ **This version (v1.2):** [10.5281/zenodo.22177402](https://doi.org/10.5281/zenodo.22177402)

**Author ID:** `m97j`
**Code & artifacts:** [https://github.com/m97j/axiom-world](https://github.com/m97j/axiom-world)

(tag `v1.0.0`; the repository was later restructured into protocol-version layers at tag `v1.0.1` without changing any artifact or result — paths below follow the post-restructure layout, and pre-restructure paths cited by v1.0 are retained as stubs)
**Protocol:** `docs/protocols/v1/experiment_protocol_v1.md` v1.4

(pre-registered, amendment-logged; published at `docs/experimental-protocol.md`, retained as a stub)
**Next protocol:** **v2-CLB (Closed-Loop Belief)**

(pre-registration written; freeze pending; see the Revision Note below and §8)

# Revision Note (v1.1 $\rightarrow$ v1.2)

**No experimental result, table, numerical statistic, or conclusion of v1.0 is changed by this revision.** All experimental content remains the published result of v1 under protocol v1.4.

This revision synchronizes the documented future research program with the current pre-freeze roadmap and successor protocol.

1. **Successor-track renumbering.** The previously documented `v3.0-CLB` Closed-Loop Belief track is now positioned as **v2.0-CLB**, the first subtrack of the broader v2 Closed-Loop World Interaction phase.

   The former `v3.0-CLB` artifact remains a historical protocol artifact and is not rewritten.
2. **v2 is now explicitly a research phase with separable axes.** The roadmap distinguishes:

   - **v2-CLB**: observation and interaction regime;
   - **v2-AP**: general capability and post-training scaling;
   - **v2-WE**: world-complexity scaling.

   This separation is intended to prevent changes in model capability, world complexity, and observation regime from being interpreted as the same causal effect.
3. **Phase-1 capability expansion is explicitly assigned to v2-AP.** The v1 Phase-1 intervention was deliberately small and math-centric and was used as a transfer intervention rather than as a frontier general-purpose training recipe.

   The successor v2-CLB protocol therefore freezes the upstream v1 `B1v2/P1` Phase-1 source while entering the canonical comparison from the frozen v1 endpoints **B4v2** (two-stage champion) and **A2v2** (direct-route control). This keeps the primary question focused on whether the v1 transfer finding survives partial observability and closed-loop interaction.

   Future expansion of Phase 1—including harder mathematics and science, coding, structured JSON and tool/function calling, API/search workflows, long-context data and methods, and appropriately designed online RL—is reserved for the separately specified v2-AP capability-scaling track.
4. **Multi-GPU positioning is clarified.** The successor protocol does not introduce multi-GPU merely as an engineering credential.

   At the 8B + LoRA scale, ordinary SFT and offline behavior cloning can remain single-GPU. Distributed multi-GPU execution is reserved for the canonical online-RL workload, where larger rollout pools, longer interactions, simulator concurrency, counterfactual probes, and repeated policy updates create a genuine distributed workload.
5. **Protocol-status language is clarified.** At the time of this report revision, v2.0-CLB is a pre-freeze successor protocol. It must not be described as a completed experiment or as a published result.

## Abstract

We study whether a **two-stage post-training recipe** — general reasoning SFT (Phase 1) followed by task-specific tuning (Phase 2) — outperforms **direct task tuning** for rule-grounded planning, using a fully verifiable synthetic environment (PlayWorld) with frozen, fingerprint-pinned evaluation suites spanning in-distribution, three OOD axes, and adversarial traps. All arms, comparisons, champion-selection rules, and stop rules were pre-registered before the first run; every deviation is logged as a numbered amendment.

**Findings.** (1) The two-stage champion (**B4v2**: general-reasoning SFT → PlayWorld SFT) beats the direct-tuning control (**A2v2**) on **all five suites across all three seeds** (Δpass +10 to +22 pp; paired permutation $p \leq 0.0004$ per seed per suite; sign-consistent 15/15). (2) Preference optimization (DPO) on top of either route is **approximately null**, while the two-stage advantage fully survives it (B5 ≫ A2v2, 10/10 suite-metrics, $p \leq 0.0002$). (3) Verifier-rewarded online RL (**GRPO**) *regressed* the champion under two reward designs; diagnostics attribute the failure to **advantage starvation** (50–70% zero-variance reward groups) and **entropy collapse** ($0.15 \rightarrow 0.008$), producing required-component specialization on a narrow training scenario pool — reward gating recovered only the adversarial suite (net +15/300 episode flips), confirming the mechanism. (4) Off-the-shelf instruct baselines fail primarily at the **format gate** (0-shot: ~96% malformed JSON), while few-shot prompting recovers adversarial trap-avoidance (.907) but not construction (ID .227).

We release all run artifacts with SHA-pinned lineage (base-model revision, dataset fingerprints, parent-adapter hashes).

# Introduction

Post-training pipelines for task-specialized LLM agents face a practical design question: **spend budget on general capability first, or tune directly on the target task?** Staged recipes are common practice, but controlled, seed-replicated comparisons on tasks with *exact* programmatic verification are rare at single-GPU scale.

- **RQ1 (headline).** Does two-stage post-training (general reasoning SFT $\rightarrow$ task SFT) transfer better to ID/OOD/adversarial planning than direct task tuning of the same base model, at matched final-stage data?
- **RQ2.** Does verifier-mined preference optimization (DPO) add measurable gains on top of either route?
- **RQ3.** Does verifier-rewarded online RL (GRPO) improve the champion further?

All experiments run on a single GPU (Colab G4 runtime), Qwen3-8B-Base, LoRA adapters, with an artifact discipline (resolved configs, canonical dataset fingerprints, adapter SHA-256 lineage, frozen eval suites, mandatory eval-identity audits) designed so that every reported number is re-derivable from pinned inputs.

**Answers:** RQ1 — yes, decisively and seed-robustly. RQ2 — no ($\approx$ null). RQ3 — no; a diagnosed negative result with a mechanistic post-mortem (§6).

The study’s framing is deliberately **comparative**: it measures *recipe deltas under matched budgets*, not absolute task mastery (see §7).

# Environment and Evaluation

## PlayWorld

PlayWorld is a synthetic, rule-parameterized planning world. Each episode presents a world description (locations, resources, rules such as movement costs and preconditions) and a goal; the model must emit a **structured JSON plan** (`actions` + `final_state`).

A deterministic verifier chain replays the plan against the true rules:

schema/format gates $\rightarrow$ per-action legality $\rightarrow$ required-component satisfaction $\rightarrow$ goal achievement.

`passed` is strict; `score` $\in [0,1]$ grants graded partial credit; failures carry a reason-code taxonomy.

**Observability and interaction regime (v1.1 clarification).** The world description is supplied *in full* in the prompt, and the episode is scored from a *single* model output. PlayWorld v1 is therefore a fully observable, open-loop planning task. The model is never required to infer unobserved world content, nor to maintain state across interaction steps. This is a deliberate v1 design choice — it isolates the recipe comparison from state-estimation confounds — and it bounds the class of claims v1 can support (§7 item 7).

## Frozen Evaluation Suites

Five suites $\times$ 300 episodes, generated once and frozen (`freeze_fingerprint sha256:3cdcbc30...`, hard-gated at every evaluation).

Splits are made at the **scenario-family** level (ruleset $\times$ world template $\times$ goal type).

| **Suite**       | **Held-out axis**                                                 | **Measures**           |
| :-------------------- | :---------------------------------------------------------------------- | :--------------------------- |
| `eval_id`           | held-out instances of training families                                 | in-distribution learning     |
| `eval_template_ood` | unseen world templates, seen rules                                      | surface robustness           |
| `eval_comp_ood`     | unseen compositions of seen rule primitives                             | compositional generalization |
| `eval_rule_ood`     | $\geq 1$ unseen rule primitive                                        | rule extrapolation           |
| `eval_adversarial`  | hand-written traps (illegal-but-plausible actions, reward-hacking bait) | trap avoidance / alignment   |

The suites are **not difficulty-ordered on an absolute scale**: the four construction suites require building a complete multi-step plan (errors compound multiplicatively), while the adversarial suite is largely a *rejection* task.

Absolute pass rates are only comparable **within** a suite, across models.

All evaluations use greedy decoding, a fixed chat template, and a fixed empty-think opener; the decoding profile is part of the frozen protocol.

## Statistics

Per-suite paired episode-level comparisons: paired bootstrap 95% CIs and sign-flip permutation tests (10,000 resamples).

Headline comparisons were pre-registered with Holm–Bonferroni correction; the family was reduced by Amendment v1.4 (§7).

Gate **G6** required per-suite **sign consistency** of the champion-vs-control delta across 3 seeds.

# Experimental Arms and Data

Base model: **Qwen3-8B-Base** (revision `49e3418f...`), LoRA adapters (v2 adapter contract: `modules_to_save = [lm_head, embed_tokens]`).

**Phase-1 (P1) data.** A pre-registered weighted mixture of public math-reasoning corpora with **dataset-derived gold answers** (never LLM-generated):

GSM8K (`openai/gsm8k`, weight 0.6) and MATH-algebra (`EleutherAI/hendrycks_math`, weight 0.4);

target 8,000 records with prompt-level dedup, 500 held out as a frozen general-retention suite (used as a Stage-1 hard constraint).

The mixture is deliberately small and math-centric: P1 is a *transfer intervention*, not a frontier general model.

**Phase-2 data.** PlayWorld SFT: 2,000 oracle-derived episodes (canonical fingerprint `54fcb1d3...`); verifier-mined preference pairs (fingerprint `5856b5c5...`).

All artifacts frozen on HF and consumed via sha-verified fetches.

| **Arm**        | **Recipe**                                          | **Role**                                    |
| :------------------- | :-------------------------------------------------------- | :------------------------------------------------ |
| **C1**         | Qwen3-8B-Instruct, 0-shot                                 | off-the-shelf reference                           |
| **C2**         | Qwen3-8B-Instruct, 3-shot                                 | prompted reference                                |
| **A1v2**       | Base$\rightarrow$ PlayWorld SFT                         | direct route, stage 1                             |
| **A2v2**       | A1v2$\rightarrow$ PlayWorld DPO                         | **direct-route endpoint (Track-A control)** |
| **B1v2 (=P1)** | Base$\rightarrow$ general reasoning SFT (mixture above) | two-stage route, phase 1                          |
| **B4v2**       | P1$\rightarrow$ PlayWorld SFT                           | **two-stage endpoint & final champion**     |
| **B5**         | B4v2$\rightarrow$ PlayWorld DPO                         | RQ2 on champion                                   |
| **B6 / B6-R**  | B4v2$\rightarrow$ GRPO (aggregate / pass-gated reward)  | RQ3                                               |

The “v2” designation marks re-runs after the training-set freeze (the original PlayWorld SFT builder was non-deterministic in oracle-path tie-breaking; v2 re-trained affected arms on a single frozen artifact — amendment-logged).

A sensitivity check found A1v2 $\approx$ A1v1 on all suites (all n.s.), confirming the freeze itself did not move results.

B6-R is *not* an engineering re-run: it is a protocol-permitted reward-design revision (pass-gated reward) made after diagnosing B6’s aggregate-reward mismatch, with everything else held fixed.

# Headline Result (RQ1): Two-Stage Beats Direct Tuning, Robustly Across Seeds

## Three-Seed Final (Gate G6)

Final-stage training of B4v2 and A2v2 was re-run with seeds 43 and 44 on identical sha-pinned parents:

P1: `b1-general-sft-v2-s42-e6e83b`

A1v2: `a1v2-playworld-sft-s42-269d0e`

Parent adapter hashes were cross-checked against child lineage pins and identical frozen data were used.

Pass rate, mean $\pm$ sd over seeds $\{42,43,44\}$:

| **Suite**       | **B4v2 (two-stage)**  | **A2v2 (direct)** | **$\Delta$ per seed (42/43/44)** | **$p$ (perm., per seed)** |                |
| :-------------------- | :-------------------------- | :---------------------- | :--------------------------------------------------------------------------- | :------------- |
| `eval_id`           | **.393 $\pm$ .013** | .186$\pm$ .002        | +.207 / +.223 / +.193                                                        | $\leq .0001$ |
| `eval_template_ood` | **.349 $\pm$ .005** | .207$\pm$ .003        | +.140 / +.150 / +.137                                                        | $\leq .0001$ |
| `eval_comp_ood`     | **.329 $\pm$ .022** | .139$\pm$ .004        | +.160 / +.203 / +.207                                                        | $\leq .0001$ |
| `eval_rule_ood`     | **.302 $\pm$ .005** | .192$\pm$ .005        | +.100 / +.117 / +.113                                                        | $\leq .0003$ |
| `eval_adversarial`  | **.851 $\pm$ .002** | .651$\pm$ .011        | +.210 / +.203 / +.187                                                        | $\leq .0001$ |

**Verdict: G6 PASS (15/15 suite$\times$seed deltas positive; all significant).**

Mean-score deltas mirror pass-rate deltas (ID +.133–.147, adversarial +.185–.207, all $p = 0.0001$; paired-bootstrap CIs exclude zero everywhere).

## Failure Analysis

**Champion seed stability (direct paired tests, s43/s44 vs the s42 run of record).**

No suite regresses in either reseed: 18 of 20 suite-metric comparisons are n.s. ($|\Delta\mathrm{pass}| \leq .013$, $p \geq .52$).

The two exceptions are *small improvements* on comp-OOD (s43 +.037, perm. $p=.063$ / bootstrap CI excl. 0; s44 +.040, $p=.041$).

Combined with the mean $\pm$ sd table, the champion is stable and the headline gap ($\sim$10–22 pp) exceeds seed noise by roughly an order of magnitude.

Failure taxonomies show *why* the routes differ.

B4v2’s failures are almost purely semantic (`required_component_failed`; format-gate failures $\approx 0$; 0 truncated outputs),

while A2v2 retains a format-fragility tail (e.g. s43: 57 malformed-JSON failures on adversarial, 55 on rule-OOD) and pervasive output truncation ($\sim$1,490/1,500 episodes hit the token cap — the direct-route model appends degenerate continuations after its JSON).

Phase-1 general SFT appears to buy:

$a$ clean sequence termination and format discipline, and

$b$ higher legal-action rates on OOD suites ($\approx .69$–$.77$ vs $.66$–$.72$),

consistent with transferable procedural competence rather than surface memorization.

**The adapter-contract prerequisite.**

The termination/format axis was itself the subject of a mid-campaign engineering finding that conditions all results above:

with a standard LoRA target set (attention/MLP projections only), fine-tuned models systematically failed to emit the chat template’s terminal tokens and think-block delimiters — outputs ran to the token cap with degenerate continuations, indistinguishable at the loss level but catastrophic at the verifier’s format gate.

A bottom-up audit chain (x09 termination audit $\rightarrow$ x10 stop-logit probe $\rightarrow$ x13 trained-token NLL) localized the failure to special/control tokens whose embeddings and output logits are frozen under adapter-only training while the surrounding distribution shifts.

Adding `lm_head` and `embed_tokens` to `modules_to_save` (the “v2 adapter contract”, applied uniformly to *all* arms) resolved termination without touching task semantics.

A detailed write-up with the audit evidence is maintained separately in

`docs/experiments/v1/adapter_contract_termination.md`.

Here we note only that the A2v2 truncation tail above is a *residual* of the same axis — the direct route, even under the v2 contract, retains weaker termination discipline than the two-stage route, suggesting Phase-1 data volume also contributes to stabilizing sequence-boundary behavior.

## Reference Rows (s42, Off-the-Shelf)

| **Suite (pass)** | **C1: Instruct 0-shot** | **C2: Instruct 3-shot** | **A1v2 (task SFT only)** | **B4v2** |
| :--------------------- | :---------------------------- | :---------------------------- | :----------------------------- | :------------- |
| `eval_id`            | .000                          | .227                          | .167                           | **.393** |
| `eval_template_ood`  | .000                          | .200                          | .190                           | **.350** |
| `eval_comp_ood`      | .000                          | .137                          | .137                           | **.303** |
| `eval_rule_ood`      | .000                          | .137                          | .183                           | **.297** |
| `eval_adversarial`   | .323                          | **.907**                | .590                           | .853           |

C1 fails at the format gate (malformed JSON on $\sim$96% of construction episodes).

C2 fixes formatting well enough to reveal a real but weak planner — and is the **best adversarial trap-avoider** (.907), plausibly because instruct-tuned caution directly serves the rejection-style adversarial task.

This reinforces the evaluation caveat in the frozen-suites description: adversarial pass is not a general-competence proxy.

# RQ2: Preference Optimization Is $\approx$ Null on This Task

Per-suite pass rates at s42 and paired tests:

| **Suite (pass)** | **B4v2** | **B5 (B4v2$\rightarrow$DPO)** | **$\Delta$ vs B4v2 ($p$)** | **A2v2** | **$\Delta$ B5$-$A2v2 ($p$)** |                            |      |                         |
| :--------------------- | :------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------- | :--- | :---------------------- |
| `eval_id`            | .393           | .380                                                                                                                                     | $-.013$ (.59)            | .187 | **+.193 (.0001)** |
| `eval_temp_ood`      | .350           | .337                                                                                                                                     | $-.013$ (.64)            | .210 | **+.127 (.0001)** |
| `eval_comp_ood`      | .303           | .293                                                                                                                                     | $-.010$ (.71)            | .143 | **+.150 (.0001)** |
| `eval_rule_ood`      | .297           | .307                                                                                                                                     | +.010 (.73)                | .197 | **+.110 (.0002)** |
| `eval_adversar`      | .853           | .817                                                                                                                                     | **$-.037$ (.047)** | .643 | **+.173 (.0001)** |

- **On the champion:** no construction suite moves (all $|\Delta| \leq .013$, n.s.); the only significant effect is a small **adversarial pass decrease** ($-.037$, $p=.047$), the single isolated significant result in the DPO family.
- **On the direct route** (A1v2 $\rightarrow$ A2v2): no suite reaches significance ($|\Delta\mathrm{pass}| \leq .017$, all $p \geq .36$).
- **The pre-registered two-stage-vs-direct contrast fully survives DPO on both sides:** B5 $\gg$ A2v2 on 10/10 suite-metrics ($p \leq .0002$; mean-score deltas $+.123$ to $+.212$).

Verifier-mined pair quality was *higher* on the champion (mining yield .675 vs .447), so the null is not explained by pair scarcity.

Interpretation: with a strict programmatic verifier and an SFT stage that already fits the format and rule surface, offline preference deltas carry little additional trainable signal at this scale.

The random-pair mining control (E-RANDPAIR) was de-scoped accordingly (Amendment v1.4): a control for a $\sim$ null effect is uninformative.

# RQ3: Verifier-Rewarded GRPO — a Diagnosed Negative Result

## Outcome

Two GRPO variants on top of B4v2, sharing frozen scenario pools and decoding.

Per-suite pass rates at s42 (B6-R rates derived exactly from the episode-paired flip matrices of the audited run of record `8791aa`; see Appendix A):

| **Suite**       | **B4v2 pass / score** | **B6 pass ($\Delta$, $p$)** | **B6-R pass / score ($\Delta$pass)** |  |
| :-------------------- | :-------------------------- | :----------------------------------------------------------------------------------- | :- |
| `eval_id`           | .393 / .593                 | .253 (**$-.140$**, .0002)   | .233 / .530 ($-.160$)                        |  |
| `eval_template_ood` | .350 / .559                 | .210 (**$-.140$**, .0001)   | .213 / .518 ($-.137$)                        |  |
| `eval_comp_ood`     | .303 / .546                 | .217 (**$-.087$**, .024)    | .233 / .533 ($-.070$)                        |  |
| `eval_rule_ood`     | .297 / .513                 | .260 ($-.037$, .35)         | .240 / .513 ($-.057$)                              |  |
| `eval_adversarial`  | .853 / .951                 | .750 (**$-.103$**, .0001)   | .800 / .929 ($-.053$)                        |  |

B6-R pass rates from the audited `8791aa` summary agree exactly with the episode-paired flip-matrix reconstruction.

Notably, B6-R’s *mean scores* sit much closer to B4v2 than its pass rates (rule-OOD score is identical at .513; legal-action rates are *higher* than B4v2 on OOD suites, .77–.99):

the GRPO policy remains a competent partial planner but loses precisely the required-component completion that strict passing demands — the score/pass wedge that the pass-gated reward was built to close at training time, visible here surviving into evaluation.

B6 is also significantly below B5 on ID/template/adversarial (e.g. ID $-.127$, $p=.0008$).

The GRPO arm was **closed with champion unchanged** (`docs/experiments/v1/b6_grpo_closure.md`).

## Mechanism

1. **Aggregate-reward mismatch (B6):** 40–50% of failing adversarial episodes scored $>0.8$ — the policy learned to *almost pass*, harvesting partial credit.

   B6-R’s pass-gated reward (passed $\rightarrow 0.5 + 0.5\cdot\text{score}$; failed $\rightarrow 0.1\cdot\text{score}$) removed this incentive.
2. **Episode-level confirmation (x19 flip analysis, B6-R vs B6):** the entire recovery is **adversarial-local** — net +15/300 flips there (fail$\rightarrow$pass 25, pass$\rightarrow$fail 10), every other suite within $\pm 6$ net flips (noise level).

   100% of pass$\rightarrow$fail regressions across all suites carry `required_component_failed`.

   A residual near-miss cluster persists under B6-R (6/10 adversarial pass$\rightarrow$fail episodes score in $[0.8,1.0)$) but is reward-gated during training.
3. **The binding failure is data/regularization, not reward shape:** 50–70% of prompt groups had zero reward variance (no gradient — *advantage starvation*), and policy entropy collapsed $0.15 \rightarrow 0.008$.

   The policy specialized to the narrow training pool’s required-component patterns and lost ID/OOD coverage.

   Reward redesign (B6-R) was precisely the controlled test of the alternative hypothesis — and it did not move the construction suites.

## Implication

Reward-function or verifier refinement alone is unlikely to rescue online RL here.

The binding constraints are scenario-pool diversity/curriculum and regularization (KL-to-parent, entropy bonus, SFT replay), all explicitly out of protocol-v1 scope.

Advantage-estimator swaps (e.g. RLOO) share the group-relative structure exposed to the same starvation mechanism; E-RLOO was de-scoped on these grounds (Amendment v1.4) rather than run as a low-prior ablation.

**(v1.2 note.)** These prerequisites are carried forward into the next research subtrack, **v2-CLB**, where advantage-starvation fraction and policy entropy are specified as pre-registered abort conditions rather than post-hoc diagnostics. The v1 post-mortem therefore informs a stopping rule before the corresponding RL arm is run; the v2-CLB protocol remains pending freeze at this report revision.

# Threats to Validity and Disclosures

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

   The v1 SFT builder broke oracle-path ties non-deterministically; discovered mid-campaign, fixed by freezing a single artifact and re-running affected arms as “v2” (A1v2 $\approx$ A1v1: all n.s.).
6. **Adversarial suite semantics.**

   Rejection-style scoring makes absolute adversarial pass rates incomparable to construction suites (see C2’s .907).
7. **Open-loop, fully observable design — a bound on the kind of claim this report can make (added in v1.1).**

   Every episode supplies the complete world state in the prompt and is scored from a single model output. The model therefore never infers unobserved content and never maintains state across steps.

   Consequently, this report’s results support claims about *which post-training recipe transfers better on a rule-grounded planning task*, and *not* claims about whether any arm has acquired a *transition model* of the world: a policy that maps a fully specified state to a correct plan is observationally indistinguishable, under this design, from one that predicts how the world changes under intervention.

   Distinguishing the two requires (i) partial observability, so that state must be inferred and carried, and (ii) a closed loop against a simulator, so that a counterfactual action can actually be executed and compared against the model’s prediction.

   Neither is present in protocol v1. Both are the subject of the successor protocol (§8). We state this explicitly because the phrase “verifiable world” can otherwise be over-read: v1’s verifier is exact, but what it verifies is a *plan*, not a *prediction*.

# Conclusions and Future Work

**Conclusions.** On a verifiable planning task with frozen evaluation:

$i$ a general-reasoning SFT phase before task SFT yields large, seed-robust ID *and* OOD gains over direct tuning at matched task data — the single strongest and most stable effect in the study;

$ii$ offline DPO adds $\approx$ nothing on either route;

$iii$ verifier-rewarded GRPO actively hurts without data-side diversity and entropy/KL regularization, and reward-shape fixes relocate, but do not remove, the failure.

## Future work (roadmap-aligned in v1.2)

The roadmap is now organized as a broader **v2 – Closed-Loop World Interaction** research phase followed by **v3 – Executable World Models**. This revision records the current research program without treating future candidate interventions as completed experiments.

| **Track**  | **Status at v1.2**                       | **Question**                                                                                                                                                       |
| :--------------- | :--------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v2**     | Current research phase                         | Closed-loop world interaction and capability scaling under separately controlled observation, model-capability, and world-complexity axes.                               |
| **v2-CLB** | Next; pre-registration written, freeze pending | Whether explicit belief-state maintenance improves closed-loop competence, counterfactual transition prediction, and calibrated uncertainty under partial observability. |
| **v2-AP**  | Planned; not protocolized                      | How far general-purpose agent capability and absolute competence can be improved while keeping the world specification fixed.                                            |
| **v2-WE**  | Planned; not protocolized                      | How capability changes as horizon, state-space complexity, and interaction structure increase.                                                                           |
| **v3**     | Exploratory                                    | Whether world dynamics can be represented, authored, and revised as executable structure while remaining exactly checkable.                                              |

#### Why v2-CLB comes first.

The v1 environment is fully observable and open-loop: the complete world state is supplied in the prompt and the model emits a single plan. Thus v1 establishes recipe transfer on a verifiable planning task, but it cannot separately identify a learned transition model from a strong prompt-to-plan mapping.

Partial observability introduces a new measurement axis: the model must maintain information about latent or previously observed state. Closed-loop interaction then makes checkpoint/restore and counterfactual transition measurement possible. The closed-loop simulator and interaction contract are also reusable infrastructure for the later v2 capability and world complexity studies.

#### Protocol v2.0-CLB — Closed-Loop Belief.

v2-CLB is the first successor protocol. PlayWorld-PO is intended to inherit v1’s rule engine, action space, legality semantics, and deterministic simulator wherever possible, while restricting the observation available to the agent. At each step the model emits $\{`belief`,`action`,`confidence`\}$; the simulator executes the action, returns the next observation, and retains the complete underlying state.

The central questions are:

- Does explicit belief-state emission improve closed-loop task success?
- Does it improve counterfactual transition prediction?
- Does the v1 two-stage transfer advantage survive partial observability?
- Is uncertainty over unobserved state calibrated, and does calibration track competence?
- Can verifier-grounded online RL improve the closed-loop policy after the failure mechanisms diagnosed in v1 are explicitly addressed?

The primary mechanism-oriented metric is **counterfactual rollout error**: at frozen probe points, the model predicts the state resulting from an action it did *not* take; the simulator executes that counterfactual from the identical checkpointed state; and the prediction is compared with the resulting state. Task success measures whether the agent achieved the objective, while counterfactual error tests agreement with the actual transition.

The protocol should include a discriminating occlusion-OOD suite with a smaller observation radius than training, replication across the v1 two-stage champion and direct-control initialization, and pre-registered guardrails for online RL. In particular, the v1 observations of advantage starvation and entropy collapse should become *pre-registered abort conditions* rather than post-hoc diagnostics.

**Phase 1 is intentionally frozen in v2-CLB.** This is a methodological control, not a claim that the v1 Phase-1 recipe is optimal. The v1 Phase-1 dataset is deliberately small and math-centric (GSM8K 0.6 + MATH-algebra 0.4, 8,000 target records with a 500-record retention holdout), and v1 identifies the resulting two-stage initialization as the strongest transfer lever. Nevertheless, changing Phase 1 at the same time as changing the observation/interaction regime would confound the primary v2-CLB question. The upstream Phase-1 checkpoint is therefore frozen to the v1 `B1v2/P1` run of record, while the canonical two-stage initialization entering v2-CLB is the frozen v1 champion endpoint **B4v2**, which already includes the subsequent PlayWorld SFT stage. The matched direct-route initialization is **A2v2**.

#### Multi-GPU and compute scaling in v2-CLB.

The successor program does *not* increase base-model parameter count solely to force a distributed configuration. At the 8B + LoRA scale, ordinary Phase-1 SFT and offline behavior-cloning stages may remain single-GPU.

The canonical *online-RL* stage, however, is designed as a genuinely distributed workload. Multi-GPU execution is therefore a required part of the canonical v2-CLB online-RL path, not merely an optional portfolio demonstration. The reason is workload scaling rather than model-size artificiality:

- many concurrent trajectories;
- longer interaction horizons;
- larger rollout pools;
- simulator concurrency;
- repeated verifier evaluation;
- counterfactual probes;
- repeated policy updates and replay.

The intended relationship is

$$
\text{research workload scaling}
\rightarrow
\text{rollout bottleneck}
\rightarrow
\text{distributed execution}.
$$

Single-GPU execution remains an explicit control where feasible, so that distributed execution is compared against a known baseline rather than being treated as an unexplained implementation detail. The scientific reason for the multi-GPU stage is the scale of closed-loop rollout/RL workload; it is not to claim that 8B + LoRA itself requires multi-GPU model parallelism.

A systems report may characterize the selected distributed implementation (FSDP, DeepSpeed ZeRO-3, or an equivalent stack) using throughput, effective memory scaling, communication share, checkpoint save/resume behavior, and cost per useful training token or rollout. Such measurements are secondary systems results and must not be presented as evidence for the belief-state efficacy hypothesis.

#### Planned — v2-AP (Agent Capability Scaling).

v2-AP is the dedicated capability-scaling axis and incorporates the former absolute-performance direction. It asks:

> Given a fixed verifiable world specification, how far can the general capability of the agent and its absolute closed-loop competence be pushed?

This is where the v1 Phase-1 intervention can be deliberately upgraded. The purpose is not merely to obtain a higher PlayWorld score, but to test whether stronger general-purpose reasoning and agent-workflow capability transfers to the world-interaction setting established by v2-CLB.

Candidate Phase-1 interventions include:

- harder mathematical reasoning and more demanding problem-solving data;
- scientific reasoning and verified scientific problem solving;
- coding and sandbox-verified code generation;
- structured JSON generation and output-contract robustness;
- function/tool calling;
- API, retrieval, and search workflows;
- longer-context training data and suitable long-context methods;
- improved instruction following and multi-step agent workflows;
- curriculum and data-mixture optimization;
- verifier-grounded online RL where the task and reward are sufficiently informative;
- reward/objective design, KL regularization, entropy control, and SFT replay.

These are *candidate interventions*, not v1 results and not a frozen v2-AP protocol.

The key methodological requirement is to distinguish

$$
\text{general capability improvement}
$$

from

$$
\text{PlayWorld-specific training improvement}.
$$

A future v2-AP protocol should therefore freeze its Phase-1 dataset and mixture, general-retention suites, capability metrics, task-transfer procedure, and intervention family before canonical experiments. It should also preserve a fixed-world anchor against the published v1 checkpoint so that absolute improvement remains interpretable.

Importantly, v2-AP should not silently become a moving target in which different experiments change Phase 1, Phase 2, the world, and the evaluator simultaneously. Capability interventions must be frozen within each comparison family, with parent checkpoints and data lineage explicitly recorded.

#### Planned — v2-WE (World Extension).

v2-WE addresses the environmental scaling axis. Candidate extensions include longer interaction horizons, larger state spaces, richer spatial structure, additional persistent state, additional action types, NPC/environment interactions, richer event/state transitions, and more demanding task dependencies.

Each major extension should be introduced as a separately frozen evaluation axis rather than modifying the original benchmark in place. The original v1 specification and relevant v2-CLB/v2-AP checkpoints should remain available as baselines.

The closed-loop contract established by v2-CLB is intended to serve as shared infrastructure for these extensions. New world complexity should extend the interaction and verification contract rather than silently replace the underlying evaluation semantics.

#### Attribution across v2 subtracks.

The three v2 subtracks may share infrastructure and implementation components, but their headline claims remain separately attributable.

v2-CLB must distinguish belief/transition-model effects from changes caused by starting from a stronger checkpoint.

v2-AP must not be used to claim improved world modeling merely because the score increases on a fixed world.

v2-WE must not be used to claim better optimization merely because the model succeeds on a harder environment.

This separation is necessary to keep the v2 phase interpretable as a set of controlled scientific axes rather than a single undifferentiated capability-improvement program.

#### Cross-track inheritance and control.

Shared infrastructure may be reused across v2 subtracks, but checkpoint or dataset reuse must not silently redefine the scientific baseline.

For **v2-CLB → v2-AP**, the frozen v1/v2-CLB initialization remains a historical control. If v2-AP introduces a stronger Phase-1 checkpoint, its downstream transfer comparison should include a matched run using the previous frozen parent under the same downstream training and evaluation conditions. A v2-AP checkpoint becomes a new capability baseline only after its own protocol is frozen and completed with lineage recorded.

For **v2-AP → v2-WE**, agent capability should be held fixed for the primary world-complexity comparison. If both a historical v2-CLB checkpoint and a later v2-AP checkpoint are evaluated, they should be reported as separate fixed-agent strata rather than pooled into a single world-complexity effect.

Frozen evaluation data are never promoted into training data. Training datasets may be reused only when their exact fingerprints and scientific roles remain unchanged; an enriched, regenerated, or reweighted dataset is a new intervention and requires an appropriate matched control when used in a causal comparison.

#### Exploratory — v3 (Worlds as Code).

v3 is reserved for a qualitatively different representation-level question: can the model emit, inspect, revise, and execute representations of world dynamics as code or other executable structure, with rollouts evaluated through exact agreement against deterministic ground truth?

The earlier exploratory direction of programmatic geometry, including SVG, can serve as one possible intermediate representation rather than defining the whole phase. The endpoint is **world-models-as-code**, where the model can author and revise executable dynamics whose behavior can be rolled out by a deterministic executor and compared against ground truth.

The closed-loop simulator established through v2-CLB is a prerequisite for this direction. Executable world representations become scientifically meaningful only when their predicted dynamics can be executed and compared against an actual world.

#### Research-program invariants.

Across the successor tracks, the following methodological principles remain fixed unless a future protocol explicitly amends them:

- deterministic simulators/verifiers remain authoritative wherever the quantity of interest is exactly computable;
- learned judges do not participate in headline reward or scoring paths;
- evaluation suites are frozen and fingerprint-pinned before canonical training;
- model and dataset lineage are recorded by stable hashes;
- failed and aborted runs are retained as research artifacts rather than silently replaced;
- changes in model capability, world complexity, and observation regime are not conflated within a single causal claim;
- future protocol amendments must state which previously frozen variable has changed and why;
- no single deterministic toy-world result is treated as evidence of general world-model or AGI capability.

# References

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

# Appendix

## Runs of Record

<table>
<thead>
<tr class="header">
<th style="text-align: left;"><strong>Artifact</strong></th>
<th style="text-align: left;"><strong>ID / value</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">Champion training run</td>
<td style="text-align: left;"><code>20260814-023603-b4v2-playworld-sft-from-p1-s42-c56ed2</code> (repo <code>m97j/aw-runs-b4</code>) ⋅ adapter <code>sha256:d4fcacdd...</code></td>
</tr>
<tr class="even">
<td style="text-align: left;">Champion eval (s42)</td>
<td style="text-align: left;"><code>20260814-032546-eval-playworld-s42-7308ee</code></td>
</tr>
<tr class="odd">
<td style="text-align: left;">Control training run</td>
<td style="text-align: left;"><code>20260814-114257-a2v2-playworld-dpo-s42-ef1e20</code> (repo <code>m97j/aw-runs-a2</code>) ⋅ adapter <code>sha256:cc23345f...</code></td>
</tr>
<tr class="even">
<td style="text-align: left;">Control eval (s42)</td>
<td style="text-align: left;"><code>20260814-120733-eval-playworld-s42-61d7cd</code></td>
</tr>
<tr class="odd">
<td style="text-align: left;">Parents (pinned)</td>
<td style="text-align: left;">P1 <code>20260807-225109-b1-general-sft-v2-s42-e6e83b</code> (<code>sha256:747e5757...</code>) ⋅ A1v2 <code>20260814-022256-a1v2-playworld-sft-s42-269d0e</code> (<code>sha256:70c2ecb9...</code>)</td>
</tr>
<tr class="even">
<td style="text-align: left;">Seed runs (repo <code>m97j/aw-runs-seeds</code>)</td>
<td style="text-align: left;">b4v2-s43 <code>20260817-151254-...-c04b9c</code> ⋅ a2v2-s43, b4v2-s44, a2v2-s44 (run cards in repo)</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Seed evals</td>
<td style="text-align: left;"><p>s43: <code>20260818-100258-...-304567</code> (b4v2), <code>20260818-101023-...-44a500</code> (a2v2) ⋅</p>
<p>s44: <code>20260818-105059-...-082b2a</code> (b4v2), <code>20260818-105846-...-40b6f2</code> (a2v2)</p></td>
</tr>
<tr class="even">
<td style="text-align: left;">Seed analyses</td>
<td style="text-align: left;"><p><code>analysis_b4v2_vs_a2v2_s43/s44.json</code>, <code>analysis_b4v2_s43/s44_vs_s42.json</code> (under the b4v2 seed-eval runs) ⋅</p>
<p><code>runs/seed_variance_report.json</code> (x21; G6 verdict PASS)</p></td>
</tr>
<tr class="odd">
<td style="text-align: left;">B5 run / eval</td>
<td style="text-align: left;"><code>20260814-115239-b5-playworld-dpo-s42-4c45da</code> / <code>20260814-124224-eval-playworld-s42-f77cd8</code> (repo <code>m97j/aw-runs-b5</code>)</td>
</tr>
<tr class="even">
<td style="text-align: left;">B6 run / eval</td>
<td style="text-align: left;"><code>20260815-150717-b6-playworld-grpo-s42-6e6c45</code> / <code>20260816-004824-...-274abd</code> (repo <code>m97j/aw-runs-b6</code>)</td>
</tr>
<tr class="odd">
<td style="text-align: left;">B6-R run / eval</td>
<td style="text-align: left;"><p><code>...-b6r-playworld-grpo-gated-s42-de963e</code> (rerun of record) / <code>20260816-201107-...-8791aa</code> (x20-audited) ⋅</p>
<p>invalid eval <code>f1854f</code> excluded (§7) ⋅</p>
<p>flip artifact <code>x19_b6r_vs_b6_flip.json</code></p></td>
</tr>
<tr class="even">
<td style="text-align: left;">C1 / C2 evals</td>
<td style="text-align: left;"><code>20260802-053440-...-e6cb05</code> ⋅ <code>20260802-060802-...-67cf20</code></td>
</tr>
<tr class="odd">
<td style="text-align: left;">Frozen suites</td>
<td style="text-align: left;"><code>freeze_fingerprint sha256:3cdcbc30c99e492c...</code>, 5 × 300 episodes</td>
</tr>
<tr class="even">
<td style="text-align: left;">Data fingerprints</td>
<td style="text-align: left;"><p>P1 mixture: GSM8K 0.6 + MATH-algebra 0.4, 8k target / 500 holdout (manifest in <code>data/p1</code>) ⋅</p>
<p>PlayWorld SFT <code>sha256:54fcb1d3...</code> (2,000 records) ⋅</p>
<p>preference <code>sha256:5856b5c5...</code> (canonical <code>fingerprint_payload</code>)</p></td>
</tr>
</tbody>
</table>

## Diagnostic Tooling (x-scripts)

x09 termination audit $\cdot$ x10 stop-logit probe $\cdot$ x11 adapter integrity $\cdot$ x12 attested probe $\cdot$ x13 trained-ids NLL $\cdot$ x15 SFT data diff $\cdot$ x16 SFT provenance resolver $\cdot$ x17 GRPO scenario audit $\cdot$ x19 regression/flip diagnostics $\cdot$ x20 eval identity audit (stale-weights guard) $\cdot$ x21 seed-variance aggregation.

Each incident that motivated a tool is preserved chronologically in the campaign notebooks (aw_01–aw_11).

## Document Revision History

| **Version** | **Date** | **Change**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| :---------------- | :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1.0              | 2026-08-23     | Initial publication under protocol v1.4. Zenodo version DOI 10.5281/zenodo.22052149 (concept DOI 10.5281/zenodo.22052148).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| v1.1              | 2026-08-27     | Documentary revision only; no result changed. (a) §7 item 7 added: the open-loop, fully observable design bounds the class of claims the report can support — an exact verifier over*plans* is not an exact verifier over *predictions*. (b) §8 Future Work re-ordered: the partial-observability stage is pulled ahead of v2a/v2b and frozen as protocol v3.0-CLB, with the rationale recorded in-text; SVG / programmatic geometry re-positioned as a later sub-track with an explicit justification for the swap. (c) §6 implication note: the v1 GRPO post-mortem is converted into pre-registered abort conditions in the successor protocol. (d) Two systems references added (FSDP, ZeRO) for the distributed characterization accompanying the successor track. (e) Documentary path updates following the repository restructure at tag `v1.0.1` (protocol-version layering); no artifact, hash, or number is affected, and the pre-restructure paths cited by v1.0 remain resolvable as stubs. |
| v1.2              | 2026-08-30     | Pre-publication documentary revision only; no v1 result, table, numerical statistic, run identifier, or conclusion is changed. The complete v1 experimental record and appendices are retained. The forward roadmap is synchronized to the pre-freeze v2 phase: v2.0-CLB is the next protocol, v2-AP explicitly owns future Phase-1/general-capability expansion, v2-WE owns world-complexity scaling, and v3 remains the executable-world-model direction. Multi-GPU is positioned as a required component of the canonical v2-CLB online-RL workload because of rollout/interaction scaling, while single-GPU remains the controlled baseline where feasible. v2.0-CLB itself remains pre-freeze at this report revision.                                                                                                                                                                                                                                                                                        |

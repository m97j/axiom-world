# Axiom-World: A Pre-Registered Study of Two-Stage Post-Training for Rule-Grounded Planning in a Verifiable Toy World

**Technical Report — Protocol v1 (v1.0-rc, 2026-07-23)**
Author: m97j · Code & artifacts: <https://github.com/m97j/axiom-world> (public at release; results reproducible at tag `v1.0.0`) · Protocol: `docs/experimental-protocol.md` v1.4 (pre-registered, amendment-logged)

---

## Abstract

We study whether a **two-stage post-training recipe** — general reasoning SFT (Phase 1) followed by task-specific tuning (Phase 2) — outperforms **direct task tuning** for rule-grounded planning, using a fully verifiable synthetic environment (PlayWorld) with frozen, fingerprint-pinned evaluation suites spanning in-distribution, three OOD axes, and adversarial traps. All arms, comparisons, champion-selection rules, and stop rules were pre-registered before the first run; every deviation is logged as a numbered amendment.

**Findings.** (1) The two-stage champion (**B4v2**: general-reasoning SFT → PlayWorld SFT) beats the direct-tuning control (**A2v2**) on **all five suites across all three seeds** (Δpass +10 to +22 pp; paired permutation p ≤ 0.0004 per seed per suite; sign-consistent 15/15). (2) Preference optimization (DPO) on top of either route is **approximately null**, while the two-stage advantage fully survives it (B5 ≫ A2v2, 10/10 suite-metrics, p ≤ 0.0002). (3) Verifier-rewarded online RL (**GRPO**) *regressed* the champion under two reward designs; diagnostics attribute the failure to **advantage starvation** (50–70 % zero-variance reward groups) and **entropy collapse** (0.15 → 0.008) producing required-component specialization on a narrow training scenario pool — reward gating recovered only the adversarial suite (net +15/300 episode flips), confirming the mechanism. (4) Off-the-shelf instruct baselines fail primarily at the **format gate** (0-shot: ~96 % malformed JSON), while few-shot prompting recovers adversarial trap-avoidance (.907) but not construction (ID .227). We release all run artifacts with sha-pinned lineage (base-model revision, dataset fingerprints, parent-adapter hashes).

---

## 1. Introduction

Post-training pipelines for task-specialized LLM agents face a practical design question: **spend budget on general capability first, or tune directly on the target task?** Staged recipes are common practice, but controlled, seed-replicated comparisons on tasks with *exact* programmatic verification are rare at single-GPU scale.

- **RQ1 (headline).** Does two-stage post-training (general reasoning SFT → task SFT) transfer better to ID/OOD/adversarial planning than direct task tuning of the same base model, at matched final-stage data?
- **RQ2.** Does verifier-mined preference optimization (DPO) add measurable gains on top of either route?
- **RQ3.** Does verifier-rewarded online RL (GRPO) improve the champion further?

All experiments run on a single Blackwell GPU (Colab G4), Qwen3-8B-Base, LoRA adapters, with an artifact discipline (resolved configs, canonical dataset fingerprints, adapter sha256 lineage, frozen eval suites, mandatory eval-identity audits) designed so that every reported number is re-derivable from pinned inputs.

**Answers:** RQ1 — yes, decisively and seed-robustly. RQ2 — no (≈ null). RQ3 — no; a diagnosed negative result with a mechanistic post-mortem (§6). The study's framing is deliberately **comparative**: it measures *recipe deltas under matched budgets*, not absolute task mastery (see §7.1).

---

## 2. Environment and evaluation

### 2.1 PlayWorld

PlayWorld is a synthetic, rule-parameterized planning world. Each episode presents a world description (locations, resources, rules such as movement costs and preconditions) and a goal; the model must emit a **structured JSON plan** (`actions` + `final_state`). A deterministic verifier chain replays the plan against the true rules: schema/format gates → per-action legality → required-component satisfaction → goal achievement. `passed` is strict; `score ∈ [0,1]` grants graded partial credit; failures carry a reason-code taxonomy.

### 2.2 Frozen evaluation suites

Five suites × 300 episodes, generated once and frozen (`freeze_fingerprint sha256:3cdcbc30…`, hard-gated at every evaluation). Splits are made at the **scenario-family** level (ruleset × world template × goal type):

| Suite | Held-out axis | Measures |
|---|---|---|
| `eval_id` | held-out instances of training families | in-distribution learning |
| `eval_template_ood` | unseen world templates, seen rules | surface robustness |
| `eval_comp_ood` | unseen compositions of seen rule primitives | compositional generalization |
| `eval_rule_ood` | ≥ 1 unseen rule primitive | rule extrapolation |
| `eval_adversarial` | hand-written traps (illegal-but-plausible actions, reward-hacking bait) | trap avoidance / alignment |

The suites are **not difficulty-ordered on an absolute scale**: the four construction suites require building a complete multi-step plan (errors compound multiplicatively), while the adversarial suite is largely a *rejection* task; absolute pass rates are only comparable **within** a suite, across models.

All evaluations use greedy decoding, a fixed chat template, and a fixed empty-think opener; the decoding profile is part of the frozen protocol.

### 2.3 Statistics

Per-suite paired episode-level comparisons: paired bootstrap 95 % CIs and sign-flip permutation tests (10,000 resamples). Headline comparisons were pre-registered with Holm–Bonferroni correction; the family was reduced by Amendment v1.4 (§7.3). Gate **G6** required per-suite **sign consistency** of the champion-vs-control delta across 3 seeds.

---

## 3. Experimental arms and data

Base model: **Qwen3-8B-Base** (revision `49e3418f…`), LoRA adapters (v2 adapter contract: `modules_to_save = [lm_head, embed_tokens]`).

**Phase-1 (P1) data.** A pre-registered weighted mixture of public math-reasoning corpora with **dataset-derived gold answers** (never LLM-generated): GSM8K (`openai/gsm8k`, weight 0.6) and MATH-algebra (`EleutherAI/hendrycks_math`, weight 0.4); target 8,000 records with prompt-level dedup, 500 held out as a frozen general-retention suite (used as a Stage-1 hard constraint). The mixture is deliberately small and math-centric: P1 is a *transfer intervention*, not a frontier general model.

**Phase-2 data.** PlayWorld SFT: 2,000 oracle-derived episodes (canonical fingerprint `54fcb1d3…`); verifier-mined preference pairs (fingerprint `5856b5c5…`). All artifacts frozen on HF and consumed via sha-verified fetches.

| Arm | Recipe | Role |
|---|---|---|
| **C1** | Qwen3-8B-Instruct, 0-shot | off-the-shelf reference |
| **C2** | Qwen3-8B-Instruct, 3-shot | prompted reference |
| **A1v2** | Base → PlayWorld SFT | direct route, stage 1 |
| **A2v2** | A1v2 → PlayWorld DPO | **direct-route endpoint (Track-A control)** |
| **B1v2 (=P1)** | Base → general reasoning SFT (mixture above) | two-stage route, phase 1 |
| **B4v2** | P1 → PlayWorld SFT | **two-stage endpoint & final champion** |
| **B5** | B4v2 → PlayWorld DPO | RQ2 on champion |
| **B6 / B6-R** | B4v2 → GRPO (aggregate / pass-gated reward) | RQ3 |

The "v2" designation marks re-runs after the training-set freeze (the original PlayWorld SFT builder was non-deterministic in oracle-path tie-breaking; v2 re-trained affected arms on a single frozen artifact — amendment-logged). A sensitivity check found A1v2 ≈ A1v1 on all suites (all n.s.), confirming the freeze itself did not move results. B6-R is *not* an engineering re-run: it is a protocol-permitted reward-design revision (pass-gated reward) made after diagnosing B6's aggregate-reward mismatch, with everything else held fixed.

---

## 4. Headline result (RQ1): two-stage beats direct tuning, robustly across seeds

### 4.1 Three-seed final (Gate G6)

Final-stage training of B4v2 and A2v2 was re-run with seeds 43 and 44 on identical sha-pinned parents (P1 `b1-general-sft-v2--s42--e6e83b`, A1v2 `a1v2-playworld-sft--s42--269d0e`; parent adapter hashes cross-checked against child lineage pins) and identical frozen data. Pass rate, mean ± sd over seeds {42, 43, 44}:

| Suite | **B4v2 (two-stage)** | **A2v2 (direct)** | Δ per seed (42/43/44) | p (perm., per seed) |
|---|---|---|---|---|
| eval_id | **.393 ± .013** | .186 ± .002 | +.207 / +.223 / +.193 | ≤ .0001 |
| eval_template_ood | **.349 ± .005** | .207 ± .003 | +.140 / +.150 / +.137 | ≤ .0001 |
| eval_comp_ood | **.329 ± .022** | .139 ± .004 | +.160 / +.203 / +.207 | ≤ .0001 |
| eval_rule_ood | **.302 ± .005** | .192 ± .005 | +.100 / +.117 / +.113 | ≤ .0003 |
| eval_adversarial | **.851 ± .002** | .651 ± .011 | +.210 / +.203 / +.187 | ≤ .0001 |

**Verdict: G6 PASS (15/15 suite×seed deltas positive; all significant).** Mean-score deltas mirror pass-rate deltas (ID +.133–.147, adversarial +.185–.207, all p = 0.0001; paired-bootstrap CIs exclude zero everywhere).

**Champion seed stability (direct paired tests, s43/s44 vs the s42 run of record).** No suite regresses in either reseed: 18 of 20 suite-metric comparisons are n.s. (|Δpass| ≤ .013, p ≥ .52); the two exceptions are *small improvements* on comp-OOD (s43 +.037, perm. p = .063 / bootstrap CI excl. 0; s44 +.040, p = .041). Combined with the mean ± sd table, the champion is stable and the headline gap (~10–22 pp) exceeds seed noise by roughly an order of magnitude.

### 4.2 Mechanism hints

Failure taxonomies show *why* the routes differ. B4v2's failures are almost purely semantic (`required_component_failed`; format-gate failures ≈ 0; 0 truncated outputs), while A2v2 retains a format-fragility tail (e.g. s43: 57 malformed-JSON failures on adversarial, 55 on rule-OOD) and pervasive output truncation (~1,490/1,500 episodes hit the token cap — the direct-route model appends degenerate continuations after its JSON). Phase-1 general SFT appears to buy (a) clean sequence termination and format discipline, and (b) higher legal-action rates on OOD suites (≈ .69–.77 vs .66–.72), consistent with transferable procedural competence rather than surface memorization.

### 4.3 Reference rows (s42, off-the-shelf)

| Suite (pass) | C1: Instruct 0-shot | C2: Instruct 3-shot | A1v2 (task SFT only) | **B4v2** |
|---|---|---|---|---|
| eval_id | .000 | .227 | .167 | **.393** |
| eval_template_ood | .000 | .200 | .190 | **.350** |
| eval_comp_ood | .000 | .137 | .137 | **.303** |
| eval_rule_ood | .000 | .137 | .183 | **.297** |
| eval_adversarial | .323 | **.907** | .590 | .853 |

C1 fails at the format gate (malformed JSON on ~96 % of construction episodes). C2 fixes formatting well enough to reveal a real but weak planner — and is the **best adversarial trap-avoider** (.907), plausibly because instruct-tuned caution directly serves the rejection-style adversarial task. This reinforces §2.2: adversarial pass is not a general-competence proxy.

---

## 5. RQ2: preference optimization is ≈ null on this task

Per-suite pass rates at s42 and paired tests:

| Suite (pass) | B4v2 | **B5 (B4v2→DPO)** | Δ vs B4v2 (p) | A2v2 | Δ B5−A2v2 (p) |
|---|---|---|---|---|---|
| eval_id | .393 | .380 | −.013 (.59) | .187 | **+.193 (.0001)** |
| eval_template_ood | .350 | .337 | −.013 (.64) | .210 | **+.127 (.0001)** |
| eval_comp_ood | .303 | .293 | −.010 (.71) | .143 | **+.150 (.0001)** |
| eval_rule_ood | .297 | .307 | +.010 (.73) | .197 | **+.110 (.0002)** |
| eval_adversarial | .853 | .817 | **−.037 (.047)** | .643 | **+.173 (.0001)** |

- **On the champion**: no construction suite moves (all |Δ| ≤ .013, n.s.); the only significant effect is a small **adversarial pass decrease** (−.037, p = .047), the single isolated significant result in the DPO family.
- **On the direct route** (A1v2 → A2v2): no suite reaches significance (|Δpass| ≤ .017, all p ≥ .36).
- **The pre-registered two-stage-vs-direct contrast fully survives DPO on both sides**: B5 ≫ A2v2 on 10/10 suite-metrics (p ≤ .0002; mean-score deltas +.123 to +.212).

Verifier-mined pair quality was *higher* on the champion (mining yield .675 vs .447), so the null is not explained by pair scarcity. Interpretation: with a strict programmatic verifier and an SFT stage that already fits the format and rule surface, offline preference deltas carry little additional trainable signal at this scale. The random-pair mining control (E-RANDPAIR) was de-scoped accordingly (Amendment v1.4): a control for a ~null effect is uninformative.

---

## 6. RQ3: verifier-rewarded GRPO — a diagnosed negative result

### 6.1 Outcome

Two GRPO variants on top of B4v2, sharing frozen scenario pools and decoding. Per-suite pass rates at s42 (B6-R rates derived exactly from the episode-paired flip matrices of the audited run of record `8791aa`; see §7.4):

| Suite (pass) | B4v2 | **B6 (aggregate reward)** | Δ vs B4v2 (p) | **B6-R (pass-gated)** | Δ vs B4v2 |
|---|---|---|---|---|---|
| eval_id | .393 | .253 | **−.140 (.0002)** | .233 | −.160 |
| eval_template_ood | .350 | .210 | **−.140 (.0001)** | .213 | −.137 |
| eval_comp_ood | .303 | .217 | **−.087 (.024)** | .233 | −.070 |
| eval_rule_ood | .297 | .260 | −.037 (.35) | .240 | −.057 |
| eval_adversarial | .853 | .750 | **−.103 (.0001)** | .800 | −.053 |

B6 is also significantly below B5 on ID/template/adversarial (e.g. ID −.127, p = .0008). The GRPO arm was **closed with champion unchanged** (`docs/experiments/b6_grpo_closure.md`).

### 6.2 Mechanism

1. **Aggregate-reward mismatch (B6):** 40–50 % of failing adversarial episodes scored > 0.8 — the policy learned to *almost pass*, harvesting partial credit. B6-R's pass-gated reward (passed → 0.5 + 0.5·score; failed → 0.1·score) removed this incentive.
2. **Episode-level confirmation (x19 flip analysis, B6-R vs B6):** the entire recovery is **adversarial-local** — net +15/300 flips there (fail→pass 25, pass→fail 10), every other suite within ±6 net flips (noise level). 100 % of pass→fail regressions across all suites carry `required_component_failed`. A residual near-miss cluster persists under B6-R (6/10 adversarial pass→fail episodes score in [0.8, 1.0)) but is reward-gated during training.
3. **The binding failure is data/regularization, not reward shape:** 50–70 % of prompt groups had zero reward variance (no gradient — *advantage starvation*), and policy entropy collapsed 0.15 → 0.008: the policy specialized to the narrow training pool's required-component patterns and lost ID/OOD coverage. Reward redesign (B6-R) was precisely the controlled test of the alternative hypothesis — and it did not move the construction suites.

### 6.3 Implication

Reward-function or verifier refinement alone is unlikely to rescue online RL here. The binding constraints are scenario-pool diversity/curriculum and regularization (KL-to-parent, entropy bonus, SFT replay), all explicitly out of protocol-v1 scope. Advantage-estimator swaps (e.g. RLOO) share the group-relative structure exposed to the same starvation mechanism; E-RLOO was de-scoped on these grounds (Amendment v1.4) rather than run as a low-prior ablation.

---

## 7. Threats to validity and disclosures

1. **Comparative, not absolute, claims.** Absolute pass rates of the champion remain modest on construction suites (.30–.39 ID/OOD at 8B/LoRA/2k-episode scale). Protocol v1 was designed to answer *which recipe transfers better under a fixed small budget*, and its gates (G1–G6) were defined over comparative deltas, never over an absolute mastery threshold. Absolute mastery of PlayWorld is future work (§8), not a claim of this report.
2. **Scale and task scope.** One base model (8B), LoRA adapters, one synthetic environment, 300 episodes/suite. External validity to real agent tasks is untested.
3. **Reseeding scope & de-scoped ablations.** Seeds cover the final-stage training of B4v2/A2v2 only; upstream stages (P1 SFT, A1v2 SFT, pair mining) and B5/B6/C rows are single-seed and labeled as such. E-RLOO, E-RANDPAIR, E-QLORA, E-ATTN, E-RULE and the LogosP breadth study were dropped under the pre-registered budget-cut order plus the §6 closure rationale (Amendment v1.4); the two dropped headline comparisons are disclosed as *not run*.
4. **Infrastructure incidents (all documented; none affect reported numbers):**
   - an early B6-R evaluation (`f1854f`) scored **stale hub weights** — caught by the x20 identity audit (1.0 prediction-identical rate with B6) and excluded; the audited B6-R run of record is `8791aa`. B6-R pass rates in §6.1 are derived exactly from `8791aa`'s episode-paired flip matrices; x20 audits are mandatory in all later stages;
   - an interim G6 aggregation consumed a wrong summary for the b4v2-s42 baseline (missing fetch in a fresh runtime), producing a spurious FAIL; the run of record uses hard-gated per-entry loading (adapter_dir + freeze fingerprint asserted per entry);
   - two champion-stability analyses initially bound a wrong baseline run; they were re-run against the verified s42 eval (`7308ee`) and the corrected versions are reported in §4.1.
5. **Dataset non-determinism (fixed).** The v1 SFT builder broke oracle-path ties non-deterministically; discovered mid-campaign, fixed by freezing a single artifact and re-running affected arms as "v2" (A1v2 ≈ A1v1: all n.s.).
6. **Adversarial suite semantics.** Rejection-style scoring makes absolute adversarial pass rates incomparable to construction suites (see C2's .907).

---

## 8. Conclusions and future work

**Conclusions.** On a verifiable planning task with frozen evaluation: (i) a general-reasoning SFT phase before task SFT yields large, seed-robust ID *and* OOD gains over direct tuning at matched task data — the single strongest and most stable effect in the study; (ii) offline DPO adds ≈ nothing on either route; (iii) verifier-rewarded GRPO actively hurts without data-side diversity and entropy/KL regularization, and reward-shape fixes relocate, but do not remove, the failure.

**Future work.** The evidence points to a staged program, ordered by what the v1 data actually license:

- **Protocol v2a — close the absolute-performance gap on the frozen spec.** Before enriching the world, push the champion recipe toward mastery of the *current* PlayWorld (construction-suite pass rates from ~.30–.39 toward a pre-registered target band), holding the frozen suites fixed so progress is measurable against v1. The levers are exactly those v1 identified as binding: **(a) Phase-1 enrichment** — the dominant transfer lever in v1 (B5 ≫ A2v2 even after DPO) — extending the math-centric mixture with harder math/logic corpora, sandbox-executed code generation with verified rewards, and structured-JSON tool-calling, all representationally adjacent to PlayWorld's output contract; **(b) Phase-2 data** — scenario-pool expansion and difficulty curricula targeting the `required_component_failed` mass; **(c) online RL retried under its prerequisites** — diverse/curriculum pools, KL-to-parent, entropy bonuses, SFT replay, with advantage-starvation and entropy metrics as pre-registered guardrails.
- **Protocol v2b — controlled world-spec extension.** Only after v2a: longer horizons, NPC/environment interactions, grid-free 2-D layouts — each added as a *new frozen suite axis* so that v1/v2a checkpoints remain comparable baselines.
- **Exploratory track — text-native spatial world modeling via vector-graphics output.** A candidate bridge from structured-JSON plans toward more general world models within a single text modality: emit **SVG** (text-serialized vector graphics) as the plan/state representation — paths in grid or grid-free 2-D worlds, composable scenes — verified by a *hybrid* stack (deterministic geometry checks on the parsed SVG first; rendering + rule-based CV, and optionally VLM judges, only as secondary graded signal). This inherits v1's core lesson: keep a deterministic verifier as the ground truth and treat learned judges as noisy auxiliaries, since verifier-rewarded RL already showed reward-mismatch failure modes with *exact* verifiers (§6). Known risks to de-risk first: LLMs' precise-geometry weaknesses in SVG generation, VLM-judge exploitability, and rendering-pipeline complexity on a single-GPU budget.

---

## Appendix A. Runs of record

| Artifact | ID / value |
|---|---|
| Champion training run | `20260814-023603--b4v2-playworld-sft-from-p1--s42--c56ed2` (repo `m97j/aw-runs-b4`) · adapter `sha256:d4fcacdd…` |
| Champion eval (s42) | `20260814-032546--eval-playworld--s42--7308ee` |
| Control training run | `20260814-114257--a2v2-playworld-dpo--s42--ef1e20` (repo `m97j/aw-runs-a2`) · adapter `sha256:cc23345f…` |
| Control eval (s42) | `20260814-120733--eval-playworld--s42--61d7cd` |
| Parents (pinned) | P1 `20260807-225109--b1-general-sft-v2--s42--e6e83b` (`sha256:747e5757…`) · A1v2 `20260814-022256--a1v2-playworld-sft--s42--269d0e` (`sha256:70c2ecb9…`) |
| Seed runs (repo `m97j/aw-runs-seeds`) | b4v2-s43 `20260817-151254--…--c04b9c` · a2v2-s43, b4v2-s44, a2v2-s44 (run cards in repo) |
| Seed evals | s43: `20260818-100258--…--304567` (b4v2), `20260818-101023--…--44a500` (a2v2) · s44: `20260818-105059--…--082b2a` (b4v2), `20260818-105846--…--40b6f2` (a2v2) |
| Seed analyses | `analysis_b4v2_vs_a2v2_s43/s44.json`, `analysis_b4v2_s43/s44_vs_s42.json` (under the b4v2 seed-eval runs) · `runs/seed_variance_report.json` (x21; G6 verdict PASS) |
| B5 run / eval | `20260814-115239--b5-playworld-dpo--s42--4c45da` / `20260814-124224--eval-playworld--s42--f77cd8` (repo `m97j/aw-runs-b5`) |
| B6 run / eval | `20260815-150717--b6-playworld-grpo--s42--6e6c45` / `20260816-004824--…--274abd` (repo `m97j/aw-runs-b6`) |
| B6-R run / eval | `…--b6r-playworld-grpo-gated--s42--de963e` (rerun of record) / `20260816-201107--…--8791aa` (x20-audited) · invalid eval `f1854f` excluded (§7.4) · flip artifact `x19_b6r_vs_b6_flip.json` |
| C1 / C2 evals | `20260802-053440--…--e6cb05` · `20260802-060802--…--67cf20` |
| Frozen suites | `freeze_fingerprint sha256:3cdcbc30c99e492c…`, 5 × 300 episodes |
| Data fingerprints | P1 mixture: GSM8K 0.6 + MATH-algebra 0.4, 8k target / 500 holdout (manifest in `data/p1`) · PlayWorld SFT `sha256:54fcb1d3…` (2,000 records) · preference `sha256:5856b5c5…` (canonical `fingerprint_payload`) |

## Appendix B. Diagnostic tooling (x-scripts)

x09 termination audit · x10 stop-logit probe · x11 adapter integrity · x12 attested probe · x13 trained-ids NLL · x15 SFT data diff · x16 SFT provenance resolver · x17 GRPO scenario audit · x19 regression/flip diagnostics · x20 eval identity audit (stale-weights guard) · x21 seed-variance aggregation. Each incident that motivated a tool is preserved chronologically in the campaign notebooks (aw_01–aw_11).

---

*v1.0-rc. Remaining before v1.0 tag: (i) optional — B6-R mean-score row from `8791aa`'s evaluation_summary.json, (ii) DOI placeholder after TechRxiv submission, (iii) LaTeX conversion.*

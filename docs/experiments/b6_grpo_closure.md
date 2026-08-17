# §6 Addendum — B6/B6-R GRPO Arm Closure (Final Conclusion)

**Status:** CLOSED (negative result, confirmed across two reward designs)
**Date:** 2026-08-16 (KST session) / documented 2026-07-23 turn
**Champion (unchanged):** `b4v2-playworld-sft-from-p1` — run `20260814-023603--b4v2-playworld-sft-from-p1--s42--c56ed2`

---

## 1. Research question

RQ2-ext: Does online GRPO with a programmatic verifier reward improve a strong SFT
parent (B4v2) on the frozen PlayWorld evaluation, and is any regression attributable
to reward design (aggregate partial credit) rather than to RL itself?

Two arms answer this as a controlled single-variable pair:

| arm | run_id | reward | adapter sha256 |
|---|---|---|---|
| B6 (aggregate) | `20260815-150717--b6-playworld-grpo--s42--6e6c45` | `verifier_reward_hybrid` (aggregate score) | `5286a89f…` |
| B6-R (pass-gated) | `20260816-155851--b6r-playworld-grpo-gated--s42--de963e` | `verifier_reward_hybrid_pass_gated` (pass→0.5+0.5·s, fail→0.1·s) | `385c58c246153ab13c18b51bd939b777456564dc47d1012df1636eaa06e753ae` |

Shared: parent=B4v2, seed=42, 2000 steps, HF `generate` rollouts, GRPO config
identical except `training.extra.reward_mode`. Freeze fingerprint
`sha256:3cdcbc30c99e492c…` identical across all eval runs cited below.

Eval-run validity: B6-R eval `20260816-201107--eval-playworld--s42--8791aa` passed
the x09jr run audit (1500 unique trace rows after dedupe, truncation_rate 0.0007,
empty_predictions 0, runaway_rate 0.0007) and scores the fresh adapter
(`runs/…de963e/artifacts/final_adapter`); the earlier stale-weights incident (x20,
identical-prediction rate 1.0) does not apply to this run.

## 2. Headline results

### 2.1 B6-R vs B4v2 (frozen eval, n=300/suite, paired permutation, 10k resamples)

| suite | pass Δ | 95% CI | p | sig | mean_score Δ | p | sig |
|---|---|---|---|---|---|---|---|
| eval_id | −0.160 | [−.227,−.090] | 0.0001 | **YES** | −0.063 | 0.013 | **YES** |
| eval_template_ood | −0.137 | [−.203,−.070] | 0.0001 | **YES** | −0.041 | 0.112 | no |
| eval_comp_ood | −0.070 | [−.140, .000] | 0.055 | no | −0.013 | 0.628 | no |
| eval_rule_ood | −0.057 | [−.123,+.010] | 0.114 | no | +0.000 | 0.993 | no |
| eval_adversarial | −0.053 | [−.100,−.007] | 0.032 | **YES** | −0.022 | 0.031 | **YES** |

B6-R is significantly worse than the SFT champion on 3/5 suites (pass) and never
better anywhere. Direction is uniformly negative on pass_rate.

### 2.2 B6-R vs B6 (effect of pass-gating, single-variable contrast)

| suite | pass Δ (gated−aggregate) | p | sig |
|---|---|---|---|
| eval_adversarial | **+0.050** | 0.016 | **YES** |
| eval_comp_ood | +0.017 | 0.491 | no |
| eval_template_ood | +0.003 | 1.000 | no |
| eval_id | −0.020 | 0.291 | no |
| eval_rule_ood | −0.020 (mean_score −0.024, p=.053) | 0.309 | no |

Pass-gating recovered performance **only** on eval_adversarial — precisely the suite
where x19 showed the densest near-miss (score≥0.8) partial-credit cluster — and left
the ID/OOD regression intact.

## 3. Mechanistic interpretation

1. **Training was healthy, transfer failed.** B6-R train reward climbed
   0.002→~0.5–0.7 with clipped_ratio→0 and stable lengths (~90 tokens, no runaway).
   The regression is therefore a train→frozen-eval generalization gap, not a broken
   run and not (any longer) an aggregate-objective mismatch.
2. **Advantage starvation + entropy collapse.** `frac_reward_zero_std` rose to
   0.5–0.7 in the second half (zero-gradient groups on the majority of prompts;
   frequent grad_norm=0 steps) while entropy collapsed 0.15→~0.008. Late training
   updates were driven by a shrinking minority of prompts, sharpening the policy on
   the training scenario distribution.
3. **Failure signature persists under gating.** In the B6-R vs B4v2 flip analysis,
   100% of parent-pass→B6R-fail episodes fail with `required_component_failed`, with
   a bimodal score histogram ([0.2,0.4) and [0.8,1.0) clusters) and positive length
   deltas (+23…+36 chars; style-shortening hypothesis remains rejected). The policy
   drops required components on frozen episodes it did not train on — a specialization
   effect, not a scoring artifact.

## 4. Decision

- **§6 final verdict:** On top of a strong SFT parent in this domain, offline DPO is
  ~null (B5) and online GRPO is harmful regardless of reward shaping
  (B6 aggregate, B6-R pass-gated). The two-stage champion **B4v2** stands.
- The B6 arm is **closed**. No further GRPO variants (KL-to-parent, entropy bonus,
  scenario-mixing) will be run inside this protocol; they are listed as future work
  in §8, since each would open a new tuning axis and break the single-variable
  discipline of the protocol.
- Positive scientific product of the arm: (a) identification and correction of
  aggregate-reward partial-credit hacking (x19 → pass-gated bridge, +5pp adversarial,
  p=.016); (b) evidence that verifier-reward GRPO on a narrow scenario pool induces
  required-component specialization under entropy collapse.

## 5. Threats to validity / notes

- Single seed (s42); all deltas are paired within a frozen 1500-episode eval set.
- Two eval invocations of the same B6-R adapter exist
  (`20260816-195129…d679ee`, `20260816-201107…8791aa`) with identical suite metrics;
  `8791aa` (audited) is the run of record and holds `analysis_b6r_vs_b4v2.json` /
  `analysis_b6r_vs_b6.json`.
- Confirmatory flip analysis between the two GRPO arms (x19; run_a=B6-R `8791aa`,
  run_b=B6 `274abd`; note the JSON's labels `b6-grpo`/`b4v2-sft` are stale cell
  labels — the data are B6-R vs B6): adversarial net gain **+15/300 (+5.0pp)**
  (fail→pass 25, pass→fail 10), exactly matching the aggregate +5pp recovery; all
  other suites are within ±6 net flips (noise-level), confirming the gating effect
  is adversarial-local. All pass→fail regressions across suites carry
  `required_component_failed` (100%), consistent with the specialization mechanism.
  Residual near-miss cluster persists under B6-R (6/10 adversarial pass→fail
  episodes score in [0.8,1.0)) but is reward-gated to ≤0.1·score during training.
  Caveat: x19 emits score histograms only for the pass→fail direction, so the
  provenance of the 25 gained episodes from the B6-era score≥0.8 cluster is
  inferred from the aggregate x19-v1 finding (40–50% of B6 adversarial failures
  scored >0.8), not shown per-episode. Artifact: `x19_b6r_vs_b6_flip.json`


## 6. Artifact registry (for §7 write-up)

- Champion: B4v2 — eval run `20260814-032546--eval-playworld--s42--7308ee`
- B6-R train: `hf://model/m97j/aw-runs-b6/runs/20260816-155851--b6r…de963e` (wandb r4b6k4lm)
- B6-R eval+analysis: `hf://model/m97j/aw-runs-b6/runs/20260816-201107--eval-playworld--s42--8791aa`
- Reward bridge modes: `src/axiom_world/training/reward_bridge.py` (`aggregate` | `pass_gated`)
- Retention: `hf_sync keep_hub_checkpoints=1` verified in-run (pruned 4–5 stale blobs per save; quota incident resolved)

# §8 Addendum — Gate G6: 3-Seed Final Confirmation (CLOSED, PASS)

**Status:** PASS — headline claim confirmed across seeds {42, 43, 44}
**Scope:** Amendment v1.4 (docs/experimental-protocol.md §13)
**Date:** 2026-07-23 turn (KST session)
**Artifact of record:** `runs/seed_variance_report.json` (synced to `m97j/aw-runs-seeds`)

---

## 1. Pre-registered question

Is the headline Track-B advantage (champion **B4v2** vs Track-A control **A2v2**)
robust to the stochastic elements of final-stage Phase-2 training (init RNG,
data shuffling), or an artifact of the single seed s42?

Reseeded unit: final-stage training only, on identical sha-pinned parents
(P1 champion `b1-general-sft-v2--s42--e6e83b`, A1v2 `a1v2-playworld-sft--s42--269d0e`;
pins verified against child lineage) and identical frozen data artifacts
(canonical `fingerprint_payload` verification). Frozen eval suites
(`freeze_fingerprint sha256:3cdcbc30…`), greedy decoding.

## 2. Verdict (pre-registered criterion: per-suite sign consistency of Δpass_rate)

**PASS — 5/5 suites sign-consistent, all positive, across all 3 seeds.**

| suite | Δ(B4v2−A2v2) s42 / s43 / s44 | B4v2 mean±sd | A2v2 mean±sd |
|---|---|---|---|
| eval_id | +.2066 / +.2234 / +.1933 | .393 ± .013 | .186 ± .002 |
| eval_template_ood | +.1400 / +.1500 / +.1366 | .349 ± .005 | .207 ± .003 |
| eval_comp_ood | +.1600 / +.2033 / +.2066 | .329 ± .022 | .139 ± .004 |
| eval_rule_ood | +.1000 / +.1166 / +.1134 | .302 ± .005 | .192 ± .005 |
| eval_adversarial | +.2100 / +.2033 / +.1867 | .851 ± .002 | .651 ± .011 |

Notable: the smallest per-suite delta across every seed (+.10, rule-OOD s42)
still exceeds every seed-induced fluctuation (max sd .022). Both arms are
remarkably seed-stable at the final stage; A2v2 in particular is near-invariant
(sd ≤ .011 everywhere).

## 3. Threats to validity / disclosures

- Upstream-stage seed variance (P1 training, A1v2 training, pair mining) is NOT
  measured — reseeding covered the final Phase-2 stage only (Amendment v1.4 scope).
- One infrastructure incident during the campaign: an interim x21 aggregation
  accidentally consumed a wrong `evaluation_summary.json` for the b4v2-s42
  baseline (the baseline eval run had not been materialized in the fresh
  runtime), producing a spurious FAIL. Root cause was a missing fetch, not
  model behavior; the final run of record uses hard-gated per-entry loading
  (adapter_dir + freeze_fingerprint printed and asserted per entry).
  No training or evaluation was re-run to obtain the PASS.
- b4v2-s43 training completed in an earlier session; its adapter was restored
  revision-pinned from `m97j/aw-runs-seeds` and hash-verified against its
  lineage (`output_adapter_sha256`) before evaluation.

## 4. Consequences

- Gate **G6 is closed**; protocol v1 experimental phase is COMPLETE.
- Final tables report mean ± sd over 3 seeds for B4v2 and A2v2; single-seed
  labels remain (disclosed) for B5/B6/C-arms.
- Next and final step: §7 tech report write-up → v1.0.0 release set.

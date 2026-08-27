# Metric Definitions

**Status:** living specification. Applies to every protocol version.

A metric named in a report must be defined here, exactly once, and computed by
one implementation. Two functions computing "pass rate" is how a project starts
disagreeing with itself.

## Outcome metrics

### `pass_rate`
Fraction of episodes whose verifier verdict is `status == "pass"`. Strict: all
gates cleared. The primary metric of protocol v1.

### `mean_score`
Mean of the verifier's partial-credit `score` over episodes.

**Always report alongside `pass_rate`, never instead of it.** Protocol v1's
B6-R arm held `mean_score` essentially constant (rule-OOD .513, identical to the
champion) and raised legal-action rate to .77–.99 while `pass_rate` collapsed.
The policy remained a competent *partial* planner and lost only completion. A
single aggregate would have reported "no meaningful change".

### `legal_action_rate`
Fraction of emitted actions legal in the state where they were emitted. A
capability floor, not a success measure — it can rise while `pass_rate` falls.

### `format_validity_rate`
Fraction of outputs clearing the format gate. Separates "cannot express" from
"cannot plan". Instruct-prompting baselines failed here (96% malformed JSON
zero-shot), not at planning.

### `truncation_rate`
Fraction of generations hitting the length cap. **Must be 0.00 for any admitted
run** — see `adapter-contract.md` A-3.

## Mechanism metrics

### `counterfactual_rollout_error` *(v3.0-CLB)*
At frozen probe points, the model predicts the state resulting from an action it
did **not** take; the simulator executes that same action from the identical
checkpointed state; the metric is cell-level disagreement between the two.

Probe points and counterfactual action choices are frozen with the suite and are
identical across all arms.

This is the only metric that measures the transition model itself rather than the
outcome of using it. Task success can rise through better search over a poor
transition model; this cannot.

### `belief_f1` *(v3.0-CLB)*
F1 of predicted contents of unobserved cells against simulator ground truth.

## Calibration

### `ece`
Expected calibration error, 10 equal-width bins unless stated otherwise. Report
bin count wherever it appears.

### `brier`
Mean squared error between stated confidence and outcome.

## Statistics

Fixed across protocols; changing any of these is an amendment.

| | |
|---|---|
| Point estimates | 95% bootstrap CI, 10,000 resamples |
| Paired comparison | paired bootstrap over per-episode outcomes, arms sharing identical episodes and probe points |
| Significance | sign-flip permutation, 10,000 permutations, α = 0.05 |
| Multiplicity | Holm–Bonferroni over the pre-registered comparison family, declared and closed in the protocol before run 1 |
| Seed replication | sign consistency across all suite × seed cells; a sign flip is a failure regardless of significance |

## Reporting rules

1. Every number carries its suite, its seed set, and its protocol version.
2. `pass_rate` and `mean_score` appear together.
3. Confidence intervals appear with point estimates, not in an appendix.
4. A metric absent from this file may not appear in a report.

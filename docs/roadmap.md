# Roadmap

## What this project is establishing

That claims about a language-model agent's competence can be made *checkable*:
by building the world it acts in, scoring it with a deterministic verifier rather
than a learned judge, freezing the evaluation before training, and publishing the
failures with the same weight as the successes.

## Track sequence

| Track | Question | Status |
|---|---|---|
| **v1 — Verifiable recipe comparison** | Which post-training recipe transfers, and why? | **Closed.** Two-stage wins on 5 suites × 3 seeds; DPO ≈ null; verifier-rewarded GRPO regresses and the mechanism is documented. |
| **v3.0-CLB — Closed-loop belief** | Does the model have a transition model, and does explicit belief maintenance produce one? | **Next.** Pre-registration written; freeze pending. |
| v2a — Absolute performance | How good can this get on the same spec? | Deferred. |
| v2b — Spec expansion | Interaction, NPCs, narrative under the same contract. | Deferred. |
| v4 — Worlds as code | Can the model emit the world's dynamics as an executable program and be scored on rollout agreement? | Exploratory. |

## Why v3 was pulled ahead of v2a

Recorded here because reordering a roadmap after seeing results is exactly the
kind of decision that needs a written reason.

1. **v2a produces no new falsifiable claim.** It raises a score on an axis
   already measured. Higher, but not different.
2. **Full observability makes the central question unaskable.** When the state is
   supplied in the prompt, one cannot ask whether the model *has* a transition
   model — only whether it can plan over a given one. Partial observability is
   the minimum condition under which the question is well-posed.
3. **The stepwise loop is a shared prerequisite.** Counterfactual measurement
   requires checkpoint/restore against the simulator. Built once for v3, reused
   by v2a and v2b.

## Invariants across all tracks

- No learned judge in any scoring path that feeds a reward, a selection, or a
  headline number.
- Evaluation suites frozen and fingerprint-pinned before training.
- Every canonical run emits its full artifact set or it did not happen.
- Adapter lineage verified by SHA-256 at load; evaluation identity audited before
  numbers are admitted.
- Negative results are published with their mechanism, and the arm is closed in
  writing.

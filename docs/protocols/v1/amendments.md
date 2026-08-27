# Protocol v1 — Amendment Log

Protocol v1.0 was frozen before run 1. Every subsequent deviation is recorded
here as a numbered entry. The protocol text itself is never edited in place;
`docs/protocols/v1/experiment_protocol_v1.md` carries the v1.4 text, and this log
is the record of how it got there.

Each entry states: what changed, when, why, and — critically — **what it does and
does not affect**. An amendment that changes what a headline number means is a
different kind of event from one that adds a diagnostic, and the log must make
that distinction legible.

> **Status of this file.** Reconstructed at repository tag `v1.0.1` from the protocol's own version history and the arm closure notes.
> If the dates or numbers differ, they must be corrected. Correcting this reconstruction is not itself an amendment.

---

## A-1 — Adapter contract widened `2026-08-02`

- **Protocol version:** v1.0 → v1.1
- **Change:** Trainable set extended to include `lm_head` and `embed_tokens`
  (`modules_to_save`), where previously LoRA adapters covered attention and MLP
  projections only.
- **Reason:** Adapter-only training on a base checkpoint leaves control-token
  embeddings and their logit rows frozen, so the model cannot emit chat-template
  terminal tokens. Observed as 100% output truncation under a normal-looking loss
  curve. Mechanism confirmed by token-level NLL replay: content-token NLL fell
  1.26 → 0.38 while terminal-token NLL stayed at 7.9.
- **Affects:** All arms. Every pre-fix run is void and is excluded from every
  reported number. Post-fix verification: truncation 100% → 0%, P1 holdout
  .844 / .834.
- **Does not affect:** Evaluation suites, scoring, statistics plan, champion
  selection rule.
- **Record:** `docs/experiments/v1/adapter_contract_termination.md`

## A-2 — GRPO arm closed `2026-08-06`

- **Protocol version:** v1.2 → v1.3
- **Change:** The verifier-rewarded RL arm (B6) is closed rather than iterated.
  A second reward design (pass-gated, B6-R) was run as a control and also failed.
- **Reason:** GRPO regressed the champion (ID −16 pp). The first hypothesis —
  reward shaping — was falsified by the B6-R control. Episode-level flip analysis
  showed recovery was adversarial-local (net +15/300 there, ±6 elsewhere) and
  that 100% of pass→fail regressions carried `required_component_failed`. The
  binding constraint was the training scenario pool: 50–70% zero-variance reward
  groups (advantage starvation) with policy entropy collapsing 0.15 → 0.008.
- **Affects:** Champion remains the two-stage SFT checkpoint; no RL-derived
  checkpoint enters the reported set.
- **Does not affect:** All non-RL arms and their statistics.
- **Record:** `docs/experiments/v1/b6_grpo_closure.md`

## A-3 — Estimator ablations descoped `2026-08-13`

- **Protocol version:** v1.3 → v1.4
- **Change:** Pre-registered ablations over the advantage estimator (E-RLOO,
  E-RANDPAIR and the remainder of that family) are withdrawn and will not be run.
- **Reason:** A-2 localised the failure to data diversity and regularisation, not
  to the estimator. RLOO shares the same group-relative structure, so the prior
  probability that swapping estimators changes the conclusion is low, and the
  compute is better spent on the 3-seed confirmation.
- **Affects:** The pre-registered experiment matrix is reduced. Withdrawal is
  recorded with its reasoning so that the descope is auditable rather than
  silent — an unrun pre-registered arm must never be simply absent.
- **Does not affect:** Any reported number.

## A-4 — Final confirmation narrowed to a 3-seed replication `2026-08-17`

- **Protocol version:** v1.4
- **Change:** Gate G6 is a 3-seed replication of the champion contrast only
  (B4v2 and A2v2 reseeded at s43 / s44), judged by sign consistency across all
  suite × seed cells, rather than a broader re-run.
- **Reason:** Compute released by A-3, directed at the claim that carries the
  report's headline.
- **Outcome:** PASS — 5/5 suites sign-consistent, Δ +.10 to +.22 throughout,
  B4v2 seed sd ≤ .022.
- **Note:** One intermediate FAIL was traced to a plumbing error — a summary read
  before the `b4v2-s42` baseline evaluation had been fetched — not to a
  scientific result. Documented rather than quietly re-run.

---

## Amendment discipline

1. Amendments are appended. Existing entries are not rewritten.
2. An amendment that voids prior runs must say so explicitly and name them.
3. A withdrawn pre-registered arm requires an amendment; silence is not
   permitted.
4. Amendments record decisions about the *protocol*. Findings belong in the
   report; mechanism write-ups belong in `docs/experiments/v1/`.

# Verifier Contract

**Status:** living specification. Applies to every protocol version.

The verifier is the only thing standing between this project and unfalsifiable
claims. Every headline number is a count of verifier verdicts. This document
states what the verifier must guarantee.

## C-1. Determinism

Given identical `(episode, output, world_config)`, the verifier returns an
identical verdict — same status, same failure code, byte-equal serialisation —
on any machine, in any process, at any time.

No randomness. No wall-clock. No network. No filesystem state outside the inputs.
No floating-point comparison without an explicitly specified tolerance.

## C-2. No learned component in the scoring path

No model output, learned classifier, embedding similarity, or LLM judgement may
influence a verdict, a reward, a champion selection, or any number that appears
in a report.

An LLM may be used for *audit-only* commentary on a sample of episodes, provided
its output cannot alter a verdict and it is labelled as such wherever it appears.
Protocol v3.0-CLB removes even this: every quantity of interest is computed
exactly from the simulator.

## C-3. Ordered gate chain, first failure wins

```
format gate  →  action legality  →  required components  →  goal
```

Evaluation stops at the first failing gate and reports that gate's code. Gates
are never reordered, never merged, and never evaluated in parallel with
tie-breaking, because the failure code is itself reported data: the v1 GRPO
post-mortem turned on the observation that 100% of pass→fail regressions carried
`required_component_failed` and not some other code.

## C-4. Total function

Every input produces a verdict. Malformed output is a `format` failure with a
code, never an exception, never a skipped episode, never a silent zero.

Silently dropping unparseable outputs inflates the scores of models that fail
loudly and is the most common way a benchmark lies.

## C-5. Versioned and hash-pinned

The verifier's version and the schema it enforces are recorded in every run's
`environment_manifest.json`. Any change to gate semantics is a version bump.
Numbers produced under different verifier versions are not comparable and must
not be placed in the same table without an explicit note.

## C-6. Golden fixtures gate every change

A fixture set covering valid outputs, each malformed class, each illegal-action
class, each required-component class, and boundary cases must reach 100%
expected-status agreement before any verifier change is merged. This is Gate G2
in both protocols.

## C-7. Verdict shape

```jsonc
{
  "status": "pass" | "fail",
  "gate":   "format" | "legality" | "required_components" | "goal" | null,
  "code":   "malformed_json" | "illegal_action" | "required_component_failed" | ...,
  "score":  0.0,          // partial credit; never substitutes for `status`
  "detail": { }           // structured, machine-readable, no free text
}
```

`score` and `status` are independent and must both be reported. Protocol v1
found a regime — the B6-R *score/pass wedge* — in which mean score and legal
action rate were unchanged or better while strict pass collapsed. Reporting a
single aggregate would have concluded "roughly equivalent" about a policy that
had lost the ability to finish.

## C-8. Protocol-specific extension, not replacement

A protocol may add gates or codes. It may not weaken C-1 through C-7. Protocol
v3.0-CLB's departure — a malformed step scores as a format failure and the
episode continues with a no-op, where v1 ended the episode — is a deliberate
contract difference recorded in that protocol, required to separate format
fragility from planning failure inside a loop.

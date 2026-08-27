# Adapter Contract

**Status:** living specification. Applies to every protocol version.

This document exists because a violation of it produced a total, loss-invisible
failure of every run in the project's first weeks. See
`docs/experiments/v1/adapter_contract_termination.md` for the full diagnosis.

## The failure this prevents

LoRA applied to attention and MLP projections only, on a **base** (non-instruct)
checkpoint, cannot raise the logits of chat-template control tokens: the
embedding rows and the corresponding `lm_head` rows are frozen, and no low-rank
update to the intervening blocks reaches them. The model therefore never emits a
terminal token and every generation runs to the length cap.

The loss curve looks healthy throughout. Content-token NLL fell 1.26 → 0.38
while terminal-token NLL sat at 7.9. Nothing in the training telemetry indicates
a problem. This class of failure is invisible to monitoring and visible only to a
contract check.

## A-1. Control tokens must be trainable when the base lacks them

If training targets a base checkpoint with a chat template whose control tokens
are untrained, the trainable set **must** include the embedding matrix and the
output head:

```yaml
peft:
  modules_to_save: [lm_head, embed_tokens]
```

## A-2. Tied embeddings are checked, not assumed

If input and output embeddings are tied, saving one and not the other is
undefined behaviour. The run configuration must record the tying state and the
loader must assert it matches what training assumed.

## A-3. Termination audit is a release gate

No adapter is admitted to evaluation until an audit over a held-out sample shows
`truncation_rate == 0.00`. This is a hard gate in both protocols. In a closed
loop it is stricter than it looks: a truncated step corrupts every subsequent
observation in that episode.

## A-4. Lineage is verified by hash at load

Every training run asserts at load time that its parent adapter's SHA-256 matches
`lineage.json`. A mismatch is a hard failure, not a warning.

## A-5. Evaluation identity is audited before numbers are admitted

Every evaluation run must pass an identity audit binding the loaded weights to
the declared adapter revision. Protocol v1 caught a stale-weights evaluation this
way, and discarded it. Inspection alone does not catch this; the audit does.

## A-6. Adapter contract version is part of the run record

Runs record which contract version they were trained under. v1 arms trained
before the fix are void and excluded from every reported number — not adjusted,
not re-weighted, excluded.

## Rationale, stated generally

The lesson is not about `modules_to_save`. It is that a training setup can
satisfy every metric being watched while violating a precondition nobody thought
to state. Adding more monitoring does not find these. Writing down what must be
true, and asserting it, does.

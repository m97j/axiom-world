# Failure Analysis — Chat-Template Termination under Adapter-Only Fine-Tuning (the "v2 Adapter Contract")

**Status:** v1.0 (CLOSED — diagnosis, fix, and post-fix verification complete)
**Scope:** Phase-1 SFT campaign (aw_05; pre-fix runs of 2026-08-03, v2 re-runs of 2026-08-07) → adapter contract applied uniformly to all v2 arms.
**Referenced from:** tech report §4.2.

---

## 1. Symptom

Early Phase-1 SFT adapters trained with the standard LoRA target set
(attention/MLP projections only: `q/k/v/o_proj`, `gate/up/down_proj`; confirmed
by x11 tensor audit) failed to emit the chat template's terminal token
`<|im_end|>` (id 151645) at generation time. Transfer-probe evaluations on the
frozen PlayWorld suites showed **catastrophic, generation-side failure**:

| pre-fix probe eval (s42) | truncated outputs | malformed-JSON gate failures (per suite) | ID pass |
|---|---|---|---|
| B1-probe (`fd3d91`) | **1,500 / 1,500** | 10–65 | .133 |
| B2-probe (`0276f7`) | **1,499 / 1,500** | 119–216 | .070 |

Every single output ran to the 1,024-token cap. Training loss showed nothing
abnormal — the failure was invisible at the loss level and surfaced only at the
verifier's format gate.

## 2. Audit chain (bottom-up), including a probe-integrity incident

### 2.1 x10 — stop-logit probe (first measurement, later superseded)

x10 teacher-forces the gold context to the terminal position and reads the
next-token distribution.

- **Base model:** `p(<|im_end|>) = 0.0`, rank 90k–141k across samples. Mass
  goes to `\n`, `####`, `<|endoftext|>` — the GSM8K surface, not the ChatML
  stop. Consistent with Qwen3-**Base** never having trained ChatML control
  tokens.
- **Adapter arms (as first probed):** B1 and B2 returned **byte-identical
  distributions** with garbage rare-token tops (p ≈ 0.001,
  `ĠForCanBeConvertedToForeach` etc.) and `p(<|im_end|>) ≈ 3×10⁻⁴`. Two
  independently trained adapters cannot share logits to the fourth decimal —
  the measurement itself was suspect.

### 2.2 x11 — adapter integrity (weights level)

x11 hash-audits files, tensors, and pairwise deltas, then runs a live forward
diff:

- On disk the adapters **differ everywhere**: 504 tensors each, 0 identical
  tensors, max |Δ| = 0.038; distinct safetensors SHA-256
  (`a67e274c…` vs `f4d158dd…`). Target-module audit confirms the pre-fix
  contract: attention/MLP only — **no `lm_head`, no `embed_tokens`**.
- Live forward diff reproduced the anomaly's signature at the loading level:
  adapter-applied entropy 8.0 nats vs base 3.7 (flattened distribution), i.e.
  the x10 numbers reflected a corrupted probe/loading path, not the models.

### 2.3 x12 — attested probe (measurement provenance)

x12 re-runs the probe while attesting *which* weights produce each number
(disk SHA vs live-LoRA SHA vs output): b1 `a67e274c…`/live `2999b9c4…`,
b2 `f4d158dd…`/live `b58df856…`, probe outputs no longer equal → **probe
healthy; the x10 identical-logits anomaly was a stale-dir/session artifact.**
Under the attested probe rendering, both pre-fix adapters still place only
p ≈ 10⁻⁴–10⁻⁶ on `<|im_end|>` (b1 rank 45–178; b2 rank 332–11,195, with mass
on `\n` or `The`).

### 2.4 x13 — trained-token NLL under attested loading (token level)

| arm (pre-fix contract) | mean NLL (content) | p(`<\|im_end\|>`) at terminal | terminal NLL | rank |
|---|---|---|---|---|
| base | 1.261 | **0.000** | **21.63** | 90k–127k |
| B1 (`b08a2c`) | 0.170 | **0.922** | 0.082 | 1 |
| B2 rehearsal (`10ecce`) | 0.338 | **0.119** | 3.44 | 2–17 |

Three findings lock the diagnosis:

1. **Base:** content NLL is fine, but the terminal `<|im_end|>` sits at
   NLL ≈ 21.6 — ~20 nats of logit deficit on one specific vocab row.
2. **B1 can place rank-1 stop mass — but only on its exact trained
   rendering.** The same adapter that reaches p = 0.92 under x13's trained
   rendering shows p ≈ 3×10⁻⁴ under x12's probe rendering: the adapter learned
   a **hidden-state workaround**, steering the residual stream toward the
   frozen output row, and that workaround is *rendering-sensitive*.
3. **B2 (rehearsal mix) collapses the margin** (0.92 → 0.12; terminal NLL
   0.08 → 3.4; worst sample p = 3×10⁻⁴): mixing data distributions breaks the
   brittle steering. Generation sampling and task transfer break it further —
   hence 1,500/1,500 truncation on PlayWorld probes.

## 3. Diagnosis

Adapter-only LoRA leaves the **embedding and lm_head rows of control tokens
frozen** at their untrained base values (near-zero output mass; ~20-nat
deficit). The adapter can compensate only indirectly, by steering hidden
states toward the frozen row — a low-margin mechanism that x12/x13 show to be
rendering- and distribution-sensitive, and that fails under generation. The
verifier's format gate converts this into total task failure. See Appendix A
for the capacity argument.

## 4. Fix — the v2 adapter contract — and post-fix verification

```yaml
peft:
  modules_to_save: [lm_head, embed_tokens]
```

applied **uniformly to all v2 arms** (the contract is part of the environment,
not a treatment; single-variable discipline preserved). Note Qwen3-8B-Base has
`tie_word_embeddings=True`, so the head and embedding rows must be handled
together — another reason to make both trainable explicitly.

**Post-fix verification (all gates PASS):**

| readout | pre-fix | post-fix (v2) |
|---|---|---|
| x13 p(`<|im_end|>`) at terminal, B1 | 0.922 (rendering-brittle) | **0.922, rank 1 (B1v2 `e6e83b`)** |
| x13 p(`<|im_end|>`), rehearsal arm | **0.119 (collapsed)** | **0.918, rank 1 (B2v2 `9293e7`)** — the rehearsal-induced collapse is *gone* |
| terminal-stop NLL (B1 / rehearsal) | 0.08 / 3.44 | 0.08 / **0.086** |
| PlayWorld probe truncation | **1,500 & 1,499 / 1,500** | 0 / 1,500 (post-fix evals `a7a47a`, `cd89fd`; runaway 0) |
| P1 held-out generation truncation | — | 4/500 (B1v2), 13/500 (B2v2); accuracy .844 / .834 |
| champion lineage (B4v2 s42/43/44) | — | truncation ≈ 0, format-gate failures ≈ 0 (tech report §4.2) |

Residual: even under the v2 contract, the *direct-route* arm retains weaker
termination discipline than the two-stage route (A2v2 ~1,490/1,500 truncated
tails vs B4v2 ≈ 0 — tech report §4.2), indicating Phase-1 data volume also
contributes to stabilizing sequence-boundary behavior beyond the parameter fix.

## 5. Generalizable lessons

1. **Loss curves do not surface termination failure.** Audit generation-side
   metrics (truncation rate, tail repetition) as first-class signals.
2. **Adapter target sets are a capability boundary, not a hyperparameter.**
   Any behavior requiring movement of specific vocab rows (control tokens, new
   special tokens) needs those rows trainable.
3. **Probes need attestation.** Byte-identical outputs across two arms is
   itself a diagnostic; x11/x12 exist because of this incident and are now
   permanent infrastructure (the same discipline later caught the stale-weights
   eval in the B6-R campaign via x20).
4. **Base checkpoints + chat templates are a known-sharp edge.** Confirm
   control-token trainability before the first long run.

---

## Appendix A. Why attention/MLP LoRA cannot cleanly fix one vocab row

This is a capacity *argument* consistent with all measurements above, not a
formal impossibility proof.

Let `h(x) ∈ ℝᵈ` be the final hidden state at the terminal position and
`E ∈ ℝ^{V×d}` the output head (tied to embeddings in Qwen3-8B). The stop logit
is `z_stop = e_stopᵀ h`. With attention/MLP-only LoRA, `e_stop` is **frozen**;
the only lever is a perturbation `Δh` of the hidden state:

```
Δz_stop = e_stopᵀ Δh ≤ ‖e_stop‖ · ‖Δh‖ · cos(e_stop, Δh)
```

Closing a ~20-nat deficit (x13: terminal NLL 21.6 → ≈0.08) *relative to all
competing rows* `e_v` requires `Δh` to grow the projection onto `e_stop`
without comparably growing projections onto the previously dominant rows
(`\n`, `####`, `<|endoftext|>`). Because control-token rows in a base
checkpoint are undertrained (weakly shaped by pre-training gradients), the
adapter must find a direction that is simultaneously (i) large enough along
`e_stop`, (ii) near-orthogonal to thousands of well-trained competing rows,
and (iii) **produced conditionally** — only at sequence-end states, since the
same `Δh` machinery must keep generating content elsewhere.

The empirical signatures match each clause: the workaround exists but only on
the exact training rendering (x13 B1: rank 1, p = 0.92 — clause iii achieved
narrowly), degrades to p ≈ 10⁻⁴ under a slightly different rendering (x12 —
clause iii fails off-distribution), and collapses under a data mixture that
disperses the steering direction (x13 B2: p = 0.12 — clause ii/iii fail). By
contrast, making `e_stop` itself trainable (`modules_to_save: [lm_head,
embed_tokens]`) turns the problem into moving **one row toward the trained
hidden states** — a low-rank-free, unconditional fix, after which the
rehearsal arm's margin is restored (B2v2: p = 0.918) and generation-side
truncation drops from 100 % to ≈0 %.

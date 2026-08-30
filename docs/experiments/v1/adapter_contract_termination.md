# Failure Analysis — Chat-Template Termination under Adapter-Only Fine-Tuning (the "v2 Adapter Contract")

**Status:** v1.1 (CLOSED — diagnosis, fix, and post-fix verification complete; explanatory wording tightened)

**Scope:** Phase-1 SFT campaign (aw_05; pre-fix runs of 2026-08-03, v2 re-runs of 2026-08-07) → adapter contract applied uniformly to all v2 arms.

**Referenced from:** tech report §4.2.

---

## 1. Symptom

Early Phase-1 SFT adapters trained with the standard LoRA target set
(attention/MLP projections only: `q/k/v/o_proj`, `gate/up/down_proj`; confirmed
by x11 tensor audit) failed to emit the chat template's terminal token
`<|im_end|>` (id 151645) reliably at generation time.

Transfer-probe evaluations on the frozen PlayWorld suites showed
**catastrophic generation-side failure**:

| pre-fix probe eval (s42) | truncated outputs | malformed-JSON gate failures (per suite) | ID pass |
|---|---:|---:|---:|
| B1-probe (`fd3d91`) | **1,500 / 1,500** | 10–65 | .133 |
| B2-probe (`0276f7`) | **1,499 / 1,500** | 119–216 | .070 |

Essentially every output ran to the 1,024-token cap. Aggregate training-loss
curves did not flag the problem: the failure became obvious only when
generation-side behavior and verifier format-gate outcomes were inspected.

---

## 2. Audit chain (bottom-up), including a probe-integrity incident

### 2.1 x10 — stop-logit probe (first measurement, later superseded)

x10 teacher-forces the gold context to the terminal position and reads the
next-token distribution.

- **Base model:** `p(<|im_end|>) = 0.0`, rank 90k–141k across samples. Mass
  goes to `\n`, `####`, `<|endoftext|>` — the GSM8K surface, not the ChatML
  stop. This is consistent with the Base checkpoint not assigning useful
  terminal probability to this chat-control token under the probed rendering.

- **Adapter arms (as first probed):** B1 and B2 returned **byte-identical
  distributions** with garbage rare-token tops (p ≈ 0.001,
  `ĠForCanBeConvertedToForeach` etc.) and `p(<|im_end|>) ≈ 3×10⁻⁴`. Two
  independently trained adapters producing the same logits to this precision
  made the measurement path itself suspect.

### 2.2 x11 — adapter integrity (weights level)

x11 hash-audits files, tensors, and pairwise deltas, then runs a live forward
diff:

- On disk the adapters **differ everywhere**: 504 tensors each, 0 identical
  tensors, max |Δ| = 0.038; distinct safetensors SHA-256
  (`a67e274c…` vs `f4d158dd…`). Target-module audit confirms the pre-fix
  contract: attention/MLP only — **no `lm_head`, no `embed_tokens`**.

- Live forward diff reproduced the anomaly's signature at the loading level:
  adapter-applied entropy 8.0 nats vs base 3.7 (flattened distribution). The
  x10 values therefore reflected a corrupted probe/loading path rather than
  trustworthy model behavior.

### 2.3 x12 — attested probe (measurement provenance)

x12 re-runs the probe while attesting *which* weights produce each number
(disk SHA vs live-LoRA SHA vs output): b1 `a67e274c…`/live `2999b9c4…`,
b2 `f4d158dd…`/live `b58df856…`; probe outputs are no longer equal →
**probe healthy; the x10 identical-logits anomaly was a stale-dir/session
artifact.**

Under the attested probe rendering, both pre-fix adapters still place only
p ≈ 10⁻⁴–10⁻⁶ on `<|im_end|>` (b1 rank 45–178; b2 rank 332–11,195, with mass
on `\n` or `The`).

### 2.4 x13 — trained-token NLL under attested loading (token level)

| arm (pre-fix contract) | mean NLL (content) | p(`<\|im_end\|>`) at terminal | terminal NLL | rank |
|---|---:|---:|---:|---:|
| base | 1.261 | **0.000** | **21.63** | 90k–127k |
| B1 (`b08a2c`) | 0.170 | **0.922** | 0.082 | 1 |
| B2 rehearsal (`10ecce`) | 0.338 | **0.119** | 3.44 | 2–17 |

Three findings constrain the diagnosis:

1. **Base:** content NLL is reasonable, while terminal `<|im_end|>` NLL is
   ≈21.6. This means the stop token has extremely low conditional probability
   at the terminal position under this rendering. The NLL value should be
   interpreted as a softmax-relative deficit, not as a direct measurement of
   a single logit's absolute distance.

2. **B1 can place rank-1 stop mass — but only on its exact trained
   rendering.** The same adapter that reaches p = 0.92 under x13's trained
   rendering shows p ≈ 3×10⁻⁴ under x12's probe rendering. This demonstrates
   that attention/MLP-only adaptation is *capable in principle* of inducing
   the desired terminal behavior through hidden-state changes, but that the
   learned correction in this run is highly rendering-sensitive.

3. **B2 (rehearsal mix) loses much of the terminal margin** (0.92 → 0.12;
   terminal NLL 0.08 → 3.4; worst sample p = 3×10⁻⁴). Together with the
   PlayWorld generation failures, this is consistent with the hidden-state-only
   correction being sensitive to representation/data distribution. It does
   **not** by itself prove that the rehearsal mixture "dispersed a steering
   direction" or identify a unique causal mechanism.

---

## 3. Diagnosis

Under the pre-fix contract, attention/MLP LoRA leaves the output/embedding
parameters associated with the chat-control token outside the trainable adapter
path. The model can still increase stop-token probability indirectly by changing
the terminal hidden representation, and B1 demonstrates that this is possible
on the exact training rendering.

The observed problem is therefore **not an impossibility of attention/MLP-only
LoRA**. Rather, in this campaign it was an indirect and brittle
parameterization for learning robust termination behavior:

```text
attention/MLP-only adaptation
        ↓
terminal hidden-state change
        ↓
fixed output-vocabulary geometry
        ↓
relative stop-vs-competitor logits
```

The required correction must improve the stop token's *relative* margin against
important competing tokens at terminal states while preserving useful content
generation at non-terminal states. x12/x13 show that the pre-fix workaround was
rendering-sensitive, and generation probes show that it did not transfer
robustly enough to the PlayWorld evaluation path.

This diagnosis is an **empirical capacity/parameterization argument**, not a
formal proof that upstream LoRA updates cannot solve the problem.

See Appendix A for the mathematical formulation.

---

## 4. Fix — the v2 adapter contract — and post-fix verification

```yaml
peft:
  modules_to_save: [lm_head, embed_tokens]
```

applied **uniformly to all v2 arms**. The adapter contract is treated as part of
the experimental environment rather than as a treatment, preserving the
single-variable comparison among v2 arms.

In the reported implementation, the model configuration uses tied word
embeddings, so the output-head / embedding relationship must be handled
consistently; both modules were therefore included explicitly in the trainable
saved-module contract.

Importantly, `modules_to_save` does **not** mean that only the single
`<|im_end|>` row is trained. It makes the selected modules directly trainable
alongside the low-rank adapter updates. The mechanistic claim is therefore
about gaining a **direct parameter path into output/embedding vocabulary
geometry**, not about literally updating only one row.

**Post-fix verification (all gates PASS):**

| readout | pre-fix | post-fix (v2) |
|---|---|---|
| x13 p(`<|im_end|>`) at terminal, B1 | 0.922 (rendering-brittle) | **0.922, rank 1 (B1v2 `e6e83b`)** |
| x13 p(`<|im_end|>`), rehearsal arm | **0.119** | **0.918, rank 1 (B2v2 `9293e7`)** — rehearsal-associated collapse no longer observed |
| terminal-stop NLL (B1 / rehearsal) | 0.08 / 3.44 | 0.08 / **0.086** |
| PlayWorld probe truncation | **1,500 & 1,499 / 1,500** | 0 / 1,500 (post-fix evals `a7a47a`, `cd89fd`; runaway 0) |
| P1 held-out generation truncation | — | 4/500 (B1v2), 13/500 (B2v2); accuracy .844 / .834 |
| champion lineage (B4v2 s42/43/44) | — | truncation ≈ 0, format-gate failures ≈ 0 (tech report §4.2) |

Residual: even under the v2 contract, the *direct-route* arm retains weaker
termination discipline than the two-stage route (A2v2 ~1,490/1,500 truncated
tails vs B4v2 ≈ 0 — tech report §4.2). This is consistent with the possibility
that training history and/or Phase-1 data volume contribute to
sequence-boundary robustness beyond the parameterization fix; the present
analysis does not isolate that effect causally.

---

## 5. Generalizable lessons

1. **Aggregate loss curves can miss termination failure.** Generation-side
   metrics such as truncation rate, tail repetition, schema validity, and
   terminal-token behavior should be audited explicitly.

2. **Adapter target sets constrain the available correction path.** A behavior
   that depends strongly on output-vocabulary/control-token geometry may be
   harder or less robust to learn when adaptation is restricted to upstream
   attention/MLP projections. This is a parameterization constraint, not a
   general impossibility theorem.

3. **Probes need attestation.** Byte-identical outputs across independently
   trained arms are themselves a diagnostic signal. x11/x12 were added because
   of this incident and became permanent infrastructure; the same provenance
   discipline later caught the stale-weights B6-R evaluation via x20.

4. **Base checkpoints combined with chat-template control tokens are a sharp
   edge worth testing early.** Before long runs, verify that the intended
   template, terminal tokens, adapter target set, loading path, and generation
   stop behavior are mutually compatible.

5. **Mechanism claims should track the level of evidence.** x12/x13 support a
   rendering-sensitive hidden-state workaround under the pre-fix contract and
   robust recovery after adding direct output/embedding trainability. They do
   not establish that control-token rows are universally undertrained, nor that
   attention/MLP-only LoRA is fundamentally incapable of robust termination.

---

## Appendix A. Why attention/MLP-only LoRA is an indirect and capacity-constrained way to correct a frozen vocabulary row

This appendix gives a **capacity/parameterization argument** consistent with the
measurements above. It is **not a formal impossibility proof**.

Let `h(x) ∈ ℝ^d` be the final hidden state at a terminal position and let
`E ∈ ℝ^(V×d)` denote the output projection. For vocabulary token `v`, write the
corresponding output row as `e_v`. Its logit is

```math
z_v = e_v^\top h.
```

For the stop token `s`,

```math
z_s = e_s^\top h.
```

### A.1 Hidden-state-only correction

With attention/MLP-only LoRA, the output row `e_s` is outside the directly
trainable adapter path. Let the adapted hidden state be

```math
h' = h + \Delta h.
```

Then the stop-logit change is exactly

```math
\Delta z_s
= e_s^\top \Delta h
= \lVert e_s\rVert \lVert\Delta h\rVert \cos\theta,
```

where `θ` is the angle between `e_s` and `Δh`.

By Cauchy–Schwarz,

```math
|\Delta z_s|
\le
\lVert e_s\rVert \lVert\Delta h\rVert.
```

This bound by itself does **not** show that a large stop-logit change is
impossible. It only makes explicit that, with a frozen output row, the stop logit
can change only through the hidden representation.

### A.2 Relative margins, not one logit in isolation

Generation depends on the softmax over all vocabulary logits. For stop token
`s` and a competing token `v`, define the pairwise margin

```math
m_{s,v}
=
z_s - z_v
=
(e_s - e_v)^\top h.
```

After the hidden-state perturbation,

```math
\Delta m_{s,v}
=
(e_s - e_v)^\top \Delta h.
```

Therefore, increasing stop probability requires more than making
`e_s^\top Δh` positive. For the competitors that carry substantial probability
mass at the terminal position, the update should improve

```math
(e_s - e_v)^\top \Delta h
```

in the stop token's favor.

This is the precise sense in which the correction is coupled to competing
vocabulary rows. It does **not** require `Δh` to be nearly orthogonal to
thousands of rows, nor is such near-orthogonality a necessary condition.

### A.3 What terminal NLL does and does not measure

For target stop token `s`,

```math
L_s
=
-\log p(s \mid x)
=
\log\sum_j e^{z_j} - z_s
=
\log\left(
1 + \sum_{v\ne s} e^{z_v-z_s}
\right).
```

Thus the x13 observation

```text
terminal NLL: 21.63 → 0.082
```

shows a very large improvement in the stop token's **softmax-relative
position**.

For example,

```math
e^{-0.082} \approx 0.92,
```

consistent with the measured B1 terminal probability.

However, the NLL decrease of roughly 21.5 nats is **not** equivalent to saying
that the stop logit itself increased by exactly 21.5. The denominator and
competing logits can change as well. The defensible statement is that the
relative logit geometry at the terminal position changed dramatically.

### A.4 Conditional representation steering

The adaptation must also be conditional.

The same model must preserve useful behavior at ordinary content positions while
favoring the stop token at the terminal state. Conceptually, the hidden-state
correction must behave like

```math
\Delta h(x)
\approx
\begin{cases}
\text{small or behavior-preserving}, & x \text{ is non-terminal},\\
\text{stop-margin-improving}, & x \text{ is terminal}.
\end{cases}
```

Thus hidden-state-only adaptation is solving two coupled requirements:

```text
increase terminal stop-vs-competitor margins
+
preserve non-terminal generation behavior
```

B1 shows that the model can satisfy this requirement on its exact training
rendering. The large drop in `<|im_end|>` probability under x12's alternative
rendering shows that the learned solution was not robust to that rendering
change. B2's weaker terminal probability under the rehearsal mixture is
consistent with additional distribution sensitivity, although the available
measurements do not identify a unique geometric cause.

### A.5 Direct output/embedding trainability

If the relevant output geometry is also trainable, write

```math
e_s' = e_s + \Delta e_s.
```

Then, even holding `h` fixed for illustration,

```math
\Delta z_s
=
\Delta e_s^\top h.
```

This provides a direct optimization path for adjusting the stop token's output
relationship to terminal hidden states rather than requiring all correction to
be expressed indirectly through upstream representation changes.

In the actual v2 adapter contract,

```yaml
modules_to_save: [lm_head, embed_tokens]
```

makes the selected modules trainable; it does **not** literally update only the
single stop-token row. The useful conceptual distinction is therefore:

```text
pre-fix:
low-rank upstream representation updates
        ↓
fixed output/embedding vocabulary geometry

post-fix:
low-rank upstream representation updates
        +
directly trainable output/embedding modules
```

The post-fix observations are consistent with this parameterization being much
more robust for the failure mode seen here: B2v2 restores
`p(<|im_end|>) = 0.918`, terminal NLL falls to 0.086, and PlayWorld probe
truncation falls from essentially 100% to ≈0%.

### A.6 Scope of the mechanism claim

The evidence supports the following claim:

> In this campaign, attention/MLP-only LoRA could learn a terminal-token
> workaround on the exact training rendering, but that workaround was brittle
> across rendering/data conditions and failed under downstream generation.
> Adding directly trainable output/embedding modules supplied a more direct
> parameterization and coincided with robust recovery of terminal-token
> likelihood and generation termination.

The evidence does **not** establish any of the following stronger claims:

- attention/MLP-only LoRA can never learn robust termination;
- a frozen vocabulary row is mathematically impossible to compensate for via
  hidden-state adaptation;
- the control-token row is universally or provably "undertrained";
- the rehearsal mixture is proven to fail because it geometrically disperses a
  single steering direction;
- the fix literally consists of moving only one vocabulary row.

Those stronger statements would require additional experiments or formal
analysis beyond the evidence recorded here.

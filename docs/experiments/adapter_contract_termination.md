# Failure Analysis — Chat-Template Termination under Adapter-Only Fine-Tuning (the "v2 Adapter Contract")

**Status:** DRAFT v0.9 — evidence chain complete for the *diagnosis*; two post-fix
confirmation numbers marked [VERIFY] below should be filled before v1.0.0.
**Scope:** Phase-1 SFT campaign (aw_05, runs of 2026-08-03) → adapter contract
revision applied uniformly to all v2 arms.
**Referenced from:** tech report §4.2.

---

## 1. Symptom

Early Phase-1 SFT adapters (B1 `b08a2c`, B2 `10ecce`; standard LoRA target set =
attention/MLP projections only) produced generations that **failed to emit the
chat template's terminal token** `<|im_end|>` (id 151645). Outputs ran past the
gold answer into degenerate continuations until the token cap. Content quality
was unaffected at the loss level — the failure was invisible in training loss
and catastrophic only at generation/eval time.

## 2. Audit chain (bottom-up)

### 2.1 x09 — termination audit (trace level)

Eval-run traces were audited for truncation, runaway tail repetition, and
prediction-length pathologies. (The x09 excerpt of record in this doc,
runs `a7a47a`/`cd89fd` of 2026-08-10, is a *post-fix* audit: truncation 0/1500,
runaway 0, mean prediction 257–261 chars — i.e., the clean state; the pre-fix
audits that motivated the chain showed the truncation/runaway signature.)

### 2.2 x10 — stop-logit probe, and a probe-integrity incident

x10 teacher-forces the gold context up to the terminal position and inspects the
next-token distribution:

- **Base model:** `p(<|im_end|>) = 0.0`, rank ~90k–140k. Mass goes to `\n`,
  `####`, `<|endoftext|>` — the GSM8K-style surface, *not* the ChatML stop.
  Consistent with Qwen3-**Base** never having trained the ChatML control tokens.
- **Adapter arms (as first probed):** B1 and B2 returned **byte-identical
  distributions** with garbage rare-token tops (p ≈ 0.001) and
  `p(<|im_end|>) ≈ 3×10⁻⁴`. Two independently trained adapters cannot produce
  identical logits — this exposed a **probe-side adapter-application fault**,
  not a model property. It motivated two integrity tools that became permanent
  infrastructure: **x11** (adapter integrity: hash + parameter-delta check) and
  **x12** (attested probe: proves *which* weights produced the measurement).
  x10's U1 verdict for the adapter arms is therefore **superseded** by x13.

### 2.3 x13 — trained-token NLL under attested loading (token level)

| arm | mean NLL (content) | p(`<|im_end|>`) at terminal | terminal NLL | rank |
|---|---|---|---|---|
| base | 1.261 | **0.000** | **21.63** | ~90k–127k |
| B1 (`b08a2c`) | 0.170 | **0.922** | 0.082 | 1 |
| B2 rehearsal (`10ecce`) | 0.338 | 0.119 | 3.44 | 2–17 |

Findings:

1. **Base:** content NLL is fine but the terminal `<|im_end|>` sits at NLL ≈ 21.6
   — the base model assigns essentially zero mass to the ChatML stop
   (hypothesis-K: the `<|im_end|>` lm_head/embedding rows are untrained in
   Qwen3-Base, and attention/MLP-only LoRA has no direct parameterization to
   lift a single vocab logit by ~20 nats).
2. **B1 under teacher forcing** *can* place rank-1 mass on the stop token on its
   own training rendering — i.e., adapter-only training moves the stop
   *conditionally*, through hidden-state steering rather than the (frozen)
   token rows.
3. **B2 (rehearsal mix)** degrades exactly this margin (0.92 → 0.12, terminal
   NLL 0.08 → 3.4): with mixed-distribution data the hidden-state workaround is
   brittle — the stop mass collapses on part of the samples (idx 3: p = 3×10⁻⁴).

## 3. Diagnosis

Adapter-only LoRA leaves the **embedding and lm_head rows of special/control
tokens frozen** at their (untrained, near-zero-mass) base values. The adapter
can compensate only *indirectly*, by steering hidden states toward the frozen
output row — a low-margin, data-distribution-sensitive mechanism (§2.3, B1 vs
B2). Under any distribution shift (rehearsal mixing, task transfer, sampling
noise at generation), the margin collapses and generation fails to terminate,
which the verifier's format gate converts into hard task failure.

## 4. Fix — the v2 adapter contract

Add the token-row parameters to the trainable set:

```yaml
peft:
  modules_to_save: [lm_head, embed_tokens]
```

applied **uniformly to all v2 arms** (single-variable discipline: the contract
is part of the environment, not a treatment). Post-fix state:

- Post-fix eval audits show 0 truncation / 0 runaway across 1,500 episodes
  (x09 on `a7a47a`, `cd89fd`), and all champion-lineage evals (B4v2 s42/s43/s44)
  hold truncation ≈ 0 with format-gate failures ≈ 0 (tech report §4.2).
- [VERIFY] x13 re-run on a v2-contract adapter (e.g., B1v2 `e6e83b`):
  p(`<|im_end|>`) at terminal = ___ , terminal NLL = ___ .
- [VERIFY] pre-fix generation-level termination/truncation rate of B1/B2
  (`b08a2c`/`10ecce`) eval runs, for the before/after table: ___ .

Residual: even under the v2 contract, the *direct-route* arm retains weaker
termination discipline than the two-stage route (A2v2: ~1,490/1,500 truncated
tails vs B4v2 ≈ 0 — tech report §4.2), indicating Phase-1 data volume also
contributes to stabilizing sequence-boundary behavior beyond the parameter fix.

## 5. Generalizable lessons

1. **Loss curves do not surface termination failure**; audit generation-side
   (truncation, tail repetition) as a first-class metric.
2. **Adapter target sets are a capability boundary, not a hyperparameter**:
   any behavior that requires moving specific vocab rows (control tokens, new
   special tokens) needs those rows trainable.
3. **Probes need attestation**: identical outputs across two arms is itself a
   diagnostic (x11/x12 exist because of this incident).
4. Base (non-instruct) checkpoints + chat templates is a known-sharp edge:
   confirm control-token trainability *before* the first long run.

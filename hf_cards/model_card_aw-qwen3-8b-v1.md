---
license: mit
base_model: Qwen/Qwen3-8B-Base
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - planning
  - reasoning
  - verifiable-environment
  - post-training
  - research-artifact
datasets:
  - m97j/aw-playworld
  - m97j/axiom-general-posttrain
language:
  - en
---

# aw-qwen3-8b-v1 — Axiom-World Protocol-v1 Champion (B4v2)

The final champion adapter of **Axiom-World protocol v1**: a pre-registered,
single-GPU comparison of post-training recipes for rule-grounded planning in a
fully verifiable toy world (PlayWorld). This is the **two-stage** recipe:
general-reasoning SFT (GSM8K + MATH-algebra) → PlayWorld task SFT.

- **Code & protocol:** https://github.com/m97j/axiom-world (tag `v1.0.0`)
- **Tech report:** docs/reports/axiom-world-tech-report-v1.md (TechRxiv DOI pending)
- **Run of record:** `20260814-023603--b4v2-playworld-sft-from-p1--s42--c56ed2`
  (full artifacts incl. resolved config & lineage: `m97j/aw-runs-b4`; 3-seed
  replications: `m97j/aw-runs-seeds`)

## Model details

| | |
|---|---|
| Base model | `Qwen/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Method | LoRA (attention + MLP projections) with the **v2 adapter contract**: `modules_to_save: [lm_head, embed_tokens]` |
| Parent (Phase 1) | `b1-general-sft-v2--s42--e6e83b` (sha-pinned; GSM8K .6 + MATH-algebra .4, 8k records) |
| Phase-2 data | 2,000 oracle-derived PlayWorld episodes, fingerprint `sha256:54fcb1d3…` |
| Adapter sha256 | `sha256:d4fcacddf21f758cdab904845ebdfee1eefde309c0edb6205bac64d5f07c76c8` |
| Precision / attn | BF16 / SDPA |

The `modules_to_save` choice is not incidental: attention/MLP-only LoRA on this
base model cannot reliably emit the chat template's terminal `<|im_end|>`
(100 % output truncation). Full failure analysis:
[`docs/experiments/adapter_contract_termination.md`](https://github.com/m97j/axiom-world/blob/main/docs/experiments/adapter_contract_termination.md).

## Evaluation (frozen PlayWorld suites, 300 episodes each, greedy decoding)

Pass rate, mean ± sd over training seeds {42, 43, 44}; control = direct task
tuning (A2v2) on the identical frozen data:

| Suite | **this model (two-stage)** | direct-tuning control |
|---|---|---|
| in-distribution | **.393 ± .013** | .186 ± .002 |
| template-OOD | **.349 ± .005** | .207 ± .003 |
| compositional-OOD | **.329 ± .022** | .139 ± .004 |
| rule-OOD | **.302 ± .005** | .192 ± .005 |
| adversarial (trap avoidance) | **.851 ± .002** | .651 ± .011 |

All 15 suite×seed deltas positive; paired permutation p ≤ 0.0004 each.

## Intended use & limitations

This is a **research artifact** supporting a comparative claim about
post-training recipes — not a general-purpose assistant and not a mastered
planner (absolute construction-suite pass rates are .30–.39 at this
budget/scale; see report §7). PlayWorld is synthetic; transfer to real agent
tasks is untested. Outputs are structured-JSON plans for PlayWorld episodes
rendered with the bundled chat template.

## How to load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Base",
    revision="49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    torch_dtype="bfloat16", attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(base, "m97j/aw-qwen3-8b-v1")
tok = AutoTokenizer.from_pretrained("m97j/aw-qwen3-8b-v1")
```

For verifier-scored evaluation on the frozen suites, use the repo's
`scripts/run_evaluation.py` (fingerprint-gated).

## Citation

```bibtex
@techreport{kim2026axiomworld,
  author = {Kim, Minjae},
  title  = {Axiom-World: A Pre-Registered Study of Two-Stage Post-Training
            for Rule-Grounded Planning in a Verifiable Toy World},
  year   = {2026},
  note   = {Technical report v1.0. Code: https://github.com/m97j/axiom-world}
}
```

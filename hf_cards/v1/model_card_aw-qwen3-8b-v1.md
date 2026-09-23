---
license: apache-2.0
base_model: Qwen/Qwen3-8B-Base
library_name: transformers
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

The standalone BF16 export of the final champion adapter of **Axiom-World protocol v1**: a pre-registered,
single-GPU comparison of post-training recipes for rule-grounded planning in a
fully verifiable toy world (PlayWorld). This is the **two-stage** recipe:
general-reasoning SFT (GSM8K + MATH-algebra) → PlayWorld task SFT.

- **Code & protocol:** [https://github.com/m97j/axiom-world](https://github.com/m97j/axiom-world) (tag `v1.0.0`)
- **Tech report:** docs/reports/v1/axiom-world-tech-report-v1.md (DOI: [10.5281/zenodo.22052149](https://doi.org/10.5281/zenodo.22052149))
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
[`docs/experiments/v1/adapter_contract_termination.md`](https://github.com/m97j/axiom-world/blob/main/docs/experiments/v1/adapter_contract_termination.md).

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

## Release layout and provenance

The repository root contains a standalone BF16 model in standard Transformers
safetensors format (5 GB maximum shard size). `adapter/` preserves the exact
original PEFT weights, config, tokenizer and chat template. The adapter identity
digest above combines the adapter config and weight-file digests; it is not the
SHA-256 of the safetensors file alone.

`provenance/manifest.json` records the pinned base and source revisions, library
versions, fixed engineering gates, and measured export verification results.
Original adapter-only revision:
`f33d2d16125e89eb38d4b668a2c20a6929ad3784` (tag `protocol-v1-adapter`).
History is preserved. Old root-adapter consumers must pin that revision or use
`subfolder="adapter"`. The model includes Qwen weights under Apache-2.0; the
Axiom-World code license remains MIT. See the original
[Qwen model card and license](https://huggingface.co/Qwen/Qwen3-8B-Base).

The table above reports the original adapter campaign, including three training
seeds; this public artifact is the seed-42 champion. Export checks are numerical
and standalone-loading checks, not rerun benchmark scores or an ARC result.
Greedy generation stops on end-of-text or `<|im_end|>`; apply the saved chat
template for instruction inputs. External benchmark prompting is a separate
choice to record with its results.

## How to load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "m97j/aw-qwen3-8b-v1"
# Pin the published commit from published.json for repeatable benchmarking.
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="bfloat16")
tok = AutoTokenizer.from_pretrained(repo)
```

To inspect the original training representation:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Base",
    revision="49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    torch_dtype="bfloat16", attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(base, repo, subfolder="adapter")
```

For verifier-scored evaluation on the frozen suites, use the code repository's
`scripts/common/run_evaluation.py` with the explicit adapter path and frozen data.
See `notebooks/protocol_v1/README.md` for historical replay boundaries.

## Citation

```bibtex
@techreport{kim2026axiomworld,
  author = {Kim, Minjae},
  title  = {Axiom-World: A Pre-Registered Study of Two-Stage Post-Training
            for Rule-Grounded Planning in a Verifiable Toy World},
  year   = {2026},
  doi    = {10.5281/zenodo.22052149},
  note   = {Technical report v1.0. Code: https://github.com/m97j/axiom-world}
}
```

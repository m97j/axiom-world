---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - math
  - reasoning
  - sft
  - preference-pairs
pretty_name: "Axiom-World Phase-1 general reasoning mixture"
---

# axiom-general-posttrain — Phase-1 general reasoning mixture (protocol v1)

The Phase-1 ("general warm-start") training mixture of the Axiom-World
project: a pre-registered, weighted blend of public math-reasoning corpora
with **dataset-derived gold answers only** (no LLM-generated labels).

- **Code:** [GitHub](https://github.com/m97j/axiom-world) (tag `v1.0.0`)

## Contents

| path | what |
|---|---|
| `v1/` | SFT mixture — GSM8K (weight 0.6) + MATH-algebra (weight 0.4), target 8,000 records, prompt-level dedup; 500-record held-out retention suite |
| `pref-v1/` | rejection-sampling preference pairs mined on the same distribution (used by the B2/B3 arms) |

Records are chat-rendered (Qwen3 template) with answers normalized to the
source datasets' gold format. Fingerprints are recorded in each consuming
run's `lineage.json`.

## Source datasets & licenses

- **GSM8K** (`openai/gsm8k`) — MIT license. Cobbe et al., 2021.
- **MATH**, algebra subject (`EleutherAI/hendrycks_math`) — MIT license.
  Hendrycks et al., 2021.

This repository redistributes preprocessed derivatives of the above under
their original MIT terms; all credit for the underlying problems goes to the
original authors.

## Citation

Cite the original datasets (Cobbe et al. 2021; Hendrycks et al. 2021) and the
Axiom-World tech report.

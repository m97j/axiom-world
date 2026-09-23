# Champion publication evidence — 2026-09-23

Published FP32 revision: [5b71e3e41d3c7630b4c6320aa67148ab653ac96d](https://huggingface.co/m97j/aw-qwen3-8b-v1/tree/5b71e3e41d3c7630b4c6320aa67148ab653ac96d).
These JSON files are copied unchanged from the user-downloaded release ZIP.
`verified.json` is the final payload inventory; `verified.before-task-audit.json`
records the payload before task evidence was attached. The Xet-timeout marker and
successful attempt marker are both retained. `published.json` records completion.

The ZIP is a **partial runtime evidence archive**, not a complete model backup.
It contains no `model/` directory, full weights, task prediction JSONL files, or
`merged_logits.safetensors`. The published revision contains the weights and
[evaluation/precision_audit](https://huggingface.co/m97j/aw-qwen3-8b-v1/tree/5b71e3e41d3c7630b4c6320aa67148ab653ac96d/evaluation/precision_audit).
Local availability of every remote payload file has not been independently checked.

Original downloaded notebook, both ZIPs, extracted command logs, the prior workspace
notebook and closeout verification are preserved under the Git-ignored
`_local/v1-notebook-closeout-20260924/`. All 29 supplied command logs are preserved
there. `archive_inventory.json` records supplied file digests and notebook checks.
The notebook preserves every downloaded code-cell object (source, outputs,
execution count and metadata), at initial closeout. Subsequent English cleanup translated four comments and two
unexecuted error-message strings; all outputs and execution counts remain unchanged.

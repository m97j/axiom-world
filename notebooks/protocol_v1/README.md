# Protocol v1 notebooks: historical record and champion export

> **B4v2 publication completed (2026-09-23).** `aw_07_b4.ipynb` now contains
> the web Colab execution archive, including failed attempts and successful recovery.
> Do not rerun publication or evaluations to remove error logs. See
> [publication evidence](evidence/champion_release_20260923/README.md).
> The release/recovery instructions below document the workflow as it developed;
> they are not pending actions for the already published checkpoint.

The historical experiment cells and their outputs are the as-run research record.
The B4 common header has been refreshed for publication; its earlier source and
output remain in Git history, while the new header has no execution output yet. They use
pre-refactor `scripts/` and `configs/experiments/` paths. Rewriting those paths
in-place would leave old outputs attached to code that was never executed.
They are therefore preserved. Do not use **Run All** for champion publication.

## Historical replay

The final pre-layout-refactor release is `v1.0.0`, resolving to commit
`511bbd7864abef9bed076cd03eb91f149b8f1747` (the annotated tag object has a
*different* SHA). `v1.0.1` already includes the layout refactor.

Use a separate directory and editable installation; PyPI publication is not needed:

```bash
git clone https://github.com/m97j/axiom-world.git axiom-world-v1-replay
cd axiom-world-v1-replay
git checkout --detach 511bbd7864abef9bed076cd03eb91f149b8f1747
python -m pip install -e . -r requirements/colab-g4.lock.txt
```

Run historical cells against that checkout instead of the historical unpinned
clone header. The final v1 code layout alone does not reproduce every earlier
cell: the notebooks deliberately include superseded runs, overwritten Hub paths,
incidents and remediations. Exact replay also requires each run's recorded code
commit, data bytes, model/adapter revisions, and compatible pinned runtime.
The B4v2 run records code commit `ff984e4bef63a2a04d3606b01e01da4152a91a21`.
Do not regenerate frozen data with a later generator and call it the same run.

## Publish the existing champion from VS Code (no training)

Run the **common header** in `aw_07_b4.ipynb`, then the three cells in its
**release appendix**. Do not run the intermediate historical training cells.
The header and release cells are unexecuted; existing experiment outputs are preserved.

The publisher is deliberately v1-specific and lives at
`scripts/publish/v1/publish_champion.py`. Its fixed champion/base/source pins are
release policy, not generic defaults. `scripts/common/fetch_run.py` remains shared.
The shared Hub transport accepts the legacy tag and commit message from the caller.
Future protocol champions need their own reviewed release policy; adding CLI
arguments alone would not make v1 probes, module checks and metadata universal.

1. Commit and push the release code and notebook from the local repository,
   excluding uncommitted CLB development. The header pins published FP32 implementation `1daef869d312dfc91a6d400d56e7851a421eeee8`, so the remote runtime loads the reviewed version rather than a moving main.
2. Open the **Windows local** notebook in VS Code and select its Colab kernel.
   Run the common header. `git clone` in that cell executes on Colab and creates
   `/content/axiom-world-fp32`, not a second clone on Windows. The editable package
   install runs on Colab too. No PyPI publication, ZIP upload, embedded source
   snapshot or browser notebook is needed. See the
   [official extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).
3. The public GitHub clone needs no token. The W&B environment-setting line is
   retained as a comment for training reproduction. HF authentication uses an
   existing environment token, Colab secrets, or a hidden prompt. A write-scoped
   HF token is needed for the last publish cell. The existing Colab dependency
   lock is used; image-owned torch/CUDA are not explicitly reinstalled.
4. Run the fetch/dry-run cell, then the prepare cell. It requires enough CPU RAM,
   GPU VRAM and disk for base cache, adapter, FP32 output and probe files. An 8B
   FP32 payload is roughly 32 GB before the saved adapter and caches; provision
   ample headroom. No paid runtime is selected or connected by these files.
5. Inspect the printed `verified.json`: 32 fixed raw/chat probes, adapter tensor
   attestation, saved-module preservation, FP32 merge, numerical gates and fresh
   offline standalone load. Default bounds are max absolute logit error 0.001,
   mean 0.0001, top-1 agreement 0.99, and exact 32-token generation agreement 0.90.
   These are conservative engineering thresholds, not established Qwen error
   bounds; a failure stops publication and needs investigation. Fresh reload must
   reproduce saved-model logits and generations exactly on the same device/backend.
   `verification.precision` separately compares historical BF16-base PEFT inference
   with FP32 inference (comparison bounds: max 0.5, mean 0.03, top-1 0.99,
   generation 0.90). These observations do not gate FP32 merge correctness and
   do not establish original benchmark equivalence. Retain failed BF16 outputs.
6. Review the precision comparison and model card before enabling the final cell.
   Set `PUBLISH_PRECISION_VARIANT=True` only to publish the disclosed FP32 variant.
   The CLI requires `--accept-precision-change` when this comparison fails. It verifies the payload hashes,
   preserves the adapter-only commit with `protocol-v1-adapter`, and atomically
   adds root merged weights plus `adapter/`, removing only the two old root
   adapter files. It guards the expected remote HEAD and retains Git history.
   A changed target stops the migration. No automatic upload retry is performed.
7. Save the notebook locally in VS Code, then commit and push **only that
   notebook** from your local terminal to record the run. No Git commit/push
   is performed by the Colab cells. Also retain `verified.json`,
   `published.json` and any failure diagnostics from the runtime before it expires.
   The public manifest includes verification metrics. Notebook outputs alone do
   not preserve the large local merge payload or all diagnostics.

If prepare fails, preserve its output directory and choose a new `--output` after
fixing the cause. Reusing the header checks the runtime checkout commit; a
different existing checkout is preserved and requires a new runtime/directory. If upload is interrupted, inspect the remote HEAD and
`upload_started.json`/`uploaded.json` before attempting recovery; do not remove
markers and blindly retry. `published.json` means remote sizes/hashes were
checked; it is not a new remote GPU inference test or a TPN scan result.

## Storage and compatibility

Keep weights and their standard index at root. The original tokenizer/chat
files are used, not a newly fetched base tokenizer. Both `lm_head` and
`embed_tokens` are full saved modules; the current adapter weight file is 2,663,975,192 bytes (about 2.66 GB), not a tiny LoRA-only payload.
A same-byte path move can reuse content-addressed storage, but quota accounting
and deduplication are Hub concerns; this tool makes no zero-extra-storage promise.
The merged weights are genuinely new bytes. Deleting paths does not erase
historical blobs. Never use `hf_prune_checkpoints.py --execute` or a history
rewrite to reorganize the champion. The old path-only prune CLI is now a
read-only run-checkpoint diagnostic because it cannot protect shared blob aliases.

The model card template switches to Transformers and Apache-2.0 metadata because
the standalone payload incorporates Qwen base weights. Code remains MIT. The
pinned base card declares Apache-2.0 but has no LICENSE file; prepare includes
the official Apache-2.0 text, the pinned base card, a modification notice and
any upstream NOTICE.
The original numerical results remain attributed to adapter evaluations.

References: [PEFT checkpoint and merge format](https://huggingface.co/docs/peft/developer_guides/checkpoint),
[Hub atomic commit API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api).

The BF16-base reference is an export diagnostic, not a replay of the original
v1 evaluation runtime. Original benchmark equivalence still needs separate evaluation.

## Stale notebook or kernel detection

If the prepare cell title says BF16, uses `/content/axiom-world`, or omits
`--dtype float32`, it is the previous notebook/kernel state. Preserve unsaved
outputs in a separate notebook copy, then reopen the updated local file and run
its common header before the release cells. GitHub push alone does not refresh
an open notebook buffer or a running kernel. No runtime reset is needed solely
for changing checkouts; the old checkout and failure evidence remain intact.
The updated header pins the published FP32 implementation and each release
command checks the checkout revision and release source cleanliness. Child
processes receive that checkout's src directory explicitly through PYTHONPATH.

## Recover the completed FP32 merge after the PEFT probe failure

The `Standalone verification forbids PEFT` failure during `find_spec` was a
verification guard defect, not a failed FP32 merge. The corrected guard permits
package discovery and rejects actual PEFT module execution. A fresh offline
process must still load the standalone model without importing PEFT.

For the existing `runs/champion-v1-release-fp32` directory, use
`--verify-existing --dtype float32 --device cuda --output runs/champion-v1-release-fp32`.
This validates pinned paths/source/probes and a successful merge report, checks
payload/evidence hashes before and after reload, and finalizes the manifest and
receipt only after standalone verification passes. It refuses an existing
standalone report, verified receipt or upload marker; investigate those states
rather than deleting evidence. Original merge evidence is retained. The manifest
identifies the reported original merge implementation separately from recovery
code; no pre-failure hash attestation is retroactively claimed.

The notebook selects recovery when its output directory exists; otherwise it
selects prepare. An incomplete directory is rejected, not silently remerged.
Preserve the current Colab runtime and files to avoid repeating the 8B merge.

The header permits updating the known previous clean FP32 checkout to the pinned
recovery implementation. It performs no reset or clean; saved runs stay in place.

## Task comparison before publication

Run `scripts/publish/v1/compare_champion.py --release runs/champion-v1-release-fp32
--output runs/champion-v1-comparison --batch-size 4` after standalone verification.
The output directory must be new. The command downloads (does not regenerate)
all five 300-episode suites from aw-playworld revision
`20a669ce2782546572dc6445e8b5cba62577ac34` and validates their original fingerprints.
It archives v1 code `ff984e4bef63a2a04d3606b01e01da4152a91a21` into the output
folder and uses its schemas, config and verifier in isolated child processes.
Reference is BF16 base + original adapter; candidate is saved FP32 standalone.
Before GPU allocation, both tokenizers must render identical prompt token IDs.
Both arms use the v1 empty-think opener, greedy SDPA and 1024-token limit.
Batch size 4 is explicit for FP32 memory headroom; the historical notebook used
100, so this is a paired current-stack comparison, not historical exact replay.

This generates 3000 completions, substantially more work than the 32 probes.
No training, model overwrite, Hub upload, or automatic publication occurs.
Progress is printed every batch; per-episode outputs/verdicts are flushed to
reference.jsonl and candidate.jsonl. An interruption leaves partial evidence,
no completion report, and does not silently resume or rerun it. Choose a new
comparison directory only after investigating any failure. Keep the runtime
and all reports until saved elsewhere.

Review comparison.json: suite pass rates, paired regressions/improvements,
format failures, truncation and exact-text agreement. Equal mean success alone
does not establish equivalence, and text changes alone do not prove regression.
No acceptance margin is invented after seeing results; publication remains a
separate reviewed decision. The evaluation does not modify verified.json or the
release payload. Bring the report back for review before enabling the final cell.

After successful FP32 prepare, run the updated header and the paired-comparison
cell only. Do not rerun prepare/recovery on an already verified directory. The
final publication cell now requires a completed comparison bound to the exact
verified receipt, plus explicit review of both task results and precision drift.

## BF16 deployment audit (decision fixed before seeing BF16 task results)

Preserve the verified FP32 release and its completed comparison. Derive a separate
BF16 candidate with `derive_bf16_candidate.py`; never modify or round the FP32
source in place. It casts the saved merged FP32 weights, reports the 32-probe
precision drift without relabelling a failed equivalence check as a pass, and
requires exact BF16 save/reload replay with no PEFT import. Its receipt status is
`verified_candidate`, which the publisher deliberately does not accept.

Run the same full paired task comparison with BF16 as candidate. This reruns the
original adapter as a control (3000 completions), then `review_precision.py`
combines both precision reports, verifies input identity and summaries, records
whether the reference repeated exactly, and adds exploratory paired-bootstrap
95% intervals (10000 resamples, seed 42; unadjusted, not equivalence tests).

Deployment rule for this audit: prefer BF16 only if every suite has at least the
reference adapter's observed pass count, and no increase in schema failures or
truncation, with the repeated reference matching the earlier control. Otherwise
retain FP32 as the already tested, disclosed precision variant, including its
observed compositional-OOD regression. If the control itself changes, do not issue
an automatic precision recommendation. This is a conservative deployment rule,
not a preregistered scientific hypothesis or proof of population non-inferiority.
It was chosen after the FP32 results but before the BF16 task results; report both.
Do not cherry-pick a new champion or compare a seed-42 candidate against a 3-seed
historical mean as an acceptance threshold.

The public release remains blocked pending final model-card/evidence packaging
and review of precision_review.json. This audit performs no training, conversion
uploads, or email sending. Once precision is selected, publish the standard root
checkpoint with the original adapter and disclose the exact dtype and both task
comparisons in the release evidence, then reply with the published repo revision.
Keep this Colab runtime: neither GitHub nor the notebook stores its model files.


### BF16 cast/reload correction

The initial candidate path called `model.to(bfloat16)`, which also rounded
non-persistent RoPE frequency buffers. Those buffers are omitted from the saved
checkpoint and regenerated by the native loader. The pre-save and reload probes
could therefore evaluate different runtime buffer values despite identical weights.
The corrected conversion preserves native non-persistent buffers and retains the
exact standalone replay gate. Old drift observations may include this buffer effect;
they must not be described as isolated evidence of weight rounding alone.
Use `runs/champion-v1-release-bf16-candidate-native-buffers` for the corrected attempt.
Preserve the failed directory and its standalone.json. Keep the current runtime;
rerun the updated header, derive cell and BF16 comparison cell, not FP32 prepare.
A failed standalone check now prints its measured differences before raising.

### Completed BF16 run with a different batch size

The observed BF16 audit used batch size 256, while the earlier FP32 audit used 4.
Strict review still rejects this as a matched precision comparison. The explicit
`--independent-controls` review mode retains both within-run comparisons, marks
cross-precision comparability false, and never automatically recommends a dtype.
The completed BF16 run is useful evidence of degradation against its own control;
it is not an incomplete evaluation and need not be rerun to reject that candidate.

For this release, the explicit deployment decision is to retain the independently
verified FP32 variant, including its disclosed compositional-OOD regression.
This decision follows the unmatched BF16 results and is not the original automatic
matched-control rule. It does not claim FP32 superiority under matched conditions.
`finalize_fp32_audit.py` consumes completed reports/traces with no GPU execution,
checks receipt binding and reaggregates paired verdicts, then attaches both runs
and their limitations to the staged model card. It preserves all model weights
and the original receipt, and updates the payload receipt only after verification.
Run this once, after the header, then review and enable the final publication cell.
Do not rerun either compare cell, derive, or prepare. The failed review traceback
should remain as execution history. Do not disconnect before saving runtime evidence.

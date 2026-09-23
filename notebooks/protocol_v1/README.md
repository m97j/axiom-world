# Protocol v1 notebooks: historical record and champion export

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
   excluding uncommitted CLB development. The header asks for the pushed full implementation commit SHA (`git rev-parse HEAD`), so the remote runtime loads the reviewed version rather than a moving main.
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

# Protocol v1 notebooks: historical record and champion export

The existing cells and their outputs are the as-run research record. They use
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

Only execute the **new release appendix** at the bottom of `aw_07_b4.ipynb`.
The new cells are unexecuted. They replace the separately reported original
`fetch_run.py; publish_champion.py --dry-run; publish_champion.py` cell with
current paths and explicit prepare/publish stages.

A local notebook can keep its outputs in VS Code while its kernel executes on
Colab. Its local project files are **not automatically present on that server**.
The official extension supports file upload via Explorer's **Upload to Colab**
(see [official user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)).

1. On your computer, from this repository, archive the reviewed local commit:

   ```bash
   mkdir -p _local
   git archive --format=zip --output=_local/v1-release-source.zip HEAD
   ```

   Use the v1 publication commit created for this change. Check `git show --stat
   HEAD` first if you have made other commits. `git archive HEAD` includes committed
   files only, excluding the uncommitted CLB work, tokens and run weights.
2. Open the **local** `aw_07_b4.ipynb`, choose your Colab kernel, then upload that
   ZIP to the runtime. Set `ARCHIVE` in the new setup cell to its remote path.
   Do not run the old common header or clone unpushed GitHub main.
3. Run the new setup cell. It extracts to a new directory, installs the project
   and existing Colab lock, and uses `HF_TOKEN` from the runtime environment or a
   hidden prompt. It never prints or saves the token. A write-scoped token is
   needed only for the last publish cell. Never reinstall image-owned torch/CUDA.
4. Run the fetch/dry-run cell, then the prepare cell. It requires enough CPU RAM,
   GPU VRAM and disk for base cache, adapter, BF16 output and probe files. An 8B
   BF16 payload is roughly 16 GB before the saved adapter and caches; provision
   ample headroom. No paid runtime is selected or connected by these files.
5. Inspect the printed `verified.json`: 32 fixed raw/chat probes, adapter tensor
   attestation, saved-module preservation, BF16 merge, numerical gates and fresh
   offline standalone load. Default bounds are max absolute logit error 0.5,
   mean 0.03, top-1 agreement 0.99, and exact 32-token generation agreement 0.90.
   These are conservative engineering thresholds, not established Qwen error
   bounds; a failure stops publication and needs investigation. Fresh reload must
   reproduce saved-model logits and generations exactly on the same device/backend.
6. Run the separate publish cell only when ready. It verifies the payload hashes,
   preserves the adapter-only commit with `protocol-v1-adapter`, and atomically
   adds root merged weights plus `adapter/`, removing only the two old root
   adapter files. It guards the expected remote HEAD and retains Git history.
   A changed target stops the migration. No automatic upload retry is performed.
7. Save the notebook locally in VS Code. Also download `verified.json`,
   `published.json` and any failure diagnostics from the runtime before it expires.
   The public manifest includes verification metrics. Notebook outputs alone do
   not preserve the large local merge payload or all diagnostics.

If prepare fails, preserve its output directory and choose a new `--output` after
fixing the cause. If upload is interrupted, inspect the remote HEAD and
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

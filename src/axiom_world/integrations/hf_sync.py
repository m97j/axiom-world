"""Hugging Face Hub persistence for ephemeral Colab sessions (protocol §3).

Colab /content is volatile and sessions cap at 24h. Durable state therefore
lives on the Hub:

- ``upload_directory``   : one-shot folder upload (datasets, final adapters,
                           run artifact dirs).
- ``HFCheckpointSync``   : a transformers ``TrainerCallback`` that pushes the
                           newest checkpoint + run artifacts to a (private)
                           model repo on every save, and prunes older
                           checkpoint dirs locally to bound disk usage.
- ``download_latest_checkpoint`` : restore the newest ``checkpoint-*`` from a
                           repo into a local dir for ``--resume-from``.

All hub calls are lazy imports; contract environments without
huggingface_hub still import this module (callback construction requires it).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def upload_directory(
    local_dir: Path | str,
    repo_id: str,
    path_in_repo: str = "",
    repo_type: str = "model",
    private: bool = True,
    commit_message: str = "axiom-world sync",
) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    return f"hf://{repo_type}/{repo_id}/{path_in_repo}".rstrip("/")


def download_run_directory(
    repo_id: str,
    run_id: str,
    workspace: Path | str = ".",
    repo_type: str = "model",
) -> Path:
    """Fetch a persisted ``runs/<run_id>/`` tree from a repo (eval runs).

    Read-path counterpart of the ``path_in_repo=f"runs/{run_id}"`` upload in
    run_evaluation.py. Training-run artifacts live at the repo ROOT under
    ``artifacts/`` (one run per repo) and are handled by fetch_run.py's
    adapter path; eval runs are nested under ``runs/`` (many per repo), so
    they need their own targeted snapshot.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id, repo_type=repo_type, local_dir=str(workspace),
        allow_patterns=[f"runs/{run_id}/*"],
    )
    return Path(workspace) / "runs" / run_id


def download_latest_checkpoint(repo_id: str, local_dir: Path | str) -> Path | None:
    """Fetch the highest-step checkpoint-* tree from a model repo, or None."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="model")
    steps: dict[int, str] = {}
    for name in files:
        match = _CHECKPOINT_RE.match(name.split("/")[0])
        if match:
            steps[int(match.group(1))] = name.split("/")[0]
    if not steps:
        return None
    newest = steps[max(steps)]
    snapshot_download(
        repo_id, repo_type="model", local_dir=str(local_dir),
        allow_patterns=[f"{newest}/*"],
    )
    return Path(local_dir) / newest


def _make_callback_base() -> Any:
    from transformers import TrainerCallback

    return TrainerCallback


class HFCheckpointSync:
    """Factory: build the TrainerCallback lazily (transformers import)."""

    def __init__(
        self,
        repo_id: str,
        run_id: str,
        artifacts_dir: Path | None = None,
        keep_local_checkpoints: int = 2,
        private: bool = True,
        keep_hub_checkpoints: int = 1,
    ) -> None:
        self.repo_id = repo_id
        self.run_id = run_id
        self.artifacts_dir = artifacts_dir
        self.keep_local_checkpoints = keep_local_checkpoints
        self.private = private
        self.keep_hub_checkpoints = keep_hub_checkpoints

    def prune_hub_checkpoints(self) -> None:
        """Permanently delete this run's hub checkpoint LFS blobs, keeping
        the newest ``keep_hub_checkpoints`` steps (0/negative disables)."""
        if self.keep_hub_checkpoints <= 0:
            return
        from huggingface_hub import HfApi

        api = HfApi()
        by_step: dict[int, list[Any]] = {}
        for info in api.list_lfs_files(self.repo_id):
            # hub layout (see on_save): checkpoints live at ROOT as
            # "checkpoint-<step>/..." and successive runs overwrite the same
            # paths — which is exactly how stale hidden revisions pile up.
            match = _CHECKPOINT_RE.match(info.filename)
            if match:
                by_step.setdefault(int(match.group(1)), []).append(info)
        drop_steps = sorted(by_step)[: -self.keep_hub_checkpoints]
        stale = [info for step in drop_steps for info in by_step[step]]
        if stale:
            api.permanently_delete_lfs_files(
                self.repo_id, stale, rewrite_history=True)
            print(f"[hf_sync] pruned {len(stale)} stale hub checkpoint blobs "
                  f"(steps {drop_steps}, kept {sorted(by_step)[-self.keep_hub_checkpoints:]})")

    def build(self) -> Any:
        base = _make_callback_base()
        sync = self

        class _Callback(base):  # type: ignore[misc, valid-type]
            def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                output_dir = Path(args.output_dir)
                checkpoints = sorted(
                    (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
                    key=lambda p: int(p.name.split("-")[-1]),
                )
                if not checkpoints:
                    return
                newest = checkpoints[-1]
                upload_directory(
                    newest, sync.repo_id,
                    path_in_repo=f"{newest.name}",
                    private=sync.private,
                    commit_message=f"{sync.run_id}: {newest.name}",
                )
                if sync.artifacts_dir and Path(sync.artifacts_dir).is_dir():
                    upload_directory(
                        sync.artifacts_dir, sync.repo_id,
                        path_in_repo="artifacts",
                        private=sync.private,
                        commit_message=f"{sync.run_id}: artifacts @ {newest.name}",
                    )
                # bound local disk: keep only the newest K checkpoints
                import shutil

                for old in checkpoints[: -sync.keep_local_checkpoints]:
                    shutil.rmtree(old, ignore_errors=True)
                # bound HUB storage (2026-08-16 quota incident): HF quota
                # counts the LFS blobs of ALL revisions, so overwritten or
                # merely 'deleted' checkpoints keep billing. Permanently
                # delete hub checkpoint blobs older than the newest
                # keep_hub_checkpoints. Best-effort: a failure here must
                # never kill the training run.
                try:
                    sync.prune_hub_checkpoints()
                except Exception as exc:  # noqa: BLE001 - telemetry only
                    print(f"[hf_sync] hub checkpoint prune skipped: {exc!r}")

        return _Callback()

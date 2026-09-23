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

Repository-owned Hub I/O is centralized here. Library-managed model/dataset
loading remains in its native SDK. Legacy optional-sync interfaces below are
preserved; they do not yet enforce the forthcoming durable-output policy or
authorize automatic local cache eviction.

All hub calls are lazy imports; contract environments without
huggingface_hub still import this module (callback construction requires it).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def download_file(
    repo_id: str, filename: str, *, repo_type: str = "model", revision: str | None = None,
) -> Path:
    """Central cached read. Unpinned reads are legacy diagnostics, not canonical evidence."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                repo_type=repo_type, revision=revision))


def download_pinned_file(
    repo_id: str, filename: str, *, revision: str, expected_sha256: str,
    repo_type: str = "model",
) -> Path:
    """Exact revision + byte-checked consumption; preserve the shared cache on failure."""
    from axiom_world.core.fingerprints import fingerprint_file

    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("An exact 40-hex repository revision is required")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("An explicit sha256: digest is required")
    path = download_file(repo_id, filename, repo_type=repo_type, revision=revision)
    actual = fingerprint_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Downloaded file SHA mismatch: expected {expected_sha256}, got {actual}")
    return path


def download_snapshot(
    repo_id: str, *, local_dir: Path | str, allow_patterns: list[str],
    repo_type: str = "model", revision: str | None = None,
) -> Path:
    """Materialize an explicit subset. Caller still validates run/artifact identity."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type,
                                  local_dir=str(local_dir), allow_patterns=allow_patterns,
                                  revision=revision))


def list_repository_files(repo_id: str, *, repo_type: str = "model") -> list[str]:
    from huggingface_hub import HfApi

    return list(HfApi().list_repo_files(repo_id, repo_type=repo_type))


def list_storage_objects(repo_id: str, *, repo_type: str = "model") -> list[Any]:
    from huggingface_hub import HfApi

    return list(HfApi().list_lfs_files(repo_id, repo_type=repo_type))


def delete_storage_objects(
    repo_id: str, objects: list[Any], *, repo_type: str = "model", rewrite_history: bool = True,
) -> None:
    """Legacy explicit execution boundary. Calling this permanently deletes objects.

    Centralization does not make the v1 caller's path-only selection CLB-safe.
    The object-aware planner is separate and does not call this function.
    """
    from huggingface_hub import HfApi

    HfApi().permanently_delete_lfs_files(
        repo_id, objects, repo_type=repo_type, rewrite_history=rewrite_history)


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
    *, revision: str | None = None,
) -> Path:
    """Fetch a persisted ``runs/<run_id>/`` tree from a repo (eval runs).

    Read-path counterpart of the ``path_in_repo=f"runs/{run_id}"`` upload in
    run_evaluation.py. Training-run artifacts live at the repo ROOT under
    ``artifacts/`` (one run per repo) and are handled by fetch_run.py's
    adapter path; eval runs are nested under ``runs/`` (many per repo), so
    they need their own targeted snapshot.
    """
    download_snapshot(
        repo_id, repo_type=repo_type, local_dir=str(workspace),
        allow_patterns=[f"runs/{run_id}/*"], revision=revision,
    )
    return Path(workspace) / "runs" / run_id


def download_latest_checkpoint(repo_id: str, local_dir: Path | str) -> Path | None:
    """Fetch the highest-step checkpoint-* tree from a model repo, or None."""
    files = list_repository_files(repo_id, repo_type="model")
    steps: dict[int, str] = {}
    for name in files:
        match = _CHECKPOINT_RE.match(name.split("/")[0])
        if match:
            steps[int(match.group(1))] = name.split("/")[0]
    if not steps:
        return None
    newest = steps[max(steps)]
    download_snapshot(
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
        by_step: dict[int, list[Any]] = {}
        for info in list_storage_objects(self.repo_id):
            # hub layout (see on_save): checkpoints live at ROOT as
            # "checkpoint-<step>/..." and successive runs overwrite the same
            # paths — which is exactly how stale hidden revisions pile up.
            match = _CHECKPOINT_RE.match(info.filename)
            if match:
                by_step.setdefault(int(match.group(1)), []).append(info)
        drop_steps = sorted(by_step)[: -self.keep_hub_checkpoints]
        stale = [info for step in drop_steps for info in by_step[step]]
        if stale:
            delete_storage_objects(
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


def repository_revision(repo_id: str, revision: str = "main") -> str:
    """Resolve a model ref once before reading a multi-file snapshot."""
    from huggingface_hub import HfApi

    sha = HfApi().repo_info(repo_id, repo_type="model", revision=revision).sha
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise ValueError("Hub did not return an exact revision")
    return sha


def release_inventory(repo_id: str, revision: str) -> dict[str, dict]:
    """Read exact-revision file sizes and LFS hashes; never download weights."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, repo_type="model", revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Release inventory revision mismatch")
    result = {}
    for entry in info.siblings:
        lfs = entry.lfs
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        result[entry.rfilename] = {"size": entry.size,
                                   "sha256": "sha256:" + digest if digest else None}
    return result


def commit_model_release(*, repo_id: str, stage: Path, expected_head: str,
                         legacy_revision: str, delete_paths: list[str]) -> str:
    """History-preserving atomic migration; no storage-object deletion or retries."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    if any(re.fullmatch(r"[0-9a-f]{40}", ref) is None for ref in [expected_head, legacy_revision]):
        raise ValueError("Exact target and legacy revisions required")
    if not set(delete_paths).issubset({"adapter_config.json", "adapter_model.safetensors"}):
        raise ValueError("Only legacy root adapter files may be removed")
    api = HfApi()
    if api.repo_info(repo_id, repo_type="model").sha != expected_head:
        raise ValueError("Target changed; prepare and review a new release")
    tags = {tag.name: tag.target_commit for tag in api.list_repo_refs(repo_id, repo_type="model").tags}
    tag_name = "protocol-v1-adapter"
    if tag_name in tags and tags[tag_name] != legacy_revision:
        raise ValueError("Legacy tag already points elsewhere")
    if tag_name not in tags:
        api.create_tag(repo_id, tag=tag_name, revision=legacy_revision, repo_type="model")
    operations = [CommitOperationDelete(path_in_repo=path) for path in delete_paths]
    operations.extend(CommitOperationAdd(path_in_repo=p.relative_to(stage).as_posix(),
                                         path_or_fileobj=str(p))
                      for p in sorted(stage.rglob("*")) if p.is_file())
    commit = api.create_commit(repo_id=repo_id, repo_type="model", revision="main",
                              parent_commit=expected_head, operations=operations,
                              commit_message="Publish verified BF16 v1 champion and preserve adapter")
    return commit.oid

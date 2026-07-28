"""Canonical run directory layout. One layout; no synonyms.

runs/{run_id}/
├── manifest.json
├── checkpoints/
├── artifacts/
├── logs/
└── events/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def events_dir(self) -> Path:
        return self.root / "events"

    def artifact(self, filename: str) -> Path:
        return self.artifacts_dir / filename

    def create(self) -> RunPaths:
        for directory in (
            self.root,
            self.checkpoints_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.events_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


# Required artifact filenames (protocol §11). A run missing any of these is
# not 'completed'; see context.assert_run_complete().
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "resolved_config.yaml",
    "environment_manifest.json",
    "dataset_manifest.json",
    "git_state.json",
    "run_card.json",
    "metrics.json",
    "lineage.json",
    "checkpoint_pointer.json",
)


def run_root(workspace: Path | str, run_id: str) -> RunPaths:
    return RunPaths(Path(workspace) / "runs" / run_id)

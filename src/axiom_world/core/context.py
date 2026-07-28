"""ExperimentContext — THE single run-scoped object.

Owns: run identity, validated config, run directories, manifest transitions,
artifact registration, and the run-completeness gate (protocol §11).
There is no ``RunContext``; any code importing one is out of contract.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from axiom_world.core.enums import ArtifactKind, RunStatus
from axiom_world.core.errors import ArtifactError, RunContractError
from axiom_world.core.fingerprints import fingerprint_file
from axiom_world.core.paths import REQUIRED_ARTIFACTS, RunPaths, run_root
from axiom_world.core.schemas import ExperimentConfig, RunCard

_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING}),
    RunStatus.RUNNING: frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.ABORTED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.ABORTED: frozenset(),
}


def _utcnow() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def make_run_id(experiment_name: str, seed: int) -> str:
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9-]", "-", experiment_name.lower()).strip("-")
    return f"{stamp}--{slug}--s{seed}--{uuid.uuid4().hex[:6]}"


class ExperimentContext:
    def __init__(
        self,
        config: ExperimentConfig,
        config_fingerprint: str,
        workspace: Path | str,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.config_fingerprint = config_fingerprint
        self.run_id = run_id or make_run_id(config.experiment_name, config.runtime.seed)
        self.paths: RunPaths = run_root(Path(workspace), self.run_id)
        self._status = RunStatus.PENDING
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._created_at = _utcnow()

    # -- lifecycle ----------------------------------------------------------

    @property
    def status(self) -> RunStatus:
        return self._status

    def initialize(self, resolved_mapping: dict[str, Any]) -> None:
        """Create directories and persist the immutable resolved config."""
        self.paths.create()
        config_path = self.paths.artifact("resolved_config.yaml")
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved_mapping, handle, sort_keys=True, allow_unicode=True)
        self.register_artifact("resolved_config.yaml", ArtifactKind.CONFIG)
        self._write_manifest()

    def transition(self, new_status: RunStatus) -> None:
        allowed = _ALLOWED_TRANSITIONS[self._status]
        if new_status not in allowed:
            raise RunContractError(
                f"Illegal status transition {self._status.value} -> {new_status.value} "
                f"(allowed: {sorted(s.value for s in allowed)})"
            )
        if new_status is RunStatus.COMPLETED:
            self.assert_run_complete()
        self._status = new_status
        self._write_manifest()

    # -- artifacts ----------------------------------------------------------

    def write_json_artifact(self, filename: str, payload: Any, kind: ArtifactKind) -> Path:
        path = self.paths.artifact(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        self.register_artifact(filename, kind)
        return path

    def register_artifact(self, filename: str, kind: ArtifactKind) -> None:
        path = self.paths.artifact(filename)
        if not path.is_file():
            raise ArtifactError(f"Cannot register missing artifact: {path}")
        self._artifacts[filename] = {
            "kind": kind.value,
            "sha256": fingerprint_file(path),
            "registered_at": _utcnow(),
        }
        self._write_manifest()

    # -- gates ---------------------------------------------------------------

    def assert_run_complete(self) -> None:
        """Protocol §11: a run without required artifacts does not exist."""
        missing = [
            name
            for name in REQUIRED_ARTIFACTS
            if name not in self._artifacts or not self.paths.artifact(name).is_file()
        ]
        if missing:
            raise RunContractError(
                "Run cannot be marked completed; missing required artifacts: "
                + ", ".join(missing)
            )
        violations = self.config.validate_canonical()
        if violations and self.config.runtime.environment_policy == "strict":
            raise RunContractError(
                "Canonical-contract violations at completion: " + " | ".join(violations)
            )

    # -- persistence ----------------------------------------------------------

    def run_card(self) -> RunCard:
        return RunCard(
            run_id=self.run_id,
            experiment_name=self.config.experiment_name,
            track=self.config.track,
            phase=self.config.phase,
            objective=self.config.objective,
            protocol_version=self.config.project.protocol_version,
            seed=self.config.runtime.seed,
            status=self._status,
            config_fingerprint=self.config_fingerprint,
            created_at=self._created_at,
            updated_at=_utcnow(),
        )

    def _write_manifest(self) -> None:
        if not self.paths.root.exists():
            return
        manifest = {
            "run_id": self.run_id,
            "status": self._status.value,
            "experiment_name": self.config.experiment_name,
            "track": self.config.track.value,
            "phase": self.config.phase.value,
            "objective": self.config.objective.value,
            "protocol_version": self.config.project.protocol_version,
            "seed": self.config.runtime.seed,
            "config_fingerprint": self.config_fingerprint,
            "created_at": self._created_at,
            "updated_at": _utcnow(),
            "artifacts": self._artifacts,
        }
        with self.paths.manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)

"""Axiom-World CLI — contract-level commands.

Commands:
  validate-config  Compose + validate a recipe; print fingerprint and
                   canonical-contract violations (exit 1 on violation).
  show-config      Print the fully composed mapping as YAML.
  runtime-audit    Print the environment manifest as JSON.
  init-run         Create the run directory, persist resolved config,
                   environment manifest, git state, run card, lineage.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from axiom_world.core.config_loader import resolve
from axiom_world.core.context import ExperimentContext
from axiom_world.core.enums import ArtifactKind
from axiom_world.core.errors import AxiomError
from axiom_world.core.lineage import build_lineage_record
from axiom_world.runtime.audit import collect_environment_manifest, enforce_environment


def _git_state() -> dict[str, object]:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except Exception:
            return None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def cmd_validate_config(args: argparse.Namespace) -> int:
    config, fingerprint, _ = resolve(args.config, args.override)
    violations = config.validate_canonical()
    print(f"experiment : {config.experiment_name}")
    print(f"track      : {config.track.value}")
    print(f"phase      : {config.phase.value}")
    print(f"objective  : {config.objective.value}")
    print(f"fingerprint: {fingerprint}")
    if violations:
        print("CANONICAL-CONTRACT VIOLATIONS:")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("canonical  : OK")
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    _config, fingerprint, mapping = resolve(args.config, args.override)
    print(yaml.safe_dump(mapping, sort_keys=True, allow_unicode=True))
    print(f"# fingerprint: {fingerprint}")
    return 0


def cmd_runtime_audit(args: argparse.Namespace) -> int:
    print(json.dumps(collect_environment_manifest(), indent=2, sort_keys=True))
    return 0


def cmd_init_run(args: argparse.Namespace) -> int:
    config, fingerprint, mapping = resolve(args.config, args.override)
    violations = config.validate_canonical()
    if violations and config.runtime.environment_policy == "strict":
        for violation in violations:
            print(f"violation: {violation}", file=sys.stderr)
        return 1

    manifest = collect_environment_manifest()
    env_violations = enforce_environment(config.runtime, manifest)
    for violation in env_violations:
        print(f"environment warning: {violation}", file=sys.stderr)

    ctx = ExperimentContext(config, fingerprint, Path(args.workspace))
    ctx.initialize(mapping)
    ctx.write_json_artifact("environment_manifest.json", manifest, ArtifactKind.MANIFEST)
    ctx.write_json_artifact("git_state.json", _git_state(), ArtifactKind.MANIFEST)
    ctx.write_json_artifact(
        "run_card.json", ctx.run_card().model_dump(mode="json"), ArtifactKind.MANIFEST
    )
    lineage = build_lineage_record(ctx.run_id, config, fingerprint)
    ctx.write_json_artifact(
        "lineage.json", lineage.model_dump(mode="json"), ArtifactKind.MANIFEST
    )
    print(f"run_id     : {ctx.run_id}")
    print(f"run_dir    : {ctx.paths.root}")
    print(f"fingerprint: {fingerprint}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="axiom", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="Path to experiment recipe YAML.")
        p.add_argument(
            "--override",
            action="append",
            default=[],
            metavar="KEY.PATH=VALUE",
            help="Dotlist override; repeatable.",
        )

    p_validate = sub.add_parser("validate-config", help="Validate a recipe.")
    _common(p_validate)
    p_validate.set_defaults(func=cmd_validate_config)

    p_show = sub.add_parser("show-config", help="Print composed config.")
    _common(p_show)
    p_show.set_defaults(func=cmd_show_config)

    p_audit = sub.add_parser("runtime-audit", help="Print environment manifest.")
    p_audit.set_defaults(func=cmd_runtime_audit)

    p_init = sub.add_parser("init-run", help="Initialize a run directory.")
    _common(p_init)
    p_init.add_argument("--workspace", default=".", help="Workspace root (default: cwd).")
    p_init.set_defaults(func=cmd_init_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except AxiomError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

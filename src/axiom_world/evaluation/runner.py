"""Evaluation runner — generator-agnostic, verifier-authoritative.

generator(prompt: str) -> str is any callable (HF pipeline, vLLM, mock).
The runner writes per-sample traces (evaluation.jsonl) and a suite summary
(evaluation_summary.json) into the run's artifact directory, satisfying two
of the protocol §11 required artifacts.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from axiom_world.core.context import ExperimentContext
from axiom_world.core.enums import ArtifactKind
from axiom_world.data.bundle import DataBundle
from axiom_world.evaluation.metrics import summarize_suite
from axiom_world.verifiers.base import Verifier


class EvaluationRunner:
    def __init__(self, verifier: Verifier, generator: Callable[[str], str]) -> None:
        self.verifier = verifier
        self.generator = generator

    def run(self, bundle: DataBundle, context: ExperimentContext | None = None) -> dict[str, Any]:
        if bundle.kind != "evaluation":
            raise ValueError(f"EvaluationRunner requires an evaluation bundle, got {bundle.kind!r}.")
        traces: list[dict[str, Any]] = []
        for record in bundle.records:
            prompt_text = "\n".join(m.content for m in record.prompt)
            prediction = self.generator(prompt_text)
            verdict = self.verifier.verify(prediction, {"scenario": record.scenario})
            traces.append(
                {
                    "id": record.id,
                    "suite": record.suite,
                    "scenario_family_id": record.scenario_family_id,
                    "prediction": prediction,
                    "verdict": verdict.model_dump(mode="json"),
                }
            )
        summary = {
            "dataset_fingerprint": bundle.fingerprint,
            "verifier": {"name": self.verifier.name, "version": self.verifier.version},
            "suites": {},
        }
        by_suite: dict[str, list[dict[str, Any]]] = {}
        for trace in traces:
            by_suite.setdefault(trace["suite"], []).append(trace)
        for suite, suite_traces in sorted(by_suite.items()):
            summary["suites"][suite] = summarize_suite(suite_traces)

        if context is not None:
            path = context.paths.artifact("evaluation.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for trace in traces:
                    handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
            context.register_artifact("evaluation.jsonl", ArtifactKind.EVALUATION)
            context.write_json_artifact(
                "evaluation_summary.json", summary, ArtifactKind.EVALUATION
            )
        return {"summary": summary, "traces": traces}

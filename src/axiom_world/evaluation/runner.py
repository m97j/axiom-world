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
    def __init__(
        self,
        verifier: Verifier,
        generator: Callable[[str], str] | None = None,
        batch_generator: Callable[[list[str]], list[str]] | None = None,
        batch_size: int = 16,
    ) -> None:
        if generator is None and batch_generator is None:
            raise ValueError("EvaluationRunner needs a generator or a batch_generator.")
        self.verifier = verifier
        self.generator = generator
        self.batch_generator = batch_generator
        self.batch_size = batch_size

    def _generate_all(self, prompts: list[str]) -> list[str]:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            tqdm = lambda x, **k: x  # noqa: E731

        if self.batch_generator is not None:
            outputs: list[str] = []
            batches = [
                prompts[i : i + self.batch_size]
                for i in range(0, len(prompts), self.batch_size)
            ]
            for batch in tqdm(batches, desc="generate(batched)"):
                outputs.extend(self.batch_generator(batch))
            return outputs
        return [self.generator(p) for p in tqdm(prompts, desc="generate")]

    def run(self, bundle: DataBundle, context: ExperimentContext | None = None) -> dict[str, Any]:
        if bundle.kind != "evaluation":
            raise ValueError(f"EvaluationRunner requires an evaluation bundle, got {bundle.kind!r}.")
        traces: list[dict[str, Any]] = []
        prompts = ["\n".join(m.content for m in record.prompt) for record in bundle.records]
        predictions = self._generate_all(prompts)
        for record, prediction in zip(bundle.records, predictions, strict=True):
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
            # Suite-scoped trace file so multi-suite eval runs don't overwrite
            # each other; run_evaluation.py writes the combined summary itself.
            first_suite = traces[0]["suite"] if traces else "empty"
            trace_name = f"evaluation_{first_suite}.jsonl"
            path = context.paths.artifact(trace_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for trace in traces:
                    handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
            context.register_artifact(trace_name, ArtifactKind.EVALUATION)
        return {"summary": summary, "traces": traces}

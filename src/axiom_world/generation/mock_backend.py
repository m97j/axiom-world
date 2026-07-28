"""Deterministic mock backend — contract tests without GPU/vLLM."""
from __future__ import annotations

from collections.abc import Callable

from axiom_world.generation.backend import Candidate, GenerationBackend, GenerationRequest


class DeterministicMockBackend(GenerationBackend):
    name = "mock"

    def __init__(self, responder: Callable[[GenerationRequest, int], str]) -> None:
        self._responder = responder

    def generate(self, requests: list[GenerationRequest]) -> list[Candidate]:
        candidates: list[Candidate] = []
        for request in requests:
            for index in range(request.n):
                candidates.append(
                    Candidate(
                        candidate_id=f"{request.request_id}-c{index}",
                        request_id=request.request_id,
                        source_record_id=request.source_record_id,
                        text=self._responder(request, index),
                        index=index,
                        backend=self.name,
                    )
                )
        return candidates

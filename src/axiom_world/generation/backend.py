"""Generation backend abstraction.

vLLM specifics never leak past this boundary. Backends are constructed by
name; the vLLM backend imports vllm lazily so training/eval environments
without vLLM can still import the package (the previous snapshot's failure
mode was eager imports crossing session boundaries).
"""
from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from axiom_world.core.errors import AxiomError


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    source_record_id: str
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict, description="Verifier context (scenario).")
    n: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    request_id: str
    source_record_id: str
    text: str
    index: int
    finish_reason: str = "stop"
    completion_tokens: int | None = None
    backend: str = "unknown"
    model_id: str | None = None


class GenerationBackend(abc.ABC):
    name: str = "backend"

    @abc.abstractmethod
    def generate(self, requests: list[GenerationRequest]) -> list[Candidate]:
        ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class VLLMBackend(GenerationBackend):
    """Offline vLLM generation. Constructed only in dedicated generation sessions."""

    name = "vllm"

    def __init__(self, model_id: str, revision: str | None = None, **engine_kwargs: Any) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise AxiomError(
                "vLLM is not installed. Install requirements/vllm.lock.txt in a "
                "dedicated generation session (never alongside training)."
            ) from exc
        self.model_id = model_id
        self._llm = LLM(model=model_id, revision=revision, **engine_kwargs)

    def generate(self, requests: list[GenerationRequest]) -> list[Candidate]:  # pragma: no cover
        from vllm import SamplingParams

        candidates: list[Candidate] = []
        for request in requests:
            params = SamplingParams(
                n=request.n,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )
            outputs = self._llm.generate([request.prompt], params)
            for output in outputs:
                for index, completion in enumerate(output.outputs):
                    candidates.append(
                        Candidate(
                            candidate_id=f"{request.request_id}-c{index}",
                            request_id=request.request_id,
                            source_record_id=request.source_record_id,
                            text=completion.text,
                            index=index,
                            finish_reason=completion.finish_reason or "stop",
                            completion_tokens=len(completion.token_ids),
                            backend=self.name,
                            model_id=self.model_id,
                        )
                    )
        return candidates

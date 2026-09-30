"""Abstract interface between the pipeline and any chatbot under evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class TargetResponse:
    """Everything the pipeline needs to observe about one chatbot answer."""

    answer: str
    contexts: list[str] = field(default_factory=list)       # retrieved chunks (post-rerank)
    sources: list[str] = field(default_factory=list)        # citations / doc ids
    retrieved_raw: list[dict] = field(default_factory=list)  # pre-rerank candidates with scores
    reranked: list[dict] = field(default_factory=list)      # post-rerank order with scores
    prompt: str = ""                                         # final prompt sent to the LLM
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


@runtime_checkable
class TargetSystem(Protocol):
    """A RAG chatbot with role-based access control."""

    def answer(self, question: str, user_role: str) -> TargetResponse:
        """Answer ``question`` on behalf of a user holding ``user_role``."""
        ...


STAGE_METHODS = ("retrieve", "rerank", "build_prompt", "generate", "assemble")


def is_staged(target: object) -> bool:
    """True if the target exposes per-stage methods so the pipeline can trace each stage as a span.

    Staged contract: ``retrieve(q, role) -> list[dict]``, ``rerank(q, cands) -> list[dict]``,
    ``build_prompt(q, reranked) -> str``, ``generate(prompt) -> LLMCompletion`` and
    ``assemble(reranked, raw, prompt, completion) -> TargetResponse``.
    """
    return all(callable(getattr(target, m, None)) for m in STAGE_METHODS)
